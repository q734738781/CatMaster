"""Rebuildable display of native async run messages, including nested workers.

This projection never writes graph state or controls runs. The native stream
and checkpoint messages are authoritative; the existing WebUI message store
and event broker provide pagination and browser reconnects.
"""
from langchain_core.messages import AIMessage, AIMessageChunk, message_chunk_to_message

from .run_projection import _content_fragments, _delta_offset, _plain_dict, _safe_token, _tool_calls, _tool_input
from .thread_models import MessagePart, ThreadMessage


class AsyncActivityProjection:
    def __init__(self, *, store, broker, thread_id, run_id, source, on_update=None):
        self.store, self.broker = store, broker
        self.thread_id, self.run_id, self.source = thread_id, run_id, source
        self.on_update = on_update
        self.chunks = {}
        self.message_ids = {}
        self.invocations = {}
        self.native_ids = {}
        self.tool_owners = {}
        self.sources = {}
        self.replay_until = ''
        self._state_updates = None
        for message in store.list_messages(thread_id, run_id=run_id):
            if not message.meta.get('native_message_id'):
                continue
            namespace = tuple(message.meta.get('namespace', []))
            ids = message.meta.get('native_message_ids') or [message.meta['native_message_id']]
            self.native_ids[message.id] = list(ids)
            for native_id in ids:
                self.message_ids[namespace, native_id] = message.id
            invocation = message.meta.get('stream_invocation')
            if invocation and message.meta.get('partial'):
                self.invocations[namespace, invocation] = message.id
            for part in message.parts:
                if part.meta.get('tool_call_id'):
                    self.tool_owners[part.meta['tool_call_id']] = (message.id, part.id)

    def restore(self, data):
        """Show the snapshot once; retain replay for transcript reconstruction.

        Native resumable streams include root ``values`` with message IDs. The
        last checkpoint message is our replay boundary, including when earlier
        nested-worker messages are absent from that root checkpoint.
        """
        self.state(data)
        messages = _plain_dict(data).get('messages') or []
        self.replay_until = str(_plain_dict(messages[-1]).get('id') or '') if messages else ''

    def _update(self, text, source, tool):
        if self.replay_until or not self.on_update:
            return
        if self._state_updates is not None:
            self._state_updates[:] = [(text, source, tool)]
        else:
            self.on_update(text, source, tool)

    def _save(self, message):
        old = self.store.get_message(self.thread_id, message.id)
        if old:
            # A resumed native stream can replay a prefix already in the UI.
            if message.meta.get('partial'):
                previous = {part.id: part for part in old.parts}
                message.parts = [previous[part.id] if (
                    part.id in previous and part.type in {'text', 'reasoning'}
                    and previous[part.id].text.startswith(part.text)
                ) else part for part in message.parts]
                if old.status == 'completed':
                    message.status = old.status
            if old.parts == message.parts and old.status == message.status and old.meta == message.meta:
                return
            saved = self.store.update_message(self.thread_id, message.id,
                parts=message.parts, status=message.status, meta=message.meta)
        else:
            self.store.append_message(message)
            saved = message
        if old and message.meta.get('partial') and old.meta.get('source') == saved.meta.get('source'):
            previous = {part.id: part for part in old.parts}
            for part in saved.parts:
                prior = previous.get(part.id)
                if prior == part:
                    continue
                if prior and part.type in {'text', 'reasoning'} and part.text.startswith(prior.text):
                    delta = part.text[len(prior.text):]
                    if delta:
                        self.broker.emit(self.thread_id,
                            'reasoning.delta' if part.type == 'reasoning' else 'message.delta',
                            message_id=saved.id, status=saved.status,
                            data={'part_id': part.id, 'delta': delta,
                                  'text_offset': _delta_offset(saved, part.id, delta), 'run_id': self.run_id})
                else:
                    self.broker.emit(self.thread_id, 'message.part.updated' if prior else 'message.part.created',
                        message_id=saved.id, status=saved.status, data={'part': part.model_dump(mode='json')})
            return
        self.broker.emit(self.thread_id, 'message.updated' if old else 'message.created',
            message_id=saved.id, status=saved.status,
            data={'message': saved.model_dump(mode='json'), 'run_id': self.run_id})

    def process(self, chunk):
        event = _plain_dict(chunk)
        mode, *suffix = str(event.get('type') or event.get('event') or '').split('|')
        namespace = tuple(event.get('ns') or suffix)
        data = event.get('data')
        if mode in {'messages', 'messages-tuple'} and isinstance(data, (tuple, list)) and len(data) == 2:
            self.message(_plain_dict(data[0]), _plain_dict(data[1]), namespace)
        elif mode in {'updates', 'values'}:
            self.state(data, namespace)
            if not namespace and self.replay_until and any(
                str(_plain_dict(message).get('id') or '') == self.replay_until
                for message in (_plain_dict(data).get('messages') or [])
            ):
                self.replay_until = ''

    def state(self, data, namespace=()):
        # A values event contains the whole conversation, not a sequence of
        # current activities. Save every message but publish only the latest
        # new update after the complete snapshot has been applied.
        updates = []
        self._state_updates = updates
        try:
            self._state(data, namespace)
        finally:
            self._state_updates = None
        if updates:
            self._update(*updates[-1])

    def _state(self, data, namespace=()):
        payload = _plain_dict(data)
        messages = payload.get('messages')
        if isinstance(messages, list):
            for message in messages:
                self.message(_plain_dict(message), {}, namespace)
        else:
            for value in payload.values():
                if isinstance(value, dict) and isinstance(value.get('messages'), list):
                    self._state(value, namespace)

    def message(self, raw, metadata, namespace=()):
        source = str(metadata.get('lc_agent_name') or self.sources.get(namespace)
                     or (self.source if not namespace else 'Worker'))
        self.sources[namespace] = source
        call_source = metadata.get('lc_source') or _plain_dict(raw.get('additional_kwargs')).get('lc_source')
        if call_source == 'summarization' or metadata.get('lc_internal_call'):
            return
        kind = str(raw.get('type') or raw.get('role') or '').lower()
        if kind in {'human', 'user'} and not namespace and raw.get('id'):
            message_id = 'activity_' + _safe_token(str(raw['id']), 'input')
            self._save(ThreadMessage(id=message_id, thread_id=self.thread_id, role='user', status='completed',
                parts=[MessagePart(id=message_id + '_text', type='text', text=_content_fragments(raw.get('content'))[0])],
                meta={'run_id': self.run_id, 'source': 'Task instructions'}))
            return
        if kind in {'tool', 'toolmessage', 'toolmessagechunk'}:
            self.tool_result(raw)
            return
        if kind not in {'ai', 'assistant', 'aimessage', 'aimessagechunk'}:
            return
        native_id = str(raw.get('id') or '')
        if not native_id:
            return
        is_chunk = kind == 'aimessagechunk'
        partial = is_chunk and raw.get('chunk_position') != 'last'
        # LangGraph's messages stream supplies the model task namespace. In
        # langchain-openai Responses streams, response.created has a provider
        # ID while later chunks can have an lc_run-- ID; chunk aggregation
        # prefers the provider ID again. Retain every alias, including empty
        # opening chunks, and keep one display ID for this invocation. State
        # updates omit stream metadata, so they resolve through those aliases.
        invocation = str(metadata.get('langgraph_checkpoint_ns') or '')
        key = (namespace, invocation)
        message_id = self.message_ids.get((namespace, native_id))
        if not message_id and invocation:
            message_id = self.invocations.get(key)
        if not message_id and not is_chunk:
            owners = {self.tool_owners[call['id']][0] for call in _tool_calls(raw)
                      if call.get('id') in self.tool_owners}
            if len(owners) == 1:
                message_id = owners.pop()
        message_id = message_id or 'activity_' + _safe_token(native_id, 'message')
        self.message_ids[namespace, native_id] = message_id
        aliases = self.native_ids.setdefault(message_id, [])
        if native_id not in aliases:
            aliases.append(native_id)
        if invocation:
            self.invocations[key] = message_id
        old = self.store.get_message(self.thread_id, message_id)
        # A replayed token prefix must not turn a full saved message back into
        # a partial one or cause its tool name to be announced again.
        if is_chunk and old and not old.meta.get('partial'):
            self.invocations = {k: v for k, v in self.invocations.items() if v != message_id}
            return
        if is_chunk:
            chunk = AIMessageChunk(**raw)
            self.chunks[message_id] = self.chunks[message_id] + chunk if message_id in self.chunks else chunk
            normalized = message_chunk_to_message(self.chunks[message_id])
            raw = normalized.model_dump(mode='json')
        else:
            normalized = AIMessage(**raw)
        if not partial:
            self.chunks.pop(message_id, None)
            self.invocations = {k: v for k, v in self.invocations.items() if v != message_id}
        if old and not metadata and old.meta.get('source'):
            source = old.meta['source']
        text, reasoning = _content_fragments(normalized.content_blocks)
        parts = []
        for part_type, content in [('reasoning', reasoning), ('text', text)]:
            if content:
                parts.append(MessagePart(id=f'{message_id}_{part_type}', type=part_type,
                    text=content, status='streaming' if partial else 'completed',
                    meta={'source': source, 'agent_name': source}))
        previous = {part.id: part for part in old.parts} if old else {}
        calls = _tool_calls(raw)
        for call in calls:
            call_id = str(call.get('id') or '')
            if not call_id:
                continue
            part_id = 'tool_' + _safe_token(call_id, 'call')
            self.tool_owners[call_id] = (message_id, part_id)
            prior = previous.get(part_id)
            meta = {**(prior.meta if prior else {}), 'tool_call_id': call_id,
                'tool': call.get('name', ''), 'input': _tool_input(call.get('args')),
                'source': source, 'agent_name': source}
            parts.append(MessagePart(id=part_id, type='tool-call',
                status=prior.status if prior else 'running', meta=meta))
        if not parts:
            return
        self._save(ThreadMessage(id=message_id, thread_id=self.thread_id, role='assistant',
            status='streaming' if partial or any(part.status == 'running' for part in parts) else 'completed',
            parts=parts, meta={'run_id': self.run_id, 'source': source,
                'native_message_id': normalized.id, 'native_message_ids': list(aliases),
                'stream_invocation': invocation or (old.meta.get('stream_invocation', '') if old else ''),
                'namespace': list(namespace), 'partial': partial}))
        if not partial and (old is None or old.meta.get('partial')):
            progress = next((call for call in calls if call.get('name') == 'notify_progress'), None)
            if progress:
                args = _tool_input(progress.get('args'))
                self._update(str(args.get('summary') or ''), source, 'notify_progress')
            elif calls:
                self._update('', source, str(calls[-1].get('name') or ''))
            elif text.strip():
                self._update(text, source, '')

    def tool_result(self, raw):
        owner = self.tool_owners.get(str(raw.get('tool_call_id') or ''))
        if not owner:
            return
        message = self.store.get_message(self.thread_id, owner[0])
        if not message:
            return
        for part in message.parts:
            if part.id == owner[1]:
                part.meta['output'] = raw.get('content')
                part.status = 'failed' if raw.get('status') == 'error' else 'completed'
        message.meta['partial'] = False
        message.status = 'streaming' if any(part.status == 'running' for part in message.parts) else 'completed'
        self._save(message)

    def finish(self, status):
        for message in self.store.list_messages(self.thread_id, run_id=self.run_id):
            if message.meta.get('run_id') != self.run_id or message.status != 'streaming':
                continue
            message.status = 'completed' if status == 'success' else 'interrupted' if status in {'interrupted', 'cancelled'} else 'failed'
            message.meta['partial'] = False
            for part in message.parts:
                if part.status in {'running', 'streaming'}:
                    part.status = message.status
            self._save(message)
