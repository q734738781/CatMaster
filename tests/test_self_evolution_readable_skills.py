from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from catmaster.runtime.self_evolution import agents
from catmaster.runtime.self_evolution.effective import EffectiveSkillsManager, candidate_version
from catmaster.runtime.self_evolution.models import LearningCandidate, ProposerResult, ReflectionBatch, ReflectionResult, ReviewerResult
from catmaster.runtime.self_evolution.storage import SelfEvolutionStore, hash_tree

TARGET = 'materials_worker/readable-method'
PATH = f'/current/skills/{TARGET}/SKILL.md'


def _skill(path: Path, marker: str) -> None:
    (path / 'references').mkdir(parents=True)
    (path / 'SKILL.md').write_text(
        f'---\nname: {path.name}\ndescription: {marker}\nallowed-tools: unavailable_advisory_tool\n---\n'
        f'{marker}\nRead references/details.md for the method.\n'
    )
    (path / 'references/details.md').write_text(f'{marker}_REFERENCE\n')


def _manager(tmp_path: Path) -> EffectiveSkillsManager:
    repo = tmp_path / 'repo'
    _skill(repo / 'skills' / TARGET, 'BASE_BODY')
    (repo / 'skills' / TARGET / 'base_only.txt').write_text('removed by selected revision')
    _skill(repo / 'skills/materials_worker/disabled-method', 'DISABLED_BODY')
    shared = repo / 'skills/materials_worker/shared_support'
    shared.mkdir()
    (shared / 'notes.md').write_text('SHARED_REFERENCE\n')
    store = SelfEvolutionStore(tmp_path / 'workspace')
    candidate = LearningCandidate(
        candidate_id='sec_readable', project_id=store.project_id,
        run_id='source', thread_id='source-thread', action='skill',
        group='materials_worker', name='readable-method', revision=1,
    )
    source = store.revision_dir(candidate.candidate_id, 1) / 'proposed' / TARGET
    _skill(source, 'SELECTED_BODY')
    candidate.bundle_hash = hash_tree(source)
    store.write_candidate(candidate)
    store.write_active_skills({'skills': {
        TARGET: {'enabled': True, 'selected_version': candidate_version(candidate), 'update_policy': 'pinned'},
        'materials_worker/disabled-method': {'enabled': False, 'selected_version': 'base'},
    }})
    assert store.compare_and_swap_memory(expected_hash=store.memory_hash(), new_text='WORKSPACE_GUIDANCE\n')[0]
    return EffectiveSkillsManager(store, repo_root=repo)


def test_catalog_paths_follow_selected_versions_and_keep_support_files(tmp_path):
    manager = _manager(tmp_path)
    with agents._reflection_guidance(manager) as (current, catalog):
        entries = []
        cursor = ''
        while True:
            page = json.loads(catalog.invoke({'after': cursor, 'limit': 1}))
            entries.extend(page['skills'])
            cursor = page['next_cursor']
            if not cursor:
                break
        by_target = {row['target']: row for row in entries}
        assert by_target[TARGET]['path'] == PATH
        assert by_target[TARGET]['selected_version'] == 'sec_readable@r0001'
        assert by_target['materials_worker/disabled-method']['enabled'] is False
        assert by_target['/memories/AGENTS.md']['path'] == '/current/AGENTS.md'
        for entry in entries:
            assert (current / entry['path'].removeprefix('/current/')).is_file()
        assert 'SELECTED_BODY' in (current / 'skills' / TARGET / 'SKILL.md').read_text()
        assert not (current / 'skills' / TARGET / 'base_only.txt').exists()
        assert (current / 'skills/materials_worker/shared_support/notes.md').read_text() == 'SHARED_REFERENCE\n'
        assert (current / 'AGENTS.md').read_text() == 'WORKSPACE_GUIDANCE\n'
        schema = convert_to_openai_tool(catalog)['function']['parameters']
        assert '"type": "null"' not in json.dumps(schema)
        assert set(schema['properties']) == {'after', 'limit'}
        # The selection may change between model calls; the in-flight catalog
        # must continue to label the already staged revision correctly.
        manager.update_target(TARGET, actor='test', selected_version='base')
        page = json.loads(catalog.invoke({}))
        assert next(row for row in page['skills'] if row['target'] == TARGET)['selected_version'] == 'sec_readable@r0001'
        assert 'SELECTED_BODY' in (current / 'skills' / TARGET / 'SKILL.md').read_text()
    assert not current.exists()
    with agents._reflection_guidance(manager) as (current, catalog):
        assert 'BASE_BODY' in (current / 'skills' / TARGET / 'SKILL.md').read_text()
        assert next(row for row in json.loads(catalog.invoke({}))['skills'] if row['target'] == TARGET)['selected_version'] == 'base'


def test_disabled_selected_revision_is_readable_without_activation(tmp_path):
    manager = _manager(tmp_path)
    manager.update_target(TARGET, actor='test', enabled=False)
    before = manager.store.read_active_skills()
    with agents._reflection_guidance(manager) as (current, catalog):
        entry = next(row for row in json.loads(catalog.invoke({}))['skills'] if row['target'] == TARGET)
        assert entry['enabled'] is False
        assert 'SELECTED_BODY' in (current / 'skills' / TARGET / 'SKILL.md').read_text()
    assert manager.store.read_active_skills() == before


def test_draft_only_target_does_not_advertise_a_nonexistent_body(tmp_path):
    manager = _manager(tmp_path)
    draft = LearningCandidate(candidate_id='sec_draft_only', project_id=manager.store.project_id,
        run_id='source', thread_id='thread', action='skill', group='materials_worker', name='new-method')
    _skill(manager.store.revision_dir(draft.candidate_id, 1) / 'proposed/materials_worker/new-method', 'DRAFT_BODY')
    manager.store.write_candidate(draft)
    with agents._reflection_guidance(manager) as (current, catalog):
        entry = next(row for row in json.loads(catalog.invoke({}))['skills'] if row['target'] == 'materials_worker/new-method')
        assert entry['path'] == '' and entry['unavailable_reason']
        assert not (current / 'skills/materials_worker/new-method').exists()


class ReadingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        result = super()._generate(messages, *args, **kwargs)
        if isinstance(messages[-1], ToolMessage) and messages[-1].name == 'query_effective_skills':
            entries = json.loads(messages[-1].content)['skills']
            path = next(row['path'] for row in entries if row['target'] == TARGET)
            # The following read uses the actual tool result, not an invented path.
            result.generations[0].message.tool_calls[0]['args']['file_path'] = path
        return result


def _call(name, args, ident):
    return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': ident}])


@pytest.mark.parametrize('stage', ['reflector', 'proposer', 'reviewer'])
def test_native_stage_and_investigator_read_full_selected_guidance(tmp_path, stage):
    manager = _manager(tmp_path)
    final = {
        'reflector': ReflectionBatch(items=[ReflectionResult(kind='no_change', rationale='Existing guidance covers this.')]),
        'proposer': ProposerResult(action='ignore', rationale='Existing guidance covers this.'),
        'reviewer': ReviewerResult(recommendation='reject', summary='Already covered.'),
    }[stage]
    first = (_call('query_effective_skills', {}, 'catalog') if stage == 'reflector'
             else _call('read_file', {'file_path': '/current/catalog.md'}, 'catalog'))
    model = ReadingModel(responses=[
        first,
        _call('read_file', {'file_path': PATH}, 'body'),
        _call('task', {'subagent_type': 'general-purpose', 'description': 'Read the relevant supporting file.'}, 'delegate'),
        _call('read_file', {'file_path': f'/current/skills/{TARGET}/references/details.md'}, 'child-reference'),
        AIMessage(content='The complete reference is present.'),
        _call('glob', {'path': '/current/skills/materials_worker', 'pattern': '**/*.md'}, 'find-support'),
        _call('grep', {'path': '/current/skills/materials_worker', 'pattern': 'SHARED_REFERENCE', 'output_mode': 'content'}, 'search-support'),
        _call('read_file', {'file_path': '/current/AGENTS.md'}, 'memory'),
        _call(type(final).__name__, final.model_dump(mode='json'), 'finish'),
    ])
    actor_class = agents.ReviewerAgent if stage == 'reviewer' else agents.ProposerAgent
    actor = actor_class(model=model, model_label='test', workspace=manager.store.workspace)
    before = manager.store.read_active_skills()
    if stage == 'reflector':
        result, meta = actor.reflect(trajectory_markdown='Inspect the relevant guidance.', skill_catalog='',
            prior_targets=[], trace_scope=SimpleNamespace(tools=lambda **kw: []), effective_skills=manager)
    else:
        root = agents.prepare_candidate_workspace(store=manager.store, candidate_id='sec_inspection', repo_root=manager.repo_root)
        if stage == 'proposer':
            result, meta = actor.propose(candidate_root=root)
        else:
            result, meta = actor.review(candidate_root=root, action='skill', group='materials_worker',
                name='readable-method', rationale='Check existing guidance.', validation={})
    assert result == final
    run = manager.store.workspace / 'metadata/runs' / meta['usage_run_id']
    with sqlite3.connect(run / 'observability.sqlite') as db:
        rows = [json.loads(row[0]) for row in db.execute("SELECT payload_json FROM observation_events WHERE name='TOOL_RAW_OUTPUT'")]
    assert all(row['tool_status'] == 'success' for row in rows)
    # Native task callbacks return plain text; filesystem/query callbacks
    # return ToolMessages. The shared projection covers both representations.
    contents = '\n'.join(row['projection']['content_text'] for row in rows)
    assert all(marker in contents for marker in ('SELECTED_BODY', 'SELECTED_BODY_REFERENCE', 'SHARED_REFERENCE', 'WORKSPACE_GUIDANCE'))
    child_reads = [row for row in rows if row['agent_name'].endswith('/general-purpose') and row['tool'] == 'read_file']
    assert len(child_reads) == 1 and 'SELECTED_BODY_REFERENCE' in child_reads[0]['raw_output']['content']
    assert manager.store.read_active_skills() == before
    assert not list((manager.store.root / 'agent_context').iterdir())


def test_proposer_edits_candidate_but_cannot_mutate_current_guidance(tmp_path):
    manager = _manager(tmp_path)
    root = agents.prepare_candidate_workspace(store=manager.store, candidate_id='sec_write_boundary', repo_root=manager.repo_root)
    final = ProposerResult(action='ignore', rationale='Boundary check only.')
    model = ReadingModel(responses=[
        _call('prepare_skill_candidate', {'group': 'materials_worker', 'name': 'readable-method'}, 'prepare'),
        _call('write_file', {'file_path': PATH, 'content': 'CORRUPTED'}, 'deny-write'),
        _call('edit_file', {'file_path': PATH, 'old_string': 'SELECTED_BODY', 'new_string': 'CORRUPTED', 'replace_all': True}, 'deny-edit'),
        _call('delete', {'file_path': f'/current/skills/{TARGET}'}, 'deny-delete'),
        _call('edit_file', {'file_path': f'/proposed/{TARGET}/SKILL.md', 'old_string': 'SELECTED_BODY', 'new_string': 'REVISED_BODY', 'replace_all': True}, 'edit-candidate'),
        _call(type(final).__name__, final.model_dump(mode='json'), 'finish'),
    ])
    actor = agents.ProposerAgent(model=model, model_label='test', workspace=manager.store.workspace)
    result, meta = actor.propose(candidate_root=root)
    assert result == final
    assert 'SELECTED_BODY' in (root / 'current/skills' / TARGET / 'SKILL.md').read_text()
    assert 'REVISED_BODY' in (root / 'proposed' / TARGET / 'SKILL.md').read_text()
    assert 'BASE_BODY' in (manager.repo_root / 'skills' / TARGET / 'SKILL.md').read_text()
    # Blocked tools may be short-circuited before tool callbacks. Inspect the
    # next model's actual input to confirm the agent receives explicit errors.
    run = manager.store.workspace / 'metadata/runs' / meta['usage_run_id']
    with sqlite3.connect(run / 'observability.sqlite') as db:
        requests = [json.loads(row[0]) for row in db.execute("SELECT payload_json FROM observation_events WHERE name='LLM_RAW_REQUEST'")]
    messages = [message for batch in requests[-1]['messages'] for message in batch]
    errors = [message for message in messages if message.get('type') == 'tool' and message.get('status') == 'error']
    assert {message['tool_call_id'] for message in errors} == {'deny-write', 'deny-edit', 'deny-delete'}
