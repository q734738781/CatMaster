"""Join prepared HTTP requests and raw provider responses for cache experiments."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

if len(sys.argv) != 2:
    raise SystemExit("Usage: analyze_codex_cache_probe.py EXPERIMENT_DIRECTORY")

root = Path(sys.argv[1])
def digest(value, ordered=True):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=not ordered, separators=(',', ':')).encode()).hexdigest()

all_rows = []
summaries = []
for arm in sorted(p for p in root.iterdir() if p.is_dir() and (p/'observability.sqlite').exists()):
    with sqlite3.connect(f'file:{arm}/observability.sqlite?mode=ro', uri=True) as db:
        response_rows = [json.loads(r[0]) for r in db.execute('SELECT payload_json FROM observation_events WHERE name="LLM_PROVIDER_RESPONSE" ORDER BY id')]
        tool_counts = dict(db.execute("SELECT name,count(*) FROM observation_events WHERE category='tool' GROUP BY name"))
    responses = {r['callback_run_id']: r['event']['response'] for r in response_rows}
    previous = previous_response = None
    after_summary = False
    last_normal = last_normal_response = None
    rows = []
    for f in sorted(arm.glob('request_*.json')):
        q = json.loads(f.read_text()); body = json.loads(q['body']); response = responses.get(q['callback_run_id'])
        if not response: continue
        usage = response.get('usage') or {}; inp = usage.get('input_tokens', 0); cached = (usage.get('input_tokens_details') or {}).get('cached_tokens', 0)
        is_summary = q['metadata'].get('lc_source') == 'summarization'
        comparison = previous if is_summary else last_normal
        comparison_response = previous_response if is_summary else last_normal_response
        same_instructions = comparison is not None and body.get('instructions') == comparison.get('instructions')
        same_tools = comparison is not None and body.get('tools') == comparison.get('tools')
        prefix = comparison is not None and body.get('input', [])[:len(comparison.get('input', []))] == comparison.get('input', [])
        prior_input = (comparison_response or {}).get('usage', {}).get('input_tokens', 0)
        prior_output = (comparison_response or {}).get('usage', {}).get('output_tokens', 0)
        row = {'arm': arm.name, 'call':q['call'], 'summary': is_summary, 'after_summary': after_summary and not is_summary, 'model':response.get('model'),
               'input_tokens':inp, 'cached_tokens':cached, 'output_tokens':usage.get('output_tokens',0),
               'cache_percent':round(cached/inp*100,2) if inp else None,
               'prior_input_tokens':prior_input,
               'cached_vs_prior_input_percent':round(cached/prior_input*100,2) if prior_input else None,
               'input_items':len(body.get('input',[])),
               'tool_outputs':sum(i.get('type')=='function_call_output' for i in body.get('input',[]) if isinstance(i,dict)),
               'tool_count':len(body.get('tools',[])),
               'tool_order':[t.get('name') or t.get('type') for t in body.get('tools',[])],
               'instructions_hash':digest(body.get('instructions')),
               'tools_hash':digest(body.get('tools',[])),
               'tools_hash_key_sorted':digest(body.get('tools',[]),ordered=False),
               'same_instructions':same_instructions, 'same_tools':same_tools, 'append_only_input':prefix,
               'key':body.get('prompt_cache_key'), 'session_id':q['headers'].get('session-id'),
               'thread_id':q['headers'].get('thread-id'), 'returned_key':response.get('prompt_cache_key'),
               'full_miss_on_unchanged_prefix':cached==0 and same_instructions and same_tools and prefix,
               'web_search_used':any(i.get('type')=='web_search_call' for i in response.get('output',[]) if isinstance(i,dict)),
               'response_id':response.get('id'), 'retention':response.get('prompt_cache_retention')}
        rows.append(row); previous,previous_response=body,response
        after_summary = is_summary
        if not is_summary: last_normal,last_normal_response=body,response
    if not rows: continue
    checkpoint = arm / 'checkpoint_messages.json'
    messages = json.loads(checkpoint.read_text()) if checkpoint.exists() else []
    tool_outcomes = [{'name': m.get('name'), 'status': m.get('status')}
                     for m in messages if m.get('type') == 'tool']
    def has_web_call(value):
        if isinstance(value, dict):
            return value.get('type') == 'web_search_call' or any(has_web_call(v) for v in value.values())
        return isinstance(value, list) and any(has_web_call(v) for v in value)
    all_rows.extend(rows)
    normal = [r for r in rows if not r['summary']]
    eligible = [r for r in normal if r['same_instructions'] and r['same_tools'] and r['append_only_input']]
    stats = {'arm':arm.name, 'calls':len(rows), 'summary_calls':sum(r['summary'] for r in rows),
             'input_range':[normal[0]['input_tokens'],normal[-1]['input_tokens']],
             'input_tokens':sum(r['input_tokens'] for r in rows),'cached_tokens':sum(r['cached_tokens'] for r in rows),
             'output_tokens':sum(r['output_tokens'] for r in rows),
             'cache_percent_all':round(sum(r['cached_tokens'] for r in rows)/sum(r['input_tokens'] for r in rows)*100,2),
             'stable_keys':len({r['key'] for r in rows})==1,
             'identity_matches':all(r['key']==r['session_id']==r['returned_key'] for r in rows),
             'normal_instructions_stable':len({r['instructions_hash'] for r in normal})==1,
             'normal_tools_stable':len({r['tools_hash'] for r in normal})==1,
             'append_only_eligible_calls':len(eligible),
             'eligible_full_miss_calls':[r['call'] for r in eligible if r['cached_tokens']==0],
             'tool_observations':tool_counts, 'tool_outcomes': tool_outcomes,
             'web_search_called_in_checkpoint': has_web_call(messages)}
    summaries.append(stats)
(root/'analysis.json').write_text(json.dumps({'summaries':summaries,'calls':all_rows},indent=2,ensure_ascii=False))
for stats in summaries: print(json.dumps(stats,ensure_ascii=False))
for row in all_rows: print(json.dumps({k:row[k] for k in ['arm','call','summary','input_tokens','cached_tokens','cache_percent','cached_vs_prior_input_percent','append_only_input','same_instructions','same_tools','full_miss_on_unchanged_prefix']},ensure_ascii=False))
