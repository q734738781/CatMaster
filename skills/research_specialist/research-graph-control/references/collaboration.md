# Collaborating on an active research question

## Discuss evidence across branches

Assess discussion against the current user objective and existing Methods and
Results. Discussion alone does not authorize new execution or override a pause.
New user instructions can authorize further work after a completed stage; reuse
the earlier findings without treating its completion flag as a permanent stop.

Use `check_async_task` to inspect a relevant task's full brief/result. Shared graph
access permits reading peers, not steering or cancelling them. Independent
questions or replications may use the same source material.

Use `post_research_message` for a consequential overlap, tentative finding, question
or correction. A new topic has a title; target a task with `target_task_id`, and reply
using the returned message ID in `reply_to`. Only Research/Persistent Research
branches accept targeted notices; leave `target_task_id` empty for other shared
discussion. Include relevant scientific nodes and refs. Messages are shared
discussion, not authoritative Results.

Full messages are queryable through `research_discussions`; use the query tool's
column definitions, a relevant node/topic/target filter and normal SQL pagination.
A discussion can be referenced with `ref_kind="message"` and its exact message ID.
Do not broadcast histories, post after every tool or poll for replies. Targeted
notices arrive at the recipient's next ordinary model call; they do not interrupt
a tool or wake an idle task. Continue independent work while awaiting a reply.

## Correct or continue an owned child

For a substantive follow-up, use `update_async_task` on the appropriate existing
child with its source refs, question, authority and stopping condition. Continue
the owning research branch for nested work. Check existing follow-ups first.

- Use `strategy="interrupt"` to retract a source claim, method, scope or premise
  the child is currently using. Supply the correction and preserve unaffected work.
- Use `enqueue` for additional evidence or a later question that leaves current
  work valid. It arrives after the current turn and synchronous children finish.
- Interrupting an agent does not cancel its remote jobs; avoid duplicate execution.

## Record a scientific decision

Record consequential answers as replies. When marking the root's decision, set
`resolves_message_id` to the replied-to message. `addressed` means an actual answer
or correction; `deferred` explains why and what would change the decision;
`follow_up` references an already accepted investigation and its returned task ID.
Assignment alone does not settle a scientific question. Do not require replies to
all messages or create work merely to clear the discussion.

## Change this task's cost

Use the actual chosen method to set `task_cost`. Once it changes, call the exposed
`set_research_task_cost` alone after prior work returns and wait for admission.
Keep the higher cost while that work remains active. Synchronous delegates share
the task's slot; independent async tasks acquire their own. Capacity is not scientific
value or authorization, and a full expensive tier need not block independent low-cost
research. Remote-job scheduling remains the execution system's responsibility.
