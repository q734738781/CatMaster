# CatMaster Prompt Assets

Package-internal prompt assets live here. They are not copied to workspace
`files/.deepagents/`, DeepAgents memory, or skill roots.

The active prompt combines role instructions, shared runtime guidance, native
capability/skill/memory guidance, and any model-specific harness profile. Specialist
and worker role instructions are built in `catmaster/specialists/runtime.py`;
self-evolution roles retain their separate files and completion interfaces.

`bundles/runtime.yaml` composes user-priority and system-usage fragments through
`PromptCatalog` and `PromptRenderer`. Role builders include this shared guidance
for every model, including Astra and the self-evolution investigators.

`bundles/mimo.yaml` adds a short emphasis on current authority, explicit bindings,
historical completion and decision-relevant checks. The resolved model from the ordinary LLM
YAML selects it automatically. DeepAgents' `HarnessProfile` appends it to each
matching root or child, including self-evolution, regardless of which entrypoint
is built first. Other models retain their native harness profiles.

Bundles list fragment order; fragments hold stable, recipient-appropriate guidance.
Specific tasks, corrections and completion notices remain user-turn inputs.
Other assets, including the research bundle, are not used to replace active role
instructions. Plain model calls such as proposal checks, titles and summaries do
not use this DeepAgents composition.
