---
name: general-remote-scripts
description: Use when the user requests a handwritten remote implementation or a needed operation is not covered by a dedicated registered task.
license: project-local
---

# General remote scripts

Use an existing dedicated task when it covers the requested operation. Its
batching, input handling and recovery are already implemented. An explicit
request to handwrite the calculation, or an operation outside those tasks,
calls for `general_execute`.

Read `get_remote_task_spec(task_name="general_execute")`. It lists available
environments and their software descriptions together with the submission
parameters; a separate resource query is unnecessary. Choose the environment
that provides the required calculator or program. `get_avail_resources` offers
the same environment discovery when needed. A generic GPU environment does not
imply that every MLFF provider is installed.

When implementation details are needed, `export_builtin_tool_source` exports
the selected tool and its static CatMaster imports recursively in one call.
Read or search the returned source tree; files retain their original package
paths and imports. Use `module_name` for a known additional module, or
`include_dependencies=false` for a single file. This is a reading reference;
dynamic imports, data assets and third-party libraries are not collected.

Write the script and its inputs inside a prepared stage. Submit with
`task_name="general_execute"` and `template_overrides` containing:

- `environment`: a name from the environment list.
- `entrypoint`: the `.py` or `.sh` script relative to that stage.
- `argv`: optional literal arguments as a list of strings.

The script runs with the stage as cwd. Python comes from the selected
environment; the platform activates it before launch. Do not search conda
environments or embed environment activation paths when the catalog already
provides a suitable environment. Investigate a concrete import or execution
failure only after submission returns it.

Use `remote_submission` for one stage. For independent stages sharing the same
environment, relative script path and argv, use `remote_submission_batch`.
It selects first-level children by default; `stage_paths` selects relative paths,
including nested stages, without copying them. Keep selected stages independent
and nonoverlapping, with separate inputs and outputs. Do not pass resource or machine overrides; the
selected environment supplies them.

While submission is pending, wait for its return. On success use its outputs.
After a failure, use the existing receipt recovery guidance if remote state is
uncertain; do not assume a transport failure cancelled the job. Changing the
execution environment does not authorize changing the scientific model.

Return the requested scientific result, reusable script and canonical output.
Do not turn environment selection or recovery records into an extra scientific
report.
