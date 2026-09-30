# Editable presentations

WritingSpecialist delegates PPTX creation, editing and reconstruction to
`presentation_worker`. This worker has the same native file operations, shell
execution and general-purpose delegation as other full workers, plus
`generate_figure` for image assets. Its `easyslides` skill supplies the presentation
workflow. The optional `presentation_worker` model role falls back to
`section_writer`; it can be bound independently under `agents` in `configs/llm.yaml`.

Titles, prose, tables and slide structure default to native editable PowerPoint
objects. EasySlides converts SVG text and shapes to DrawingML; the worker selects
`--only native` to deliver that deck. Individual illustrations or data plots can
remain image assets. Native charts and tables are used when their data or cells
must be editable. The worker renders the PPTX, inspects the slides, and repairs
visible layout problems before returning it.

For scientific decks, the shared `scientific-communication` skill guides evidence
selection and the relationship between pages and the spoken explanation. Progress
talks retain meaningful negative findings and research decisions; they need not
claim a finished manuscript contribution. Core results and comparisons belong on
the slides, not solely in speaker notes. Humanizer covers authored slide text and
scripts, while the plotting root supplies the validated Origin/NPG guidance for
direct data-figure work. An explicit user style or chart-editability requirement
takes precedence. Technical rendering checks do not establish scientific content
quality, and format-only reconstruction preserves the approved narrative.

Handoffs distinguish an explanatory talk, a technical briefing for informed
readers, and manuscript-based presentation requirements. Without an explicit
audience, a scientific talk assumes an advisor or experimental collaborator
unfamiliar with the project's computational theory and history. The slides explain
why the comparisons were needed and how the results bear on the scientific
question. Explicitly internal technical briefings may stay compact. A short
completion message does not imply a sparse deck or reduced scientific coverage.

Revision briefs preserve the target and scope of user feedback. Editorial choices
remain distinct from explicit requirements: criticism of isolated colored numbers
does not impose a monochrome theme. Method comparisons explain what discrepancies
mean for the scientific question, with the necessary conditions and reference
types retained. Shared qualifications accompany the comparison they govern;
they recur only where another claim would otherwise mislead.

WritingSpecialist owns editorial acceptance of the actual returned artifact. It
reads the authored content and rendered pages against the audience and explicit
feedback, reusing the worker's previews and completed scientific checks. A worker
completion message is not the acceptance decision. Material content or visual
problems go back to the responsible authoring worker as a bounded correction;
Writing inspects the affected result before delivery. This does not add a review
agent, fixed revision count or separate quality report, and optional design
alternatives do not keep an otherwise suitable artifact running indefinitely.

## Installation

[EasySlides](https://github.com/Rimagination/easyslides) is a repository-backed
skill/runtime, not a PyPI distribution. Its upstream installation requires the
scripts, templates and reference files as well as Python dependencies. CatMaster
uses the upstream production-bundle builder and retains its MIT license.

For a source checkout, complete installation before starting CatMaster:

```bash
conda env create -f requirements/pc-conda.yml
conda activate catmaster
python scripts/install_easyslides.py
```

For an existing environment, use `conda env update -n catmaster -f
requirements/pc-conda.yml` in place of environment creation. Maintainers test
dependency changes in `catmaster-dev` before applying the tested pins to the main
environment.

The environment file contains the Python dependencies for native PPTX export,
SVG raster rendering, document import, the SVG editor and narration. Cairo comes
from conda. Linux slide preview also requires LibreOffice, Poppler and suitable
fonts; for Debian/Ubuntu:

```bash
sudo apt-get install libreoffice-impress poppler-utils fonts-noto-cjk
```

Image generation needs the configured OpenRouter key; see
[figure generation](figure_generation.md). Narration and online source acquisition
also use their respective network services. Experimental upstream capabilities
can have extra dependencies described in their own documentation.

`scripts/install_easyslides.py` installs a pinned upstream revision into
`third_party/easyslides/`. It uses upstream's distributable layout, including
scripts, workflows, skills, templates, references and assets. The generated
directory is ignored by Git. Re-running the installer reuses the matching local
installation without downloading it. A failed download or build leaves the
previous installation in place.

Both `deploy_runtime.sh` and `package_remote_deploy.sh` prepare this bundle and
include it in the runtime deployment, including when frontend rebuilding is
skipped. A packaged installation therefore needs no EasySlides download at
startup or during a presentation task. Python dependencies are installed through
the environment file on the destination machine.

## Runtime access

Native file tools expose the whole installed bundle at `/.easyslides/`. Shell
commands use `$CATMASTER_EASYSLIDES_ROOT`, which points to the same physical
directory. This is an explicitly supplied software library, reachable alongside
the usual workspace and skill mounts. The workspace Python environment inherits
CatMaster's installed packages.

The worker reads and executes the shared library, and writes deck projects and
adapted assets in the workspace. It does not copy the entire dependency into every
workspace. The CatMaster skill adapts upstream's onboarding and defaults:
editable native output is the default, routine implementation choices do not
require a separate confirmation, and image calls use `generate_figure`.

For Linux preview, use a separate writable LibreOffice profile per render task,
as shown in the skill. This avoids default-profile permission failures and
concurrent rendering conflicts. Preview images belong in workspace scratch;
the requested PPTX and any useful editing sources are the deliverables.
