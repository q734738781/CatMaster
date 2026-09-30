# Figure generation

`generate_figure` generates or edits one image through OpenRouter and saves it
under the current workspace's `files/` directory. WritingSpecialist,
`writing_worker_agent`, `presentation_worker`, and `materials_worker` bind the tool directly. The
quantitative `plot_worker` uses data-based plotting.

## Inputs and outputs

| Input | Meaning |
| --- | --- |
| `prompt` | Complete generation or editing instruction, passed through without added style rules. |
| `output_path` | Workspace-relative output path. A missing suffix follows the returned image format. |
| `model` | Configured model label or OpenRouter image-model ID. An empty value uses the configured image model. |
| `reference_images` | Ordered workspace-relative image paths. Reuse a previous output here for iterative editing. Defaults to `[]`. |
| `image_options` | Image API options overriding configured defaults. Defaults to `{}`. |

Common options are `aspect_ratio`, `quality`, `size`, `resolution`,
`output_format`, `background`, `output_compression`, `seed`, and `provider`.
Their accepted values depend on the selected model. `model`, `prompt`, and
reference paths have their own inputs; each call requests one image.

```json
{
  "prompt": "Keep the supplied diagram's objects and labels. Use a dark blue background and improve spacing.",
  "output_path": "writing/figures/mechanism-revised.png",
  "model": "openai/gpt-image-2.5-sunburst",
  "reference_images": ["writing/figures/mechanism.png"],
  "image_options": {"aspect_ratio": "16:9", "quality": "high"}
}
```

Reference files are attached as image data in the actual request. Outputs
include the saved image path, media type, and model used. Inspect the image with
the normal file-reading capability before using it. Source images remain
available for further edits.

## Configuration

```yaml
models:
  figure-generation:
    provider: openrouter
    model: openai/gpt-image-2.5-sunburst
    api_key_env: OPENROUTER_API_KEY
    base_url: https://openrouter.ai/api/v1

image_generation:
  model_label: figure-generation
  image_config:
    aspect_ratio: "4:3"
```

Public profile templates select GPT Image 2.5 Sunburst. To use Flare or another
OpenRouter image model, set `model` on a call or change the configured model.
A configured label uses that entry's connection, credentials, and provider
options. A raw model ID reuses the default image model's connection and
credentials without inheriting another model's provider routing. Shared
`image_config` defaults still apply; per-call `image_options` override them.
An explicitly supplied `size` replaces inherited aspect-ratio and resolution
defaults. Explicitly supplied conflicting geometry is checked by OpenRouter.

The image tool uses OpenRouter independently of the language model provider.
A Codex OAuth writing model therefore still uses `OPENROUTER_API_KEY` for this
tool. Python dependencies come from the existing CatMaster environment.

## Editable presentations and scientific figures

The [EasySlides presentation worker](easyslides.md) uses this tool for image assets
within editable decks.

Generated illustrations are image assets. For editable PPTX, keep titles, body
text, ordinary tables, connectors, and page layout as native presentation
objects. Place complex illustrations and data plots within that layout.
Rendering an entire slide with an image model or matplotlib does not preserve
those editable objects. Quantitative plots must retain their source data and
scientific meaning; generation is suitable for conceptual or illustrative
material.

## API integration

The implementation uses `POST /api/v1/images`, top-level image options,
`input_references`, and the returned `data[].b64_json` and `media_type` fields.
These follow the current [OpenRouter Image API documentation](https://openrouter.ai/docs/guides/overview/multimodal/image-generation).
It makes one generation request, with a ten-minute read timeout, and reports
provider failures without automatically submitting another paid request.

The separation of tool inputs, provider configuration, and saved image artifacts
follows the approach in [nanobot's image-generation implementation](https://github.com/HKUDS/nanobot/blob/main/nanobot/agent/tools/image_generation.py).
Its older OpenRouter chat-completions transport is not used here.

The previous name `generate_nanobanana_figure` resolves to `generate_figure`
through the registry's compatibility alias; the active tool schema exposes the
general name.
