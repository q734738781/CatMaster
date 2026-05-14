---
name: nature-reader
description: Build full-text bilingual, figure-aware, source-grounded Markdown reading files for journal or conference papers from PDF, DOI, arXiv, publisher HTML, or pasted text. Use whenever the user asks to translate an entire paper, make a complete markdown reader, preserve figure or table placement near the relevant prose, or keep exact source anchors for every block. Do not use this for summaries, bullet-keyword notes, or citation-only tasks.
---

# Full-Paper Markdown Reader

Use this skill to turn a research paper into a complete Markdown reading artifact.

The default output should read like a paper companion, not a summary dump:

- keep the full prose, paragraph structure, and section flow
- show original text and Chinese translation together
- keep figures and tables close to the discussion that introduces them
- preserve stable page and block anchors for traceability
- write a complete `paper.md` by default

This skill is for papers, preprints, and conference proceedings across disciplines. It is not limited to Nature-family journals.

## When to use

Use this skill when the user wants any of the following:

- translate an entire paper into a complete Markdown document
- make a paper easier to read without losing the original wording
- generate a full-paper reading file with original/translation alignment
- keep figures or tables visually close to the claims they support
- preserve exact source locations for every substantive block
- build a source-grounded markdown artifact rather than a slide deck or short summary

If the user only wants a summary, use a summarization skill instead. If the user only wants citation search, use a citation skill instead.

## Core principle

Translate for meaning, not for style. Preserve the paper's structure, evidence, hedging, terminology, equations, units, and citation markers. Keep the output in prose paragraphs unless the source itself is tabular or list-like. Do not collapse the paper into keyword bullets or slide-style notes.

The reading file should help a reader move between:

- original text
- translated text
- source location
- figure or table evidence

## Workflow

### 1. Identify the source and paper type

Determine whether the source is:

- selectable-text PDF
- scanned PDF
- publisher HTML
- DOI or arXiv link
- pasted text or notes

Then identify the paper type at a high level:

- discovery or mechanism paper
- methods or algorithm paper
- resource or dataset paper
- conference paper
- review or perspective

This helps decide how tightly to couple text, figures, and captions.

### 2. Build a full-document source map before translating

If the user provides a full paper, process the entire document. Do not stop at the abstract, introduction, or a few representative pages unless the user explicitly asks for a preview.

Create stable IDs for source blocks:

- `S001`, `S002`, ... for body text
- `C001`, `C002`, ... for captions
- `F001`, `F002`, ... for figures
- `T001`, `T002`, ... for tables

For each block, capture:

- page number
- block type
- original text
- translation
- nearby figure or table references
- confidence level when extraction is uncertain

Keep the source map stable so later questions can point back to the same IDs.
For long papers, add a page index so the reader can jump across the whole document without losing location.

### 3. Translate conservatively

Translate each block with these rules:

- preserve technical terms unless a standard Chinese equivalent is clearly better
- keep gene names, protein names, formulas, model names, and symbols intact
- keep citations, superscripts, subscripts, and numeric values unchanged
- do not collapse methods details into vague prose
- keep paragraph order and section order unless the user asks for restructuring
- mark uncertain text instead of guessing when OCR or layout extraction is weak
- keep the source's paragraph form; do not convert dense prose into bullet-point keywords

If a sentence contains multiple claims, keep the translation readable but do not split away the original evidence chain.

### 4. Place figures and tables near the relevant discussion

Do not try to recreate the PDF pixel-for-pixel. Preserve semantic proximity instead.

Default placement rule:

- show a figure near its first substantive mention in the body text
- keep the caption attached to the figure
- if the caption contains critical details, keep caption and figure together
- if a table is central to the claim, keep it near the paragraph that interprets it

If the paper has a complex multi-column layout, prefer a clean reading layout over exact visual mimicry.

### 4b. Crop figures and tables tightly

When extracting a figure or table image:

- crop only the figure or table content area, not the whole page
- use the smallest rectangle that fully contains the visual object
- exclude page headers, footers, surrounding prose, and unrelated margins
- keep the caption separate unless the caption is part of the requested visual crop
- if the crop box is uncertain, mark it as approximate instead of enlarging it

Precision matters more than convenience here. A slightly smaller but correct crop is better than a wider crop that includes unrelated page content.

### 5. Generate the Markdown file

Default output is a single full-paper `paper.md` file.

The Markdown should usually include:

- metadata header
- page-level sections for long papers
- body prose in paragraph form
- figure and table blocks placed near the relevant discussion
- clickable source anchors on every substantive block
- short uncertainty notes only when extraction is weak

Do not add an interactive Q&A panel or follow-up widget in the Markdown deliverable. If the user later asks a question, answer it in chat using the source map rather than embedding a conversational panel in the artifact.

If a browser preview is explicitly requested, a companion `reader.html` can be generated as a secondary artifact, but the Markdown file remains the primary output.

### 6. Answer follow-up questions with source grounding

When the user asks a question after the file is created:

- identify the most relevant source blocks first
- answer from the paper, not from memory
- cite the exact block IDs and page numbers
- if the answer depends on a figure or table, cite that too
- if the paper does not support the claim, say so plainly

Every substantive answer should include a source pointer such as:

- `p.4 S012-S013`
- `Fig. 2 caption`
- `Table 1`

If the answer is a synthesis across several blocks, list all supporting locations.

## Output contract

Prefer these outputs:

- `paper.md` for the full-paper Markdown artifact
- `source_map.json` for stable source anchors
- `translation_notes.md` for terminology, uncertainty, and layout notes
- `assets/` for extracted figures or cropped snippets when needed
- `reader.html` only when the user explicitly wants a browser preview

Do not hide missing information. If the source is incomplete, label the output as draft mode.

## Tooling guidance

If the input is a PDF, load the `pdf` skill first for extraction and OCR guidance.
If the user asks for a richer browser view, use `web-artifacts-builder` or `frontend-design` only as a preview layer on top of the Markdown workflow.
If the user wants citation-level grounding to original text, keep the source map explicit and do not lose the page or block IDs.
If the user asks for a model backend, treat the provider as configurable and keep the prompt format provider-neutral.

## Model backends

Use official APIs from the provider the user has available. Prefer OpenAI-compatible chat or responses interfaces when they exist, because that keeps the paper reader portable across vendors.

- `DeepSeek`: official OpenAI-compatible API at `https://api.deepseek.com`
- `GLM / Zhipu`: official OpenAI-compatible API at `https://open.bigmodel.cn/api/paas/v4`
- `Qwen / DashScope`: official OpenAI-compatible API at `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `Kimi / Moonshot`: official OpenAI-compatible API at `https://api.moonshot.cn/v1`

Keep model names provider-specific, but keep the app contract the same: `base_url`, `api_key`, `model`, and chat-completions-style messages.

## Quality bar

Good output feels like a paper reader, not a machine translation dump.

It should let a reader:

- read the paper in two languages
- see where a claim came from
- inspect the nearby figure or table
- move through a complete Markdown file without losing source traceability
