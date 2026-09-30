---
name: literature-figure-guided-structure-reconstruction
description: Use this skill when reconstructing a simple atomistic model from a clear literature figure or schematic and a visual morphology brief would reduce modeling errors; it extracts observable topology, placement, termination, distribution, and contacts without treating the image as a validation gate or a source of exact coordinates.
license: project-local
---

# Literature-figure-guided structure reconstruction

## Purpose and boundary

Translate a clear literature image and its caption into a short morphology brief before writing the construction script. This is optional modeling evidence, not a structure-acceptance gate. It does not replace crystallographic data, reported coordinates, numerical geometry checks, or physical validation.

Use it for figures that visibly communicate a simple construction, such as a stepped site, exposed termination, cyclic reaction motif, supported cluster contact, adsorbate anchoring pattern, uniform distribution, or substitutional placement. If the image is crowded, stylized, perspective-ambiguous, low resolution, or inconsistent with the text, do not force one reconstruction. Prefer reported structures or build the few alternatives that remain scientifically plausible.

## Optional visual branch

Use the exact figure or page, caption, nearby explanatory text, target model, and authoritative component paths. If `general-purpose` is available, one self-contained image-reading branch can help isolate the visual evidence. Keep scientific interpretation, construction and validation within the modeling task.

Give the visual branch a neutral, observation-first brief. Ask it to separate:

- features directly visible in the image;
- assignments stated by the caption or nearby text;
- reasonable but unconfirmed interpretation;
- details that cannot be recovered from the figure.

Ask specifically about the host facet, step, edge, pore, or termination; which species occupy, replace, or decorate which sites; intended inter-species contacts; visible rings or reaction-center topology; sidedness and orientation; and whether a distribution appears isolated, clustered, ordered, or approximately uniform. Do not ask the VLM to invent Cartesian coordinates, bond lengths, atom indices, hidden atoms, coverage, oxidation states, or a unique three-dimensional structure that the image does not establish.

A suitable delegated brief is:

```text
Inspect only [image path], Figure [identifier], together with the supplied caption and nearby text.
Describe the morphology needed to reconstruct [target model]. Separate directly visible features,
caption-supported assignments, interpretation, and unknowns. Focus on termination or step identity,
species placement or substitution, named contacts, ring/topology order, orientation, and distribution.
Do not generate coordinates or fill in atoms and chemistry that the source does not show.
Return one concise morphology brief.
```

## Morphology brief

Return a compact handoff containing:

1. the source image path and figure/page identifier;
2. the directly visible morphology in plain language;
3. caption-supported species and site assignments;
4. construction requirements that the script must preserve;
5. unresolved alternatives or information absent from the figure.

A useful requirement is concrete enough to map into code, for example: "the substituted atom occupies the upper step row rather than the terrace", "the adsorbate anchor contacts both labeled edge atoms", or "the four named reaction-center atoms form the depicted ring order". Avoid vague conclusions such as "the model looks reasonable".

## Apply the brief to construction

At the top of the construction script, add a short `Literature-derived modeling intent` comment block. Map every construction requirement to named atom selections, fragment anchors, site choices, ring order, termination edits, substitution operations, or distribution rules in the code. Prefer geometry-derived placement and explicit indices over unexplained Cartesian literals.

When the image leaves a material ambiguity, generate labeled alternatives instead of burying one guess in the script. When independent fragments must be combined, continue with `constraint-guided-atomic-assembly`. After construction, use `atomic-structure-validation-and-recovery` exactly as usual.

The literature figure may motivate the model and help compare a reconstruction render with the intended morphology, but visual resemblance is not PASS/FAIL evidence. Numerical geometry validation remains the hard gate, and energetic or mechanistic claims still require the appropriate physical calculation.

## Handoff

Return the morphology brief, construction-script path, image-to-script mapping, generated structure paths, and any alternatives kept because the figure was ambiguous. State which details came from the image, which came from text, and which were modeling choices.
