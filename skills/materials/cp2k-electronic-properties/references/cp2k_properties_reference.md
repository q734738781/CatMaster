# CP2K electronic-properties reference notes

Checked: 2026-05-29

This file is a local, source-grounded note for `cp2k-electronic-properties`. It is not a full copy of the upstream documentation.

## Upstream sources

- CP2K FORCE_EVAL/PROPERTIES: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/PROPERTIES.html
- CP2K BANDSTRUCTURE: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/PROPERTIES/BANDSTRUCTURE.html
- CP2K DOS: https://manual.cp2k.org/cp2k-2024_1-branch/CP2K_INPUT/FORCE_EVAL/PROPERTIES/BANDSTRUCTURE/DOS.html

## Practical notes

- CP2K electronic-property output is opt-in. A completed energy calculation is not evidence that DOS, PDOS, band, or population files exist.
- `FORCE_EVAL/PROPERTIES` is the main namespace for many property calculations, including `BANDSTRUCTURE` and population-analysis style subsections.
- DOS output is controlled under the band-structure/property hierarchy in the linked CP2K input reference, so DOS parser assumptions must match the exact generated sections and filenames.
- Energy window alignment, spin treatment, k-point choices, atom/orbital projections, and smearing choices are method choices; they should be reported with plots or tables.

## SOP implications for the skill body

- Use `cp2k_prepare(recipe="dos")` or explicit `settings.properties` only when the user requested that property family.
- After execution, identify exact files before parsing; do not grep a directory and present unrelated files as property evidence.
- Prefer a focused parser for the requested property over a generic CP2K analyzer.
