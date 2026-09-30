---
name: nebts-and-irc
description: Use this skill to prepare mapped ORCA NEB endpoints and native IRC follow-up stages for a molecular reaction path.
allowed-tools: "orca_nebts_prepare orca_prepare remote_submission remote_submission_batch analyze_orca_results execute"
---

# ORCA NEB-TS and IRC

## Workflow

1. Confirm that reactant and product have identical atom count and per-index element ordering.
2. Call `orca_nebts_prepare` with complete `simple_keywords`, explicit charge/multiplicity, and a complete native `%neb ... end` block that references staged `product.xyz`.
3. Submit with `orca_execute` and inspect the accepted TS candidate.
4. For IRC follow-up, call generic `orca_prepare` on the accepted TS with an explicit `IRC` token and any complete `%irc` block in `input_blocks`.
5. Analyze endpoint connection, task convergence, and requested frequency evidence separately.

## Scientific choices

- Select `NEB`, `NEB-CI`, `NEB-TS`, `FAST-NEB-TS`, or another documented variant intentionally; the tool does not choose it.
- Set image and preoptimization controls in the native `%neb` block.
- Include frequency or optimization keywords in an IRC input only when those operations are intended.
- A pathway-search level can generate a candidate without being the final barrier-energy level.

## References

- [ORCA NEB](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/neb.html)
- [ORCA IRC](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/irc.html)
