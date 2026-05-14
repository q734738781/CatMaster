# HEO DeepAgent Scaffold

This package implements the two-stage agent split described in `Agent_struct.md`:

- `stage1`: low-cost screening, posterior updates, and artifact export
- `stage2`: prior loading, candidate proposal, Pareto archive maintenance, and queue export

Run it with:

```bash
python -m Agent_Optimization.heo_agent campaign-init HEO_Na_001
python -m Agent_Optimization.heo_agent stage1-run HEO_Na_001
python -m Agent_Optimization.heo_agent stage2-run HEO_Na_001
```

If `TAVILY_API_KEY` is set, both stages also have a `researcher` search subagent available for chemistry background facts and literature context.

Artifacts are written under `Agent_Optimization/runtime/campaigns/<campaign_id>/`.

Stage1 Agent:
python -m Agent_Optimization.heo_agent stage1-run HEO_STAGE1_FORMAL --task "Run stage1 for NFPP HEO screening. First build a literature-informed prior over the 23 candidate elements and choose an initial active pool of at most 15 elements. Then use up to 15 screening rounds on the fixed S416 NFPP base structure with x_total=0.5 and five-dopant combinations, using lightweight MACE relax by default. Use literature evidence, raw screening results, the ideal-configurational Delta G_mix(298 K) proxy, and uncertainty together to gradually shrink toward a relatively stable pool. At the end, explicitly reconcile the literature-informed prior pool and the screening-suggested stable pool, then write a full analysis report and export all required stage1 artifacts."