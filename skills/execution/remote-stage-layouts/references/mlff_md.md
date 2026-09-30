# mlff_md

Prepare exactly one independent trajectory source per stage:

```text
stage/
  input/
    start.vasp or start.xyz or restart.traj
```

One stage holds one segment of one trajectory lineage; a lineage may span multiple dependent stages. Set grouped dynamics, thermostat, barostat, and output controls through `template_overrides.task_config`, not a duplicate params file. Use one stage per independent lineage at the same segment (first-level children by default, or explicit relative `stage_paths`), and submit later segments only after the preceding `restart.traj` is available.
