# orca_execute

```text
stage/
  job.inp
  input.xyz
  any explicitly referenced native files
```

Build this stage with `orca_prepare` or `orca_nebts_prepare`. The generic preparer receives complete `simple_keywords`, complete native blocks, and explicit charge/multiplicity. The canonical `job.inp` remains unchanged during execution.
