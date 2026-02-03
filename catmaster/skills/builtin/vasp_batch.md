id: vasp_batch
keywords: ["vasp", "batch", "kpoints"]
tools:
  - vasp_execute_batch
  - bash_exec
prompt: |
  When running VASP batch jobs, always create a dedicated folder per system.
  Verify INCAR/KPOINTS/POTCAR consistency before submitting.
