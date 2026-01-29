id: vasp_batch
keywords: ["vasp", "batch", "kpoints"]
tools:
  - vasp_execute
  - vasp_execute_batch
  - workspace_read_file
  - workspace_write_file
prompt: |
  When running VASP batch jobs, always create a dedicated folder per system.
  Verify INCAR/KPOINTS/POTCAR consistency before submitting.
