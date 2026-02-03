id: neb_workflow
keywords: ["neb", "nudged elastic band", "reaction path"]
tools:
  - make_neb_geometry
  - make_neb_incar
  - vasp_execute_batch
  - bash_exec
prompt: |
  For NEB workflows, ensure initial and final structures are well-relaxed before generating images.
  Keep a clear mapping between image indices and filesystem paths.
