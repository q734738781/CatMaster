id: neb_workflow
keywords: ["neb", "nudged elastic band", "reaction path"]
tools:
  - make_neb_geometry
  - make_neb_incar
  - vasp_execute
  - workspace_read_file
  - workspace_write_file
prompt: |
  For NEB workflows, ensure initial and final structures are well-relaxed before generating images.
  Keep a clear mapping between image indices and filesystem paths.
