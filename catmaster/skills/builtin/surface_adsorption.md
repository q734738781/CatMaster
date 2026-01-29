id: surface_adsorption
keywords: ["adsorption", "surface", "slab"]
tools:
  - build_slab
  - enumerate_adsorption_sites
  - place_adsorbate
  - relax_prepare
  - workspace_read_file
  - workspace_write_file
prompt: |
  For surface adsorption workflows, generate clean slab structures before placing adsorbates.
  Always record surface termination and adsorption site labels in outputs.
