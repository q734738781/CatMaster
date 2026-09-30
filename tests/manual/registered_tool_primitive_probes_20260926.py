"""Local, offline audit probes. Records observations, does not endorse defects."""
import importlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.io import read, write
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar
from catmaster.tools.base import workspace_scope
from catmaster.tools.registry import ToolRegistry

root = Path(tempfile.mkdtemp(prefix="catmaster-primitive-audit-"))
observations = {}

def record(name, function):
    try:
        observations[name] = function()
    except Exception as exc:
        observations[name] = {"probe_error": f"{type(exc).__name__}: {exc}"}

def module(name):
    return importlib.import_module(name)

with workspace_scope(root):
    files = root / "files"
    def save(path, text):
        destination = files / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
        return destination

    def xyz(path, distance=0.7):
        return save(path, f"2\nfixture\nH 0 0 0\nH 0 0 {distance}\n")

    mq = module("catmaster.tools.geometry_inputs.molecular_qchem")
    crystal = module("catmaster.tools.geometry_inputs.crystal_tool")
    registry = ToolRegistry()
    tools = {tool.name: tool for tool in registry.as_langchain_tools()}

    def schema_and_bindings():
        import ast
        source = Path("catmaster/specialists/runtime.py").read_text()
        bound = {node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        schemas = registry.as_openai_tools()
        absent = sorted(set(tools) - bound)
        missing = {item["name"]: [k for k, v in item["parameters"].get("properties", {}).items() if not v.get("description")] for item in schemas}
        return {"registered_count": len(schemas), "no_runtime_literal": absent,
                "top_level_missing_descriptions": {k:v for k,v in missing.items() if v},
                "langchain_schema_count": len(tools),
                "schema_surface_equal_without_duplicate_description": all({k:v for k,v in item["parameters"].items() if k != "description"} == tools[item["name"]].args_schema for item in schemas)}
    record("R00_registry", schema_and_bindings)

    def molecule_name():
        fn = module("catmaster.tools.geometry_inputs.molecule").create_molecule_from_smiles
        _, a = fn({"smiles":"O", "name":"water_A", "fmt":"xyz", "output_path":"name/mol"})
        _, b = fn({"smiles":"C", "name":"methane_B", "fmt":"xyz", "output_path":"name/mol"})
        return {"first":a["data"]["xyz_file_rel"], "second":b["data"]["xyz_file_rel"], "remaining_files":[p.name for p in (files/"name").iterdir()], "final_formula":read(files/"name/mol.xyz").get_chemical_formula()}
    record("R01_ignored_name", molecule_name)

    def absolute_energy():
        xyz("energy/a.xyz", 0.7)
        xyz("energy/b.xyz", 1.7)
        save("energy/conformers.json", json.dumps({"records":[{"structure_rel":"a.xyz", "energy_kcal_mol":100}, {"structure_rel":"b.xyz", "energy_kcal_mol":102}]}))
        content, artifact = mq.filter_conformer_ensemble({"input_dir":"energy", "output_dir":"energy_out", "energy_window_kcal_mol":5, "rmsd_threshold_angstrom":0})
        return {"expected_count":2, "observed_count":artifact["data"]["count"], "content":content}
    record("R02_absolute_energy", absolute_energy)

    def extraction():
        xyz("extract/trajectory/job_trj.xyz", .7)
        with (files/"extract/trajectory/job_trj.xyz").open("a") as f:
            f.write("2\nfinal\nH 0 0 0\nH 0 0 1.4\n")
        save("extract/trajectory/status.json", '{"returncode":0}')
        xyz("extract/runtime/job.runtime.xyz", 1.4)
        save("extract/runtime/status.json", '{"returncode":0}')
        _, a = mq.extract_optimized_molecules({"input_dir":"extract", "output_dir":"extracted", "source":"orca"})
        rows = json.loads((files/"extracted/optimized_molecules.json").read_text())["records"]
        _, b = mq.extract_optimized_molecules({"input_dir":"extract/trajectory", "output_dir":"extracted_single", "source":"orca"})
        return {"count_two_runs":a["data"]["count"], "copied_bond":float(read(files/rows[0]["structure_rel"]).get_distance(0,1)), "expected_last_bond":1.4, "single_root_count":b["data"]["count"]}
    record("R03_extraction", extraction)

    structure = Structure(Lattice.cubic(4.0), ["Na", "Cl"], [[0,0,0], [.5,.5,.5]])
    (files/"structures").mkdir()
    Poscar(structure).write_file(files/"structures/same.vasp")
    # Same stem, scientifically different inputs: the second has a different lattice.
    Structure(Lattice.cubic(5.0), ["Na", "Cl"], [[0,0,0], [.5,.5,.5]]).to(filename=str(files/"structures/same.cif"))
    def collision():
        content, artifact = crystal.supercell({"structure_dir":"structures", "supercell":[1,1,1], "output_dir":"supercells"})
        rows=json.loads((files/"supercells/batch_supercell.json").read_text())["results"]
        return {"processed":artifact["data"]["structures_processed"], "output_paths":[r["output_rel"] for r in rows], "distinct_output_count":len({r["output_rel"] for r in rows}), "content":content}
    record("R04_supercell_collision", collision)

    def orca_collision():
        xyz("mols/a/b.xyz",.7)
        xyz("mols/a_b.xyz",1.4)
        fn=module("catmaster.tools.geometry_inputs.orca_prepare").orca_prepare
        content, a=fn({"input_path":"mols", "output_root":"orca_stages", "simple_keywords":["HF", "STO-3G"], "charge":0, "multiplicity":1})
        rows=json.loads((files/"orca_stages/orca_prepare_manifest.json").read_text())["records"]
        return {"prepared_count":a["data"]["prepared_count"], "stage_paths":[r["stage_path"] for r in rows], "distinct_stage_count":len({r["stage_path"] for r in rows}), "content":content}
    record("R05_orca_collision", orca_collision)

    def truncated_trajectory():
        path=xyz("md/broken.xyz")
        with path.open("a") as f: f.write("2\nincomplete\nH 0 0 0\n")
        fn=module("catmaster.tools.dynamics.lammps_tools").md_trajectory_summary
        content,a=fn({"path":"md/broken.xyz", "output_dir":"md_summary"})
        return {"content":content, "data":a["data"]}
    record("R06_truncated_xyz", truncated_trajectory)

    def cp2k_ambiguous():
        cp=module("catmaster.tools.dynamics.cp2k_analysis")
        save("cp2k/a_scheduler.out", "batch task started\n")
        save("cp2k/z_science.out", "ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -12.5\nPROGRAM ENDED AT\n")
        content,a=cp.cp2k_output_summary({"result_root":"cp2k", "output_dir":"cp_summary"})
        return {"picked":cp._find_cp2k_output_file(files/"cp2k").name, "content":content, "output":{p.name:json.loads(p.read_text()) for p in (files/"cp_summary").glob("*.json")}}
    record("R07_cp2k_ambiguity", cp2k_ambiguous)

    def custom_names():
        qc=module("catmaster.tools.analysis.qchem_analysis")
        lm=module("catmaster.tools.dynamics.lammps_tools")
        save("custom_orca/molecule.out", "FINAL SINGLE POINT ENERGY -1.0\nORCA TERMINATED NORMALLY\n")
        save("custom_xtb/xtb_stdout.out", "normal termination of xtb\n")
        save("custom_lammps/anneal.log", "Step Temp TotEng\n0 300 -1\nLoop time of 1\n")
        result={}
        for key,fn in [("orca",qc.analyze_orca_results),("xtb",qc.analyze_xtb_results),("lammps",lm.lammps_log_summary)]:
            try: result[key]=fn({"result_root":f"custom_{key}"})[0]
            except Exception as exc: result[key]=str(exc)
        return result
    record("R08_custom_result_names", custom_names)

    def phonon_manual_index():
        generated, meta=crystal._manual_phonon_displacements(structure, supercell=[2,1,1], displacement=.01, plus_minus=False, symprec=.01, angle_tolerance=5)
        reference=structure.copy()
        reference.make_supercell([2,1,1])
        moved=[]
        for label, item in generated:
            indices=np.flatnonzero(np.linalg.norm(item.cart_coords-reference.cart_coords,axis=1)>1e-8)
            moved.append({"label":label,"moved_species":[str(reference[int(i)].specie) for i in indices]})
        return {"input_species":[str(s.specie) for s in structure],"supercell_species":[str(s.specie) for s in reference],"metadata":meta,"moves":moved}
    record("R09_manual_phonon_mapping", phonon_manual_index)

    def fallback():
        with patch.dict("sys.modules", {"phonopy":SimpleNamespace(Phonopy=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected phonopy failure")))}):
            content,a=crystal.generate_phonon_displacements({"structure_file":"structures/same.vasp", "output_dir":"phonon", "supercell":[2,1,1]})
        return {"generator":a["data"]["generator"], "content":content, "summary":json.loads((files/"phonon/phonon_displacements.json").read_text())}
    record("R10_phonon_fallback_injected", fallback)

    def mp_conversion():
        mp=module("catmaster.tools.retrieval.matdb")
        bcc=Structure(Lattice([[ -1.5,1.5,1.5],[1.5,-1.5,1.5],[1.5,1.5,-1.5]]), ["Fe"], [[0,0,0]])
        class Client:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def get_structure_by_material_id(self, key): return bcc
        with patch.object(mp,"_mpr",return_value=Client()):
            content,a=mp.mp_download_structure({"mp_ids":["mp-fixture"], "output_dir":"mp"})
        saved=Structure.from_file(files/"mp/mp-fixture.vasp")
        return {"provider_atom_count":len(bcc),"saved_atom_count":len(saved),"reported_atom_count":a["data"]["results"][0]["metadata"]["natoms"],"content":content}
    record("R11_mp_conventional_stub", mp_conversion)

    def graph_handles():
        graph=module("catmaster.tools.misc.research_graph")
        service=SimpleNamespace(store=SimpleNamespace(get_graph=lambda _: {"revision":2}))
        content,a=graph._mutation_result(service=service, tool_name="add_research_experiment",graph_id="graph_fixture",changed={"node":{"node_id":"exp_fixture", "title":"Example"}},message="Added experiment proposal Example.")
        return {"content":content,"node_id_in_content":"exp_fixture" in content,"node_id_in_artifact":"exp_fixture" in json.dumps(a)}
    record("R12_graph_handle_surface", graph_handles)

    def no_mmff():
        from rdkit.Chem import AllChem
        with patch.object(AllChem,"MMFFGetMoleculeProperties",return_value=None):
            content,a=mq.enumerate_molecular_conformers({"smiles":"CC", "output_dir":"no_mmff", "max_conformers":1,"optimize":"mmff"})
        return {"content":content,"summary":json.loads((files/"no_mmff/conformers.json").read_text())}
    record("R13_mmff_failure_injected", no_mmff)

    def casefold_offsets():
        lit=module("catmaster.runtime.literature.tools")
        page=SimpleNamespace(model_dump=lambda **k:{"text":"Straße target", "requested_url":"https://fixture.invalid", "final_url":"https://fixture.invalid"})
        with patch.object(lit,"_literature_components",return_value=(None,None,None,None)), patch.object(lit,"_load_or_fetch_public_page",return_value=(page,"snapshot.json")):
            _,a=lit.find_in_page({"source_path":"snapshot.json", "pattern":"target"})
        return {"expected_start":7, "result":a["data"]["result"]}
    record("R14_unicode_find_offsets", casefold_offsets)

    def acquisition_cache():
        ac=module("catmaster.runtime.literature.acquisition")
        ac._reset_acquisition_cache_for_tests()
        calls=[]
        def fake(**kwargs):
            calls.append(kwargs["provided_title"])
            return "saved", {"tool_name":"acquire_literature_source","data":{"status":"downloaded_pdf","path":"fixture.pdf"}}
        with patch.object(ac,"_require_scansci_version"), patch.object(ac,"_acquire_normalized_source",side_effect=fake),patch.object(ac,"_with_supplementary",side_effect=lambda result,**kwargs:result):
            ac.acquire_literature_source({"identifier":"10.1234/fixture", "expected_title":"title A"})
            ac.acquire_literature_source({"identifier":"10.1234/fixture", "expected_title":"title B"})
        return {"validation_calls":calls,"second_title_revalidated":"title B" in calls}
    record("R15_acquisition_cache_stub", acquisition_cache)

    def slab_collisions():
        slab=module("catmaster.tools.geometry_inputs.slab_tools")
        result={}
        cases=[("build_slab", {"bulk_dir":"structures", "output_root":"slabs", "miller_index":[1,0,0], "slab_thickness":4}),
               ("fix_atoms_by_layers", {"structure_dir":"structures", "output_dir":"fixed_layers", "freeze_layers":1}),
               ("fix_atoms_by_height", {"structure_dir":"structures", "output_dir":"fixed_height", "z_ranges":[{"z_min":0,"z_max":1}]}),
               ("fix_atoms_by_indices", {"structure_dir":"structures", "output_dir":"fixed_indices", "indices":[0]})]
        for name,args in cases:
            content,a=getattr(slab,name)(args)
            directory=files/(args.get("output_root") or args["output_dir"])
            result[name]={"content":content,"vasp_outputs":[str(p.relative_to(directory)) for p in directory.rglob("*.vasp")],"data":a["data"]}
        return result
    record("R16_slab_batch_collisions", slab_collisions)

result={"workspace":str(root), "observations":observations}
output=Path("/tmp/catmaster_primitive_probe_results_20260926.json")
output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(result,ensure_ascii=False,indent=2))
