from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from random import Random

import pytest
from ase import Atoms
from ase.io import write as ase_write
from fastapi.testclient import TestClient
from pymatgen.core import Lattice, Molecule, Species, Structure
from rdkit import Chem
from rdkit.Chem import AllChem

from catmaster.structures.models import (
    DefectCandidatesParams,
    DefectCandidatesRequest,
    MakeSupercellRequest,
    MoleculeConformersParams,
    MoleculeConformersRequest,
    SaveStructureRequest,
    SourceVersion,
    StructureOpenRequest,
    SupercellParams,
    TRANSFORM_REQUEST_ADAPTER,
)
from catmaster.structures.adsorption import (
    enumerate_adsorption_sites,
    place_adsorbate_at_site,
)
from catmaster.structures.molecules import snapshot_from_viewer
from catmaster.structures.operations import transform_structure
from catmaster.structures.serialization import (
    StructureFormatLossError,
    StructureSerializationError,
    StructureVersionConflict,
    format_loss_warnings,
    load_structure_document,
    save_structure_document,
    snapshot_from_molecule,
    snapshot_from_structure,
    snapshot_to_molecule,
    snapshot_to_structure,
    viewer_structure,
)
from catmaster.structures.trajectory import (
    MAX_TEXT_INDEX_CACHE,
    _TEXT_INDEX_CACHE,
    trajectory_frame,
    trajectory_frame_count,
    trajectory_metadata,
)
from catmaster.structures.surfaces import generate_slab_candidates
from catmaster.webui.server import _entry_preview_kind, create_app
from catmaster.webui.structure_api import open_structure


def _periodic_structure() -> Structure:
    return Structure(
        Lattice.from_parameters(4.2, 5.1, 6.3, 78, 83, 72),
        ["Si", "O"],
        [[0.0, 0.0, 0.0], [0.23, 0.37, 0.41]],
        site_properties={
            "selective_dynamics": [[False, True, False], [True, True, True]],
        },
    )


def _molecule() -> Chem.Mol:
    molecule = Chem.AddHs(Chem.MolFromSmiles("C[C@H](O)c1ccccc1[NH3+]"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 7
    assert AllChem.EmbedMolecule(molecule, params) == 0
    return molecule


def test_frontend_mic_matches_seeded_pymatgen_closest_image_oracle() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the frontend scientific oracle test.")
    rng = Random(1729)
    cases = []
    for _ in range(96):
        matrix = [
            [rng.uniform(2.0, 8.0), 0.0, 0.0],
            [rng.uniform(-4.0, 4.0), rng.uniform(2.0, 8.0), 0.0],
            [
                rng.uniform(-4.0, 4.0),
                rng.uniform(-4.0, 4.0),
                rng.uniform(2.0, 8.0),
            ],
        ]
        delta = [rng.uniform(-2.5, 2.5) for _axis in range(3)]
        lattice = Lattice(matrix)
        expected, _image = lattice.get_distance_and_image([0, 0, 0], delta)
        cases.append(
            {
                "matrix": matrix,
                "delta": delta,
                "expected": float(expected),
            }
        )

    module = (
        Path("catmaster/webui/frontend/src/v2/structure/structureModel.js")
        .resolve()
        .as_uri()
    )
    script = f"""
import fs from "node:fs";
import {{ minimumImageDisplacement }} from {json.dumps(module)};
const cases = JSON.parse(fs.readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(cases.map((item) =>
  Math.hypot(...minimumImageDisplacement(item.matrix, item.delta))
)));
"""
    completed = subprocess.run(
        [node, "--input-type=module", "-e", script],
        input=json.dumps(cases),
        text=True,
        capture_output=True,
        check=True,
    )
    actual = json.loads(completed.stdout)
    assert actual == pytest.approx(
        [case["expected"] for case in cases],
        abs=1e-9,
    )


def test_periodic_snapshot_and_poscar_constraints_roundtrip(tmp_path: Path) -> None:
    original = _periodic_structure()
    original.properties["energy"] = -1.0
    snapshot = snapshot_from_structure(original, fmt="poscar")
    projected = viewer_structure(snapshot)
    assert projected["properties"] == {}
    assert projected["sites"][0]["properties"]["selective_dynamics"] == [
        False,
        True,
        False,
    ]
    target = tmp_path / "POSCAR"

    version, warnings = save_structure_document(
        snapshot,
        target,
        overwrite=False,
        expected_version=SourceVersion(),
        accept_format_loss=False,
    )
    reloaded, open_warnings = load_structure_document(target)
    structure = snapshot_to_structure(reloaded)

    assert warnings == []
    assert open_warnings == []
    assert version.size == target.stat().st_size
    assert [
        [bool(value) for value in row]
        for row in structure.site_properties["selective_dynamics"]
    ] == [[False, True, False], [True, True, True]]
    assert structure.lattice.matrix == pytest.approx(original.lattice.matrix)
    assert structure.frac_coords == pytest.approx(original.frac_coords)

    trajectory_target = tmp_path / "constraints.traj"
    save_structure_document(
        snapshot,
        trajectory_target,
        overwrite=False,
        expected_version=SourceVersion(),
        accept_format_loss=False,
    )
    trajectory_snapshot, trajectory_warnings = load_structure_document(trajectory_target)
    trajectory_structure = snapshot_to_structure(trajectory_snapshot)
    assert trajectory_warnings == []
    assert [
        [bool(value) for value in row]
        for row in trajectory_structure.site_properties["selective_dynamics"]
    ] == [[False, True, False], [True, True, True]]


def test_periodic_format_loss_audits_only_information_the_format_cannot_keep(
    tmp_path: Path,
) -> None:
    structure = Structure(
        Lattice.from_parameters(4.2, 5.1, 6.3, 78, 83, 72),
        [Species("Fe", 2), Species("O", -2)],
        [[0, 0, 0], [0.23, 0.37, 0.41]],
        site_properties={
            "selective_dynamics": [[False, True, False], [True, True, True]],
            "site_score": [1.25, 2.5],
            "force_hint": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "annotation": [{"source": "manual"}, {"source": "fit"}],
        },
    )
    snapshot = snapshot_from_structure(structure)

    poscar = "\n".join(format_loss_warnings(snapshot, Path("POSCAR")))
    assert "site properties" in poscar
    assert "annotation" in poscar
    assert "force_hint" in poscar
    assert "site_score" in poscar
    assert "oxidation states" in poscar
    assert "selective-dynamics" not in poscar

    cif = "\n".join(format_loss_warnings(snapshot, Path("structure.cif")))
    assert "selective-dynamics constraints" in cif
    assert "annotation" in cif
    assert "force_hint" in cif
    assert "site_score" in cif
    assert "oxidation states" not in cif

    trajectory = "\n".join(format_loss_warnings(snapshot, Path("structure.traj")))
    assert "annotation" in trajectory
    assert "force_hint" in trajectory
    assert "site_score" in trajectory
    assert "oxidation states" in trajectory
    assert "selective-dynamics" not in trajectory

    lossless_structure = structure.copy()
    lossless_structure.remove_site_property("annotation")
    lossless_snapshot = snapshot_from_structure(lossless_structure)
    target = tmp_path / "properties.xyz"
    assert format_loss_warnings(lossless_snapshot, target) == []
    save_structure_document(
        lossless_snapshot,
        target,
        overwrite=False,
        expected_version=SourceVersion(),
        accept_format_loss=False,
    )
    reloaded, _warnings = load_structure_document(target)
    restored = snapshot_to_structure(reloaded)
    assert restored.site_properties["site_score"] == pytest.approx([1.25, 2.5])
    assert restored.site_properties["force_hint"] == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    assert [float(site.specie.oxi_state) for site in restored] == pytest.approx([2, -2])

    accepted_target = tmp_path / "properties-with-loss.xyz"
    save_structure_document(
        snapshot,
        accepted_target,
        overwrite=False,
        expected_version=SourceVersion(),
        accept_format_loss=True,
    )
    accepted, _warnings = load_structure_document(accepted_target)
    accepted_structure = snapshot_to_structure(accepted)
    assert "annotation" not in accepted_structure.site_properties
    assert accepted_structure.site_properties["site_score"] == pytest.approx([1.25, 2.5])
    assert [float(site.specie.oxi_state) for site in accepted_structure] == pytest.approx(
        [2, -2]
    )


def test_disorder_and_partial_occupancy_require_explicit_loss_for_ordered_formats(
    tmp_path: Path,
) -> None:
    structure = Structure(
        Lattice.cubic(4.5),
        [{Species("Fe", 2): 0.4, Species("Mn", 3): 0.6}],
        [[0.25, 0.25, 0.25]],
    )
    snapshot = snapshot_from_structure(structure)
    cif_warnings = "\n".join(format_loss_warnings(snapshot, Path("ordered.cif")))
    assert "partial occupancies" not in cif_warnings
    assert "oxidation states" not in cif_warnings

    target = tmp_path / "POSCAR"
    warnings = format_loss_warnings(snapshot, target)
    combined = "\n".join(warnings)
    assert "partial occupancies" in combined
    assert "oxidation states" in combined
    with pytest.raises(StructureFormatLossError):
        save_structure_document(
            snapshot,
            target,
            overwrite=False,
            expected_version=SourceVersion(),
            accept_format_loss=False,
        )
    save_structure_document(
        snapshot,
        target,
        overwrite=False,
        expected_version=SourceVersion(),
        accept_format_loss=True,
    )
    restored, _warnings = load_structure_document(target)
    restored_structure = snapshot_to_structure(restored)
    assert restored_structure.is_ordered
    assert restored_structure[0].specie.symbol == "Mn"


def test_save_as_conflict_and_format_loss_are_explicit(tmp_path: Path) -> None:
    snapshot = snapshot_from_structure(_periodic_structure(), fmt="poscar")
    target = tmp_path / "structure.cif"
    target.write_text("external\n", encoding="utf-8")
    version = SourceVersion(mtime_ns=target.stat().st_mtime_ns, size=target.stat().st_size)

    with pytest.raises(StructureVersionConflict, match="explicitly confirm overwrite"):
        save_structure_document(
            snapshot,
            target,
            overwrite=False,
            expected_version=version,
            accept_format_loss=False,
        )
    with pytest.raises(StructureFormatLossError, match="discard scientific"):
        save_structure_document(
            snapshot,
            target,
            overwrite=True,
            expected_version=version,
            accept_format_loss=False,
        )

    stale = version
    target.write_text("changed outside\n", encoding="utf-8")
    os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 1_000))
    with pytest.raises(StructureVersionConflict, match="changed on disk"):
        save_structure_document(
            snapshot,
            target,
            overwrite=True,
            expected_version=stale,
            accept_format_loss=True,
        )


def test_general_integer_supercell_and_defect_candidates() -> None:
    snapshot = snapshot_from_structure(_periodic_structure(), fmt="poscar")
    result = transform_structure(
        MakeSupercellRequest(
            operation="make_supercell",
            input=snapshot,
            params=SupercellParams(matrix=[[2, 1, 0], [0, 1, 0], [0, 0, 1]]),
        )
    )
    expanded = snapshot_to_structure(result["snapshot"])
    assert len(expanded) == 4
    assert len(result["atom_mapping"]) == 4
    assert result["change"]["determinant"] == 2

    defects = transform_structure(
        DefectCandidatesRequest(
            operation="defect_candidates",
            input=snapshot,
            params=DefectCandidatesParams(kind="vacancy"),
        )
    )
    assert defects["candidate_type"] == "vacancy"
    assert defects["candidates"]
    assert all(len(snapshot_to_structure(item["snapshot"])) == 1 for item in defects["candidates"])


def test_slab_gallery_and_adsorption_share_constraint_preserving_services() -> None:
    bulk = Structure(
        Lattice.cubic(4.2),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    terminations = generate_slab_candidates(
        bulk,
        miller_index=[1, 0, 0],
        min_slab_size=8,
        min_vacuum_size=12,
        center_slab=True,
        symmetrize=False,
        orthogonal=True,
        lll_reduce=False,
        surface_supercell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    assert terminations
    assert all(candidate["surface_area"] > 0 for candidate in terminations)
    assert all(candidate["top_composition"] for candidate in terminations)
    assert all(candidate["bottom_composition"] for candidate in terminations)

    slab = snapshot_to_structure(terminations[0]["snapshot"])
    mobility = [[False, False, False] for _ in slab]
    slab.add_site_property("selective_dynamics", mobility)
    sites = enumerate_adsorption_sites(
        slab,
        distance=2.0,
        site_kinds=["ontop", "bridge", "hollow"],
    )
    assert sites
    placed = place_adsorbate_at_site(
        slab,
        Molecule(["H"], [[0, 0, 0]]),
        sites[0]["cart_coords"],
        reorient=False,
    )
    assert len(placed) == len(slab) + 1
    assert placed.site_properties["selective_dynamics"][:-1] == mobility
    assert placed.site_properties["selective_dynamics"][-1] == [True, True, True]


def test_molecule_roundtrip_preserves_chemistry_and_generates_conformers(tmp_path: Path) -> None:
    original = _molecule()
    snapshot = snapshot_from_molecule(original, fmt="sdf")
    target = tmp_path / "charged_chiral.sdf"
    save_structure_document(
        snapshot,
        target,
        overwrite=False,
        expected_version=SourceVersion(),
        accept_format_loss=False,
    )
    reloaded, _warnings = load_structure_document(target)
    molecule = snapshot_to_molecule(reloaded)

    assert Chem.GetFormalCharge(molecule) == 1
    assert any(bond.GetIsAromatic() for bond in molecule.GetBonds())
    assert Chem.FindMolChiralCenters(molecule, includeUnassigned=False)
    assert Chem.MolToSmiles(molecule, isomericSmiles=True) == Chem.MolToSmiles(
        original,
        isomericSmiles=True,
    )

    conformers = transform_structure(
        MoleculeConformersRequest(
            operation="molecule_conformers",
            input=reloaded,
            params=MoleculeConformersParams(count=3, random_seed=11),
        )
    )
    assert conformers["candidate_type"] == "conformer"
    assert 1 <= len(conformers["candidates"]) <= 3
    for candidate in conformers["candidates"]:
        candidate_molecule = snapshot_to_molecule(candidate["snapshot"])
        assert candidate_molecule.GetNumConformers() == 1
        assert Chem.GetFormalCharge(candidate_molecule) == 1


def test_sdf_data_properties_are_preserved_or_reported_before_loss(
    tmp_path: Path,
) -> None:
    molecule = _molecule()
    molecule.SetProp("_Name", "lead compound 017")
    molecule.SetProp("assay_id", "A-017")
    molecule.SetProp("temperature_K", "325")
    molecule.SetIntProp("replicate_count", 7)
    molecule.SetProp("_private_cache", "not document data")
    molecule.SetProp("computed_hint", "not document data", computed=True)
    snapshot = snapshot_from_molecule(molecule, fmt="sdf")
    restored = snapshot_to_molecule(snapshot)
    assert restored.GetProp("_Name") == "lead compound 017"
    assert restored.GetProp("assay_id") == "A-017"
    assert restored.GetProp("temperature_K") == "325"
    assert restored.GetProp("replicate_count") == "7"
    assert restored.GetProp("_private_cache") == "not document data"
    assert not restored.HasProp("computed_hint")
    assert format_loss_warnings(snapshot, Path("copy.sdf")) == []

    viewer = viewer_structure(snapshot)
    original_point = restored.GetConformer().GetAtomPosition(0)
    for site in viewer["sites"]:
        site["xyz"] = [
            float(site["xyz"][0]) + 3.0,
            float(site["xyz"][1]) + 2.0,
            float(site["xyz"][2]) + 1.0,
        ]
        site["abc"] = list(site["xyz"])
    edited_snapshot = snapshot_from_viewer(snapshot, viewer)
    edited = snapshot_to_molecule(edited_snapshot)
    assert edited.GetProp("_Name") == "lead compound 017"
    assert edited.GetProp("assay_id") == "A-017"
    assert edited.GetProp("temperature_K") == "325"
    assert edited.GetProp("replicate_count") == "7"
    assert edited.GetProp("_private_cache") == "not document data"
    assert not edited.HasProp("computed_hint")

    target = tmp_path / "copy.sdf"
    save_structure_document(
        edited_snapshot,
        target,
        overwrite=False,
        expected_version=SourceVersion(),
        accept_format_loss=False,
    )
    reloaded, _warnings = load_structure_document(target)
    reloaded_molecule = snapshot_to_molecule(reloaded)
    assert reloaded_molecule.GetProp("assay_id") == "A-017"
    assert reloaded_molecule.GetProp("temperature_K") == "325"
    assert reloaded_molecule.GetProp("replicate_count") == "7"
    assert reloaded_molecule.GetProp("_Name") == "lead compound 017"
    assert reloaded_molecule.GetProp("_private_cache") == "not document data"
    assert not reloaded_molecule.HasProp("computed_hint")
    point = reloaded_molecule.GetConformer().GetAtomPosition(0)
    assert [point.x, point.y, point.z] == pytest.approx(
        [original_point.x + 3.0, original_point.y + 2.0, original_point.z + 1.0],
        abs=1e-4,
    )

    mol_warning = "\n".join(format_loss_warnings(snapshot, Path("copy.mol")))
    assert "SDF data properties" in mol_warning
    assert "assay_id" in mol_warning
    assert "temperature_K" in mol_warning


def test_multi_record_sdf_is_never_silently_reduced_to_one_record(
    tmp_path: Path,
) -> None:
    source = tmp_path / "library.sdf"
    writer = Chem.SDWriter(str(source))
    try:
        first = Chem.MolFromSmiles("CCO")
        first.SetProp("record_id", "first")
        second = Chem.MolFromSmiles("c1ccccc1")
        second.SetProp("record_id", "second")
        writer.write(first)
        writer.write(second)
    finally:
        writer.close()

    snapshot, open_warnings = load_structure_document(source)
    assert "contains 2 SDF records" in "\n".join(open_warnings)
    assert snapshot_to_molecule(snapshot).GetProp("record_id") == "first"
    loss = format_loss_warnings(snapshot, tmp_path / "first-only.sdf")
    assert "source contains 2 SDF records" in "\n".join(loss)

    target = tmp_path / "first-only.sdf"
    with pytest.raises(StructureFormatLossError):
        save_structure_document(
            snapshot,
            target,
            overwrite=False,
            expected_version=SourceVersion(),
            accept_format_loss=False,
        )
    save_structure_document(
        snapshot,
        target,
        overwrite=False,
        expected_version=SourceVersion(),
        accept_format_loss=True,
    )
    records = [item for item in Chem.SDMolSupplier(str(target)) if item is not None]
    assert len(records) == 1
    assert records[0].GetProp("record_id") == "first"


def test_molecule_2d_3d_projection_preserves_chemistry_and_coordinates() -> None:
    original = _molecule()
    snapshot = snapshot_from_molecule(original, fmt="sdf")
    viewer = viewer_structure(snapshot)
    viewer["sites"][0]["xyz"] = [1.25, -0.5, 2.75]
    viewer["sites"][0]["abc"] = [1.25, -0.5, 2.75]

    rebuilt_snapshot = snapshot_from_viewer(snapshot, viewer)
    rebuilt = snapshot_to_molecule(rebuilt_snapshot)

    assert Chem.MolToSmiles(rebuilt, isomericSmiles=True) == Chem.MolToSmiles(
        original,
        isomericSmiles=True,
    )
    assert Chem.GetFormalCharge(rebuilt) == 1
    assert any(bond.GetIsAromatic() for bond in rebuilt.GetBonds())
    assert Chem.FindMolChiralCenters(rebuilt, includeUnassigned=False)
    position = rebuilt.GetConformer().GetAtomPosition(0)
    assert [position.x, position.y, position.z] == pytest.approx([1.25, -0.5, 2.75])

    assert format_loss_warnings(rebuilt_snapshot, Path("molecule.sdf")) == []
    assert "3D coordinates" in " ".join(
        format_loss_warnings(rebuilt_snapshot, Path("molecule.smi"))
    )

    inverted = viewer_structure(snapshot)
    inverted["sites"][0]["xyz"] = [3.0, 2.0, 1.0]
    inverted["sites"][0]["abc"] = [3.0, 2.0, 1.0]
    with pytest.raises(StructureSerializationError, match="stereochemistry at atom"):
        snapshot_from_viewer(snapshot, inverted)


def test_molecule_3d_bond_and_atom_edits_rebuild_a_valid_connection_table() -> None:
    original = Chem.MolFromSmiles("CC")
    snapshot = snapshot_from_molecule(original, fmt="mol")
    viewer = viewer_structure(snapshot)
    viewer["properties"]["bonds"][0]["order"] = 2
    double_bond = snapshot_to_molecule(snapshot_from_viewer(snapshot, viewer))
    assert Chem.MolToSmiles(double_bond) == "C=C"

    viewer["sites"] = viewer["sites"][:1]
    viewer["properties"]["bonds"] = []
    deleted = snapshot_to_molecule(snapshot_from_viewer(snapshot, viewer))
    assert Chem.MolToSmiles(deleted) == "C"

    carbon = viewer_structure(snapshot_from_molecule(Chem.MolFromSmiles("C"), fmt="mol"))
    carbon["sites"].append(
        {
            "species": [{"element": "O", "occu": 1.0, "oxidation_state": 0.0}],
            "label": "O",
            "abc": [1.2, 0.0, 0.0],
            "xyz": [1.2, 0.0, 0.0],
            "properties": {"formal_charge": 0, "isotope": 0},
        }
    )
    carbon["properties"]["bonds"] = [{"site_idx_1": 0, "site_idx_2": 1, "order": 2}]
    carbonyl = snapshot_to_molecule(
        snapshot_from_viewer(snapshot_from_molecule(Chem.MolFromSmiles("C"), fmt="mol"), carbon)
    )
    assert Chem.MolToSmiles(carbonyl) == "C=O"


def test_structure_transform_and_save_schemas_keep_optional_objects_non_nullable() -> None:
    transform_schema = TRANSFORM_REQUEST_ADAPTER.json_schema()
    transform_params_schema = transform_schema["$defs"]["MoleculeFromViewerParams"]
    viewer_schema = transform_params_schema["properties"]["viewer_structure"]
    assert viewer_schema["type"] == "object"
    assert "anyOf" not in viewer_schema
    assert "viewer_structure" not in transform_params_schema.get("required", [])

    save_schema = SaveStructureRequest.model_json_schema()
    save_viewer_schema = save_schema["properties"]["viewer_structure"]
    assert save_viewer_schema["type"] == "object"
    assert "anyOf" not in save_viewer_schema
    assert "viewer_structure" not in save_schema.get("required", [])


def test_trajectory_reports_true_count_and_frames_beyond_240(tmp_path: Path) -> None:
    path = tmp_path / "long.traj"
    frames = [
        Atoms(
            "H2",
            positions=[[0, 0, 0], [0, 0, 0.7 + index / 10_000]],
            cell=[10, 10, 10],
            pbc=True,
            info={"energy": -1.0 + index / 1000},
        )
        for index in range(301)
    ]
    ase_write(path, frames, format="traj")

    meta = trajectory_metadata(path)
    assert meta["total_frames"] == 301
    assert meta["property_stride"] == 1
    assert trajectory_frame(path, 0)["index"] == 0
    assert trajectory_frame(path, 239)["index"] == 239
    assert trajectory_frame(path, 240)["index"] == 240
    assert trajectory_frame(path, 300)["index"] == 300
    with pytest.raises(IndexError):
        trajectory_frame(path, 301)


def test_extxyz_and_xdatcar_use_versioned_bounded_random_access_indices(tmp_path: Path) -> None:
    frames = [
        Atoms(
            "H2",
            positions=[[0, 0, 0], [0, 0, 0.7 + index / 10_000]],
            cell=[8, 8, 8],
            pbc=True,
            info={"energy": -1.0 + index / 1000},
        )
        for index in range(301)
    ]
    extxyz = tmp_path / "long.extxyz"
    ase_write(extxyz, frames, format="extxyz")

    meta = trajectory_metadata(extxyz)
    assert meta["total_frames"] == 301
    assert trajectory_frame_count(extxyz) == 301
    assert meta["random_access"] is True
    for index in (0, 239, 240, 300):
        assert trajectory_frame(extxyz, index)["index"] == index
    with pytest.raises(IndexError):
        trajectory_frame(extxyz, 301)

    xdatcar = tmp_path / "XDATCAR"
    ase_write(xdatcar, frames[:4], format="vasp-xdatcar")
    xdatcar_meta = trajectory_metadata(xdatcar)
    assert xdatcar_meta["total_frames"] == 4
    assert xdatcar_meta["random_access"] is True
    assert trajectory_frame(xdatcar, 3)["positions"][1][2] == pytest.approx(
        frames[3].positions[1][2]
    )

    xyz = tmp_path / "plain.xyz"
    ase_write(xyz, frames[:3], format="xyz")
    assert trajectory_frame_count(xyz) == 3
    assert trajectory_frame(xyz, 2)["index"] == 2

    for index in range(MAX_TEXT_INDEX_CACHE + 2):
        one_frame = tmp_path / f"cache-{index}.extxyz"
        ase_write(one_frame, frames[:1], format="extxyz")
        trajectory_metadata(one_frame)
    assert len(_TEXT_INDEX_CACHE) <= MAX_TEXT_INDEX_CACHE


def test_volume_artifacts_are_not_misclassified_as_structures(tmp_path: Path) -> None:
    assert _entry_preview_kind(tmp_path / "density.cube") == "volume"
    assert _entry_preview_kind(tmp_path / "CHGCAR") == "volume"
    assert _entry_preview_kind(tmp_path / "LOCPOT") == "volume"
    assert _entry_preview_kind(tmp_path / "ELFCAR") == "volume"
    assert _entry_preview_kind(tmp_path / "field.xsf") == "volume"


def test_structure_api_open_transform_save_and_trajectory(tmp_path: Path) -> None:
    workspace = tmp_path / "demo"
    (workspace / "files").mkdir(parents=True)
    (workspace / "metadata").mkdir()
    snapshot = snapshot_from_structure(_periodic_structure(), fmt="poscar")
    save_structure_document(
        snapshot,
        workspace / "files" / "POSCAR",
        overwrite=False,
        expected_version=SourceVersion(),
        accept_format_loss=False,
    )
    ase_write(
        workspace / "files" / "small.traj",
        [Atoms("He", positions=[[0, 0, index * 0.1]]) for index in range(3)],
        format="traj",
    )
    ase_write(
        workspace / "files" / "single.extxyz",
        [Atoms("He", positions=[[0, 0, 0]])],
        format="extxyz",
    )
    ase_write(
        workspace / "files" / "multi.extxyz",
        [Atoms("He", positions=[[0, 0, index * 0.1]]) for index in range(3)],
        format="extxyz",
    )
    ase_write(
        workspace / "files" / "multi.xyz",
        [Atoms("He", positions=[[0, 0, index * 0.1]]) for index in range(3)],
        format="xyz",
    )
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)

    opened = client.post(
        "/api/structures/open",
        json={"workspace": "demo", "path": "files/POSCAR"},
    )
    assert opened.status_code == 200
    body = opened.json()
    assert body["snapshot"]["mode"] == "periodic"
    assert body["summary"]["atom_count"] == 2

    single = client.post(
        "/api/structures/open",
        json={"workspace": "demo", "path": "files/single.extxyz"},
    )
    assert single.status_code == 200
    assert single.json()["capabilities"]["trajectory"] is False
    assert single.json()["capabilities"]["editable"] is True

    multi = client.post(
        "/api/structures/open",
        json={"workspace": "demo", "path": "files/multi.extxyz"},
    )
    assert multi.status_code == 200
    assert multi.json()["capabilities"]["trajectory"] is True
    assert multi.json()["capabilities"]["editable"] is False

    multi_xyz = client.post(
        "/api/structures/open",
        json={"workspace": "demo", "path": "files/multi.xyz"},
    )
    assert multi_xyz.status_code == 200
    assert multi_xyz.json()["capabilities"]["trajectory"] is True
    assert multi_xyz.json()["capabilities"]["editable"] is False

    transformed = client.post(
        "/api/structures/transform",
        json={
            "operation": "make_supercell",
            "input": body["snapshot"],
            "params": {"matrix": [[1, 1, 0], [0, 2, 0], [0, 0, 1]]},
        },
    )
    assert transformed.status_code == 200
    assert transformed.json()["summary"]["atom_count"] == 4

    saved = client.post(
        "/api/structures/save",
        json={
            "workspace": "demo",
            "destination_path": "files/supercell.vasp",
            "snapshot": transformed.json()["snapshot"],
        },
    )
    assert saved.status_code == 200
    assert (workspace / "files" / "supercell.vasp").is_file()

    meta = client.get(
        "/api/trajectories/meta",
        params={"workspace": "demo", "path": "files/small.traj"},
    )
    assert meta.status_code == 200
    assert meta.json()["total_frames"] == 3
    frame = client.get(
        "/api/trajectories/frame",
        params={"workspace": "demo", "path": "files/small.traj", "index": 2},
    )
    assert frame.status_code == 200
    assert frame.json()["index"] == 2


def test_structure_api_saves_the_current_3d_molecule_without_losing_topology(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "demo"
    (workspace / "files").mkdir(parents=True)
    (workspace / "metadata").mkdir()
    original = _molecule()
    save_structure_document(
        snapshot_from_molecule(original, fmt="sdf"),
        workspace / "files" / "source.sdf",
        overwrite=False,
        expected_version=SourceVersion(),
        accept_format_loss=False,
    )
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    opened = client.post(
        "/api/structures/open",
        json={"workspace": "demo", "path": "files/source.sdf"},
    )
    assert opened.status_code == 200
    body = opened.json()
    viewer = body["viewer_structure"]
    for site in viewer["sites"]:
        site["xyz"] = [
            float(site["xyz"][0]) + 3.0,
            float(site["xyz"][1]) + 2.0,
            float(site["xyz"][2]) + 1.0,
        ]
        site["abc"] = list(site["xyz"])

    saved = client.post(
        "/api/structures/save",
        json={
            "workspace": "demo",
            "destination_path": "files/edited.sdf",
            "snapshot": body["snapshot"],
            "viewer_structure": viewer,
        },
    )
    assert saved.status_code == 200
    saved_molecule = snapshot_to_molecule(saved.json()["snapshot"])
    assert Chem.MolToSmiles(saved_molecule, isomericSmiles=True) == Chem.MolToSmiles(
        original,
        isomericSmiles=True,
    )
    point = saved_molecule.GetConformer().GetAtomPosition(0)
    original_point = original.GetConformer().GetAtomPosition(0)
    assert [point.x, point.y, point.z] == pytest.approx(
        [original_point.x + 3.0, original_point.y + 2.0, original_point.z + 1.0],
        abs=1e-4,
    )


def test_existing_save_as_target_requires_version_probe_and_fresh_confirmation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "demo"
    (workspace / "files").mkdir(parents=True)
    (workspace / "metadata").mkdir()
    source_snapshot = snapshot_from_structure(_periodic_structure(), fmt="poscar")
    target = workspace / "files" / "existing.vasp"
    save_structure_document(
        snapshot_from_structure(
            Structure(Lattice.cubic(3.0), ["H"], [[0, 0, 0]]),
            fmt="poscar",
        ),
        target,
        overwrite=False,
        expected_version=SourceVersion(),
        accept_format_loss=False,
    )
    original_bytes = target.read_bytes()
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    request = {
        "workspace": "demo",
        "destination_path": "files/existing.vasp",
        "snapshot": source_snapshot.model_dump(mode="json"),
        "overwrite": True,
        "expected_source_version": {"mtime_ns": 0, "size": 0},
    }

    probe = client.post("/api/structures/save", json=request)
    assert probe.status_code == 200
    assert probe.json()["requires_overwrite_confirmation"] is True
    verified_version = probe.json()["source_version"]
    assert target.read_bytes() == original_bytes

    target.write_bytes(original_bytes + b"\nexternal change\n")
    os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 1_000))
    stale = client.post(
        "/api/structures/save",
        json={**request, "expected_source_version": verified_version},
    )
    assert stale.status_code == 409
    assert b"external change" in target.read_bytes()

    refreshed = client.post("/api/structures/save", json=request)
    assert refreshed.status_code == 200
    confirmed = client.post(
        "/api/structures/save",
        json={
            **request,
            "expected_source_version": refreshed.json()["source_version"],
        },
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["summary"]["atom_count"] == 2


def test_molecule_save_rejects_stale_viewer_topology_from_2d_sync_race(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "demo"
    (workspace / "files").mkdir(parents=True)
    (workspace / "metadata").mkdir()
    stale_snapshot = snapshot_from_molecule(Chem.MolFromSmiles("CC"), fmt="sdf")
    edited_snapshot = snapshot_from_molecule(Chem.MolFromSmiles("C=C"), fmt="sdf")
    stale_viewer = viewer_structure(stale_snapshot)
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))

    response = client.post(
        "/api/structures/save",
        json={
            "workspace": "demo",
            "destination_path": "files/racy.sdf",
            "snapshot": edited_snapshot.model_dump(mode="json"),
            "viewer_structure": stale_viewer,
        },
    )

    assert response.status_code == 400
    assert "stale relative to the MolBlock" in response.json()["detail"]
    assert not (workspace / "files" / "racy.sdf").exists()


def test_outcar_open_contract_is_explicitly_read_only_vibration_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcar = tmp_path / "OUTCAR"
    outcar.write_text("fixture", encoding="utf-8")
    snapshot = snapshot_from_structure(_periodic_structure(), fmt="outcar")
    monkeypatch.setattr(
        "catmaster.webui.structure_api.load_structure_document",
        lambda _path, relative_path="": (snapshot, []),
    )

    payload = open_structure(
        tmp_path,
        StructureOpenRequest(workspace="demo", path="OUTCAR"),
    )

    assert payload["capabilities"] == {
        "editable": False,
        "trajectory": False,
        "vibration_fallback": True,
    }
