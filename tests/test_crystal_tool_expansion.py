from __future__ import annotations

import json
from pathlib import Path

from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Kpoints, Poscar

from catmaster.tools.base import workspace_scope
from catmaster.tools.geometry_inputs.crystal_tool import (
    create_vacancy,
    enumerate_unique_sites,
    generate_kpath,
    generate_phonon_displacements,
    generate_strained_structures,
    insert_interstitial_at_coords,
    substitute_species,
)
from catmaster.tools.geometry_inputs.vasp_band_prepare import vasp_band_prepare


def _write_structure(path: Path, structure: Structure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Poscar(structure).write_file(str(path))


def test_enumerate_unique_sites_and_vacancy_generation(tmp_path: Path) -> None:
    structure = Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    with workspace_scope(tmp_path):
        structure_path = tmp_path / "files" / "inputs" / "nacl.vasp"
        _write_structure(structure_path, structure)

        _, enum_artifact = enumerate_unique_sites({"structure_file": "inputs/nacl.vasp"})
        enum_data = enum_artifact["data"]
        assert enum_data["groups_count"] == 2
        output_json = tmp_path / "files" / enum_data["output_json_rel"]
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert {group["species"] for group in payload["groups"]} == {"Na", "Cl"}

        _, vac_artifact = create_vacancy(
            {
                "structure_file": "inputs/nacl.vasp",
                "mode": "one_per_group",
            }
        )
        vac_data = vac_artifact["data"]
        assert vac_data["structures_generated"] == 2
        batch = json.loads((tmp_path / "files" / vac_data["batch_json_rel"]).read_text(encoding="utf-8"))
        assert len(batch["results"]) == 2


def test_substitute_insert_strain_kpath_and_phonon_tools(tmp_path: Path) -> None:
    structure = Structure(
        Lattice.cubic(3.2),
        ["Li"],
        [[0.0, 0.0, 0.0]],
    )
    with workspace_scope(tmp_path):
        structure_path = tmp_path / "files" / "inputs" / "li.vasp"
        _write_structure(structure_path, structure)

        substitute_species(
            {
                "structure_file": "inputs/li.vasp",
                "group_id": 0,
                "new_species": "Na",
                "output_path": "outputs/li_to_na.vasp",
            }
        )
        substituted = Structure.from_file(tmp_path / "files" / "outputs" / "li_to_na.vasp")
        assert substituted[0].species_string == "Na"

        _, insert_artifact = insert_interstitial_at_coords(
            {
                "structure_file": "inputs/li.vasp",
                "species": "H",
                "coords": [[0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
            }
        )
        assert insert_artifact["data"]["structures_generated"] == 2

        _, strain_artifact = generate_strained_structures(
            {
                "structure_file": "inputs/li.vasp",
                "output_dir": "outputs/strains",
                "strain_modes": ["xx", "isotropic"],
                "strain_values": [-0.01, 0.01],
            }
        )
        assert strain_artifact["data"]["structures_generated"] == 4

        generate_kpath(
            {
                "structure_file": "inputs/li.vasp",
                "output_path": "outputs/KPOINTS",
            }
        )
        kpoints = Kpoints.from_file(tmp_path / "files" / "outputs" / "KPOINTS")
        assert kpoints.style == Kpoints.supported_modes.Line_mode

        _, phonon_artifact = generate_phonon_displacements(
            {
                "structure_file": "inputs/li.vasp",
                "output_dir": "outputs/phonons",
                "supercell": [1, 1, 1],
                "displacement": 0.02,
            }
        )
        assert phonon_artifact["data"]["structures_generated"] >= 1


def test_vasp_band_prepare_preserves_line_mode_kpoints_and_chgcar(tmp_path: Path) -> None:
    structure = Structure(Lattice.cubic(3.2), ["Li"], [[0.0, 0.0, 0.0]])
    with workspace_scope(tmp_path):
        structure_path = tmp_path / "files" / "inputs" / "li.vasp"
        _write_structure(structure_path, structure)
        generate_kpath(
            {
                "structure_file": "inputs/li.vasp",
                "output_path": "outputs/band.KPOINTS",
            }
        )
        (tmp_path / "files" / "inputs" / "CHGCAR").write_text("chgcar\n", encoding="utf-8")
        _, artifact = vasp_band_prepare(
            {
                "input_path": "inputs/li.vasp",
                "output_root": "outputs/band_job",
                "kpoints_path": "outputs/band.KPOINTS",
                "charge_density_path": "inputs/CHGCAR",
            }
        )
        data = artifact["data"]
        kpoints = Kpoints.from_file(tmp_path / "files" / data["kpoints_rel"])
        incar_text = (tmp_path / "files" / data["output_root_rel"] / "INCAR").read_text(encoding="utf-8")
        assert kpoints.style == Kpoints.supported_modes.Line_mode
        assert "ICHARG = 11" in incar_text
        assert (tmp_path / "files" / data["output_root_rel"] / "CHGCAR").is_file()
