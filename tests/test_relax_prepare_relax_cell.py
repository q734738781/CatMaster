from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from pymatgen.core import Lattice, Structure

pytest.importorskip("pymatgen")

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import workspace_scope
from catmaster.tools.geometry_inputs.vasp_inputs import StructWriter
from catmaster.tools.geometry_inputs.vasp_prepare import (
    VaspPrepareInput,
    vasp_prepare,
)


def test_vasp_prepare_input_rejects_legacy_lattice_regime() -> None:
    with pytest.raises(ValidationError):
        VaspPrepareInput(
            input_path="tests/assets/Fe.cif",
            output_root="tests/test_output/relax_prepare",
            preset="relax",
            regime="lattice",
        )


def test_vasp_prepare_input_accepts_user_incar_patch_dict() -> None:
    params = VaspPrepareInput(
        input_path="tests/assets/Fe.cif",
        output_root="tests/test_output/relax_prepare",
        preset="relax",
        regime="bulk",
        user_incar_patch={"magmom": {"Fe": 2.2}, "NUPDOWN": 2},
    )
    assert params.user_incar_patch["MAGMOM"] == {"Fe": 2.2}
    assert params.user_incar_patch["NUPDOWN"] == 2


def test_user_incar_patch_rejects_magmom_list_with_clear_message() -> None:
    with pytest.raises(
        ValidationError,
        match="MAGMOM must be an element-map in this tool due to pymatgen constraints",
    ):
        VaspPrepareInput(
            input_path="tests/assets/Fe.cif",
            output_root="tests/test_output/relax_prepare",
            preset="relax",
            regime="bulk",
            user_incar_patch={"MAGMOM": [1, 1]},
        )


def test_user_incar_patch_rejects_symbol_value_list_form() -> None:
    with pytest.raises(
        ValidationError,
        match="MAGMOM must be an element-map in this tool due to pymatgen constraints",
    ):
        VaspPrepareInput(
            input_path="tests/assets/Fe.cif",
            output_root="tests/test_output/relax_prepare",
            preset="relax",
            regime="bulk",
            user_incar_patch={"MAGMOM": [{"symbol": "Fe", "value": 2.2}]},
        )


def test_user_incar_patch_rejects_ldauu_pair_list_form() -> None:
    with pytest.raises(
        ValidationError,
        match="LDAUU must be an element-map in this tool due to pymatgen constraints",
    ):
        VaspPrepareInput(
            input_path="tests/assets/Fe.cif",
            output_root="tests/test_output/relax_prepare",
            preset="relax",
            regime="bulk",
            user_incar_patch={"LDAUU": [["Fe", 4.0]]},
        )


def test_vasp_prepare_input_rejects_legacy_kv_list_user_incar_patch() -> None:
    with pytest.raises(ValidationError):
        VaspPrepareInput(
            input_path="tests/assets/Fe.cif",
            output_root="tests/test_output/relax_prepare",
            preset="relax",
            regime="bulk",
            user_incar_patch=[{"key": "MAGMOM", "value": {"Fe": 2.2}}],
        )


def test_user_incar_patch_normalizes_keys_and_preserves_none() -> None:
    params = VaspPrepareInput(
        input_path="tests/assets/Fe.cif",
        output_root="tests/test_output/relax_prepare",
        preset="relax",
        regime="bulk",
        user_incar_patch={
            "isym": 0,
            "ISYM": None,
            "MAGMOM": {"O": 1},
        },
    )
    assert params.user_incar_patch == {"ISYM": None, "MAGMOM": {"O": 1}}


def test_vasp_prepare_input_rejects_dos_knobs_outside_dos() -> None:
    with pytest.raises(ValidationError, match="dos_charge_density_path is only allowed"):
        VaspPrepareInput(
            input_path="tests/assets/Fe.cif",
            output_root="tests/test_output/relax_prepare",
            preset="static",
            regime="bulk",
            dos_charge_density_path="tests/assets/CHGCAR",
        )


@pytest.mark.parametrize(
    ("preset", "regime"),
    [("relax", "gas"), ("relax", "slab"), ("static", "bulk"), ("freq", "bulk"), ("dos", "bulk"), ("md", "bulk")],
)
def test_vasp_prepare_input_rejects_relax_cell_conflict(preset: str, regime: str) -> None:
    with pytest.raises(ValidationError, match="relax_cell=True is only allowed"):
        VaspPrepareInput(
            input_path="tests/assets/Fe.cif",
            output_root="tests/test_output/relax_prepare",
            preset=preset,
            regime=regime,
            relax_cell=True,
        )


def test_struct_writer_plan_bulk_isif_controlled_by_relax_cell() -> None:
    writer = StructWriter()
    structure = Structure.from_file("tests/assets/Fe.cif")

    bulk_fixed_cell = writer.plan_vasp_inputs(
        structure=structure,
        output_dir=Path("unused"),
        preset="relax",
        regime="bulk",
        relax_cell=False,
    )
    bulk_relax_cell = writer.plan_vasp_inputs(
        structure=structure,
        output_dir=Path("unused"),
        preset="relax",
        regime="bulk",
        relax_cell=True,
    )

    assert bulk_fixed_cell.user_incar_settings["ISIF"] == 2
    assert bulk_relax_cell.user_incar_settings["ISIF"] == 3


def test_struct_writer_plan_lorbit_controlled_by_compute_dos() -> None:
    writer = StructWriter()
    structure = Structure.from_file("tests/assets/Fe.cif")
    settings_no_dos = writer.plan_vasp_inputs(
        structure=structure,
        output_dir=Path("unused"),
        preset="relax",
        regime="bulk",
        compute_dos=False,
    ).user_incar_settings
    settings_with_dos = writer.plan_vasp_inputs(
        structure=structure,
        output_dir=Path("unused"),
        preset="relax",
        regime="bulk",
        compute_dos=True,
    ).user_incar_settings
    assert settings_no_dos["LORBIT"] == 0
    assert settings_with_dos["LORBIT"] == 11


def test_struct_writer_static_defaults() -> None:
    writer = StructWriter()
    structure = Structure.from_file("tests/assets/Fe.cif")
    settings_sp = writer.plan_vasp_inputs(
        structure=structure,
        output_dir=Path("unused"),
        preset="static",
        regime="bulk",
    ).user_incar_settings
    assert settings_sp["NSW"] == 1
    assert settings_sp["IBRION"] == -1
    assert settings_sp["NELM"] == 150
    assert "EDIFFG" not in settings_sp


def test_struct_writer_freq_defaults_include_frequency_controls() -> None:
    writer = StructWriter()
    structure = Structure.from_file("tests/assets/Fe.cif")
    settings = writer.plan_vasp_inputs(
        structure=structure,
        output_dir=Path("unused"),
        preset="freq",
        regime="slab",
    ).user_incar_settings
    assert settings["IBRION"] == 5
    assert settings["NSW"] == 1
    assert settings["POTIM"] == pytest.approx(0.015)
    assert settings["NFREE"] == 2
    assert settings["ISYM"] == 0


def test_struct_writer_dos_defaults_follow_tetrahedron_style() -> None:
    writer = StructWriter()
    structure = Structure.from_file("tests/assets/Fe.cif")
    settings = writer.plan_vasp_inputs(
        structure=structure,
        output_dir=Path("unused"),
        preset="dos",
        regime="bulk",
    ).user_incar_settings
    assert settings["IBRION"] == -1
    assert settings["NSW"] == 0
    assert settings["ISMEAR"] == -5
    assert settings["NEDOS"] == 2001
    assert settings["LORBIT"] == 11
    assert "ICHARG" not in settings


def test_struct_writer_dos_with_chgcar_sets_icharg() -> None:
    writer = StructWriter()
    structure = Structure.from_file("tests/assets/Fe.cif")
    settings = writer.plan_vasp_inputs(
        structure=structure,
        output_dir=Path("unused"),
        preset="dos",
        regime="bulk",
        dos_use_chgcar=True,
    ).user_incar_settings
    assert settings["ICHARG"] == 11


def test_struct_writer_fixed_density_defaults_add_lmaxmix_for_d_block_systems() -> None:
    writer = StructWriter()
    structure = Structure.from_file("tests/assets/Fe.cif")
    settings = writer.plan_vasp_inputs(
        structure=structure,
        output_dir=Path("unused"),
        preset="dos",
        regime="bulk",
        dos_use_chgcar=True,
    ).user_incar_settings
    assert settings["LMAXMIX"] == 4


def test_struct_writer_fixed_density_respects_explicit_lmaxmix_override() -> None:
    writer = StructWriter()
    structure = Structure.from_file("tests/assets/Fe.cif")
    settings = writer.plan_vasp_inputs(
        structure=structure,
        output_dir=Path("unused"),
        preset="dos",
        regime="bulk",
        dos_use_chgcar=True,
        user_incar_patch={"LMAXMIX": 6},
    ).user_incar_settings
    assert settings["LMAXMIX"] == 6


def test_struct_writer_md_defaults_use_nose_hoover() -> None:
    writer = StructWriter()
    structure = Structure.from_file("tests/assets/Fe.cif")
    settings = writer.plan_vasp_inputs(
        structure=structure,
        output_dir=Path("unused"),
        preset="md",
        regime="slab",
    ).user_incar_settings
    assert settings["IBRION"] == 0
    assert settings["NSW"] == 1000
    assert settings["POTIM"] == pytest.approx(1.0)
    assert settings["MDALGO"] == 2
    assert settings["SMASS"] == pytest.approx(0.0)
    assert settings["TEBEG"] == pytest.approx(300.0)
    assert settings["TEEND"] == pytest.approx(300.0)
    assert settings["ISYM"] == 0


def test_struct_writer_dimer_defaults_follow_relax_with_ibrion_44() -> None:
    writer = StructWriter()
    structure = Structure.from_file("tests/assets/Fe.cif")
    settings = writer.plan_vasp_inputs(
        structure=structure,
        output_dir=Path("unused"),
        preset="dimer",
        regime="slab",
    ).user_incar_settings
    assert settings["IBRION"] == 44
    assert settings["NSW"] == 500
    assert settings["EDIFFG"] == pytest.approx(-0.02)


def test_struct_writer_gas_defaults_keep_isym_zero_and_1x1x1() -> None:
    writer = StructWriter()
    structure = Structure(
        lattice=Lattice.cubic(12.0),
        species=["H", "H"],
        coords=[[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]],
    )
    plan = writer.plan_vasp_inputs(
        structure=structure,
        output_dir=Path("unused"),
        preset="static",
        regime="gas",
    )
    assert plan.user_incar_settings["ISIF"] == 2
    assert plan.user_incar_settings["ISYM"] == 0
    assert plan.user_incar_settings["ISMEAR"] == 0
    assert plan.user_incar_settings["SIGMA"] == 0.01
    assert plan.k_grid == (1, 1, 1)


def test_struct_writer_use_d3_defaults_to_ivdw_12() -> None:
    writer = StructWriter()
    structure = Structure.from_file("tests/assets/Fe.cif")
    settings = writer.plan_vasp_inputs(
        structure=structure,
        output_dir=Path("unused"),
        preset="relax",
        regime="slab",
        use_d3=True,
    ).user_incar_settings
    assert settings["IVDW"] == 12


def test_struct_writer_safe_patch_blocks_protected_override() -> None:
    writer = StructWriter()
    structure = Structure.from_file("tests/assets/Fe.cif")
    with pytest.raises(ValueError, match="patch_policy='safe'"):
        writer.plan_vasp_inputs(
            structure=structure,
            output_dir=Path("unused"),
            preset="static",
            regime="bulk",
            user_incar_patch={"IBRION": 2},
            patch_policy="safe",
        )


def test_struct_writer_safe_patch_allows_dos_method_override() -> None:
    writer = StructWriter()
    structure = Structure.from_file("tests/assets/Fe.cif")
    plan = writer.plan_vasp_inputs(
        structure=structure,
        output_dir=Path("unused"),
        preset="dos",
        regime="bulk",
        user_incar_patch={"NEDOS": 4001, "ISMEAR": 0},
        patch_policy="safe",
    )
    assert plan.user_incar_settings["NEDOS"] == 4001
    assert plan.user_incar_settings["ISMEAR"] == 0


def test_struct_writer_safe_patch_allows_md_method_override() -> None:
    writer = StructWriter()
    structure = Structure.from_file("tests/assets/Fe.cif")
    plan = writer.plan_vasp_inputs(
        structure=structure,
        output_dir=Path("unused"),
        preset="md",
        regime="bulk",
        user_incar_patch={"MDALGO": 3, "NSW": 2500, "TEBEG": 500},
        patch_policy="safe",
    )
    assert plan.user_incar_settings["MDALGO"] == 3
    assert plan.user_incar_settings["NSW"] == 2500
    assert plan.user_incar_settings["TEBEG"] == 500


def test_struct_writer_safe_patch_allows_non_protected_override() -> None:
    writer = StructWriter()
    structure = Structure.from_file("tests/assets/Fe.cif")
    plan = writer.plan_vasp_inputs(
        structure=structure,
        output_dir=Path("unused"),
        preset="static",
        regime="bulk",
        user_incar_patch={"ALGO": "Fast", "LORBIT": 11},
        patch_policy="safe",
    )
    assert plan.user_incar_settings["ALGO"] == "Fast"
    assert plan.user_incar_settings["LORBIT"] == 11


def test_struct_writer_force_patch_allows_canonical_override_and_removal() -> None:
    writer = StructWriter()
    structure = Structure.from_file("tests/assets/Fe.cif")
    plan = writer.plan_vasp_inputs(
        structure=structure,
        output_dir=Path("unused"),
        preset="static",
        regime="bulk",
        user_incar_patch={"IBRION": 8, "LWAVE": None},
        patch_policy="force",
    )
    assert plan.user_incar_settings["IBRION"] == 8
    assert "LWAVE" not in plan.user_incar_settings
    assert "LWAVE" in plan.removal_keys


def test_vasp_prepare_input_defaults() -> None:
    params = VaspPrepareInput(
        input_path="tests/assets/Fe.cif",
        output_root="tests/test_output/sp_prepare",
        preset="static",
        regime="bulk",
    )
    assert params.preset == "static"
    assert params.regime == "bulk"
    assert params.compute_dos is False


def test_vasp_prepare_dos_copies_chgcar_and_reports_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path("tests/assets/Fe.cif").read_text(encoding="utf-8")
    input_path = tmp_path / "files" / "inputs" / "Fe.cif"
    chgcar_path = tmp_path / "files" / "references" / "CHGCAR"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    chgcar_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(source, encoding="utf-8")
    chgcar_path.write_text("fake chgcar\n", encoding="utf-8")

    def _fake_write(self, structure, output_dir, **kwargs):
        _ = (self, structure, kwargs)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "INCAR").write_text("SYSTEM = test\n", encoding="utf-8")
        return self.plan_vasp_inputs(
            structure=Structure.from_file(input_path),
            output_dir=out,
            preset="dos",
            regime="bulk",
            dos_use_chgcar=True,
        )

    monkeypatch.setattr(StructWriter, "write_vasp_inputs", _fake_write)

    with workspace_scope(tmp_path):
        _content, artifact = vasp_prepare(
            {
                "input_path": "inputs/Fe.cif",
                "output_root": "jobs/dos_single",
                "preset": "dos",
                "regime": "bulk",
                "dos_charge_density_path": "references/CHGCAR",
            }
        )

    output_root = tmp_path / "files" / "jobs" / "dos_single"
    data = artifact["data"]
    assert data["dos_charge_density_mode"] == "fixed_chgcar"
    assert data["dos_charge_density_path_rel"] == "references/CHGCAR"
    assert (output_root / "CHGCAR").read_text(encoding="utf-8") == "fake chgcar\n"


def test_vasp_prepare_single_file_writes_directly_into_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path("tests/assets/Fe.cif").read_text(encoding="utf-8")
    input_path = tmp_path / "files" / "inputs" / "Fe.cif"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(source, encoding="utf-8")

    def _fake_write(self, structure, output_dir, **kwargs):
        _ = (self, structure, kwargs)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "INCAR").write_text("SYSTEM = test\n", encoding="utf-8")
        return self.plan_vasp_inputs(
            structure=Structure.from_file(input_path),
            output_dir=out,
            preset="relax",
            regime="bulk",
        )

    monkeypatch.setattr(StructWriter, "write_vasp_inputs", _fake_write)

    with workspace_scope(tmp_path):
        _content, artifact = vasp_prepare(
            {
                "input_path": "inputs/Fe.cif",
                "output_root": "jobs/relax_single",
                "preset": "relax",
                "regime": "bulk",
            }
        )

    data = artifact["data"]
    assert data["prepared_directory_rel"] == "jobs/relax_single"
    assert (tmp_path / "files" / "jobs" / "relax_single" / "INCAR").is_file()
    assert not (tmp_path / "files" / "jobs" / "relax_single" / "Fe").exists()


def test_vasp_prepare_single_file_refuses_existing_incar_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path("tests/assets/Fe.cif").read_text(encoding="utf-8")
    input_path = tmp_path / "files" / "inputs" / "Fe.cif"
    output_root = tmp_path / "files" / "jobs" / "occupied_target"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    input_path.write_text(source, encoding="utf-8")
    (output_root / "INCAR").write_text("SYSTEM = existing\n", encoding="utf-8")

    def _should_not_write(self, structure, output_dir, **kwargs) -> None:
        raise AssertionError("write_vasp_inputs should not run when target INCAR already exists")

    monkeypatch.setattr(StructWriter, "write_vasp_inputs", _should_not_write)

    with workspace_scope(tmp_path):
        with pytest.raises(CatMasterToolExecutionError, match="refused to overwrite existing VASP inputs"):
            vasp_prepare(
                {
                    "input_path": "inputs/Fe.cif",
                    "output_root": "jobs/occupied_target",
                    "preset": "relax",
                    "regime": "bulk",
                }
            )


def test_vasp_prepare_directory_input_is_rejected(tmp_path: Path) -> None:
    inputs_dir = tmp_path / "files" / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    with workspace_scope(tmp_path):
        with pytest.raises(CatMasterToolExecutionError, match="single structure file input"):
            vasp_prepare(
                {
                    "input_path": "inputs",
                    "output_root": "jobs/dir_case",
                    "preset": "relax",
                    "regime": "bulk",
                }
            )

def test_struct_writer_compute_com_frac_dipol_matches_mvlslabset_style_average() -> None:
    writer = StructWriter()
    structure = Structure.from_file("tests/assets/Fe_bcc_111__CONTCAR_h111_t0.vasp")

    dipol = writer._compute_com_frac_dipol(structure)
    weights = [float(site.species.weight) for site in structure.sites]
    expected = [
        float(v % 1.0)
        for v in __import__("numpy").average(structure.frac_coords, weights=weights, axis=0).tolist()
    ]

    assert dipol == pytest.approx(expected)
    assert isinstance(dipol, list)


def test_struct_writer_writes_dipol_as_space_separated_values(tmp_path: Path) -> None:
    writer = StructWriter()
    structure = Structure.from_file("tests/assets/Fe_bcc_111__CONTCAR_h111_t0.vasp")
    with workspace_scope(tmp_path):
        writer.write_vasp_inputs(
            structure=structure,
            output_dir=Path("vasp_inputs"),
            preset="relax",
            regime="slab",
            enable_dipole=True,
        )
    incar_text = (tmp_path / "files" / "vasp_inputs" / "INCAR").read_text(encoding="utf-8")
    dipol_line = next(line for line in incar_text.splitlines() if line.startswith("DIPOL"))
    assert "(" not in dipol_line
    assert ")" not in dipol_line
    assert "," not in dipol_line


def test_struct_writer_preserves_input_species_order_in_poscar(tmp_path: Path) -> None:
    writer = StructWriter()
    structure = Structure(
        lattice=Lattice.cubic(10.0),
        species=["Pt", "Pt", "H"],
        coords=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.25, 0.25, 0.75],
        ],
    )

    with workspace_scope(tmp_path):
        writer.write_vasp_inputs(
            structure=structure,
            output_dir=Path("vasp_inputs_order"),
            preset="relax",
            regime="slab",
        )

    poscar_lines = (tmp_path / "files" / "vasp_inputs_order" / "POSCAR").read_text(encoding="utf-8").splitlines()
    assert poscar_lines[5].split() == ["Pt", "H"]
    assert poscar_lines[6].split() == ["2", "1"]


def test_struct_writer_preserves_duplicate_species_groups_for_potcar(tmp_path: Path) -> None:
    writer = StructWriter()
    structure = Structure(
        lattice=Lattice.cubic(10.0),
        species=["H", "O", "H"],
        coords=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.75],
        ],
    )

    with workspace_scope(tmp_path):
        writer.write_vasp_inputs(
            structure=structure,
            output_dir=Path("vasp_inputs_duplicate_groups"),
            preset="relax",
            regime="gas",
        )

    output_dir = tmp_path / "files" / "vasp_inputs_duplicate_groups"
    poscar_lines = (output_dir / "POSCAR").read_text(encoding="utf-8").splitlines()
    assert poscar_lines[5].split() == ["H", "O", "H"]
    assert poscar_lines[6].split() == ["1", "1", "1"]

    potcar_text = (output_dir / "POTCAR").read_text(encoding="utf-8")
    titel_lines = [line for line in potcar_text.splitlines() if "TITEL" in line]
    assert len(titel_lines) == 3
    assert "H" in titel_lines[0]
    assert "O" in titel_lines[1]
    assert "H" in titel_lines[2]
