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
    VaspRelaxPrepareInput,
    VaspSPPrepareInput,
    vasp_relax_prepare,
)


def test_vasp_relax_prepare_input_rejects_legacy_lattice_calc_type() -> None:
    with pytest.raises(ValidationError):
        VaspRelaxPrepareInput(
            input_path="tests/assets/Fe.cif",
            output_root="tests/test_output/relax_prepare",
            calc_type="lattice",
        )


def test_vasp_relax_prepare_input_accepts_user_incar_settings_dict() -> None:
    params = VaspRelaxPrepareInput(
        input_path="tests/assets/Fe.cif",
        output_root="tests/test_output/relax_prepare",
        calc_type="bulk",
        user_incar_settings={"magmom": {"Fe": 2.2}, "NUPDOWN": 2},
    )
    assert params.user_incar_settings["MAGMOM"] == {"Fe": 2.2}
    assert params.user_incar_settings["NUPDOWN"] == 2


def test_user_incar_settings_rejects_magmom_list_with_clear_message() -> None:
    with pytest.raises(
        ValidationError,
        match="MAGMOM must be an element-map in this tool due to pymatgen constraints",
    ):
        VaspRelaxPrepareInput(
            input_path="tests/assets/Fe.cif",
            output_root="tests/test_output/relax_prepare",
            calc_type="bulk",
            user_incar_settings={"MAGMOM": [1, 1]},
        )


def test_user_incar_settings_rejects_symbol_value_list_form() -> None:
    with pytest.raises(
        ValidationError,
        match="MAGMOM must be an element-map in this tool due to pymatgen constraints",
    ):
        VaspRelaxPrepareInput(
            input_path="tests/assets/Fe.cif",
            output_root="tests/test_output/relax_prepare",
            calc_type="bulk",
            user_incar_settings={"MAGMOM": [{"symbol": "Fe", "value": 2.2}]},
        )


def test_user_incar_settings_rejects_ldauu_pair_list_form() -> None:
    with pytest.raises(
        ValidationError,
        match="LDAUU must be an element-map in this tool due to pymatgen constraints",
    ):
        VaspRelaxPrepareInput(
            input_path="tests/assets/Fe.cif",
            output_root="tests/test_output/relax_prepare",
            calc_type="bulk",
            user_incar_settings={"LDAUU": [["Fe", 4.0]]},
        )


def test_vasp_relax_prepare_input_rejects_legacy_kv_list_user_incar_settings() -> None:
    with pytest.raises(ValidationError):
        VaspRelaxPrepareInput(
            input_path="tests/assets/Fe.cif",
            output_root="tests/test_output/relax_prepare",
            calc_type="bulk",
            user_incar_settings=[{"key": "MAGMOM", "value": {"Fe": 2.2}}],
        )


def test_user_incar_settings_normalizes_keys_and_preserves_none() -> None:
    params = VaspRelaxPrepareInput(
        input_path="tests/assets/Fe.cif",
        output_root="tests/test_output/relax_prepare",
        calc_type="bulk",
        user_incar_settings={
            "isym": 0,
            "ISYM": None,
            "MAGMOM": {"O": 1},
        },
    )
    assert params.user_incar_settings == {"ISYM": None, "MAGMOM": {"O": 1}}


@pytest.mark.parametrize("calc_type", ["gas", "slab"])
def test_vasp_relax_prepare_input_rejects_relax_cell_conflict(calc_type: str) -> None:
    with pytest.raises(ValidationError, match="relax_cell=True is not allowed"):
        VaspRelaxPrepareInput(
            input_path="tests/assets/Fe.cif",
            output_root="tests/test_output/relax_prepare",
            calc_type=calc_type,
            relax_cell=True,
        )


def test_struct_writer_required_overrides_bulk_isif_controlled_by_relax_cell() -> None:
    writer = StructWriter()

    bulk_fixed_cell = writer._required_overrides("bulk", False, {})
    bulk_relax_cell = writer._required_overrides("bulk", True, {})

    assert bulk_fixed_cell["ISIF"] == 2
    assert bulk_relax_cell["ISIF"] == 3


def test_struct_writer_lorbit_controlled_by_compute_dos() -> None:
    writer = StructWriter()
    required = writer._required_overrides("bulk", False, {})
    settings_no_dos = writer._build_user_incar_settings(
        calc_type="bulk",
        required_overrides=required,
        use_d3=False,
        use_dft_plus_u=False,
        user_incar_overrides={},
        single_point=False,
        compute_dos=False,
    )
    settings_with_dos = writer._build_user_incar_settings(
        calc_type="bulk",
        required_overrides=required,
        use_d3=False,
        use_dft_plus_u=False,
        user_incar_overrides={},
        single_point=False,
        compute_dos=True,
    )
    assert settings_no_dos["LORBIT"] == 0
    assert settings_with_dos["LORBIT"] == 11


def test_struct_writer_single_point_defaults() -> None:
    writer = StructWriter()
    required = writer._required_overrides("bulk", False, {})
    settings_sp = writer._build_user_incar_settings(
        calc_type="bulk",
        required_overrides=required,
        use_d3=False,
        use_dft_plus_u=False,
        user_incar_overrides={},
        single_point=True,
        compute_dos=False,
    )
    assert settings_sp["NSW"] == 1
    assert settings_sp["IBRION"] == -1
    assert settings_sp["NELM"] == 150
    assert "EDIFFG" not in settings_sp


def test_struct_writer_gas_required_overrides_do_not_force_lreal() -> None:
    writer = StructWriter()
    required = writer._required_overrides("gas", False, {})
    assert required["ISIF"] == 2
    assert required["ISYM"] == 0
    assert required["ISMEAR"] == 0
    assert required["SIGMA"] == 0.01
    assert "LREAL" not in required


def test_struct_writer_use_d3_defaults_to_ivdw_12() -> None:
    writer = StructWriter()
    required = writer._required_overrides("slab", False, {})
    settings = writer._build_user_incar_settings(
        calc_type="slab",
        required_overrides=required,
        use_d3=True,
        use_dft_plus_u=False,
        user_incar_overrides={},
        single_point=False,
        compute_dos=False,
    )
    assert settings["IVDW"] == 12


@pytest.mark.parametrize("calc_type", ["gas", "slab"])
def test_struct_writer_required_overrides_rejects_relax_cell_conflict(calc_type: str) -> None:
    writer = StructWriter()
    with pytest.raises(ValueError, match="relax_cell=True is not allowed"):
        writer._required_overrides(calc_type, True, {})


def test_vasp_sp_prepare_input_defaults() -> None:
    params = VaspSPPrepareInput(input_path="tests/assets/Fe.cif", output_root="tests/test_output/sp_prepare")
    assert params.calc_type == "bulk"
    assert params.compute_dos is False


def test_vasp_relax_prepare_single_file_writes_directly_into_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path("tests/assets/Fe.cif").read_text(encoding="utf-8")
    input_path = tmp_path / "files" / "inputs" / "Fe.cif"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(source, encoding="utf-8")

    def _fake_write(self, structure, output_dir, **kwargs) -> None:
        _ = (self, structure, kwargs)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "INCAR").write_text("SYSTEM = test\n", encoding="utf-8")

    monkeypatch.setattr(StructWriter, "write_vasp_inputs", _fake_write)

    with workspace_scope(tmp_path):
        _content, artifact = vasp_relax_prepare(
            {
                "input_path": "inputs/Fe.cif",
                "output_root": "jobs/relax_single",
                "calc_type": "bulk",
            }
        )

    data = artifact["data"]
    assert data["prepared_directories_rel"] == ["jobs/relax_single"]
    assert (tmp_path / "files" / "jobs" / "relax_single" / "INCAR").is_file()
    assert not (tmp_path / "files" / "jobs" / "relax_single" / "Fe").exists()


def test_vasp_relax_prepare_single_file_refuses_existing_incar_target(
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
            vasp_relax_prepare(
                {
                    "input_path": "inputs/Fe.cif",
                    "output_root": "jobs/occupied_target",
                    "calc_type": "bulk",
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
            calc_type="slab",
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
            calc_type="slab",
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
            calc_type="gas",
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
