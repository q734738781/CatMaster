from __future__ import annotations

import pytest
from pydantic import ValidationError

pytest.importorskip("pymatgen")

from catmaster.tools.geometry_inputs.vasp_inputs import StructWriter
from catmaster.tools.geometry_inputs.vasp_prepare import (
    VaspRelaxPrepareInput,
    VaspSPPrepareInput,
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
    assert settings_sp["NSW"] == 0
    assert settings_sp["IBRION"] == -1
    assert "EDIFFG" not in settings_sp


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
