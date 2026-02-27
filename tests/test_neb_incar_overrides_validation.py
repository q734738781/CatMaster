from __future__ import annotations

import pytest
from pydantic import ValidationError

from catmaster.tools.geometry_inputs.neb_tools import MakeNebIncarInput


def _base_kwargs() -> dict:
    return {
        "template_incar_path": "tests/assets/O2_VASP_inputs/INCAR",
        "images": 5,
    }


def test_additional_overrides_accepts_element_map_for_magmom() -> None:
    params = MakeNebIncarInput(
        **_base_kwargs(),
        additional_overrides={"magmom": {"O": 1}, "nupdown": 2},
    )
    assert params.additional_overrides["MAGMOM"] == {"O": 1}
    assert params.additional_overrides["NUPDOWN"] == 2


def test_additional_overrides_rejects_magmom_list() -> None:
    with pytest.raises(
        ValidationError,
        match="MAGMOM must be an element-map in this tool due to pymatgen constraints",
    ):
        MakeNebIncarInput(
            **_base_kwargs(),
            additional_overrides={"MAGMOM": [1, 1]},
        )


def test_additional_overrides_rejects_ldauj_symbol_value_form() -> None:
    with pytest.raises(
        ValidationError,
        match="LDAUJ must be an element-map in this tool due to pymatgen constraints",
    ):
        MakeNebIncarInput(
            **_base_kwargs(),
            additional_overrides={"LDAUJ": [{"symbol": "Fe", "value": 0.1}]},
        )


def test_additional_overrides_allows_non_element_map_list_for_other_keys() -> None:
    params = MakeNebIncarInput(
        **_base_kwargs(),
        additional_overrides={"LDAUL": [2, 2]},
    )
    assert params.additional_overrides == {"LDAUL": [2, 2]}
