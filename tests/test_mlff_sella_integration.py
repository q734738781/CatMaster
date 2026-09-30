"""Run with the Sella dependency set from a remote MLFF environment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms, FixCartesian, FixScaled
from ase.io import read, write

pytest.importorskip("sella")

from catmaster.remote.mlff import mlff_ts


class QuadraticSaddle(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, reference: np.ndarray, hessian: np.ndarray):
        super().__init__()
        self.reference = reference
        self.hessian = hessian

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        displacement = (self.atoms.positions - self.reference).ravel()
        self.results = {
            "energy": float(displacement @ self.hessian @ displacement / 2),
            "forces": -(self.hessian @ displacement).reshape(-1, 3),
        }


@pytest.mark.parametrize("method", ["analytic", "finite_difference"])
@pytest.mark.parametrize("constraint", ["cartesian", "scaled"])
def test_real_sella_refines_and_validates_constrained_saddle(
    tmp_path: Path, method: str, constraint: str,
) -> None:
    reference = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    hessian = np.diag([1.0] * 9 + [-1.0, 2.0, 3.0])
    atoms = Atoms("H4", positions=reference.copy(),
                  cell=[[4, 0, 0], [1, 4, 0], [0.5, 0.2, 4]])
    atoms.positions[3] += [0.1, 0.1, 0.05]
    partial = (FixCartesian(3, mask=[False, False, True]) if constraint == "cartesian"
               else FixScaled(3, mask=[False, True, False]))
    atoms.set_constraint([FixAtoms(indices=[0, 1, 2]), partial])
    source = tmp_path / "input.traj"
    write(source, atoms)
    calculator = QuadraticSaddle(reference, hessian)

    class Adapter:
        provider_version = "test"

        def calculator_for(self, atoms, config):
            return calculator

        def hessian_for(self, atoms, config, calculator):
            return hessian

        def provider_metadata(self, atoms, config, calculator):
            return {}

    output = tmp_path / "output"
    summary = mlff_ts.run_single(
        source=source, output_dir=output, adapter=Adapter(), item_config={},
        config={"backend": "test", "operation": "ts", "task_config": {
            "steps": 40, "fmax": 1e-3, "hessian_method": method,
            "hessian_delta": 1e-3, "imaginary_threshold_cm1": 20.0,
        }},
    )
    assert summary["converged"]
    assert summary["validated_first_order_saddle"]
    assert summary["significant_imaginary_mode_count"] == 1
    assert summary["free_dof"] == 2
    assert summary["max_projected_force_eVA"] < 1e-3
    assert summary["max_cartesian_constraint_drift_A"] < 1e-10
    assert summary["max_scaled_constraint_drift"] < 1e-10
    assert summary["validation_hessian_method"] == method
    if method == "finite_difference":
        assert summary["validation_hessian_force_evaluations"] == 4
    final = read(output / summary["output_structure"])
    np.testing.assert_allclose(final.positions[:3], reference[:3], atol=1e-10)
