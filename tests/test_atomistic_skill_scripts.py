from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write as ase_write


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = (
    REPO_ROOT
    / "skills"
    / "atomistic"
    / "atomic-structure-validation-and-recovery"
    / "scripts"
    / "check_atomic_structure.py"
)
ASSEMBLER = (
    REPO_ROOT
    / "skills"
    / "atomistic"
    / "constraint-guided-atomic-assembly"
    / "scripts"
    / "rigid_fragment_assembly.py"
)
LITERATURE_RECONSTRUCTION_SKILL = (
    REPO_ROOT
    / "skills"
    / "atomistic"
    / "literature-figure-guided-structure-reconstruction"
    / "SKILL.md"
)


def _run_script(*args: object) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, *(str(arg) for arg in args)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_literature_figure_guidance_is_optional_and_non_gating() -> None:
    text = LITERATURE_RECONSTRUCTION_SKILL.read_text(encoding="utf-8")
    assert "optional modeling evidence, not a structure-acceptance gate" in text
    assert "Do not ask the VLM to invent Cartesian coordinates" in text
    assert "The parent worker remains responsible" in text


def test_checker_fails_cross_boundary_overlap_even_when_contact_is_expected(
    tmp_path: Path,
) -> None:
    structure_path = tmp_path / "periodic_overlap.extxyz"
    context_path = tmp_path / "periodic_overlap.context.json"
    report_path = tmp_path / "periodic_overlap.report.json"
    atoms = Atoms(
        "HH",
        positions=[[0.10, 5.0, 5.0], [9.95, 5.0, 5.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    ase_write(structure_path, atoms)
    context_path.write_text(
        json.dumps(
            {
                "body_ranges": [
                    {"name": "host", "start": 0, "stop": 1},
                    {"name": "guest", "start": 1, "stop": 2},
                ],
                "expected_contacts": [
                    {
                        "label": "deliberately_too_short",
                        "i": 0,
                        "j": 1,
                        "target": 0.15,
                        "tolerance": 0.01,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_script(
        CHECKER,
        structure_path,
        "--context",
        context_path,
        "--output",
        report_path,
    )

    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert report["expected_contacts"][0]["status"] == "PASS"
    shortest = report["shortest_pairs"][0]
    assert shortest["scope"] == "inter_body"
    assert shortest["severity"] == "FAIL"
    assert shortest["distance_A"] == pytest.approx(0.15)
    assert shortest["image_shift"] != [0, 0, 0]
    assert "flagged_extreme=1-2 H-H" in result.stdout
    assert "scope=inter_body" in result.stdout
    assert "flagged_pairs" not in result.stdout


@pytest.mark.parametrize("verbose", [False, True])
def test_checker_pass_keeps_pair_table_in_report(tmp_path: Path, verbose: bool) -> None:
    structure = tmp_path / "chain.xyz"
    report_path = tmp_path / "report.json"
    ase_write(structure, Atoms("C30", positions=[[i * 1.5, 0, 0] for i in range(30)]))
    result = _run_script(CHECKER, structure, "--output", report_path, *(["--verbose"] if verbose else []))
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text())
    assert report["status"] == "PASS"
    assert len(report["shortest_pairs"]) == 20
    assert len(report["coordination"]["counts"]) == 30
    assert "status=PASS" in result.stdout and str(report_path) in result.stdout
    assert ("shortest_pairs" in result.stdout) == verbose
    if verbose:
        assert json.dumps(report, indent=2, ensure_ascii=False) in result.stdout


def test_checker_reports_contact_failure_when_pair_distances_are_normal(tmp_path: Path) -> None:
    structure = tmp_path / "pair.xyz"
    context = tmp_path / "context.json"
    report_path = tmp_path / "report.json"
    ase_write(structure, Atoms("HH", positions=[[0, 0, 0], [0.74, 0, 0]]))
    context.write_text(json.dumps({"expected_contacts": [
        {"i": 0, "j": 1, "target": 1.0, "tolerance": 0.05}
    ]}))
    result = _run_script(CHECKER, structure, "--context", context, "--output", report_path)
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads(report_path.read_text())
    assert report["flagged_pairs"] == []
    assert "failed_expected_contacts=1 worst=1-2" in result.stdout
    assert "target=1.000000A" in result.stdout


@pytest.mark.parametrize("verbose", [False, True])
def test_assembly_failure_reports_violations_without_dumping_metrics(tmp_path: Path, verbose: bool) -> None:
    host = tmp_path / "host.xyz"
    guest = tmp_path / "guest.xyz"
    ase_write(host, Atoms("C", positions=[[0, 0, 0]]))
    ase_write(guest, Atoms("C", positions=[[0, 0, 0]]))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "host": {"path": str(host)},
        "fragments": [{"name": "guest", "path": str(guest)}],
        "anchors": [
            {"a": {"body": "host", "atom": 0}, "b": {"body": "guest", "atom": 0},
             "target": target, "tolerance": 0.0} for target in (1.0, 3.0)
        ],
        "settings": {"starts": 1, "maxiter": 5, "seed": 1},
    }))
    output = tmp_path / "assembled"
    result = _run_script(ASSEMBLER, spec, "--output-dir", output, *(["--verbose"] if verbose else []))
    assert result.returncode == 2, result.stdout + result.stderr
    report = json.loads((output / "assembly_report.json").read_text())
    assert len(report["best_trial"]["metrics"]["anchors"]) == 2
    assert "max_anchor_violation_A=" in result.stdout
    assert ("anchors" in result.stdout) == verbose
    if verbose:
        assert json.dumps(report, indent=2, ensure_ascii=False) in result.stdout
    assert (output / "best_infeasible.extxyz").exists()


def test_rigid_assembly_emits_candidate_and_validation_context(tmp_path: Path) -> None:
    host_path = tmp_path / "pt_host.extxyz"
    fragment_path = tmp_path / "co.extxyz"
    spec_path = tmp_path / "assembly.json"
    output_dir = tmp_path / "assembled"
    checker_report_path = tmp_path / "candidate_geometry.json"

    ase_write(
        host_path,
        Atoms(
            "Pt",
            positions=[[5.0, 5.0, 5.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
    )
    ase_write(fragment_path, Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.13]]))
    spec_path.write_text(
        json.dumps(
            {
                "host": {"path": str(host_path)},
                "fragments": [
                    {
                        "name": "co",
                        "path": str(fragment_path),
                        "placement": {
                            "fragment_atom": 0,
                            "target": {"body": "host", "atom": 0},
                            "distance": 1.85,
                            "direction": [0.0, 0.0, 1.0],
                        },
                        "align": {
                            "axis_atoms": [0, 1],
                            "target_vector": [0.0, 0.0, 1.0],
                        },
                    }
                ],
                "anchors": [
                    {
                        "label": "Pt-C",
                        "a": {"body": "host", "atom": 0},
                        "b": {"body": "co", "atom": 0},
                        "target": 1.85,
                        "tolerance": 0.03,
                        "weight": 200.0,
                    }
                ],
                "orientations": [
                    {
                        "body": "co",
                        "axis_atoms": [0, 1],
                        "target_vector": [0.0, 0.0, 1.0],
                        "target_angle_deg": 0.0,
                        "tolerance_deg": 5.0,
                        "weight": 20.0,
                    }
                ],
                "settings": {
                    "starts": 2,
                    "keep": 1,
                    "maxiter": 80,
                    "seed": 7,
                    "output_format": "extxyz",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assembly_result = _run_script(ASSEMBLER, spec_path, "--output-dir", output_dir)
    assert assembly_result.returncode == 0, assembly_result.stdout + assembly_result.stderr
    assembly_report = json.loads(
        (output_dir / "assembly_report.json").read_text(encoding="utf-8")
    )
    assert assembly_report["status"] == "PASS"
    assert assembly_report["selected_candidate_count"] == 1
    candidate = assembly_report["candidates"][0]
    candidate_path = Path(candidate["structure_path"])
    context_path = Path(candidate["validation_context_path"])
    assert candidate_path.is_file()
    assert context_path.is_file()

    checker_result = _run_script(
        CHECKER,
        candidate_path,
        "--context",
        context_path,
        "--output",
        checker_report_path,
    )
    assert checker_result.returncode == 0, checker_result.stdout + checker_result.stderr
    checker_report = json.loads(checker_report_path.read_text(encoding="utf-8"))
    assert checker_report["status"] in {"PASS", "WARN"}
    assert checker_report["severity_counts"]["FAIL"] == 0
    assert checker_report["expected_contacts"] == [
        {
            "label": "Pt-C",
            "i": 0,
            "j": 1,
            "i_1based": 1,
            "j_1based": 2,
            "target_A": 1.85,
            "tolerance_A": 0.03,
            "distance_A": pytest.approx(1.85),
            "absolute_error_A": pytest.approx(0.0, abs=1.0e-7),
            "status": "PASS",
        }
    ]


def test_rigid_assembly_resolves_an_unintended_single_frame_overlap(
    tmp_path: Path,
) -> None:
    host_path = tmp_path / "two_sites.extxyz"
    fragment_path = tmp_path / "guest.xyz"
    spec_path = tmp_path / "overlap_assembly.json"
    output_dir = tmp_path / "overlap_assembled"

    ase_write(
        host_path,
        Atoms("Pt2", positions=[[0.0, 0.0, 0.0], [1.85, 0.0, 0.0]]),
    )
    ase_write(fragment_path, Atoms("H", positions=[[0.0, 0.0, 0.0]]))
    spec_path.write_text(
        json.dumps(
            {
                "host": {"path": str(host_path)},
                "fragments": [
                    {
                        "name": "guest",
                        "path": str(fragment_path),
                        "placement": {
                            "fragment_atom": 0,
                            "target": {"body": "host", "atom": 0},
                            "distance": 1.85,
                            "direction": [1.0, 0.0, 0.0],
                        },
                    }
                ],
                "anchors": [
                    {
                        "a": {"body": "host", "atom": 0},
                        "b": {"body": "guest", "atom": 0},
                        "target": 1.85,
                        "tolerance": 0.03,
                        "weight": 200.0,
                    }
                ],
                "settings": {
                    "starts": 16,
                    "keep": 1,
                    "seed": 19,
                    "maxiter": 250,
                    "translation_jitter": 0.6,
                    "output_format": "extxyz",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_script(ASSEMBLER, spec_path, "--output-dir", output_dir)

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(
        (output_dir / "assembly_report.json").read_text(encoding="utf-8")
    )
    best = report["best_trial"]
    assert report["status"] == "PASS"
    assert report["feasible_trial_count"] >= 1
    assert best["metrics"]["max_penetration_A"] <= 0.01
    assert best["metrics"]["losses"]["clash"] == pytest.approx(0.0, abs=1.0e-12)


def test_rigid_assembly_rejects_a_positionally_unconstrained_fragment(
    tmp_path: Path,
) -> None:
    host_path = tmp_path / "host.extxyz"
    fragment_path = tmp_path / "fragment.xyz"
    spec_path = tmp_path / "unconstrained.json"
    ase_write(host_path, Atoms("Pt", positions=[[0.0, 0.0, 0.0]]))
    ase_write(fragment_path, Atoms("H", positions=[[0.0, 0.0, 0.0]]))
    spec_path.write_text(
        json.dumps(
            {
                "host": {"path": str(host_path)},
                "fragments": [{"name": "guest", "path": str(fragment_path)}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_script(
        ASSEMBLER,
        spec_path,
        "--output-dir",
        tmp_path / "unconstrained_output",
    )

    assert result.returncode != 0
    assert "Every fragment needs an anchor or region" in result.stderr
