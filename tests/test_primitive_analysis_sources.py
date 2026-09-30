import json

import pytest

from catmaster.tools.base import workspace_scope
from catmaster.tools.analysis.qchem_analysis import analyze_orca_results, analyze_xtb_results
from catmaster.tools.dynamics.cp2k_analysis import cp2k_output_summary
from catmaster.tools.dynamics.lammps_tools import lammps_log_summary, md_trajectory_summary


@pytest.fixture
def files(tmp_path):
    with workspace_scope(tmp_path):
        root = tmp_path / "files"
        root.mkdir(exist_ok=True)
        yield root


@pytest.mark.parametrize("tool,filename,body,key", [
    (cp2k_output_summary, "custom.txt", "ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -12.5\nPROGRAM ENDED AT\n", "summary_json_rel"),
    (lammps_log_summary, "anneal.txt", "LAMMPS (fixture)\nStep Temp TotEng\n0 300 -1\nLoop time of 1\nTotal wall time: 0:00:01\n", "summary_path"),
    (analyze_xtb_results, "xtb.txt", "TOTAL ENERGY -1.5\nnormal termination of xtb\n", "summary_json"),
    (analyze_orca_results, "molecule.txt", "FINAL SINGLE POINT ENERGY -1.5\nORCA TERMINATED NORMALLY\n", "summary_json"),
])
def test_analysis_accepts_explicit_native_file_and_preserves_batch_failure(files, tool, filename, body, key):
    (files / filename).write_text(body)
    content, result = tool({"result_files": [filename, "missing.output"], "output_dir": "out"})
    rows = json.loads((files / result["data"][key]).read_text())["records"]
    assert len(rows) == 2
    assert rows[0]["execution_state"] == "completed"
    assert rows[1]["parse_state"] == "failed"
    assert "parse_failed_count=1" in content


def test_cp2k_discovery_ignores_scheduler_and_keeps_nested_runs(files):
    (files / "runs/child").mkdir(parents=True)
    (files / "runs/a_scheduler.out").write_text("submitted\n")
    body = "ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -12.5\nPROGRAM ENDED AT\n"
    (files / "runs/z_science.out").write_text(body)
    (files / "runs/child/second.out").write_text(body)
    _, result = cp2k_output_summary({"result_root": "runs", "output_dir": "out"})
    rows = json.loads((files / result["data"]["summary_json_rel"]).read_text())["records"]
    assert len(rows) == 2
    assert all(row["energies"]["last"]["hartree"] == -12.5 for row in rows)


def test_cp2k_ambiguity_is_a_visible_per_run_error(files):
    (files / "run").mkdir()
    for name in ("a.out", "b.out"):
        (files / "run" / name).write_text("ENERGY| Total energy -1\n")
    content, result = cp2k_output_summary({"result_root": "run", "output_dir": "out"})
    rows = json.loads((files / result["data"]["summary_json_rel"]).read_text())["records"]
    assert rows[0]["parse_state"] == "failed"
    assert "Multiple matching" in rows[0]["errors"][0]
    assert "parse_failed_count=1" in content


def test_orca_geometry_extraction_does_not_modify_source_directory(files):
    (files / "run").mkdir()
    (files / "run/science.out").write_text("FINAL SINGLE POINT ENERGY -1\nORCA TERMINATED NORMALLY\n")
    (files / "run/science.xyz").write_text("2\ninitial\nH 0 0 0\nH 0 0 .7\n2\nlast\nH 0 0 0\nH 0 0 1.4\n")
    before = {path.name for path in (files / "run").iterdir()}
    _, result = analyze_orca_results({"result_root": "run/science.out", "output_dir": "out"})
    assert {path.name for path in (files / "run").iterdir()} == before
    row = json.loads((files / result["data"]["summary_json"]).read_text())["records"][0]
    assert row["final_structure"].startswith("out/")
    assert "1.400000" in (files / row["final_structure"]).read_text()


def test_orca_input_is_not_reported_as_final_geometry(files):
    (files / "run").mkdir()
    (files / "run/job.out").write_text("FINAL SINGLE POINT ENERGY -1\nORCA TERMINATED NORMALLY\n")
    (files / "run/input.xyz").write_text("1\ninput\nH 0 0 0\n")
    _, result = analyze_orca_results({"result_root": "run", "output_dir": "out"})
    row = json.loads((files / result["data"]["summary_json"]).read_text())["records"][0]
    assert row["final_structure"] is None


@pytest.mark.parametrize("suffix,complete,broken", [
    ("xyz", "2\nframe\nH 0 0 0\nH 0 0 0.7\n", "2\nbroken\nH 0 0 0\n"),
    ("lammpstrj", "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp pp\n0 4\n0 4\n0 4\nITEM: ATOMS id type x y z\n1 1 0 0 0\n2 1 0 0 .7\n", "ITEM: TIMESTEP\n1\nITEM: NUMBER OF ATOMS\n2\n"),
])
def test_truncated_trajectory_retains_complete_frame_without_claiming_final(files, suffix, complete, broken):
    (files / f"input.{suffix}").write_text(complete + broken)
    content, result = md_trajectory_summary({"path": f"input.{suffix}", "output_dir": "out"})
    data = result["data"]
    assert data["nframes"] == 1
    assert data["state"] == "partial"
    assert data["final_frame"] == ""
    assert (files / data["last_complete_frame"]).read_text() == complete
    assert "parse_error=" in content
