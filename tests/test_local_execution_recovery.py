import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


@pytest.mark.parametrize("phase", ["inside_graph", "before_delivery", "before_step_commit", "after_child_acceptance"])
def test_process_restart_preserves_completed_work_and_input(tmp_path, phase):
    script = Path(__file__).parent / "fixtures/local_execution_recovery.py"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    with (tmp_path / "launch.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(script), str(tmp_path), "launch", phase],
                                   env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 25
            marker = tmp_path / ("waiting" if phase == "inside_graph" else "crashed")
            while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.03)
            assert marker.exists(), (tmp_path / "launch.log").read_text()[-5000:]
            if phase == "inside_graph":
                process.kill()
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    (tmp_path / "release").write_text("resume")
    result = subprocess.run([sys.executable, str(script), str(tmp_path), "recover", phase],
                            env=env, capture_output=True, text=True, timeout=50)
    assert result.returncode == 0, result.stderr[-5000:]
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["result"]["status"] == "success"
    assert report["human_messages"] == 1
    assert report["completed_work_count"] == 1
    assert report["final"] == "Recovered saved evidence"
    assert report["child_threads"] == (1 if phase == "after_child_acceptance" else 0)
