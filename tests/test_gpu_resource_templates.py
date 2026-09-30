from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from dpdispatcher import Machine, Submission, Task
import pytest
import yaml

from catmaster.tools.execution.dpdispatcher_runner import _build_resources


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "dpdispatcher"
RESOURCES = yaml.safe_load((CONFIG / "resources_template.yaml").read_text())
MLFF_CARDS = ("mace_gpu", "uma_gpu", "mattersim_gpu", "orb_gpu")


def _job_script(root: Path, name: str, commands: list[str], lock_dir: Path,
                *, gpus: int = 2, blas_threads: str = "") -> Path:
    root.mkdir()
    cfg = dict(RESOURCES[name])
    cfg["gpu_per_node"] = gpus
    # Exercise the actual provider script, with conda activation stubbed only.
    env = root / "environment.sh"
    provider = name.removesuffix("_gpu")
    env.write_text((CONFIG / "env_templates" / f"catmaster_env_{provider}.sh").read_text())
    cfg["source_list"] = [str(env)]
    cfg["prepend_script"] = [
        line.replace('"$HOME/.cache/catmaster"', shlex.quote(str(lock_dir)))
        .replace('"$HOME/.cache/catmaster/gpu-batch.lock"', shlex.quote(str(lock_dir / "gpu-batch.lock")))
        for line in cfg["prepend_script"]
    ]
    if blas_threads:
        cfg["envs"] = {"OPENBLAS_NUM_THREADS": blas_threads, "MKL_NUM_THREADS": blas_threads}
    resources = _build_resources(cfg, "conda() { :; }")
    machine = Machine.load_from_dict({
        "batch_type": "Shell", "context_type": "LocalContext",
        "local_root": str(root), "remote_root": str(root),
    })
    submission = Submission(work_base=".", machine=machine, resources=resources,
                            task_list=[Task(command=c, task_work_path=".") for c in commands])
    submission.generate_jobs()
    assert len(submission.belonging_jobs) == 1
    job = submission.belonging_jobs[0]
    # Shell.do_submit writes the command body beside the job header.
    remote = Path(machine.context.remote_root)
    remote.mkdir(parents=True, exist_ok=True)
    (remote / (job.script_file_name + ".run")).write_text(machine.gen_script_command(job))
    script = root / "job.sh"
    script.write_text(machine.gen_script(job))
    return script


@pytest.mark.parametrize("card", MLFF_CARDS)
def test_native_gpu_waves_bind_devices_and_limit_blas(tmp_path: Path, card: str) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text('''import fcntl, json, os, pathlib, sys, time
root = pathlib.Path(sys.argv[1])
gpu = os.environ['CUDA_VISIBLE_DEVICES']
with (root / ('gpu-' + gpu)).open('w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / ('ready-' + gpu)).touch()
    deadline = time.monotonic() + 10
    while len(list(root.glob('ready-*'))) < 2:
        if time.monotonic() > deadline:
            raise TimeoutError('The second GPU task did not start concurrently')
        time.sleep(0.01)
    with (root / ('result-' + sys.argv[2])).open('w') as out:
        json.dump(dict(gpu=gpu, blas=os.environ['OPENBLAS_NUM_THREADS'],
                       mkl=os.environ['MKL_NUM_THREADS']), out)
    time.sleep(0.05)
''')
    commands = [shlex.join([sys.executable, str(probe), str(tmp_path), str(i)]) for i in range(5)]
    script = _job_script(tmp_path / "job", card, commands, tmp_path / "locks")
    # Resource settings must override an inherited host-wide thread count.
    env = dict(os.environ, OPENBLAS_NUM_THREADS="64", MKL_NUM_THREADS="64")
    subprocess.run(["bash", str(script)], env=env, check=True, capture_output=True, timeout=20)
    results = [json.loads(p.read_text()) for p in tmp_path.glob("result-*")]
    assert len(results) == 5
    assert sorted(r["gpu"] for r in results) == ["0", "0", "0", "1", "1"]
    assert all(r["blas"] == r["mkl"] == "1" for r in results)


def test_resource_blas_override_survives_provider_activation(tmp_path: Path) -> None:
    output = tmp_path / "threads.txt"
    command = 'printf "%s %s" "$OPENBLAS_NUM_THREADS" "$MKL_NUM_THREADS" > ' + shlex.quote(str(output))
    script = _job_script(tmp_path / "job", "uma_gpu", [command], tmp_path / "locks", blas_threads="3")
    subprocess.run(["bash", str(script)], check=True, capture_output=True, timeout=10)
    assert output.read_text() == "3 3"


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + 10
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(0.01)


def test_shell_batches_share_lock_across_providers(tmp_path: Path) -> None:
    held = tmp_path / "held"
    release = tmp_path / "release"
    events = tmp_path / "events"
    probe = tmp_path / "hold.py"
    probe.write_text('''import pathlib, sys, time
held, release, events = map(pathlib.Path, sys.argv[1:])
held.touch()
deadline = time.monotonic() + 10
while not release.exists():
    if time.monotonic() > deadline: raise TimeoutError('release')
    time.sleep(0.01)
with events.open('a') as out: out.write('first finished\\n')
''')
    first = _job_script(tmp_path / "first", "mace_gpu", [
        shlex.join([sys.executable, str(probe), str(held), str(release), str(events)])
    ], tmp_path / "locks", gpus=1)
    second = _job_script(tmp_path / "second", "uma_gpu", [
        "echo 'second started' >> " + shlex.quote(str(events))
    ], tmp_path / "locks", gpus=1)
    activated = tmp_path / "second-activated"
    with (second.parent / "environment.sh").open("a") as env:
        env.write("touch " + shlex.quote(str(activated)) + "\n")
    processes = []
    try:
        processes.append(subprocess.Popen(["bash", str(first)], stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        _wait_for(held)
        processes.append(subprocess.Popen(["bash", str(second)], stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        _wait_for(activated)
        release.touch()
        for process in processes:
            stdout, stderr = process.communicate(timeout=15)
            assert process.returncode == 0, (stdout, stderr)
    finally:
        release.touch()
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    assert events.read_text().splitlines() == ["first finished", "second started"]


def test_cpu_cards_do_not_inherit_gpu_thread_limits_or_binding() -> None:
    for cfg in RESOURCES.values():
        if cfg["machine"] == "gpu_server":
            # Every card sharing the Shell GPU pool participates in its lock.
            assert cfg["prepend_script"] == RESOURCES["uma_gpu"]["prepend_script"]
            assert cfg["strategy"]["if_cuda_multi_devices"] is True
            continue
        assert "OPENBLAS_NUM_THREADS" not in cfg.get("envs", {})
        assert not cfg.get("strategy", {}).get("if_cuda_multi_devices")
