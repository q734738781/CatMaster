from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import paramiko
from paramiko.proxy import ProxyCommand
from paramiko.config import SSHConfig
import yaml


def load_worker_host(project: str, worker: str, *, config_home: Optional[Path] = None) -> str:
    """
    Load host alias for a jobflow-remote worker from ~/.jfremote/{project}.yaml.

    Args:
        project: Project name in jobflow-remote config.
        worker: Worker key within the project config.
        config_home: Optional override for ~/.jfremote.

    Returns:
        Host alias string that should match an entry in ~/.ssh/config.
    """
    base = config_home or (Path.home() / ".jfremote")
    cfgp = base / f"{project}.yaml"
    data = yaml.safe_load(cfgp.read_text()) if cfgp.is_file() else {}
    return data.get("workers", {}).get(worker, {}).get("host", worker)


def connect_via_ssh_config(host_alias: str) -> paramiko.SSHClient:
    """
    Open a Paramiko SSHClient using settings resolved from ~/.ssh/config for host_alias.

    - Supports ProxyCommand
    - Loads system known_hosts and accepts new keys automatically if missing
    """
    ssh_cfg_path = os.path.expanduser("~/.ssh/config")
    cfg = SSHConfig()
    looked: Dict[str, str] = {}
    if os.path.exists(ssh_cfg_path):
        with open(ssh_cfg_path) as f:
            cfg.parse(f)
        looked = cfg.lookup(host_alias)
    hostname = looked.get("hostname", host_alias)
    username = looked.get("user")
    port = int(looked.get("port", 22))
    keyfiles = looked.get("identityfile", [])
    proxycmd = looked.get("proxycommand")
    sock = ProxyCommand(proxycmd) if proxycmd else None

    cli = paramiko.SSHClient()
    cli.load_system_host_keys()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(
        hostname=hostname,
        port=port,
        username=username,
        key_filename=keyfiles or None,
        sock=sock,
        allow_agent=True,
        look_for_keys=True,
        banner_timeout=60,
        auth_timeout=60,
        timeout=60,
    )
    return cli


def sftp_mkdirs(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    """
    Create remote_dir and parents if missing. Idempotent.
    """
    cur = "/"
    for p in remote_dir.strip("/").split("/"):
        cur = os.path.join(cur, p)
        try:
            sftp.stat(cur)
        except IOError:
            sftp.mkdir(cur, mode=0o755)


def sftp_upload_tree(host_alias: str, local_dir: Path, remote_dir: str) -> None:
    """
    Recursively upload a local directory tree to remote_dir via SFTP.
    """
    cli = connect_via_ssh_config(host_alias)
    try:
        sftp = cli.open_sftp()
        sftp_mkdirs(sftp, remote_dir)
        for root, dirs, files in os.walk(local_dir):
            rel = os.path.relpath(root, local_dir)
            rdir = remote_dir if rel == "." else os.path.join(remote_dir, rel)
            sftp_mkdirs(sftp, rdir)
            for d in dirs:
                sftp_mkdirs(sftp, os.path.join(rdir, d))
            for f in files:
                lp = os.path.join(root, f)
                rp = os.path.join(rdir, f)
                sftp.put(lp, rp)
        sftp.close()
    finally:
        cli.close()


def _sftp_listdir(sftp: paramiko.SFTPClient, remote_dir: str) -> Iterable[Tuple[str, bool]]:
    """
    List directory entries returning (path, is_dir).
    """
    for attr in sftp.listdir_attr(remote_dir):
        name = attr.filename
        path = os.path.join(remote_dir, name)
        is_dir = paramiko.S_ISDIR(attr.st_mode)
        yield path, is_dir


def sftp_download_tree(host_alias: str, remote_dir: str, local_dir: Path) -> None:
    """
    Recursively download a remote directory tree into local_dir via SFTP.
    """
    cli = connect_via_ssh_config(host_alias)
    try:
        sftp = cli.open_sftp()
        local_dir.mkdir(parents=True, exist_ok=True)
        stack = [remote_dir]
        while stack:
            cur = stack.pop()
            rel = os.path.relpath(cur, remote_dir)
            ldir = local_dir if rel in (".", os.curdir) else (local_dir / rel)
            ldir.mkdir(parents=True, exist_ok=True)
            for path, is_dir in _sftp_listdir(sftp, cur):
                if is_dir:
                    stack.append(path)
                else:
                    dest = ldir / os.path.basename(path)
                    sftp.get(path, str(dest))
        sftp.close()
    finally:
        cli.close()


def run_remote_cmd(
    host_alias: str,
    command: str,
    *,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    get_pty: bool = False,
    timeout: Optional[float] = None,
) -> Tuple[int, str, str]:
    """
    Execute a command on the remote host, optionally in a working directory with environment variables.

    Returns:
        (exit_status, stdout, stderr)
    """
    cli = connect_via_ssh_config(host_alias)
    try:
        full_cmd = command
        if env:
            exports = " ".join(f'{k}={sh_quote(v)}' for k, v in env.items())
            full_cmd = f"{exports} {full_cmd}"
        if cwd:
            full_cmd = f'cd {sh_quote(cwd)} && {full_cmd}'
        chan = cli.get_transport().open_session()
        if get_pty:
            chan.get_pty()
        chan.exec_command(full_cmd)
        if timeout is not None:
            chan.settimeout(timeout)
        stdout = chan.makefile("r").read()
        stderr = chan.makefile_stderr("r").read()
        code = chan.recv_exit_status()
        chan.close()
        return code, stdout, stderr
    finally:
        cli.close()


def ensure_remote_dir(host_alias: str, remote_dir: str) -> None:
    """
    Ensure a remote directory exists.
    """
    cli = connect_via_ssh_config(host_alias)
    try:
        sftp = cli.open_sftp()
        sftp_mkdirs(sftp, remote_dir)
        sftp.close()
    finally:
        cli.close()


def sh_quote(s: str) -> str:
    """
    Minimal POSIX shell quoting.
    """
    if s == "":
        return "''"
    if all(c.isalnum() or c in "@%_+=:,./-" for c in s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


__all__ = [
    "load_worker_host",
    "connect_via_ssh_config",
    "sftp_mkdirs",
    "sftp_upload_tree",
    "sftp_download_tree",
    "run_remote_cmd",
    "ensure_remote_dir",
    "sh_quote",
]


