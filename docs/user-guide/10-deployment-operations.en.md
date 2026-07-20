# 10. Deployment, operations, and security

[Previous](09-tools-skills-evolution.en.md) | [Contents](README.en.md) | [Next](11-reference-troubleshooting.en.md)

This chapter covers the control plane and local helper programs. VASP, CP2K,
LAMMPS, ORCA, xTB, CREST, and MLFF providers belong in managed remote
environments and should not be mixed into the CatMaster control plane.

## 10.1 Three common deployments

### Local workstation

The WebUI and browser are on one machine:

```bash
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
CATMASTER_HOST=127.0.0.1 \
CATMASTER_PORT=7991 \
./start_webui.sh
```

### Server with SSH tunnel

The server still listens only on loopback:

```bash
CATMASTER_PROJECT_SPACE_ROOT=/srv/catmaster/projects \
CATMASTER_HOST=127.0.0.1 \
CATMASTER_PORT=7991 \
./start_webui.sh
```

On the user's computer:

```bash
ssh -L 7991:127.0.0.1:7991 <USER>@<SERVER>
```

Then open `http://127.0.0.1:7991`.

### Shared web service

Place a shared service behind a TLS reverse proxy, VPN, IP allowlist, or external
identity layer. Built-in authentication allows self-registration by default,
the session cookie lacks the Secure flag, and the application does not terminate
TLS. Do not expose `0.0.0.0:7991` directly to the public internet, and never
publish `--no-login` mode.

## 10.2 Account and file permissions

The system user running the WebUI can access every project-space file and active
configuration. Therefore:

- Use a dedicated non-root system user.
- Restrict the project root, `.webui_auth`, active YAML, SSH keys, and browser
  profile to that user.
- Do not grant unnecessary write access to repository source or shared secret
  directories.
- Keep the project root outside any web-server static directory.
- Add external access control for a multi-user deployment and review the open
  registration risk.

## 10.3 `.env.local` and secrets

Create a private file from the checklist:

```bash
cp .env.example .env.local
chmod 600 .env.local
```

Load it with exports:

```bash
set -a
source .env.local
set +a
```

For production, a systemd EnvironmentFile or secret manager is more suitable.
Do not put keys in `configs/llm.yaml`, and never put SSH keys, licenses, tokens,
or cookies in a workspace. Confirm that `.env.local`, active
`configs/dpdispatcher/*.yaml`, and the project root are outside version control
and deployment packages.

## 10.4 Runtime directory and logs

Default background state:

```text
.runtime/webui.pid
.runtime/webui.log
```

It can be overridden:

```bash
export CATMASTER_RUNTIME_DIR=/var/tmp/catmaster-runtime
export CATMASTER_WEBUI_LOG=/var/log/catmaster/webui.log
export CATMASTER_WEBUI_PID=/var/run/user/<UID>/catmaster-webui.pid
```

The runtime user must create and write those directories. Common commands:

```bash
./start_webui.sh --status
tail -f .runtime/webui.log
./start_webui.sh --stop
```

The launcher waits up to 30 seconds while stopping, then forcibly terminates the
recorded process if needed. Stopping the local WebUI does not cancel remote
scheduler jobs.

## 10.5 Runtime sync deployment

`scripts/deploy_runtime.sh` updates another runtime directory. Its default target
is `../CatMaster_Run`. It performs a runtime-only sync, deletes target files
removed from the source, builds the frontend, starts automatically, and preserves
existing target configs and launcher.

Begin with a non-destructive preview:

```bash
scripts/deploy_runtime.sh \
  --target /path/to/CatMaster_Run \
  --project-space-root /path/to/catmaster_projects \
  --dry-run \
  --no-delete \
  --no-autorun
```

Remove `--dry-run` after inspection. Important options:

- `--sync-configs` overwrites target `configs/` and may destroy private LLM or
  machine configuration.
- `--sync-start-webui` overwrites the target launcher.
- Without `--no-delete`, runtime files removed from source are deleted from the
  target.
- `--autorun` is the default. Use `--no-autorun` during a maintenance window.
- `--full-repo` expands scope and is not the normal runtime-update default.

## 10.6 Offline deployment package

Build an archive:

```bash
scripts/package_remote_deploy.sh --output-dir dist
```

The default package contains public DPDispatcher templates, not active machine
configs, `.env`, project space, logs, or large calculation intermediates. Inspect
the archive listing and checksum before transfer.

`--include-path` adds another path. Check that it contains no key, token,
POTCAR, WAVECAR, CHGCAR, personal browser state, or unauthorized data.
`--no-verify` skips post-package checks and is not recommended for a formal
delivery.

## 10.7 Upgrade and rollback

Upgrade in this order:

1. Stop the WebUI and confirm no run is writing local state.
2. Back up the project root, account database, and private configuration.
3. Update the source or deployment package.
4. Update the control-plane environment:

   ```bash
   conda env update -n catmaster -f requirements/pc-conda.yml
   ```

5. Parse the LLM YAML without a network call.
6. Start in the foreground and smoke-test sign-in, threads, file reads, and one
   short LLM turn.
7. For remote execution, run only `--list` and one minimal case first.
8. Return to background mode after acceptance.

Do not roll project-space data back blindly to an older incompatible layout.
Keep a pre-upgrade snapshot and record code version, configuration version, and
data snapshot time separately.

## 10.8 Backup

A complete backup includes:

```text
<PROJECT_SPACE_ROOT>/users/.../<workspace>/files/
<PROJECT_SPACE_ROOT>/users/.../<workspace>/metadata/
<PROJECT_SPACE_ROOT>/.webui_auth/auth.sqlite
configs/llm.yaml
active private configs under configs/dpdispatcher/
.env.local or external secret definitions
```

Secrets and project data may use different encrypted backup policies. Restore
directory permissions before starting the WebUI. Restoring only `files/` loses
threads and checkpoints; restoring only `metadata/` loses actual artifacts.

## 10.9 agent-browser

The required version is:

```bash
npm install -g agent-browser@0.31.1
agent-browser install
agent-browser doctor --offline --quick
agent-browser mcp --help
```

Keep its profile outside workspaces:

```bash
export CATMASTER_AGENT_BROWSER_PROFILE="$HOME/.config/catmaster/browser-profile"
```

A headless server can use local corpora and web search, but must not claim access
to institutional full text requiring interactive sign-in. Use a secure graphical
session or have the user legitimately upload the full text.

## 10.10 JSmol

The WebUI uses JSmol for structure and trajectory previews. The launcher invokes
an installer and downloads fixed assets when the cache is missing. Prewarm a
persistent cache for an offline server:

```bash
CATMASTER_JSMOL_CACHE_DIR=/persistent/cache/jsmol \
python scripts/install_jsmol_assets.py
```

Use the same `CATMASTER_JSMOL_CACHE_DIR` in deployment and grant read access to
the runtime user. Missing JSmol affects the corresponding preview, not the LLM
or remote execution.

## 10.11 VASPKIT and VESTA

VASPKIT resolution order is:

1. `CATMASTER_VASPKIT_BIN`.
2. `vaspkit` on `PATH`.
3. Common user paths such as `~/vaspkit/bin/vaspkit`.

```bash
export CATMASTER_VASPKIT_BIN=/opt/vaspkit/bin/vaspkit
```

VESTA uses a similar search and can be set explicitly:

```bash
export CATMASTER_VESTA_BIN=/opt/VESTA/VESTA
export CATMASTER_XVFB_RUN=/usr/bin/xvfb-run
```

Rendering on a server without DISPLAY normally needs Xvfb. VESTA and VASPKIT
are optional local helpers, not the managed VASP engine, and CatMaster does not
supply their licenses.

## 10.12 Pandoc, Chrome, fonts, and TeX

The Markdown PDF route creates HTML5/MathML with Pandoc, then prints it with
headless Chrome or Chromium. Paths can be explicit:

```bash
export CATMASTER_PANDOC_BIN=/usr/bin/pandoc
export CATMASTER_CHROME_BIN=/usr/bin/chromium
```

Checks:

```bash
pandoc --version
chromium --version
fc-match sans
fc-match "Noto Sans CJK SC"
pdflatex --version
bibtex --version
```

Without a CJK font, compilation may still succeed while producing boxes or font
substitution. Inspect the final PDF visually.

## 10.13 PySR and Julia

The first PySR import may download Julia and precompile. During an online
maintenance window run:

```bash
python scripts/pysr_julia_smoke.py --fit
```

For an offline host, install Julia and set:

```bash
export PYTHON_JULIACALL_BINDIR=/opt/julia/bin
python scripts/pysr_julia_smoke.py \
  --julia-bindir "$PYTHON_JULIACALL_BINDIR" --fit
```

Do not make the first user task pay the startup download and compilation cost.

## 10.14 Remote scientific engines

Configure these programs inside resource-controlled remote environments:

- VASP, normally `vasp_std`, with `vasp_gam` available for suitable Gamma cases.
- CP2K, whose template command uses `cp2k.psmp`.
- LAMMPS, whose boot script detects common CPU/GPU/KOKKOS binaries and also
  accepts remote `CATMASTER_LAMMPS_BIN`.
- ORCA, with a correct MPI launcher for multiple ranks.
- xTB and CREST.
- Isolated MACE, FairChem UMA, MatterSim, and ORB-v3 environments.

An executable on PATH does not prove that licenses, model weights, potential
files, or queue policy authorize its use. The administrator must complete site
acceptance before exposing a task.
