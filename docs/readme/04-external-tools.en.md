# External Materials Programs

CatMaster does not redistribute VASPKIT, VESTA, VASP, or other external programs in deployment packages. Install each program under its own license on the control-plane host or compute node; CatMaster only discovers and invokes an installed executable.

## 1. Executable Discovery

VASPKIT is resolved in this order:

1. The executable or install directory in `CATMASTER_VASPKIT_BIN`.
2. `vaspkit` on `PATH`.
3. The compatibility path `~/vaspkit/bin/vaspkit`.

VESTA is resolved in this order:

1. The launcher or install directory in `CATMASTER_VESTA_BIN`.
2. `VESTA` or `vesta` on `PATH`.
3. A small set of conventional user-local paths.

Explicit configuration avoids differences between WebUI, IDE, and terminal environments:

```bash
export CATMASTER_VASPKIT_BIN="$HOME/vaspkit/bin/vaspkit"
export CATMASTER_VESTA_BIN="$HOME/.local/opt/VESTA-gtk3/VESTA"
```

The variables are also listed in `.env.example`. CatMaster does not load `.env` automatically; export them in the shell that starts the WebUI or in your shell profile.

## 2. VASPKIT

After following the official VASPKIT installation instructions, verify the configured launcher:

```bash
test -x "$CATMASTER_VASPKIT_BIN" && echo "VASPKIT executable found"
```

CatMaster currently uses VASPKIT tasks 501 and 502 for adsorbate and gas-phase thermochemistry. Some tools can use an explicitly labeled ASE approximation when VASPKIT is absent; reports must preserve the actual backend label.

## 3. VESTA

VESTA exports atomistic figures suitable for reports and multimodal model inspection. The `materials_worker` tool `render_vesta_views` creates standardized top, side, and isometric views, a combined PNG panel, and JSON metadata. The agent then opens the panel with `read_file` for visual inspection.

Download the appropriate build from the [official VESTA download page](https://jp-minerals.org/vesta/en/download.html). A Linux example is:

```bash
mkdir -p "$HOME/.local/opt"
tar -xjf VESTA-gtk3.tar.bz2 -C "$HOME/.local/opt"
export CATMASTER_VESTA_BIN="$HOME/.local/opt/VESTA-gtk3/VESTA"
test -x "$CATMASTER_VESTA_BIN" && echo "VESTA executable found"
```

Linux also needs GTK/OpenGL runtime libraries. Common Ubuntu/Debian packages are:

```bash
sudo apt-get update
sudo apt-get install -y libglu1-mesa xvfb xauth
```

VESTA image-export commands require GUI/X11 operation. CatMaster runs VESTA directly when `DISPLAY` is available and automatically uses `xvfb-run` on a headless host. Configure a nonstandard launcher explicitly:

```bash
export CATMASTER_XVFB_RUN=/usr/bin/xvfb-run
```

VESTA `-nogui` is not a substitute for Xvfb during image export. Before starting the WebUI, inspect the effective environment with:

```bash
printf 'VESTA=%s\n' "$CATMASTER_VESTA_BIN"
printf 'DISPLAY=%s\n' "${DISPLAY:-<unset>}"
command -v xvfb-run || true
```

The tool uses VESTA auto-fit by default. For slab context, request `supercell=[2,2,1]`; do not repeat the vacuum direction merely to fill the image. Visual evidence supports geometry sanity checks, site and termination context, and reporting, while critical distances, coordination, and rankings still require numerical validation.

## 4. Markdown to PDF

The Writing Agent's `render_markdown_pdf` tool uses Pandoc to produce embedded HTML5/MathML and then prints it with headless Chrome. A plain Markdown-to-PDF request does not rewrite the source as LaTeX. The control-plane host needs:

```bash
conda env update -n catmaster -f requirements/pc-conda.yml
command -v pandoc
command -v google-chrome || command -v chromium
fc-match "Microsoft YaHei"
fc-match "Noto Sans CJK SC"
```

`requirements/pc-conda.yml` pins Pandoc. Chrome/Chromium and fonts are host-installed dependencies and are not assumed to exist on compute nodes. Configure nonstandard executable locations explicitly:

```bash
export CATMASTER_PANDOC_BIN=/path/to/pandoc
export CATMASTER_CHROME_BIN=/path/to/google-chrome
```

Chinese and mixed Chinese-English reports default to Microsoft YaHei. If it is unavailable, the tool explicitly falls back to Noto Sans CJK SC and records the resolved font in metadata. CatMaster does not redistribute Microsoft YaHei font files. On Ubuntu/Debian, install the portable fallback with:

```bash
sudo apt-get install -y fontconfig fonts-noto-cjk
fc-cache -f
```

## 5. License and Deployment

The official VESTA license prohibits redistribution of its distributed files without written permission. CatMaster therefore does not include the VESTA binary in the main repository, Deploy directory, or remote archive. Install it separately on every host that performs rendering.

VESTA drawings used in publications must explicitly acknowledge the program and cite K. Momma and F. Izumi, *J. Appl. Crystallogr.* **44**, 1272-1276 (2011). `render_vesta_views` records this requirement and citation in every render metadata file.

DPDispatcher compute nodes normally do not need VESTA because rendering runs on the CatMaster control-plane host. Configure it in a remote boot environment only when rendering is deliberately moved to that machine.
