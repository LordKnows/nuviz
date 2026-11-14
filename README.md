# NuViz

[![CI](https://github.com/LordKnows/nuviz/actions/workflows/ci.yml/badge.svg)](https://github.com/LordKnows/nuviz/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nuviz)](https://pypi.org/project/nuviz/)
[![crates.io](https://img.shields.io/crates/v/nuviz-cli)](https://crates.io/crates/nuviz-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Terminal-native ML training visualization. Monitor experiments, compare runs, and generate paper-ready tables — all without leaving the terminal.

<!-- ![NuViz Demo](docs/assets/demo.gif) -->

## Why NuViz?

Existing tools (TensorBoard, W&B, Aim) all require a browser. NuViz keeps everything in the terminal where your training is already running — offline, zero network dependency, SSH-friendly out of the box.

| Feature | TensorBoard | W&B | Aim | **NuViz** |
|---------|:-----------:|:---:|:---:|:---------:|
| Terminal native | | | | **Y** |
| Two-line integration | ~ | ~ | ~ | **Y** |
| Fully offline | Y | | Y | **Y** |
| Real-time monitoring | ~ | Y | Y | **Y** |
| Ablation matrix | | ~ | | **Y** |
| Per-scene breakdown | | | | **Y** |
| LaTeX table export | | | | **Y** |
| PLY point cloud stats | | ~ | | **Y** |

## Installation

```bash
pip install "nuviz[cli]"       # Python logger + prebuilt `nuviz` command
```

The logger itself is pure Python with no dependencies; pick what you need:

```bash
pip install nuviz              # logger only (e.g. on training nodes)
pip install "nuviz[images]"    # + numpy and Pillow for image / point-cloud logging
pip install "nuviz[yaml]"      # + PyYAML for ablation export
```

The CLI can also be installed on its own:

```bash
cargo install nuviz-cli                # from crates.io (Rust 1.88+)
brew install LordKnows/tap/nuviz       # Homebrew (macOS)
# Pre-built binaries with checksums: see GitHub Releases
```

## Quick Start

### 1. Log metrics from your training script

```python
from nuviz import Logger

with Logger("exp-001", project="gaussian_splatting",
            hparams={"lr": 1e-3, "sh_degree": 2}, max_steps=30000) as log:
    for step in range(30000):
        loss = train_step()
        psnr, ssim, lpips = evaluate()
        log.step(step, loss=loss, psnr=psnr, ssim=ssim, lpips=lpips)

        if step % 500 == 0:
            log.image("render", pred_image)
```

Tensors and numpy scalars can be passed directly. Already using wandb-style
calls, Lightning or the Hugging Face `Trainer`?

```python
import nuviz
nuviz.init(project="gaussian_splatting", name="exp-001", config=cfg)
nuviz.log({"loss": loss, "psnr": psnr})

from nuviz.integrations.lightning import NuvizLogger          # Trainer(logger=NuvizLogger(...))
from nuviz.integrations.transformers import NuvizCallback     # Trainer(callbacks=[NuvizCallback()])
```

### 2. Visualize in the terminal

```bash
nuviz watch                               # Live dashboard for the latest run
nuviz watch exp-001 exp-002               # Several runs overlaid
nuviz ls --project gaussian_splatting     # List experiments and their status
nuviz leaderboard --sort psnr --top 10    # Ranked table
nuviz compare exp-001 exp-002 --metric psnr  # One large chart
```

### 3. Generate paper-ready outputs

```bash
nuviz leaderboard --format latex --aggregate    # mean ± std over seeds
nuviz leaderboard --value final --format latex  # final-step values instead of best
nuviz matrix --rows lr --cols sh_degree --metric psnr --format latex
nuviz breakdown exp-001 --latex                 # Per-scene table
nuviz export exp-001 --format csv               # Raw data export
```

## Features

### Logging (Python)

- **Metrics** — `log.step(step, loss=loss, psnr=psnr)`; accepts tensors and numpy scalars, written by a background thread that never blocks training
- **Hyperparameters** — `Logger(hparams=...)` feeds `nuviz matrix`, seed grouping and `nuviz reproduce`
- **Run status** — running / done / failed / crashed, via a heartbeat; failures record the exception
- **Images** — `log.image("render", tensor)` with auto numpy/torch conversion
- **Point clouds** — `log.pointcloud("gaussians", xyz, colors)` as binary PLY
- **Per-scene** — `log.scene("bicycle", psnr=28.4, ssim=0.92)`
- **Ablation configs** — `Ablation().vary("lr", [1e-3, 1e-4]).generate()` with YAML export
- **GPU metrics** — Automatic nvidia-smi polling (utilization, memory, temperature)
- **Environment snapshots** — Git hash, installed packages, GPU info, command line (captured in the background)
- **Anomaly detection** — NaN/Inf detection and spike alerts against a moving average, shown live in `nuviz watch`
- **Integrations** — wandb-style `init`/`log`/`finish`, Lightning logger, Hugging Face callback
- **JSONL rotation** — Automatic file rotation at 50MB/500k lines

### Visualization (Rust CLI)

| Command | Description |
|---------|-------------|
| `nuviz watch` | Live dashboard: multi-run curves, status, speed/ETA, GPU, alerts |
| `nuviz ls` | List all experiments with status |
| `nuviz leaderboard` | Metric ranking, best or final values, mean ± std over seeds (table/latex/markdown/csv) |
| `nuviz compare` | Multi-experiment curve overlay (same dashboard, one large chart) |
| `nuviz matrix` | Ablation hyperparameter matrix |
| `nuviz breakdown` | Per-scene metric decomposition |
| `nuviz export` | Export raw data as CSV/JSON |
| `nuviz image` | Browse experiment images in terminal |
| `nuviz diff` | Visual diff between experiments |
| `nuviz view` | Point cloud (PLY) statistics |
| `nuviz tag` | Tag experiments for organization |
| `nuviz cleanup` | Remove low-value experiments (never running or tagged ones) |
| `nuviz reproduce` | Print reproduction guide from metadata |

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `NUVIZ_DIR` | `~/.nuviz/experiments` | Base directory for all experiments |
| `NUVIZ_FLUSH_INTERVAL` | `2.0` | Seconds between background flushes |
| `NUVIZ_ALERTS` | `1` | Set `0` to disable anomaly alerts |
| `NUVIZ_SNAPSHOT` | `1` | Set `0` to disable environment capture |
| `NUVIZ_GPU` | `1` | Set `0` to disable GPU polling |
| `NUVIZ_IMAGE_PROTOCOL` | auto | Force `kitty`, `iterm2`, `sixel` or `halfblock` in the CLI |

The full list is in the [usage guide](docs/USAGE.md#configuration).

## Architecture

```
Python (training)                   Rust (visualization)
┌──────────────────┐               ┌──────────────────────┐
│ Logger           │─metrics.jsonl─>│ watch / compare      │
│  writer thread   │─events.jsonl──>│  (alerts, live)      │
│  heartbeat       │─heartbeat.json>│ ls (run status)      │
│  snapshot thread │─meta.json─────>│ leaderboard / matrix │
│  images / PLY    │─PNG / PLY─────>│ image / diff / view  │
└──────────────────┘               └──────────────────────┘
```

Data flows one way: Python writes structured files, Rust reads them. No server, no network, no database. The file format is specified in [docs/FORMAT.md](docs/FORMAT.md) and both sides are tested against the same fixtures.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions.

```bash
# Python
cd python && pip install -e ".[dev,images,yaml]"
pytest tests/ --cov=nuviz --cov-report=term-missing

# Rust
cd cli && cargo test
cargo clippy --all-targets -- -D warnings
```

## Documentation

- [Usage Guide](docs/USAGE.md) — Full command reference and API docs
- [Data Format](docs/FORMAT.md) — The on-disk contract between logger and CLI
- [Changelog](CHANGELOG.md)
- [Design Document](docs/nuviz_design.md) — Architecture and roadmap

## License

[MIT](LICENSE)
