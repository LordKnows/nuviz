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

### Python Logger

```bash
pip install nuviz

# Optional extras
pip install nuviz[images]    # Pillow for image logging
pip install nuviz[yaml]      # PyYAML for ablation export
```

### Rust CLI

```bash
# From crates.io
cargo install nuviz-cli

# From Homebrew (macOS)
brew install LordKnows/tap/nuviz

# Pre-built binaries: see GitHub Releases
```

## Quick Start

### 1. Log metrics from your training script

```python
from nuviz import Logger

log = Logger("exp-001", project="gaussian_splatting")

for step in range(30000):
    loss = train_step()
    psnr, ssim, lpips = evaluate()
    log.step(step, loss=loss, psnr=psnr, ssim=ssim, lpips=lpips)

    if step % 500 == 0:
        log.image("render", pred_image)

log.finish()
```

### 2. Visualize in the terminal

```bash
nuviz watch exp-001                       # Live TUI dashboard
nuviz ls --project gaussian_splatting     # List experiments
nuviz leaderboard --sort psnr --top 10    # Ranked table
nuviz compare exp-001 exp-002 --metric psnr  # Curve overlay
```

### 3. Generate paper-ready outputs

```bash
nuviz leaderboard --format latex --aggregate  # LaTeX table with mean +/- std
nuviz matrix --rows lr --cols depth --metric psnr --format latex
nuviz breakdown exp-001 --latex               # Per-scene table
nuviz export exp-001 --format csv             # Raw data export
```

## Features

### Logging (Python)

- **Metrics** — `log.step(step, loss=loss, psnr=psnr)` with background JSONL writing
- **Images** — `log.image("render", tensor)` with auto numpy/torch conversion
- **Point clouds** — `log.pointcloud("gaussians", xyz, colors)` as binary PLY
- **Per-scene** — `log.scene("bicycle", psnr=28.4, ssim=0.92)`
- **Ablation configs** — `Ablation().vary("lr", [1e-3, 1e-4]).generate()` with YAML/JSON export
- **GPU metrics** — Automatic nvidia-smi polling (utilization, memory, temperature)
- **Environment snapshots** — Git hash, pip freeze, GPU info, Python version
- **Anomaly detection** — NaN/Inf detection + 3-sigma spike alerts
- **JSONL rotation** — Automatic file rotation at 50MB/500k lines

### Visualization (Rust CLI)

| Command | Description |
|---------|-------------|
| `nuviz watch` | Real-time TUI dashboard with braille charts |
| `nuviz ls` | List all experiments with status |
| `nuviz leaderboard` | Metric ranking table (table/latex/markdown/csv) |
| `nuviz compare` | Multi-experiment curve overlay (TUI) |
| `nuviz matrix` | Ablation hyperparameter matrix |
| `nuviz breakdown` | Per-scene metric decomposition |
| `nuviz export` | Export raw data as CSV/JSON |
| `nuviz image` | Browse experiment images in terminal |
| `nuviz diff` | Visual diff between experiments |
| `nuviz view` | Point cloud (PLY) statistics |
| `nuviz tag` | Tag experiments for organization |
| `nuviz cleanup` | Remove low-value experiments |
| `nuviz reproduce` | Print reproduction guide from metadata |

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `NUVIZ_DIR` | `~/.nuviz/experiments` | Base directory for all experiments |
| `NUVIZ_FLUSH_INTERVAL` | `2.0` | Seconds between background flushes |
| `NUVIZ_FLUSH_COUNT` | `50` | Records before triggering flush |
| `NUVIZ_ALERTS` | `1` | Set `0` to disable anomaly alerts |
| `NUVIZ_SNAPSHOT` | `1` | Set `0` to disable environment capture |
| `NUVIZ_GPU` | `1` | Set `0` to disable GPU polling |
| `NUVIZ_GPU_POLL` | `5.0` | GPU polling interval in seconds |
| `NUVIZ_ROTATE_SIZE` | `50000000` | JSONL rotation size threshold (bytes) |
| `NUVIZ_ROTATE_LINES` | `500000` | JSONL rotation line threshold |

## Architecture

```
Python (training)              Rust (visualization)
┌──────────────┐              ┌──────────────────┐
│   Logger     │──JSONL──────>│  watch / ls /    │
│   + Writer   │  metrics     │  leaderboard ... │
│   + Images   │──PNG────────>│  image / diff    │
│   + PLY      │──PLY────────>│  view            │
│   + Snapshot  │──meta.json──>│  reproduce       │
└──────────────┘              └──────────────────┘
```

Data flows one way: Python writes structured files, Rust reads them. No server, no network, no database.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions.

```bash
# Python
cd python && pip install -e ".[dev,images,yaml]"
pytest tests/ --cov=nuviz --cov-report=term-missing

# Rust
cd cli && cargo test --lib
cargo clippy -- -D warnings
```

## Documentation

- [Usage Guide](docs/USAGE.md) — Full command reference and API docs
- [Design Document](docs/nuviz_design.md) — Architecture and roadmap

## License

[MIT](LICENSE)
