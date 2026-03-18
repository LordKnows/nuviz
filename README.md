# NuViz

Terminal-native ML training visualization. Monitor experiments, compare runs, and generate paper-ready tables — all without leaving the terminal.

## Overview

NuViz consists of two components:

- **`nuviz` (Python library)** — Lightweight structured logger that embeds in training scripts to record metrics, images, and point clouds
- **`nuviz` (Rust CLI)** — Terminal TUI for real-time monitoring, experiment comparison, ablation analysis, and LaTeX table generation

## Why NuViz?

Existing tools (TensorBoard, W&B, Aim) all require a browser. NuViz keeps everything in the terminal where your training is already running — offline, zero network dependency, SSH-friendly out of the box.

## Quick Start

### Python Logger

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

### Rust CLI

```bash
nuviz watch exp-001              # Real-time TUI dashboard
nuviz compare exp-001 exp-002    # Multi-experiment curve overlay
nuviz leaderboard --sort psnr    # Experiment ranking table
nuviz matrix --metric psnr       # Ablation matrix view
nuviz table --latex              # Paper-ready LaTeX table
```

## Features

- **Real-time monitoring** — Live loss/metric curves with braille characters
- **Multi-experiment comparison** — Overlay curves with automatic annotation
- **Ablation matrix** — Visualize hyperparameter sweep results
- **Per-scene breakdown** — Scene-level metric decomposition (NeRF/3DGS)
- **LaTeX/Markdown export** — Paper-ready tables with bold-best formatting
- **Terminal image viewing** — Kitty/Sixel/half-block fallback
- **PLY statistics** — Gaussian count, opacity distribution, bounding box
- **Environment snapshots** — Git hash, pip freeze, GPU info, full config
- **Anomaly detection** — NaN/Inf and loss spike alerts

## Tech Stack

| Component | Technology |
|-----------|------------|
| Python library | Python 3.10+, numpy |
| CLI | Rust, ratatui, clap |
| Data format | JSONL (metrics), PNG (images), PLY (point clouds) |
| File watching | notify (inotify/kqueue/FSEvents) |
| Terminal images | viuer (auto protocol detection) |

## Development Status

Under active development. See [docs/nuviz_design.md](docs/nuviz_design.md) for the full design document.

## License

TBD
