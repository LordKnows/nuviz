# NuViz Usage Guide

## Python Logger API

### Logger

```python
from nuviz import Logger

log = Logger(
    name="exp-001",           # Experiment name (auto-generated if None)
    project="my_project",     # Optional project grouping
    seed=42,                  # Optional seed for multi-seed grouping
    config_hash="abc123",     # Optional hash for ablation grouping
    snapshot=True,            # Capture environment info
    alerts=True,              # Enable anomaly detection
)
```

#### `log.step(step, **metrics)`

Record metrics for a training step. Metrics are buffered and flushed in the background.

```python
log.step(step, loss=0.05, psnr=28.4, ssim=0.92, lpips=0.12)
```

#### `log.image(name, data, cmap=None)`

Save an image. Accepts numpy arrays or torch tensors. Saved as PNG in `images/`.

```python
log.image("render", pred_image)        # HWC or CHW, float [0,1] or uint8
log.image("depth", depth_map, cmap="turbo")  # Single channel with colormap
```

#### `log.pointcloud(tag, xyz, colors=None, opacities=None)`

Save a point cloud as binary PLY. Saved in `pointclouds/`.

```python
log.pointcloud("gaussians", xyz, colors=rgb, opacities=alpha)
```

#### `log.scene(scene_name, **metrics)`

Record per-scene evaluation metrics. Written to `scenes.jsonl`.

```python
log.scene("bicycle", psnr=28.4, ssim=0.92)
log.scene("garden", psnr=26.1, ssim=0.88)
```

#### `log.finish()`

Flush all data, stop background threads, write final metadata. Called automatically when using `Logger` as a context manager.

```python
log.finish()

# Or use as context manager:
with Logger("exp-001") as log:
    for step in range(1000):
        log.step(step, loss=loss)
# finish() called automatically
```

### Ablation

```python
from nuviz import Ablation

sweep = Ablation("lr_depth_sweep")
sweep.vary("lr", [1e-3, 1e-4, 1e-5])
sweep.vary("depth", [2, 4, 8])
sweep.toggle("augmentation")  # True/False

configs = sweep.generate()  # List of config combinations
sweep.export("configs.yaml")  # Export to YAML (requires PyYAML)
sweep.export("configs.json")  # Export to JSON
```

### GPU Metrics

Automatic when nvidia-smi is available. Each JSONL record includes:

```json
{"step": 100, "timestamp": 1234567890.0, "metrics": {"loss": 0.05}, "gpu": {"gpu_util": 95, "mem_used_mb": 8192, "mem_total_mb": 24576, "temperature_c": 72}}
```

Disable with `NUVIZ_GPU=0` or `NuvizConfig(enable_gpu=False)`.

---

## CLI Commands

All commands accept `--dir <path>` to override the base directory (default: `~/.nuviz/experiments`).

### `nuviz watch`

Real-time TUI dashboard with braille-rendered metric curves.

```bash
nuviz watch exp-001                    # Watch a single experiment
nuviz watch exp-001 exp-002            # Watch multiple
nuviz watch --latest 3 --project gs    # Watch 3 most recent in project
nuviz watch --poll                     # Use polling (NFS/WSL)
```

**Keyboard**: `q` to quit, arrow keys to scroll, `Tab` to cycle metrics.

### `nuviz ls`

List all discovered experiments with status and step count.

```bash
nuviz ls                          # All experiments
nuviz ls --project gaussian       # Filter by project
nuviz ls --sort name              # Sort by name (default: date)
```

### `nuviz leaderboard`

Ranked experiment table by a chosen metric.

```bash
nuviz leaderboard --sort psnr              # Best PSNR first
nuviz leaderboard --sort loss --asc        # Lowest loss first
nuviz leaderboard --top 10                 # Show top 10 only
nuviz leaderboard --format latex           # LaTeX table output
nuviz leaderboard --format markdown        # Markdown table
nuviz leaderboard --aggregate              # Group multi-seed runs (mean +/- std)
```

### `nuviz compare`

Multi-experiment curve overlay in a TUI.

```bash
nuviz compare exp-001 exp-002 --metric loss
nuviz compare --project gs --metric psnr    # All experiments in project
nuviz compare exp-001 exp-002 --align wall_time  # Align by wall time
```

### `nuviz matrix`

Ablation hyperparameter matrix view.

```bash
nuviz matrix --rows lr --cols depth --metric psnr
nuviz matrix --rows lr --cols depth --metric psnr --format latex
nuviz matrix --project sweep_v2
```

### `nuviz breakdown`

Per-scene metric decomposition table.

```bash
nuviz breakdown exp-001                      # Default table
nuviz breakdown exp-001 --latex              # LaTeX format
nuviz breakdown exp-001 --diff exp-002       # Show deltas
```

### `nuviz export`

Export raw experiment data.

```bash
nuviz export exp-001 --format csv            # CSV output
nuviz export exp-001 --format json           # JSON output
nuviz export exp-001 --metric loss --metric psnr  # Specific metrics
```

### `nuviz image`

Browse experiment images in the terminal.

```bash
nuviz image exp-001                          # Browse all images
nuviz image exp-001 --step 5000              # Specific step
nuviz image exp-001 --tag render --latest    # Latest render
nuviz image exp-001 --tag render --side-by-side gt  # Side by side
```

Rendering auto-detects terminal protocol: Kitty > iTerm2 > Sixel > half-block fallback.

### `nuviz diff`

Visual diff between two experiments.

```bash
nuviz diff exp-001 exp-002 --tag render
nuviz diff exp-001 exp-002 --step 10000 --heatmap  # Error heatmap
```

### `nuviz view`

Point cloud (PLY) statistics and inspection.

```bash
nuviz view exp-001                           # Latest PLY in experiment
nuviz view path/to/file.ply                  # Direct path
nuviz view exp-001 --histogram               # Attribute distributions
```

### `nuviz tag`

Tag experiments for organization.

```bash
nuviz tag exp-001 baseline                   # Add tag
nuviz tag exp-001 --list                     # List tags
nuviz tag exp-001 --remove baseline          # Remove tag
```

### `nuviz cleanup`

Identify and remove low-value experiments.

```bash
nuviz cleanup --project gs                   # Dry run (default)
nuviz cleanup --project gs --keep-top 3      # Keep top 3 by loss
nuviz cleanup --metric psnr --keep-top 5     # Rank by PSNR
nuviz cleanup --project gs --force           # Actually delete
```

### `nuviz reproduce`

Print a reproduction guide from experiment metadata.

```bash
nuviz reproduce exp-001
# Outputs: git checkout, pip install, GPU info, config, seed
```

---

## Data Format

Each experiment lives in `<base_dir>/<project?>/<experiment_name>/`:

| File | Format | Written By |
|------|--------|-----------|
| `metrics.jsonl` | One JSON object per step | `Logger.step()` |
| `meta.json` | Environment snapshot + final stats | `Logger.__init__()` / `Logger.finish()` |
| `scenes.jsonl` | Per-scene metrics | `Logger.scene()` |
| `images/<tag>_<step>.png` | Rendered images | `Logger.image()` |
| `pointclouds/<tag>_step_<step>.ply` | Binary PLY | `Logger.pointcloud()` |

JSONL files rotate automatically at 50MB/500k lines. Rotated files are named `metrics.1.jsonl`, `metrics.2.jsonl`, etc. The Rust CLI merges all segments transparently.
