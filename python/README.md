# NuViz

Terminal-native ML training visualization — the Python logging library.

## Install

```bash
pip install nuviz

# Optional: image logging (Pillow) and ablation YAML export
pip install nuviz[images,yaml]
```

## Usage

```python
from nuviz import Logger

log = Logger("exp-001", project="my_project")

for step in range(10000):
    loss = train_step()
    log.step(step, loss=loss, psnr=psnr)

log.finish()
```

### Images and Point Clouds

```python
log.image("render", predicted_image)         # numpy/torch tensor -> PNG
log.pointcloud("gaussians", xyz, colors)     # -> binary PLY
log.scene("bicycle", psnr=28.4, ssim=0.92)  # per-scene metrics
```

### Ablation Experiments

```python
from nuviz import Ablation

sweep = Ablation("lr_sweep")
sweep.vary("lr", [1e-3, 1e-4, 1e-5])
sweep.toggle("use_augmentation")

for config in sweep.generate():
    log = Logger(config.name, seed=42, config_hash=config.hash)
    # ... train with config.params ...
```

### GPU Metrics

GPU utilization, memory, and temperature are collected automatically via `nvidia-smi` polling. Disable with `NUVIZ_GPU=0`.

## Visualize

Install the [Rust CLI](https://github.com/LordKnows/nuviz) to visualize logged data:

```bash
cargo install nuviz-cli

nuviz watch exp-001           # Live dashboard
nuviz leaderboard --sort psnr # Ranked table
nuviz compare exp-001 exp-002 # Curve overlay
```

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `NUVIZ_DIR` | `~/.nuviz/experiments` | Data directory |
| `NUVIZ_GPU` | `1` | Set `0` to disable GPU collection |
| `NUVIZ_GPU_POLL` | `5.0` | GPU poll interval (seconds) |
| `NUVIZ_ALERTS` | `1` | Set `0` to disable anomaly alerts |

## Requirements

- Python 3.10+
- No required dependencies (numpy, Pillow, PyYAML are optional)

## License

[MIT](https://github.com/LordKnows/nuviz/blob/main/LICENSE)
