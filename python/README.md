# NuViz

Terminal-native ML training visualization — the Python logging library.

## Install

```bash
pip install nuviz                 # logger only, no dependencies
pip install "nuviz[cli]"          # + the prebuilt `nuviz` command-line tool
pip install "nuviz[images,yaml]"  # + numpy/Pillow for images, PyYAML for ablation export
```

## Usage

```python
from nuviz import Logger

with Logger("exp-001", project="my_project", hparams={"lr": 1e-3}) as log:
    for step in range(10000):
        loss = train_step()
        log.step(step, loss=loss, psnr=psnr)   # tensors and numpy scalars are fine
```

A wandb-style API and framework adapters are included:

```python
import nuviz
nuviz.init(project="my_project", name="exp-001", config={"lr": 1e-3})
nuviz.log({"loss": loss})
nuviz.finish()

from nuviz.integrations.lightning import NuvizLogger        # Lightning
from nuviz.integrations.transformers import NuvizCallback   # Hugging Face Trainer
```

### Images and Point Clouds

```python
log.image("render", predicted_image)         # numpy/torch tensor -> PNG
log.pointcloud("gaussians", xyz, colors)     # -> binary PLY
log.scene("bicycle", psnr=28.4, ssim=0.92)  # per-scene metrics
```

### Ablation Experiments

```python
from nuviz import Ablation, Logger

sweep = Ablation("lr_sweep", base_config={"lr": 1e-3})
sweep.vary("lr", [1e-3, 1e-4, 1e-5])
sweep.toggle("use_augmentation")

for config in sweep.generate():          # plain dicts
    for seed in range(3):
        name = f"lr{config['lr']}_aug{config['use_augmentation']}_seed{seed}"
        with Logger(name, hparams=config, seed=seed) as log:
            ...                          # train with `config`
```

Runs that share hyperparameters but differ in `seed` are aggregated by
`nuviz leaderboard --aggregate` and `nuviz matrix`.

### GPU Metrics

GPU utilization, memory, and temperature are collected automatically via `nvidia-smi` polling. Disable with `NUVIZ_GPU=0`.

## Visualize

Install the [CLI](https://github.com/LordKnows/nuviz) to visualize logged data:

```bash
pip install "nuviz[cli]"      # or: cargo install nuviz-cli

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
- No required dependencies (numpy, Pillow, PyYAML are optional and imported only when used)

## License

[MIT](https://github.com/LordKnows/nuviz/blob/main/LICENSE)
