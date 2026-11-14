# NuViz Usage Guide

- [Python logger](#python-logger)
- [wandb-style API](#wandb-style-api)
- [Framework integrations](#framework-integrations)
- [CLI commands](#cli-commands)
- [Configuration](#configuration)
- [Data format](FORMAT.md)

## Python logger

```python
from nuviz import Logger

log = Logger(
    name="exp-001",              # Defaults to a UTC timestamp
    project="my_project",        # Optional grouping (a parent directory)
    hparams={"lr": 1e-3, "sh": 2},   # dict, argparse.Namespace or dataclass
    seed=42,                     # Recorded for multi-seed aggregation
    max_steps=30_000,            # Lets `nuviz watch` show progress and ETA
    tags=["baseline"],
    snapshot=True,               # Capture git / packages / GPU in the background
    alerts=True,                 # NaN/Inf and spike detection
)
```

If `exp-001` already exists the run is stored as `exp-001_1` (then `_2`, …)
and a notice is printed; `log.name` is the name actually used.

`import nuviz` has no third-party dependencies. numpy and Pillow are needed
only for `log.image()` / `log.pointcloud()` (`pip install "nuviz[images]"`).

### `log.step(step, **metrics)`

Record scalars for a step. Never blocks on disk: records are queued and
written by a background thread.

```python
log.step(step, loss=loss, psnr=psnr, lr=scheduler.get_last_lr()[0])
```

Values may be Python numbers, numpy scalars or 0-dim tensors (`loss`, not
only `loss.item()`). NaN/Inf are recorded and raise an alert. A value that is
not a scalar is skipped with a one-time warning.

### `log.define_metric(name, goal)`

Declare whether lower (`"min"`) or higher (`"max"`) is better. Without it the
CLI guesses from the name (`loss`, `error`, `fid`, … are minimized, everything
else maximized).

```python
log.define_metric("regret", "min")
```

### `log.image(name, data, cmap=None, step=None)`

Save an image (numpy array or torch tensor; HWC float in [0, 1] or uint8) as
`images/step_<step>_<name>.png`.

```python
log.image("render", pred_image)
log.image("depth", depth_map, cmap="turbo")   # single channel with colormap
```

### `log.pointcloud(tag, xyz, colors=None, opacities=None, step=None)`

Save a point cloud as binary PLY in `pointclouds/`.

### `log.scene(scene_name, **metrics)`

Record per-scene evaluation metrics for `nuviz breakdown`.

```python
log.scene("bicycle", psnr=28.4, ssim=0.92)
```

### `log.flush()` / `log.finish()`

`flush()` blocks until everything logged so far is on disk. `finish()`
flushes, stops background threads and writes the final summary.

```python
with Logger("exp-001") as log:      # finish() is called for you
    for step in range(1000):
        log.step(step, loss=loss)
```

How a run ends is recorded as its status:

| Situation | Status |
|---|---|
| `finish()` called, or the `with` block exits normally | `done` |
| The `with` block raises, or the script dies with an unhandled exception | `failed` (with the error) |
| The script ends without `finish()` | `done` (finalized at interpreter exit) |
| The process is killed (`SIGKILL`, OOM, node failure) | shown as `crashed` once its heartbeat stops |

### Alerts

* **NaN / Inf** — reported for every metric.
* **Spike** — a value much worse than the recent moving average (more than
  3σ *and* more than 50 % of the mean). Watched for metrics that are
  minimized, and for metrics whose goal you declared with `define_metric`.
  Schedules such as the learning rate are therefore not flagged.

Alerts are printed to stderr, stored in `events.jsonl` and shown live in
`nuviz watch`. `log.alerts` returns them.

### Ablation

```python
from nuviz import Ablation, Logger

ab = Ablation("lr_depth", base_config={"lr": 1e-3, "depth": 4})
ab.vary("lr", [1e-3, 1e-4])
ab.vary("depth", [2, 4, 8])
ab.toggle("augmentation")            # True / False

for cfg in ab.generate():            # 12 config dicts
    for seed in range(3):
        with Logger(f"lr{cfg['lr']}_d{cfg['depth']}_seed{seed}",
                    project="sweep", hparams=cfg, seed=seed) as log:
            train(cfg, seed, log)

ab.export("configs/sweep/")          # one YAML file per config (needs PyYAML)
```

Runs that share hyperparameters but differ in `seed` are grouped
automatically by `nuviz leaderboard --aggregate` and `nuviz matrix`.

## wandb-style API

For scripts written against `wandb.init` / `wandb.log`:

```python
import nuviz

run = nuviz.init(project="my_project", name="exp-001", config={"lr": 1e-3})
for batch in loader:
    nuviz.log({"loss": loss, "train": {"acc": acc}})   # step auto-increments
nuviz.log({"val/render": image_array}, step=1000)      # arrays become images
nuviz.finish()
```

`log(data, step=None, commit=True)` follows wandb's semantics: nested dicts
are flattened to `a/b`, `commit=False` holds values for the next call.
`run.define_metric("x", summary="min")` sets the direction. wandb-only
arguments to `init` (`entity`, `mode`, …) are accepted and ignored.

## Framework integrations

```python
# Lightning
from nuviz.integrations.lightning import NuvizLogger
trainer = Trainer(logger=NuvizLogger("exp-001", project="my_project"))

# Hugging Face Trainer
from nuviz.integrations.transformers import NuvizCallback
trainer = Trainer(..., callbacks=[NuvizCallback(project="my_project")])
```

Both log from the main process only and record the hyperparameters. A run
that ends with an error is marked `failed`.

---

## CLI commands

Global option: `--dir <path>` overrides the base directory.

**Selecting experiments.** Wherever a command takes an experiment you can
give:

* its name (`exp-001`),
* `project/name` when the name exists in several projects,
* a prefix (≥ 4 characters) of its run id,
* a path to its directory, even outside the base directory.

A name that matches several experiments is an error that lists them; add
`--project` or use `project/name`.

### `nuviz watch`

Live dashboard.

```bash
nuviz watch                            # the most recent experiment
nuviz watch exp-001 exp-002            # several, overlaid
nuviz watch --latest 3 --project gs    # the 3 most recent in a project
nuviz watch --poll                     # no filesystem notifications (NFS/WSL)
```

Two charts, a table of runs (status, step / max steps, speed, ETA, last and
best values) and a panel with GPU load and alerts.

| Key | Action |
|---|---|
| `q`, `Esc` | quit |
| `Tab` | focus the other chart |
| `m` / `M`, `↓` / `↑` | next / previous metric in the focused chart |
| `a` | x axis: step ↔ wall time |
| `l` | linear ↔ log y axis |
| `s` / `S` | more / less smoothing |
| `+` / `-` | zoom the x axis |
| `←` / `→` | pan when zoomed in |
| `r` | reset zoom and pan |
| `v` | one chart ↔ two charts |
| `?` | help |

### `nuviz compare`

The same dashboard, opened on one large chart.

```bash
nuviz compare exp-001 exp-002 --metric loss
nuviz compare --project gs --metric psnr          # everything in a project
nuviz compare exp-001 exp-002 --align wall_time
```

### `nuviz ls`

```bash
nuviz ls
nuviz ls --project gs --sort steps      # sort: date (default), name, steps
nuviz ls --status running               # running, done, failed, crashed, unknown
nuviz ls --tag baseline --metric psnr   # metric shown in the "Best" column
```

### `nuviz leaderboard`

```bash
nuviz leaderboard --sort psnr                 # best first
nuviz leaderboard --sort psnr --asc           # worst first
nuviz leaderboard --value final               # value at the last step, not the best step
nuviz leaderboard --steps                     # show the step each value comes from
nuviz leaderboard --aggregate                 # mean ± std over seeds
nuviz leaderboard --metric psnr --metric ssim --top 10 --precision 2
nuviz leaderboard --format latex              # table (default), markdown, latex, csv
```

Experiments that never logged the sort metric are listed last. `--value
best` reports the best value seen at any step; for test metrics prefer
`--value final`, or select the checkpoint on a validation metric.

### `nuviz matrix`

Ablation matrix from logged hyperparameters (`Logger(hparams=…)`), with dot
notation for nested keys. Cells with several runs show mean ± std.

```bash
nuviz matrix --rows lr --cols model.depth --metric psnr
nuviz matrix --rows lr --cols model.depth --metric psnr --format latex --value final
```

### `nuviz breakdown`

```bash
nuviz breakdown exp-001
nuviz breakdown exp-001 --latex               # or --markdown
nuviz breakdown exp-001 --diff exp-002        # per-scene deltas
```

### `nuviz export`

```bash
nuviz export exp-001                          # CSV to stdout
nuviz export exp-001 --format json
nuviz export exp-001 --metric loss --metric psnr > curves.csv
```

Every step is exported; NaN/Inf and metrics not logged at a step are empty
cells.

### `nuviz image` / `nuviz diff`

```bash
nuviz image exp-001                           # browse (←/→, q)
nuviz image exp-001 --tag render --latest
nuviz image exp-001 --tag render --side-by-side gt
nuviz diff exp-001 exp-002 --tag render       # latest common step
nuviz diff exp-001 exp-002 --step 10000 --heatmap
```

The image protocol (Kitty, iTerm2, Sixel) is detected by querying the
terminal; anything else gets a half-block rendering that works everywhere.
Force one with `--protocol kitty|iterm2|sixel|halfblock` or
`NUVIZ_IMAGE_PROTOCOL`.

### `nuviz view`

```bash
nuviz view exp-001                            # latest PLY of an experiment
nuviz view path/to/file.ply --histogram
```

### `nuviz tag`

```bash
nuviz tag exp-001 baseline
nuviz tag exp-001 --list
nuviz tag exp-001 --remove baseline
```

### `nuviz cleanup`

```bash
nuviz cleanup --project gs                    # dry run: shows what would go
nuviz cleanup --project gs --keep-top 3 --metric psnr
nuviz cleanup --project gs --force            # delete
```

Running experiments are never removed. Tagged experiments are kept unless
`--include-tagged` is given. Runs without the ranking metric are removed
first.

### `nuviz reproduce`

Prints a shell script with the commit, packages, hyperparameters, seed and
original command line of a run.

---

## Configuration

### Logger (environment variables or `NuvizConfig`)

| Variable | Default | Description |
|---|---|---|
| `NUVIZ_DIR` | `~/.nuviz/experiments` | Base directory |
| `NUVIZ_FLUSH_INTERVAL` | `2.0` | Seconds between background flushes |
| `NUVIZ_FLUSH_COUNT` | `50` | Queued records that trigger a flush |
| `NUVIZ_QUEUE_SIZE` | `100000` | Maximum queued records before new ones are dropped |
| `NUVIZ_CLOSE_TIMEOUT` | `30` | Seconds `finish()` waits for the writer |
| `NUVIZ_HEARTBEAT` | `10` | Seconds between heartbeats |
| `NUVIZ_ALERTS` | `1` | `0` disables anomaly alerts |
| `NUVIZ_SNAPSHOT` | `1` | `0` disables the environment snapshot |
| `NUVIZ_GPU` | `1` | `0` disables GPU polling |
| `NUVIZ_GPU_POLL` | `5.0` | GPU polling interval in seconds |
| `NUVIZ_ROTATE_SIZE` | `50000000` | Rotate `metrics.jsonl` at this many bytes |
| `NUVIZ_ROTATE_LINES` | `500000` | Rotate `metrics.jsonl` at this many lines |

### CLI

| Variable | Description |
|---|---|
| `NUVIZ_DIR` | Base directory (`--dir` takes precedence) |
| `NUVIZ_IMAGE_PROTOCOL` | `kitty`, `iterm2`, `sixel` or `halfblock` |
| `NUVIZ_CACHE_DIR` | Where per-experiment summaries are cached |
| `NUVIZ_NO_CACHE` | `1` disables the summary cache |
