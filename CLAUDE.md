# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NuViz — terminal-native ML training visualization. Two components:
- **Python library** (`python/`): Structured logger that records metrics (JSONL), images (PNG), and environment snapshots into `~/.nuviz/experiments/<name>/`
- **Rust CLI** (`cli/`): TUI for real-time monitoring, experiment comparison, and LaTeX table export

Data flows one way: Python writes files → Rust reads them. The on-disk contract is specified in `docs/FORMAT.md`; both sides are tested against the golden files in `spec/fixtures/`. The default data directory is `~/.nuviz/experiments/` (override with `NUVIZ_DIR` env var).

## Build & Test Commands

### Python (working directory: `python/`)

```bash
pip install -e ".[dev,images,yaml]"  # Install with all dev + optional deps
pytest tests/                        # Run all tests
pytest tests/test_writer.py          # Run a single test file
pytest tests/ --cov=nuviz --cov-report=term-missing --cov-fail-under=80
ruff check src/ tests/               # Lint
mypy src/nuviz/ --ignore-missing-imports  # Type check
```

### Rust (working directory: `cli/`)

```bash
cargo build                          # Build
cargo test --lib                     # Unit tests only
cargo test                           # Everything (integration/e2e tests need the Python package installed)
cargo fmt -- --check                 # Format check
cargo clippy --all-targets -- -D warnings  # Lint
maturin build --release              # Build the `nuviz-cli` wheel (bindings = "bin")
```

Test targets: `integration_test`, `hardening_test`, `conformance_test` (golden files), `e2e_test` (Python writes, the real binary reads).

## Architecture

### Python Logger (`python/src/nuviz/`)

`Logger` is the public entry point — it orchestrates all other modules:
- `config.py` — `NuvizConfig` frozen dataclass, env var overrides (`NUVIZ_DIR`, `NUVIZ_FLUSH_INTERVAL`, `NUVIZ_GPU`, `NUVIZ_HEARTBEAT`, `NUVIZ_QUEUE_SIZE`, etc.)
- `naming.py` — experiment name sanitization, timestamp fallback, atomic directory claim
- `writer.py` — `JsonlWriter`: one background thread owns the file (write, rotate); bounded queue; `flush()`/`close()` signal and wait
- `meta.py` — `MetaStore`: atomic read-modify-write of `meta.json` (preserves keys the CLI writes, e.g. `tags`)
- `events.py` — `EventWriter` for `events.jsonl` (alerts, image / point-cloud references)
- `goals.py` — metric direction heuristic (mirrored in `cli/src/data/goal.rs`)
- `gpu.py` — `GpuCollector` daemon thread polling nvidia-smi (graceful fallback when unavailable)
- `anomaly.py` — NaN/Inf detection + direction-aware spike alerts against an exponentially weighted moving average
- `snapshot.py` — best-effort capture of git hash, installed packages, GPU info, argv (runs in a background thread)
- `image.py` — numpy/torch tensor → PNG conversion (numpy and Pillow imported lazily)
- `types.py` — frozen dataclasses for metric records and alerts
- `ablation.py` — `Ablation` class: `vary()`, `toggle()`, `generate()` configs, `export()` to YAML (PyYAML optional)
- `scene_writer.py` — synchronous writer for per-scene metrics to `scenes.jsonl`
- `pointcloud.py` — numpy/torch → binary PLY conversion
- `run.py` — wandb-style `init` / `log` / `finish` on top of `Logger`
- `integrations/` — `lightning.NuvizLogger`, `transformers.NuvizCallback` (frameworks imported only by these modules)

### Rust CLI (`cli/src/`)

Entry: `main.rs` (uses the `nuviz_cli` library crate) → `cli.rs` (clap derive, typed enums) → command modules.

- `data/` — wire format and line parser (`metrics.rs`), columnar in-memory store (`store.rs`), per-run summaries + incremental cache (`summary.rs`), lenient `meta.json` parsing (`meta.rs`), run status from heartbeat (`status.rs`), experiment discovery (`experiment.rs`), name/run-id/path resolution (`resolve.rs`), metric direction (`goal.rs`), mean/std (`aggregation.rs`), scene records (`scenes.rs`), image discovery (`images.rs`), PLY parser with 3DGS support (`ply.rs`)
- `commands/` — `watch.rs`, `compare.rs` (both open the dashboard), `ls.rs`, `leaderboard.rs`, `matrix.rs`, `breakdown.rs`, `export.rs`, `image.rs`, `diff.rs`, `view.rs`, `tag.rs`, `cleanup.rs`, `reproduce.rs`, shared helpers in `common.rs`
- `tui/` — dashboard state, rendering and event loop (`dashboard.rs`), braille chart renderer with downsampling and axes (`chart.rs`), terminal guard + panic hook (`terminal.rs`)
- `watcher/` — incremental, rotation-aware JSONL tail reader (`tail.rs`), live experiment source (`source.rs`), optional filesystem notifications (`signal.rs`)
- `terminal/` — image protocol detection by querying the terminal (`capability.rs`), Kitty/Sixel/iTerm2/half-block encoders (`render.rs`), error heatmap generation (`heatmap.rs`)
- `output.rs` — LaTeX / CSV / Markdown escaping

### Data Format

Specified in `docs/FORMAT.md`. Each experiment lives in `<base_dir>/[<project>/]<experiment_name>/`:
- `metrics.jsonl` — one JSON object per step with timestamp, step number, and metric values (`null` = NaN/Inf)
- `meta.json` — written at start (`schema_version`, `run_id`, `status`, `config`, …), snapshot fields merged in, final summary at finish
- `heartbeat.json` — refreshed while the run is alive; used to tell `running` from `crashed`
- `events.jsonl` — alerts and references to saved images / point clouds
- `images/step_<step>_<tag>.png` — rendered images
- `scenes.jsonl` — per-scene metrics (separate from step-level data)
- `pointclouds/<tag>_step_<step>.ply` — binary PLY point cloud files

## Key Design Decisions

- Zero required Python dependencies beyond stdlib; numpy, Pillow and PyYAML are optional and imported lazily
- Python 3.10+ minimum (uses `match`, `slots=True`, `frozen=True` dataclasses)
- All dataclasses are frozen (immutable) — never mutate, always create new instances
- The writer thread is the only code that touches `metrics.jsonl`; nothing else may write or rotate it
- Metric values are coerced to plain numbers in `Logger.step()` on the caller's thread; the writer thread never dies on a bad value
- A full write queue applies brief backpressure, then drops and counts — it never blocks training indefinitely
- `meta.json` is always updated by read-modify-write with an atomic replace; `update_tags()` on the Rust side patches via `serde_json::Value` for the same reason
- Run status = declared status in `meta.json` + heartbeat freshness (`data/status.rs`)
- Experiment discovery reads only `meta.json` and `heartbeat.json`; summaries come from `meta.json` (finished runs) or the incremental cache in the user's cache dir (`NUVIZ_CACHE_DIR`, `NUVIZ_NO_CACHE`)
- Metric direction: `meta.metric_goals` first, then one shared name heuristic (fixture: `spec/fixtures/metric_goals.json`)
- All commands resolve experiments through `data::resolve` (name, `project/name`, run-id prefix, path); ambiguity is an error
- The tail reader consumes complete lines only and follows rotation by inode; it owns the initial history read, so callers never pre-load
- The dashboard redraws only when data, input or size changed; charts bucket points per pixel column
- Filesystem notifications are an optimisation; sources poll on a timer regardless (NFS/WSL safe)
- Multi-seed grouping via `config_hash` in meta.json (derived from `hparams` without `seed`; fallback: strip `_seed\d+` suffix)
- Image protocol is queried from the terminal (ratatui-image `Picker`), never guessed from `TERM`; half-block is the fallback
- JSONL rotation at 50MB/500k lines; readers merge `metrics.jsonl` + `metrics.N.jsonl` segments
- PLY parser handles binary LE + ASCII, supports 3DGS attributes (SH, scale, rotation, opacity)
- GPU metrics collected automatically via nvidia-smi polling; graceful no-op on CPU-only machines
- `cleanup --force` required for actual deletion; running experiments are never removed, tagged ones only with `--include-tagged`
- The CLI is also packaged as the `nuviz-cli` wheel (`cli/pyproject.toml`, maturin `bindings = "bin"`); the logger stays pure Python
- MSRV is Rust 1.88 (`rust-version` in `cli/Cargo.toml`, checked in CI); CI lints with a pinned toolchain
- Full design doc (Chinese + English) at `docs/nuviz_design.md` — covers roadmap through Phase 4; Phase 5 is tracked in `PROGRESS.md`
