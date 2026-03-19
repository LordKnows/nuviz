# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NuViz — terminal-native ML training visualization. Two components:
- **Python library** (`python/`): Structured logger that records metrics (JSONL), images (PNG), and environment snapshots into `~/.nuviz/experiments/<name>/`
- **Rust CLI** (`cli/`): TUI for real-time monitoring, experiment comparison, and LaTeX table export

Data flows one way: Python writes JSONL → Rust reads it. The default data directory is `~/.nuviz/experiments/` (override with `NUVIZ_DIR` env var).

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
cargo test --test integration_test --test hardening_test  # Integration tests (requires Python package installed)
cargo fmt -- --check                 # Format check
cargo clippy -- -D warnings          # Lint
```

## Architecture

### Python Logger (`python/src/nuviz/`)

`Logger` is the public entry point — it orchestrates all other modules:
- `config.py` — `NuvizConfig` frozen dataclass, env var overrides (`NUVIZ_DIR`, `NUVIZ_FLUSH_INTERVAL`, `NUVIZ_GPU`, etc.)
- `naming.py` — experiment name sanitization, timestamp fallback
- `writer.py` — background daemon thread writing JSONL with deque + lock, time/count-based flushing, `atexit` hook
- `gpu.py` — `GpuCollector` daemon thread polling nvidia-smi (graceful fallback when unavailable)
- `anomaly.py` — NaN/Inf detection + Welford's online algorithm for 3-sigma spike alerts
- `snapshot.py` — best-effort capture of git hash, pip freeze, GPU info, Python version
- `image.py` — numpy/torch tensor → PNG conversion (Pillow optional)
- `types.py` — frozen dataclasses for metric records and alerts
- `ablation.py` — `Ablation` class: `vary()`, `toggle()`, `generate()` configs, `export()` to YAML/JSON (PyYAML optional)
- `scene_writer.py` — synchronous writer for per-scene metrics to `scenes.jsonl`
- `pointcloud.py` — numpy/torch → binary PLY conversion (no extra deps)

### Rust CLI (`cli/src/`)

Entry: `main.rs` → `cli.rs` (clap derive) → command modules.

- `data/` — JSONL parsing with rotation support (`metrics.rs`), experiment discovery (`experiment.rs`), metadata (`meta.rs`), scene records (`scenes.rs`), multi-seed aggregation (`aggregation.rs`), image discovery (`images.rs`), PLY parser with 3DGS support (`ply.rs`)
- `commands/` — `watch.rs`, `ls.rs`, `leaderboard.rs`, `compare.rs`, `matrix.rs`, `breakdown.rs`, `export.rs`, `image.rs`, `diff.rs`, `view.rs`, `tag.rs`, `cleanup.rs`, `reproduce.rs`
- `tui/` — ratatui app loop (`app.rs`), braille chart renderer (`chart.rs`), dashboard widgets (`widgets.rs`)
- `watcher/` — `notify`-based file watching (`file_watcher.rs`) + incremental JSONL tail reader (`tail.rs`)
- `terminal/` — capability detection (`capability.rs`), image rendering with Kitty/Sixel/iTerm2/half-block protocols (`render.rs`), error heatmap generation (`heatmap.rs`)

### Data Format

Each experiment lives in `<base_dir>/<experiment_name>/`:
- `metrics.jsonl` — one JSON object per step with timestamp, step number, and metric values
- `images/<tag>_<step>.png` — rendered images
- `meta.json` — environment snapshot (git, pip, GPU, config) with optional `seed` and `config_hash`
- `scenes.jsonl` — per-scene metrics (separate from step-level data)
- `pointclouds/<tag>_step_<step>.ply` — binary PLY point cloud files

## Key Design Decisions

- Zero required Python dependencies beyond stdlib (numpy and Pillow are optional)
- Python 3.10+ minimum (uses `match`, `slots=True`, `frozen=True` dataclasses)
- All dataclasses are frozen (immutable) — never mutate, always create new instances
- Writer uses a daemon thread so it doesn't block training loops
- Rust reads JSONL lazily: tail-follow for `watch`, seek-to-end for `ls`/`leaderboard`
- `notify` crate with poll fallback for NFS/WSL compatibility
- Multi-seed grouping via `config_hash` in meta.json (fallback: strip `_seed\d+` suffix)
- PyYAML as optional dep (`yaml` extras), same gating pattern as Pillow
- Image rendering uses protocol priority: Kitty > iTerm2 > Sixel > half-block fallback
- JSONL rotation at 50MB/500k lines; Rust reader merges `metrics.jsonl` + `metrics.N.jsonl` segments
- PLY parser handles binary LE + ASCII, supports 3DGS attributes (SH, scale, rotation, opacity)
- GPU metrics collected automatically via nvidia-smi polling; graceful no-op on CPU-only machines
- `update_tags()` patches meta.json via `serde_json::Value` to avoid clobbering Python-written fields
- `cleanup --force` required for actual deletion; default mode is dry-run with size estimates
- Full design doc (Chinese + English) at `docs/nuviz_design.md` — covers roadmap through Phase 4
