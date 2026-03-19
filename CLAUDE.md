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
pip install -e ".[dev,images]"       # Install with all dev + optional deps
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
- `config.py` — `NuvizConfig` frozen dataclass, env var overrides (`NUVIZ_DIR`, `NUVIZ_FLUSH_INTERVAL`, etc.)
- `naming.py` — experiment name sanitization, timestamp fallback
- `writer.py` — background daemon thread writing JSONL with deque + lock, time/count-based flushing, `atexit` hook
- `anomaly.py` — NaN/Inf detection + Welford's online algorithm for 3-sigma spike alerts
- `snapshot.py` — best-effort capture of git hash, pip freeze, GPU info, Python version
- `image.py` — numpy/torch tensor → PNG conversion (Pillow optional)
- `types.py` — frozen dataclasses for metric records and alerts

### Rust CLI (`cli/src/`)

Entry: `main.rs` → `cli.rs` (clap derive) → command modules.

- `data/` — JSONL parsing (`metrics.rs`), experiment directory discovery (`experiment.rs`), metadata reading (`meta.rs`)
- `commands/` — `watch.rs`, `ls.rs`, `leaderboard.rs`
- `tui/` — ratatui app loop (`app.rs`), braille chart renderer (`chart.rs`), dashboard widgets (`widgets.rs`)
- `watcher/` — `notify`-based file watching (`file_watcher.rs`) + incremental JSONL tail reader (`tail.rs`)
- `terminal/` — Kitty/Sixel/iTerm2 capability detection (`capability.rs`)

### Data Format

Each experiment lives in `<base_dir>/<experiment_name>/`:
- `metrics.jsonl` — one JSON object per step with timestamp, step number, and metric values
- `images/<tag>_<step>.png` — rendered images
- `meta.json` — environment snapshot (git, pip, GPU, config)

## Key Design Decisions

- Zero required Python dependencies beyond stdlib (numpy and Pillow are optional)
- Python 3.10+ minimum (uses `match`, `slots=True`, `frozen=True` dataclasses)
- All dataclasses are frozen (immutable) — never mutate, always create new instances
- Writer uses a daemon thread so it doesn't block training loops
- Rust reads JSONL lazily: tail-follow for `watch`, seek-to-end for `ls`/`leaderboard`
- `notify` crate with poll fallback for NFS/WSL compatibility
