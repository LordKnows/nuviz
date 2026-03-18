# NuViz Phase 1 Implementation Progress

## Plan Overview

Phase 1 delivers the MVP: Python Logger + Rust CLI basics (watch, ls, leaderboard).

## Phase 1A: Python Logger (Steps 1-9)

- [x] Step 1: Project scaffolding (`pyproject.toml`, `__init__.py`)
- [x] Step 2: Types + config (frozen dataclasses, `NuvizConfig`, env var)
- [x] Step 3: Experiment naming (user name + timestamp fallback, sanitize)
- [x] Step 4: Background JSONL writer (daemon thread, deque + lock, time/count flush)
- [x] Step 5: Anomaly detection (NaN/Inf + Welford's 3-sigma spike)
- [x] Step 6: Environment snapshot (git, pip, GPU, Python — all best-effort)
- [x] Step 7: Image saving (numpy/torch -> PNG, Pillow optional dep)
- [x] Step 8: Core Logger class (orchestrates steps 3-7, context manager)
- [x] Step 9: Integration tests (full training loop simulation)

## Phase 1B: Rust CLI (Steps 10-17)

- [x] Step 10: Cargo scaffolding (clap, ratatui, crossterm, notify, serde)
- [x] Step 11: Data layer — JSONL parsing + experiment discovery
- [x] Step 12: `nuviz ls` — tabular experiment listing
- [x] Step 13: `nuviz leaderboard` — ranked metrics table
- [x] Step 14: Braille chart renderer (2x4 dot grid, Bresenham, auto-scale)
- [x] Step 15: File watcher + tail reader (notify + incremental JSONL read)
- [x] Step 16: `nuviz watch` — TUI dashboard (charts, info, keyboard)
- [x] Step 17: Terminal capability detection (Kitty/Sixel/iTerm2 — detect only)

## Phase 1C: Integration & Polish (Steps 18-20)

- [x] Step 18: Cross-component integration test (Python writes, Rust reads)
- [x] Step 19: Error handling hardening (NaN serialization fix, edge case tests)
- [x] Step 20: GitHub Actions CI (Python lint/test/coverage + Rust fmt/clippy/test + integration)

## Dependency Graph

```
Steps 1-2 ──┬─> 3 (naming)
             ├─> 4 (writer)
             ├─> 5 (anomaly)     ──┐
             ├─> 6 (snapshot)      ├─> 8 (Logger) ──> 9 (integration)
             └─> 7 (image)      ──┘

Step 10 ────┬─> 11 (data) ──┬─> 12 (ls)
            │               ├─> 13 (leaderboard)
            │               └─> 15 (watcher)  ──┐
            ├─> 14 (chart)  ────────────────────┼─> 16 (watch TUI)
            └─> 17 (terminal caps)              ┘
```

## Key Decisions

- Pillow as optional dep for `log.image()`
- Writer daemon thread with `atexit` flush
- Rust reads JSONL lazily (tail reader for watch, seek-to-end for ls/leaderboard)
- `notify` crate with `--poll` fallback for NFS/WSL
- Python 3.10+ minimum
- Zero required deps beyond stdlib + numpy (optional)
