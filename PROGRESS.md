# NuViz Implementation Progress

## Phase 1 Overview

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

## Phase 1 Key Decisions

- Pillow as optional dep for `log.image()`
- Writer daemon thread with `atexit` flush
- Rust reads JSONL lazily (tail reader for watch, seek-to-end for ls/leaderboard)
- `notify` crate with `--poll` fallback for NFS/WSL
- Python 3.10+ minimum
- Zero required deps beyond stdlib + numpy (optional)

---

## Phase 2 Overview

Phase 2 adds ablation experiment workflows and paper-ready output.

## Phase 2A: Python — Ablation & Scene API (Steps 1-3)

- [x] Step 1: `log.scene()` API — per-scene metrics to `scenes.jsonl`
- [x] Step 2: `Ablation` class — `vary()`, `toggle()`, `generate()`, `export()` with PyYAML optional dep
- [x] Step 3: Multi-seed metadata — `seed` and `config_hash` fields in Logger + meta.json

## Phase 2B: Rust Data Layer (Steps 4-6)

- [x] Step 4: Scene data parser — `SceneRecord` + `read_scenes()` in `data/scenes.rs`
- [x] Step 5: Multi-seed aggregation — `group_by_config()`, `AggregatedMetrics`, mean ± std
- [x] Step 6: Alignment support — `AlignMode::Step` / `AlignMode::WallTime`, `align_series()`

## Phase 2C: Rust Commands (Steps 7-9)

- [x] Step 7: `nuviz compare` — multi-experiment curve overlay TUI with colors, legend, alignment
- [x] Step 8: `nuviz matrix` — ablation matrix view with `--rows`/`--cols`/`--metric`, key findings
- [x] Step 9: `nuviz breakdown` — per-scene metrics with `--latex`, `--markdown`, `--diff`

## Phase 2D: Paper Assistance (Steps 10-11)

- [x] Step 10: Enhanced LaTeX/Markdown — bold best (`\textbf{}`), underline 2nd best (`\underline{}`)
- [x] Step 11: `nuviz export` — raw data dump as CSV or JSON

## Phase 2 Dependency Graph

```
Step 1 (scene API) ─────────────────────> Step 4 (scene parser) ──> Step 9 (breakdown)
Step 2 (ablation) ──> Step 3 (seed meta) ──> Step 5 (aggregation) ──> Step 8 (matrix)
                                                                   ──> Step 10 (latex)
                                          ──> Step 6 (alignment)  ──> Step 7 (compare)
                                              Step 11 (export)        [independent]
```

## Phase 2 Key Decisions

- Scene data in separate `scenes.jsonl` (keeps step-level data clean)
- Multi-seed grouping via `config_hash` in meta.json (fallback: strip `_seed\d+` suffix)
- `nuviz table` enhances existing `leaderboard` (no new command)
- `nuviz export` is a new command (raw time-series vs summary)
- Compare TUI reuses `BrailleCanvas` with shared canvas, 8-color palette
- PyYAML as optional dep (`yaml` extras), same pattern as Pillow
