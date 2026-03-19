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

---

## Phase 3 Overview

Phase 3 adds visualization enhancement: terminal image browsing, visual diff, point cloud support, and JSONL rotation.

## Phase 3A: Terminal Image Rendering (Steps 1-3)

- [x] Step 1: Add `image` + `base64` crates to Cargo.toml
- [x] Step 2: Image rendering backend (`terminal/render.rs`) — Kitty, iTerm2, Sixel, half-block
- [x] Step 3: Image file discovery (`data/images.rs`) — parse `step_NNNNNN_<tag>.png`

## Phase 3B: `nuviz image` Command (Steps 4-5)

- [x] Step 4: CLI args for `nuviz image` (--step, --tag, --latest, --side-by-side)
- [x] Step 5: Command implementation with arrow-key navigation

## Phase 3C: `nuviz diff` Command (Steps 6-8)

- [x] Step 6: CLI args for `nuviz diff` (--heatmap, --scene)
- [x] Step 7: Error heatmap generation (`terminal/heatmap.rs`) — turbo colormap, PSNR/MAE
- [x] Step 8: Diff command with side-by-side + heatmap modes

## Phase 3D: Python Pointcloud API (Steps 9-11)

- [x] Step 9: `PointcloudRecord` frozen dataclass in types.py
- [x] Step 10: PLY writer module (`pointcloud.py`) — numpy/torch → binary PLY
- [x] Step 11: `Logger.pointcloud()` method

## Phase 3E: PLY Parser & `nuviz view` (Steps 12-14)

- [x] Step 12: PLY parser (`data/ply.rs`) — binary LE + ASCII, 3DGS attributes
- [x] Step 13: CLI args for `nuviz view` (--histogram)
- [x] Step 14: `nuviz view` command — stats table, attribute histograms

## Phase 3F: JSONL Rotation (Steps 15-17)

- [x] Step 15: Python rotation logic in writer.py (size + line limits)
- [x] Step 16: Rotation config fields + env vars in NuvizConfig
- [x] Step 17: Rust multi-file JSONL reader + tail rotation detection

## Phase 3G: Tests (Steps 18-22)

- [x] Step 18: Python pointcloud tests (test_pointcloud.py)
- [x] Step 19: Python rotation tests (test_rotation.py)
- [x] Step 20: Rust PLY parser unit tests (inline in ply.rs)
- [x] Step 21: Rust image discovery + multi-JSONL tests (inline)
- [x] Step 22: Integration tests verified

## Phase 3H: CI (Step 23)

- [x] Step 23: CI picks up all new tests, no structural changes needed

## Phase 3 Dependency Graph

```
Step 1 (deps) ──┬──> Step 2 (render) ──┬──> Step 5 (image cmd)
                │                       └──> Step 8 (diff cmd)
                ├──> Step 7 (heatmap) ─────> Step 8
                └──> Step 12 (PLY parser) ──> Step 14 (view cmd)

Step 3 (image discovery) ──> Step 5, Step 8
Step 9 (type) ──> Step 10 (PLY writer) ──> Step 11 (Logger)
Step 16 (config) ──> Step 15 (rotation) ──> Step 17 (multi-JSONL reader)
```

## Phase 3 Key Decisions

- Image rendering priority: Kitty > iTerm2 > Sixel > half-block (ANSI 24-bit color)
- PLY parser in-house (no `ply-rs` dep) for full 3DGS attribute control
- Sixel uses 216-color (6×6×6) uniform quantization
- JSONL rotation: rename-and-shift strategy (metrics.jsonl → metrics.1.jsonl → metrics.2.jsonl)
- Rotation check per-record (not per-batch) to respect limits precisely
- `nuviz view` computes stats eagerly (streaming pass planned for 500MB+ files)
