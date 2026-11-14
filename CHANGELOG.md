# Changelog

## Unreleased

### Fixed

- **Metrics passed as numpy scalars or tensors no longer kill the writer
  thread.** Previously `log.step(loss=np.float32(...))` stopped all further
  writes until `finish()`, and lost them if the process was killed.
- **A NaN at one step no longer hides the whole step in the CLI.** The CLI
  failed to parse the `null` the logger writes for NaN and dropped the record,
  including its other metrics.
- `nuviz watch` / `compare` showed every record twice after start-up.
- The live reader lost records that were half-written when it polled, and
  records around a file rotation.
- `finish()` with a large backlog wrote from two threads at once, which could
  reorder records and overwrite rotated files.
- `nuviz cleanup --force` could delete running and tagged experiments.
- Experiments without the sort metric were ranked at arbitrary positions in
  `nuviz leaderboard`; they are now last.
- LaTeX output did not escape metric or experiment names (`val_loss` broke
  compilation). CSV and Markdown output are escaped too.
- Running experiments were listed as `unknown`; killed runs were
  indistinguishable from finished ones.
- `import nuviz` required numpy despite being documented as dependency-free.
- Re-running a script with the same experiment name produced two experiments
  the CLI could not tell apart; two runs started simultaneously could share a
  directory.
- `nuviz image` sent Sixel data to terminals that cannot display it, and the
  half-block fallback wrapped wide images.
- Learning-rate schedules triggered a "spike" alert at every step change.
- `nuviz matrix` sorted parameter values as text (`128` before `64`).

### Added

- Data contract ([docs/FORMAT.md](docs/FORMAT.md)) with fixtures shared by the
  Python and Rust test suites.
- `Logger(hparams=...)`, `Logger(max_steps=...)`, `Logger(tags=...)`,
  `Logger.define_metric()`, `Logger.flush()`.
- Run status: `running`, `done`, `failed`, `crashed`, driven by a heartbeat.
- `events.jsonl`: alerts and references to saved images / point clouds.
- wandb-style `nuviz.init` / `nuviz.log` / `nuviz.finish`.
- `nuviz.integrations.lightning.NuvizLogger` and
  `nuviz.integrations.transformers.NuvizCallback`.
- Dashboard: all watched runs overlaid in color, metric switching, step or
  wall-time axis, log scale, smoothing, zoom/pan, speed and ETA, GPU and
  alert panels, help overlay.
- `nuviz leaderboard --value best|final --steps --aggregate --metric --tag
  --precision`; `nuviz ls --status --tag --metric`; `nuviz matrix --value`.
- Experiments can be addressed as `project/name`, by run-id prefix or by
  path; ambiguous names are reported instead of silently resolved.
- `--protocol` / `NUVIZ_IMAGE_PROTOCOL` for `nuviz image` and `nuviz diff`.
- `pip install "nuviz[cli]"` installs the CLI as a prebuilt binary wheel
  (`nuviz-cli`).

### Changed

- The CLI no longer parses every JSONL file for every command: summaries come
  from `meta.json` or an incremental cache, and metrics are stored
  column-wise in memory.
- The dashboard redraws only when something changed.
- The environment snapshot is captured in a background thread and lists
  packages via `importlib.metadata` instead of `pip freeze`.
- `git_dirty` now also reflects staged changes (and ignores untracked files).
- Spike alerts compare against a moving average and only watch metrics that
  are minimized or have a declared goal.
- `--format`, `--sort` and `--align` reject unknown values instead of
  falling back to a default.
- `meta.json` `name` is the (unique) directory name.
- Minimum supported Rust version is 1.88.
- Releases publish to PyPI via Trusted Publishing and fail loudly instead of
  ignoring publish errors.
