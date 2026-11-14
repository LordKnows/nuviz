# NuViz Data Format

This document is the contract between the Python logger (the only writer)
and the Rust CLI (the reader). Both sides are tested against the shared
fixtures in [`spec/fixtures/`](../spec/fixtures), so a change here must come
with a fixture change and pass on both sides.

Current `schema_version`: **1**

## Layout

```
<base_dir>/[<project>/]<experiment>/
├── meta.json            identity, status, hyperparameters, summary
├── heartbeat.json       liveness of a running experiment
├── metrics.jsonl        one record per logged step (newest segment)
├── metrics.<N>.jsonl    rotated segments; a higher N is older
├── events.jsonl         alerts and references to saved media
├── scenes.jsonl         per-scene evaluation metrics
├── images/step_<SSSSSS>_<tag>.png
└── pointclouds/<tag>_step_<SSSSSS>.ply
```

`<base_dir>` is `$NUVIZ_DIR`, else `~/.nuviz/experiments`. A directory is an
experiment if it contains `metrics.jsonl` or `meta.json`; a directory that
does not, directly under the base directory, is a project. Directories whose
name starts with `.` are ignored.

`<experiment>` is unique within its project. If the requested name is taken
the logger appends `_1`, `_2`, …; the directory is claimed atomically, so two
runs started at the same moment never share one.

## `metrics.jsonl`

One JSON object per line, newline terminated, strict JSON (no `NaN` /
`Infinity` literals).

```json
{"step":12,"timestamp":1700000000.5,"metrics":{"loss":0.25,"psnr":21.5},"gpu":{"gpu_util":87,"mem_used_mb":10240,"mem_total_mb":24576,"temperature_c":72}}
```

| Field | Type | Notes |
|---|---|---|
| `step` | non-negative integer | Not required to be unique or increasing. |
| `timestamp` | number | Unix time in seconds when the step was logged. |
| `metrics` | object: name → number or `null` | `null` means the value was NaN or ±Inf. May be empty. |
| `gpu` | object, optional | Latest GPU sample: `gpu_util` (%), `mem_used_mb`, `mem_total_mb`, `temperature_c`. |

Reader rules:

* A `null` metric is read as "not a finite number"; the record and its other
  metrics are kept.
* Unknown top-level keys are ignored.
* A line that is not valid JSON, or lacks `step`/`timestamp`/`metrics`, is
  skipped.
* A final line without a trailing newline is still being written and must not
  be consumed yet.

### Rotation

When `metrics.jsonl` reaches 50 MB or 500 000 lines (configurable) it is
renamed to `metrics.1.jsonl`, existing segments shift up by one, and a new
`metrics.jsonl` is started. Chronological order is therefore
`metrics.<max>.jsonl … metrics.1.jsonl, metrics.jsonl`. Only the logger's
single writer thread writes or rotates these files.

## `meta.json`

Written when the run starts and merged (read–modify–write, atomic replace) on
every update. Writers must preserve keys they do not own: the CLI's
`nuviz tag` patches `tags` while the logger owns the rest.

Written at start:

| Key | Type | Notes |
|---|---|---|
| `schema_version` | integer | `1` |
| `run_id` | string | 12 hex characters, unique per run. |
| `name` | string | Equals the directory name. |
| `project` | string or null | |
| `status` | string | `running`, later `done` or `failed`. |
| `start_time` | string | RFC 3339, UTC. |
| `hostname`, `pid`, `python_version` | | |
| `seed` | integer, optional | |
| `config` | object, optional | Hyperparameters (`Logger(hparams=…)`). |
| `config_hash` | string, optional | Groups runs that differ only by seed. Derived from `config` without `seed` when not given. |
| `max_steps` | integer, optional | Planned number of steps, for ETA. |
| `tags` | array of strings, optional | |
| `metric_goals` | object: name → `"min"` \| `"max"`, optional | From `Logger.define_metric()`. |

Added by the background environment snapshot: `git_hash`, `git_dirty`
(tracked files differ from `HEAD`), `pip_packages` (`name==version`),
`cuda_version`, `gpu_model`, `pytorch_version`, `argv`, `cwd`.

Written at finish:

| Key | Type | Notes |
|---|---|---|
| `status` | string | `done` or `failed`. |
| `error` | string, optional | Short description when failed. |
| `end_time` | string | RFC 3339, UTC. |
| `total_steps` | integer | The last step logged. |
| `metric_summary` | object: name → summary | See below. |
| `best_metrics`, `best_steps` | object | Best value per metric (by goal) and its step. |
| `final_metrics` | object | Last value per metric. |
| `num_alerts` | integer | |
| `dropped_records` | integer, optional | Records dropped because the write queue was full. |

A metric summary covers finite values only:

```json
{"count":3,"min":0.2,"min_step":1,"max":1.0,"max_step":0,"last":0.5,"last_step":2}
```

Readers treat every field as optional and tolerate wrong types field by
field.

## `heartbeat.json`

Rewritten atomically by the logger's writer thread while the run is alive.

```json
{"timestamp":1700000123.4,"step":1840,"interval":10.0,"pid":4242}
```

`interval` is the longest gap the writer promises between heartbeats.

### Status as shown by the CLI

| `meta.status` | Heartbeat | Shown as |
|---|---|---|
| `done` / `failed` | any | `done` / `failed` |
| `running` | newer than `max(6 × interval, 60 s)` | `running` |
| `running` | older | `crashed` |
| `running` | missing | `running` if files changed in the last 120 s, else `crashed` |
| absent | — | `running` if files changed in the last 120 s, else `unknown` |

## `events.jsonl`

Append-only, one strict-JSON object per line. Every event has `timestamp`
and `type`; most have `step`. Readers ignore unknown types and fields.

```json
{"timestamp":1700000050.1,"type":"alert","step":150,"alert_type":"nan","metric":"loss","value":null,"message":"loss is NaN at step 150"}
{"timestamp":1700000060.2,"type":"image","step":500,"tag":"render","path":"images/step_000500_render.png"}
{"timestamp":1700000070.3,"type":"pointcloud","step":5000,"tag":"gaussians","path":"pointclouds/gaussians_step_005000.ply"}
```

`alert_type` is `nan`, `inf` or `spike`. `path` is relative to the
experiment directory.

## `scenes.jsonl`

```json
{"scene":"bicycle","metrics":{"psnr":28.4,"ssim":0.92},"timestamp":1700000200.0}
```

The latest record per scene wins.

## Metric direction

Whether lower or higher is better comes from `meta.metric_goals`. For
metrics without a declaration both sides apply the same name heuristic,
checked against `spec/fixtures/metric_goals.json`:

* minimized if the lower-cased name contains `loss`, `lpips`, `error`, `mse`
  or `mae`, or has one of these as a whole token: `err`, `rmse`, `fid`, `kid`,
  `ppl`, `perplexity`, `wer`, `cer`, `nll`, `kl`, `bpd`;
* maximized otherwise.

## Compatibility

* Readers accept data without `schema_version` (written before the contract
  was versioned): no heartbeat, no events, `meta.json` possibly without
  `status`.
* Adding optional keys or event types does not change `schema_version`.
  Changing the meaning or type of an existing key does.

## Reader-side cache

The CLI caches per-experiment summaries outside the experiment directory, in
`$NUVIZ_CACHE_DIR` or the platform cache directory
(`~/.cache/nuviz/summaries` on Linux). It is safe to delete and is disabled
with `NUVIZ_NO_CACHE=1`.
