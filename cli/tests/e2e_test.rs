//! End-to-end tests: the Python logger writes experiments, the `nuviz`
//! binary reads them. Skipped when the Python package is not installed.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

fn python() -> &'static str {
    if Command::new("python3").arg("--version").output().is_ok() {
        "python3"
    } else {
        "python"
    }
}

fn nuviz_available() -> bool {
    Command::new(python())
        .args(["-c", "import nuviz"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Experiments generated once and shared (read-only) by the tests.
struct World {
    _dir: tempfile::TempDir,
    base: PathBuf,
    cache: PathBuf,
}

const GENERATE: &str = r#"
import json, sys, pathlib
from nuviz import Logger
from nuviz.config import NuvizConfig

base = pathlib.Path(sys.argv[1])
cfg = NuvizConfig(base_dir=base, flush_interval_seconds=0.05, flush_count=10,
                  enable_snapshot=False, enable_gpu=False)

# Ablation grid with three seeds per configuration.
for lr in (0.001, 0.01):
    for width in (64, 128):
        for seed in range(3):
            hp = {"lr": lr, "model": {"width": width}, "seed": seed}
            name = f"lr{lr}_w{width}_seed{seed}"
            with Logger(name, project="grid", config=cfg, seed=seed, hparams=hp) as log:
                for step in range(20):
                    quality = (0.5 if lr == 0.001 else 0.3) + width / 1000 + seed * 0.01
                    log.step(step, val_loss=1.0 / (step + 1) / quality, psnr=20 + step * quality)

# A run with a NaN in the middle and an explicitly declared goal.
with Logger("nan_run", project="misc", config=cfg) as log:
    log.define_metric("regret", "min")
    for step in range(10):
        log.step(step, loss=float("nan") if step == 4 else 1.0 / (step + 1), regret=10 - step)

# A run that fails.
try:
    with Logger("broken", project="misc", config=cfg, tags=["keep-me"]) as log:
        log.step(0, loss=5.0)
        raise RuntimeError("diverged")
except RuntimeError:
    pass

# Same name in two projects.
for project in ("alpha", "beta"):
    with Logger("baseline", project=project, config=cfg) as log:
        log.step(0, loss=0.5 if project == "alpha" else 0.7)

# A run that is still going: status stays "running", heartbeat is fresh.
live = Logger("live", project="misc", config=cfg, max_steps=1000)
for step in range(5):
    live.step(step, loss=2.0 - step * 0.1)
live.flush()

# A run whose process died: status "running" but the heartbeat stopped long ago.
dead = Logger("dead", project="misc", config=cfg)
dead.step(0, loss=9.0)
dead.flush()
hb = dead.experiment_dir / "heartbeat.json"
data = json.loads(hb.read_text()); data["timestamp"] -= 86400
hb.write_text(json.dumps(data))

import os
os._exit(0)  # skip atexit: leave "live" and "dead" unfinished on disk
"#;

fn world() -> Option<&'static World> {
    static WORLD: OnceLock<Option<World>> = OnceLock::new();
    WORLD
        .get_or_init(|| {
            if !nuviz_available() {
                eprintln!("Skipping: nuviz Python package not installed");
                return None;
            }
            let dir = tempfile::tempdir().unwrap();
            let base = dir.path().join("experiments");
            let cache = dir.path().join("cache");
            let out = Command::new(python())
                .args(["-c", GENERATE, base.to_str().unwrap()])
                .output()
                .expect("failed to run python");
            assert!(
                out.status.success(),
                "generator failed:\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
            Some(World {
                _dir: dir,
                base,
                cache,
            })
        })
        .as_ref()
}

fn nuviz_in(base: &Path, cache: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_nuviz"))
        .arg("--dir")
        .arg(base)
        .args(args)
        .env("NUVIZ_CACHE_DIR", cache)
        .env_remove("NUVIZ_DIR")
        .output()
        .expect("failed to run nuviz")
}

fn nuviz(w: &World, args: &[&str]) -> (String, String, bool) {
    let out = nuviz_in(&w.base, &w.cache, args);
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.success(),
    )
}

/// The row of a comfy-table / markdown table that mentions `needle`.
fn row<'a>(table: &'a str, needle: &str) -> &'a str {
    table
        .lines()
        .find(|l| l.contains(needle))
        .unwrap_or_else(|| panic!("no row containing {needle:?} in:\n{table}"))
}

#[test]
fn ls_reports_live_dead_failed_and_done_runs() {
    let Some(w) = world() else { return };
    let (out, err, ok) = nuviz(w, &["ls", "--project", "misc"]);
    assert!(ok, "{err}");
    assert!(row(&out, " live ").contains("running"), "{out}");
    assert!(row(&out, " live ").contains("4/1000"), "{out}");
    assert!(row(&out, " dead ").contains("crashed"), "{out}");
    assert!(row(&out, " broken ").contains("failed"), "{out}");
    assert!(row(&out, " broken ").contains("keep-me"), "{out}");
    assert!(row(&out, " nan_run ").contains("done"), "{out}");
    // Best loss for the live run comes from scanning its JSONL (no final meta yet).
    assert!(row(&out, " live ").contains("1.6000"), "{out}");

    let (out, _, _) = nuviz(w, &["ls", "--status", "running"]);
    assert!(out.contains(" live ") && !out.contains(" dead "), "{out}");
}

#[test]
fn nan_step_is_kept_in_export() {
    let Some(w) = world() else { return };
    let (out, err, ok) = nuviz(w, &["export", "nan_run"]);
    assert!(ok, "{err}");
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines[0], "step,timestamp,loss,regret");
    assert_eq!(lines.len(), 11, "all 10 steps exported:\n{out}");
    let step4: Vec<&str> = lines[5].split(',').collect();
    assert_eq!((step4[0], step4[2], step4[3]), ("4", "", "6"));

    let (json, _, ok) = nuviz(w, &["export", "misc/nan_run", "--format", "json"]);
    assert!(ok);
    let rows: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(rows.as_array().unwrap().len(), 10);
}

#[test]
fn explicit_goal_drives_leaderboard_direction() {
    let Some(w) = world() else { return };
    let (out, err, ok) = nuviz(
        w,
        &[
            "leaderboard",
            "--project",
            "misc",
            "--format",
            "markdown",
            "--metric",
            "regret",
            "--steps",
        ],
    );
    assert!(ok, "{err}");
    assert!(out.contains("regret ↓"), "{out}");
    assert!(row(&out, "nan_run").contains("**1.0000 @9**"), "{out}");
}

#[test]
fn aggregate_leaderboard_groups_seeds() {
    let Some(w) = world() else { return };
    let (out, err, ok) = nuviz(
        w,
        &[
            "leaderboard",
            "--project",
            "grid",
            "--aggregate",
            "--sort",
            "psnr",
            "--format",
            "csv",
            "--metric",
            "psnr",
            "--precision",
            "2",
        ],
    );
    assert!(ok, "{err}");
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines[0], "rank,experiment,psnr,psnr_std,status,runs");
    assert_eq!(lines.len(), 5, "4 configurations:\n{out}");
    assert!(lines[1].starts_with("1,lr0.001_w128 (n=3),"), "{out}");
    assert!(
        lines.iter().skip(1).all(|l| l.ends_with(",done,3")),
        "{out}"
    );

    // Final psnr for lr=0.001,w=128: 20 + 19*(0.5+0.128+seed*0.01), seeds 0..2
    let mean: f64 = lines[1].split(',').nth(2).unwrap().parse().unwrap();
    assert!((mean - (20.0 + 19.0 * 0.638)).abs() < 1e-4, "{mean}");
    let std: f64 = lines[1].split(',').nth(3).unwrap().parse().unwrap();
    assert!((std - 0.19).abs() < 1e-4, "{std}");
}

#[test]
fn matrix_uses_logged_hyperparameters() {
    let Some(w) = world() else { return };
    let (out, err, ok) = nuviz(
        w,
        &[
            "matrix",
            "--project",
            "grid",
            "--rows",
            "lr",
            "--cols",
            "model.width",
            "--metric",
            "val_loss",
            "--format",
            "csv",
        ],
    );
    assert!(ok, "{err}");
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines[0], "lr,64,128");
    assert_eq!(lines.len(), 3, "{out}");
    assert!(
        lines[1].starts_with("0.001,") && lines[2].starts_with("0.01,"),
        "{out}"
    );
    // Best val_loss is 1/20/quality; the cell is the mean over the three seeds.
    assert_eq!(lines[1], "0.001,0.087126,0.078383");

    // The human-readable table shows the spread across seeds.
    let (table, _, ok) = nuviz(
        w,
        &[
            "matrix",
            "--project",
            "grid",
            "--rows",
            "lr",
            "--cols",
            "model.width",
            "--metric",
            "val_loss",
        ],
    );
    assert!(ok);
    assert!(row(&table, " 0.001 ").contains("0.0871±0.0015"), "{table}");
}

#[test]
fn ambiguous_name_is_an_error_and_can_be_qualified() {
    let Some(w) = world() else { return };
    let (_, err, ok) = nuviz(w, &["export", "baseline"]);
    assert!(!ok);
    assert!(err.contains("ambiguous"), "{err}");
    assert!(
        err.contains("alpha/baseline") && err.contains("beta/baseline"),
        "{err}"
    );

    let (out, _, ok) = nuviz(w, &["export", "beta/baseline"]);
    assert!(ok);
    assert!(out.lines().nth(1).unwrap().ends_with(",0.7"), "{out}");
    let (out, _, ok) = nuviz(w, &["export", "baseline", "--project", "alpha"]);
    assert!(ok);
    assert!(out.lines().nth(1).unwrap().ends_with(",0.5"), "{out}");

    let (_, err, ok) = nuviz(w, &["export", "basline"]);
    assert!(!ok);
    assert!(err.contains("Did you mean"), "{err}");
}

#[test]
fn reproduce_shows_hyperparameters() {
    let Some(w) = world() else { return };
    let (out, err, ok) = nuviz(w, &["reproduce", "lr0.01_w64_seed2"]);
    assert!(ok, "{err}");
    assert!(out.contains("\"lr\": 0.01"), "{out}");
    assert!(out.contains("# Seed: 2"), "{out}");
}

#[test]
fn cleanup_protects_running_and_tagged_runs() {
    let Some(w) = world() else { return };
    // Work on a private copy: this test deletes things.
    let scratch = tempfile::tempdir().unwrap();
    let base = scratch.path().join("experiments");
    copy_dir(&w.base.join("misc"), &base.join("misc"));
    let cache = scratch.path().join("cache");
    let run = |args: &[&str]| {
        let out = nuviz_in(&base, &cache, args);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    };

    // Dry run: nothing is deleted.
    let out = run(&["cleanup", "--keep-top", "0"]);
    assert!(out.contains("1 running, 1 tagged"), "{out}");
    assert!(
        out.contains("misc/nan_run") && out.contains("misc/dead"),
        "{out}"
    );
    assert!(
        !out.contains("misc/live") && !out.contains("misc/broken"),
        "{out}"
    );
    assert!(out.contains("Dry run"), "{out}");
    assert!(base.join("misc/nan_run").exists());

    let out = run(&["cleanup", "--keep-top", "0", "--force"]);
    assert!(out.contains("Deleted: misc/nan_run"), "{out}");
    assert!(!base.join("misc/nan_run").exists());
    assert!(!base.join("misc/dead").exists());
    assert!(
        base.join("misc/live").exists(),
        "running experiment must survive"
    );
    assert!(
        base.join("misc/broken").exists(),
        "tagged experiment must survive"
    );
}

#[test]
fn tag_roundtrip_preserves_python_fields() {
    let Some(w) = world() else { return };
    let scratch = tempfile::tempdir().unwrap();
    let base = scratch.path().join("experiments");
    copy_dir(&w.base.join("alpha"), &base.join("alpha"));
    let cache = scratch.path().join("cache");

    let out = nuviz_in(&base, &cache, &["tag", "baseline", "paper"]);
    assert!(out.status.success());
    let meta: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(base.join("alpha/baseline/meta.json")).unwrap())
            .unwrap();
    assert_eq!(meta["tags"], serde_json::json!(["paper"]));
    assert_eq!(meta["status"], "done");
    assert_eq!(meta["best_metrics"]["loss"], 0.5);
    assert!(meta["run_id"].is_string());
}

fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap().flatten() {
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}
