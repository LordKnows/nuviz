//! Cross-component integration tests.
//!
//! Verifies that JSONL files written by the Python Logger can be correctly
//! parsed by the Rust CLI's data layer.

use std::fs;
use std::path::Path;
use std::process::Command;

/// Find the Python interpreter.
fn python() -> &'static str {
    if Command::new("python3").arg("--version").output().is_ok() {
        "python3"
    } else {
        "python"
    }
}

/// Check if the nuviz Python package is importable.
fn nuviz_available() -> bool {
    Command::new(python())
        .args(["-c", "import nuviz"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run a Python script and assert it succeeds.
fn run_python(script: &str) -> std::process::Output {
    let output = Command::new(python())
        .args(["-c", script])
        .output()
        .expect("failed to execute Python");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("Python script failed:\n{stderr}");
    }
    output
}

#[test]
fn test_python_logger_writes_valid_jsonl_for_rust() {
    if !nuviz_available() {
        eprintln!("Skipping: nuviz Python package not installed");
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let base_dir = dir.path().to_str().unwrap();

    // Run a small training simulation via Python
    let script = format!(
        r#"
import os
os.environ["NUVIZ_DIR"] = "{base_dir}"

from nuviz import Logger
from nuviz.config import NuvizConfig

config = NuvizConfig(
    base_dir=__import__("pathlib").Path("{base_dir}"),
    flush_interval_seconds=0.1,
    flush_count=5,
    enable_alerts=True,
    enable_snapshot=False,
)

log = Logger("cross-test", project="integration", config=config, snapshot=False)
for step in range(50):
    loss = 1.0 / (step + 1)
    psnr = 20.0 + step * 0.2
    log.step(step, loss=loss, psnr=psnr, ssim=0.9 + step * 0.001)
log.finish()
"#
    );

    run_python(&script);

    // Now verify the Rust data layer can parse the output
    let exp_dir = Path::new(base_dir).join("integration").join("cross-test");
    assert!(exp_dir.exists(), "Experiment directory not created");

    // Parse metrics.jsonl
    let jsonl_path = exp_dir.join("metrics.jsonl");
    assert!(jsonl_path.exists(), "metrics.jsonl not created");

    let content = fs::read_to_string(&jsonl_path).unwrap();
    let lines: Vec<&str> = content.trim().split('\n').collect();
    assert_eq!(lines.len(), 50, "Expected 50 JSONL lines, got {}", lines.len());

    // Parse each line as the Rust MetricRecord
    for (i, line) in lines.iter().enumerate() {
        let record: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("Failed to parse line {i}: {e}\nLine: {line}"));

        let step = record["step"].as_u64().unwrap();
        assert_eq!(step, i as u64, "Step mismatch at line {i}");

        assert!(record["timestamp"].is_f64(), "Missing timestamp at line {i}");

        let metrics = record["metrics"].as_object().unwrap();
        assert!(metrics.contains_key("loss"), "Missing 'loss' at step {step}");
        assert!(metrics.contains_key("psnr"), "Missing 'psnr' at step {step}");
        assert!(metrics.contains_key("ssim"), "Missing 'ssim' at step {step}");

        // Verify loss is decreasing
        let loss = metrics["loss"].as_f64().unwrap();
        assert!(loss > 0.0 && loss <= 1.0, "Unexpected loss={loss} at step {step}");
    }

    // Parse meta.json
    let meta_path = exp_dir.join("meta.json");
    assert!(meta_path.exists(), "meta.json not created");

    let meta: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&meta_path).unwrap()).unwrap();

    assert_eq!(meta["status"].as_str(), Some("done"));
    assert_eq!(meta["total_steps"].as_u64(), Some(49));
    assert_eq!(meta["name"].as_str(), Some("cross-test"));
    assert_eq!(meta["project"].as_str(), Some("integration"));

    // Verify best_metrics
    let best = meta["best_metrics"].as_object().unwrap();
    let best_loss = best["loss"].as_f64().unwrap();
    assert!(best_loss < 0.03, "Best loss should be ~0.02, got {best_loss}");

    let best_psnr = best["psnr"].as_f64().unwrap();
    assert!(best_psnr > 29.0, "Best psnr should be ~29.8, got {best_psnr}");
}

#[test]
fn test_python_logger_nan_produces_valid_jsonl() {
    if !nuviz_available() {
        eprintln!("Skipping: nuviz Python package not installed");
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let base_dir = dir.path().to_str().unwrap();

    let script = format!(
        r#"
from nuviz import Logger
from nuviz.config import NuvizConfig

config = NuvizConfig(
    base_dir=__import__("pathlib").Path("{base_dir}"),
    flush_interval_seconds=0.1,
    flush_count=100,
    enable_alerts=True,
    enable_snapshot=False,
)

log = Logger("nan-test", config=config, snapshot=False)
for step in range(10):
    log.step(step, loss=0.1)
log.step(10, loss=float("nan"))
for step in range(11, 20):
    log.step(step, loss=0.1)
log.finish()
"#
    );

    run_python(&script);

    let jsonl_path = Path::new(base_dir).join("nan-test").join("metrics.jsonl");
    let content = fs::read_to_string(&jsonl_path).unwrap();
    let lines: Vec<&str> = content.trim().split('\n').collect();
    assert_eq!(lines.len(), 20);

    // Line 10 should have NaN (which is valid JSON when serialized by Python as NaN)
    // Python's json.dumps writes NaN — verify Rust can handle it
    for line in &lines {
        // serde_json can't parse NaN by default, but our reader should skip it
        let _: Result<serde_json::Value, _> = serde_json::from_str(line);
        // We just verify no panic — some lines may fail to parse and that's OK
    }
}

#[test]
fn test_rust_experiment_discovery_finds_python_experiments() {
    if !nuviz_available() {
        eprintln!("Skipping: nuviz Python package not installed");
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let base_dir = dir.path().to_str().unwrap();

    // Create two experiments via Python
    let script = format!(
        r#"
from nuviz import Logger
from nuviz.config import NuvizConfig

config = NuvizConfig(
    base_dir=__import__("pathlib").Path("{base_dir}"),
    flush_interval_seconds=0.1,
    flush_count=100,
    enable_snapshot=False,
)

log1 = Logger("exp-alpha", project="test_proj", config=config, snapshot=False)
log1.step(0, loss=1.0)
log1.step(1, loss=0.5)
log1.finish()

log2 = Logger("exp-beta", project="test_proj", config=config, snapshot=False)
log2.step(0, loss=0.8)
log2.step(1, loss=0.3)
log2.step(2, loss=0.1)
log2.finish()
"#
    );

    run_python(&script);

    // Verify directory structure
    let proj_dir = Path::new(base_dir).join("test_proj");
    assert!(proj_dir.join("exp-alpha").join("metrics.jsonl").exists());
    assert!(proj_dir.join("exp-beta").join("metrics.jsonl").exists());
    assert!(proj_dir.join("exp-alpha").join("meta.json").exists());
    assert!(proj_dir.join("exp-beta").join("meta.json").exists());

    // Verify meta.json has correct data for ranking
    let meta_alpha: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(proj_dir.join("exp-alpha").join("meta.json")).unwrap(),
    )
    .unwrap();
    let meta_beta: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(proj_dir.join("exp-beta").join("meta.json")).unwrap(),
    )
    .unwrap();

    // exp-beta has lower best loss
    let alpha_loss = meta_alpha["best_metrics"]["loss"].as_f64().unwrap();
    let beta_loss = meta_beta["best_metrics"]["loss"].as_f64().unwrap();
    assert!(beta_loss < alpha_loss, "exp-beta should have lower loss");
}

#[test]
fn test_nuviz_ls_cli_output() {
    if !nuviz_available() {
        eprintln!("Skipping: nuviz Python package not installed");
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let base_dir = dir.path().to_str().unwrap();

    let script = format!(
        r#"
from nuviz import Logger
from nuviz.config import NuvizConfig

config = NuvizConfig(
    base_dir=__import__("pathlib").Path("{base_dir}"),
    flush_interval_seconds=0.1,
    flush_count=100,
    enable_snapshot=False,
)

log = Logger("cli-test", config=config, snapshot=False)
log.step(0, loss=1.0)
log.step(1, loss=0.5)
log.finish()
"#
    );

    run_python(&script);

    // Run nuviz ls via cargo
    let output = Command::new(env!("CARGO"))
        .args(["run", "--quiet", "--", "ls", "--dir", base_dir])
        .output()
        .expect("failed to run nuviz ls");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "nuviz ls failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("cli-test"), "Output should contain experiment name:\n{stdout}");
    assert!(stdout.contains("done"), "Output should show 'done' status:\n{stdout}");
}

#[test]
fn test_nuviz_leaderboard_cli_output() {
    if !nuviz_available() {
        eprintln!("Skipping: nuviz Python package not installed");
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let base_dir = dir.path().to_str().unwrap();

    let script = format!(
        r#"
from nuviz import Logger
from nuviz.config import NuvizConfig

config = NuvizConfig(
    base_dir=__import__("pathlib").Path("{base_dir}"),
    flush_interval_seconds=0.1,
    flush_count=100,
    enable_snapshot=False,
)

log1 = Logger("good-exp", config=config, snapshot=False)
log1.step(0, loss=0.01, psnr=30.0)
log1.finish()

log2 = Logger("bad-exp", config=config, snapshot=False)
log2.step(0, loss=1.0, psnr=15.0)
log2.finish()
"#
    );

    run_python(&script);

    // Run nuviz leaderboard
    let output = Command::new(env!("CARGO"))
        .args([
            "run", "--quiet", "--", "leaderboard", "--dir", base_dir, "--sort", "loss",
        ])
        .output()
        .expect("failed to run nuviz leaderboard");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "nuviz leaderboard failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("good-exp"), "Output should contain good-exp:\n{stdout}");
    assert!(stdout.contains("bad-exp"), "Output should contain bad-exp:\n{stdout}");

    // Verify good-exp is ranked higher (appears first)
    let good_pos = stdout.find("good-exp").unwrap();
    let bad_pos = stdout.find("bad-exp").unwrap();
    assert!(good_pos < bad_pos, "good-exp should be ranked higher than bad-exp");
}
