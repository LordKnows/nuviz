//! Error handling and edge case tests for the Rust CLI data layer.

use std::fs;
use std::path::Path;

#[test]
fn test_empty_jsonl_file() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("metrics.jsonl"), "").unwrap();

    let records = nuviz_cli::data::metrics::read_metrics(&dir.path().join("metrics.jsonl"));
    assert!(records.is_empty());
}

#[test]
fn test_jsonl_with_only_whitespace() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("metrics.jsonl"), "\n\n  \n\n").unwrap();

    let records = nuviz_cli::data::metrics::read_metrics(&dir.path().join("metrics.jsonl"));
    assert!(records.is_empty());
}

#[test]
fn test_jsonl_with_extra_fields_ignored() {
    let dir = tempfile::tempdir().unwrap();
    let line = r#"{"step":0,"timestamp":1.0,"metrics":{"loss":0.5},"extra_field":"hello","nested":{"a":1}}"#;
    fs::write(dir.path().join("metrics.jsonl"), format!("{line}\n")).unwrap();

    let records = nuviz_cli::data::metrics::read_metrics(&dir.path().join("metrics.jsonl"));
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].step, 0);
}

#[test]
fn test_jsonl_with_very_large_step_numbers() {
    let dir = tempfile::tempdir().unwrap();
    let line = r#"{"step":9999999999,"timestamp":1.0,"metrics":{"loss":0.001}}"#;
    fs::write(dir.path().join("metrics.jsonl"), format!("{line}\n")).unwrap();

    let records = nuviz_cli::data::metrics::read_metrics(&dir.path().join("metrics.jsonl"));
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].step, 9_999_999_999);
}

#[test]
fn test_jsonl_with_nan_value() {
    let dir = tempfile::tempdir().unwrap();
    // Python json.dumps writes NaN as "NaN" which is not valid JSON
    // Our reader should skip these lines gracefully
    let line = r#"{"step":0,"timestamp":1.0,"metrics":{"loss":NaN}}"#;
    fs::write(dir.path().join("metrics.jsonl"), format!("{line}\n")).unwrap();

    let records = nuviz_cli::data::metrics::read_metrics(&dir.path().join("metrics.jsonl"));
    // NaN is not valid JSON, so this line should be skipped
    assert_eq!(records.len(), 0);
}

#[test]
fn test_jsonl_with_infinity() {
    let dir = tempfile::tempdir().unwrap();
    let line = r#"{"step":0,"timestamp":1.0,"metrics":{"loss":Infinity}}"#;
    fs::write(dir.path().join("metrics.jsonl"), format!("{line}\n")).unwrap();

    let records = nuviz_cli::data::metrics::read_metrics(&dir.path().join("metrics.jsonl"));
    // Infinity is not valid JSON, so this line should be skipped
    assert_eq!(records.len(), 0);
}

#[test]
fn test_meta_json_with_unknown_fields() {
    let dir = tempfile::tempdir().unwrap();
    let meta = r#"{"name":"test","unknown_field":"value","nested":{"x":1}}"#;
    fs::write(dir.path().join("meta.json"), meta).unwrap();

    let result = nuviz_cli::data::meta::read_meta(dir.path());
    assert!(result.is_some());
    assert_eq!(result.unwrap().name.as_deref(), Some("test"));
}

#[test]
fn test_meta_json_completely_empty() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("meta.json"), "{}").unwrap();

    let result = nuviz_cli::data::meta::read_meta(dir.path());
    assert!(result.is_some());
    let meta = result.unwrap();
    assert!(meta.name.is_none());
    assert!(meta.status.is_none());
}

#[test]
fn test_meta_json_invalid_json() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("meta.json"), "not json at all").unwrap();

    let result = nuviz_cli::data::meta::read_meta(dir.path());
    assert!(result.is_none());
}

#[test]
fn test_large_jsonl_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("metrics.jsonl");

    let mut content = String::new();
    for i in 0..10_000 {
        content.push_str(&format!(
            r#"{{"step":{},"timestamp":{},"metrics":{{"loss":{}}}}}"#,
            i,
            i as f64,
            1.0 / (i as f64 + 1.0)
        ));
        content.push('\n');
    }
    fs::write(&path, &content).unwrap();

    let records = nuviz_cli::data::metrics::read_metrics(&path);
    assert_eq!(records.len(), 10_000);
    assert_eq!(records[0].step, 0);
    assert_eq!(records[9_999].step, 9_999);
}

#[test]
fn test_experiment_with_metrics_but_no_meta() {
    let dir = tempfile::tempdir().unwrap();
    let exp_dir = dir.path().join("orphan-exp");
    fs::create_dir_all(&exp_dir).unwrap();
    fs::write(
        exp_dir.join("metrics.jsonl"),
        r#"{"step":0,"timestamp":1.0,"metrics":{"loss":0.5}}"#,
    )
    .unwrap();

    let experiments = nuviz_cli::data::experiment::discover_experiments(dir.path());
    assert_eq!(experiments.len(), 1);
    assert_eq!(experiments[0].status, "running"); // No meta = assumed running
}

#[test]
fn test_experiment_with_meta_but_no_metrics() {
    let dir = tempfile::tempdir().unwrap();
    let exp_dir = dir.path().join("meta-only");
    fs::create_dir_all(&exp_dir).unwrap();
    fs::write(
        exp_dir.join("meta.json"),
        r#"{"name":"meta-only","status":"crashed"}"#,
    )
    .unwrap();

    let experiments = nuviz_cli::data::experiment::discover_experiments(dir.path());
    assert_eq!(experiments.len(), 1);
    assert_eq!(experiments[0].status, "crashed");
}

#[test]
fn test_best_metrics_empty_records() {
    let best = nuviz_cli::data::metrics::best_metrics(&[]);
    assert!(best.is_empty());
}
