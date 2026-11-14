//! `meta.json`: experiment metadata shared with the Python logger.
//!
//! Parsing is deliberately lenient and field-by-field: one field with an
//! unexpected type must not hide the rest of an experiment's metadata.

use anyhow::Result;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use crate::data::goal::Goal;
use crate::data::summary::MetricSummary;

/// Experiment metadata from meta.json.
#[derive(Debug, Clone, Default)]
pub struct ExperimentMeta {
    pub schema_version: Option<u64>,
    pub run_id: Option<String>,
    pub name: Option<String>,
    pub project: Option<String>,
    pub status: Option<String>,
    pub error: Option<String>,
    pub start_time: Option<String>,
    pub end_time: Option<String>,
    /// Last step logged (written when the run finishes).
    pub total_steps: Option<u64>,
    /// Planned number of steps, if the user declared it.
    pub max_steps: Option<u64>,
    pub best_metrics: HashMap<String, f64>,
    pub best_steps: HashMap<String, u64>,
    pub final_metrics: HashMap<String, f64>,
    pub metric_summary: Option<BTreeMap<String, MetricSummary>>,
    pub metric_goals: HashMap<String, Goal>,
    pub num_alerts: Option<u64>,

    pub seed: Option<i64>,
    pub config_hash: Option<String>,
    pub config: Option<Value>,
    pub tags: Vec<String>,

    pub git_hash: Option<String>,
    pub git_dirty: Option<bool>,
    pub hostname: Option<String>,
    pub gpu_model: Option<String>,
    pub python_version: Option<String>,
    pub pytorch_version: Option<String>,
}

impl ExperimentMeta {
    pub fn from_value(doc: &Value) -> Self {
        let s = |key: &str| doc.get(key).and_then(Value::as_str).map(String::from);
        let u = |key: &str| doc.get(key).and_then(as_u64_lenient);
        let floats = |key: &str| -> HashMap<String, f64> {
            doc.get(key)
                .and_then(Value::as_object)
                .map(|m| {
                    m.iter()
                        .filter_map(|(k, v)| Some((k.clone(), v.as_f64()?)))
                        .collect()
                })
                .unwrap_or_default()
        };

        ExperimentMeta {
            schema_version: u("schema_version"),
            run_id: s("run_id"),
            name: s("name"),
            project: s("project"),
            status: s("status"),
            error: s("error"),
            start_time: s("start_time"),
            end_time: s("end_time"),
            total_steps: u("total_steps"),
            max_steps: u("max_steps"),
            best_metrics: floats("best_metrics"),
            best_steps: doc
                .get("best_steps")
                .and_then(Value::as_object)
                .map(|m| {
                    m.iter()
                        .filter_map(|(k, v)| Some((k.clone(), as_u64_lenient(v)?)))
                        .collect()
                })
                .unwrap_or_default(),
            final_metrics: floats("final_metrics"),
            metric_summary: doc
                .get("metric_summary")
                .and_then(Value::as_object)
                .map(|m| {
                    m.iter()
                        .filter_map(|(k, v)| {
                            Some((k.clone(), serde_json::from_value(v.clone()).ok()?))
                        })
                        .collect()
                }),
            metric_goals: doc
                .get("metric_goals")
                .and_then(Value::as_object)
                .map(|m| {
                    m.iter()
                        .filter_map(|(k, v)| {
                            Some((k.clone(), serde_json::from_value(v.clone()).ok()?))
                        })
                        .collect()
                })
                .unwrap_or_default(),
            num_alerts: u("num_alerts"),
            seed: doc.get("seed").and_then(Value::as_i64),
            config_hash: s("config_hash"),
            config: doc.get("config").filter(|v| !v.is_null()).cloned(),
            tags: doc
                .get("tags")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
            git_hash: s("git_hash"),
            git_dirty: doc.get("git_dirty").and_then(Value::as_bool),
            hostname: s("hostname"),
            gpu_model: s("gpu_model"),
            python_version: s("python_version"),
            pytorch_version: s("pytorch_version"),
        }
    }
}

fn as_u64_lenient(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| {
        v.as_f64()
            .filter(|f| *f >= 0.0 && f.fract() == 0.0)
            .map(|f| f as u64)
    })
}

/// Read meta.json from an experiment directory. `None` if it is missing or
/// not a JSON object.
pub fn read_meta(dir: &Path) -> Option<ExperimentMeta> {
    let doc = read_meta_value(dir)?;
    doc.is_object().then(|| ExperimentMeta::from_value(&doc))
}

/// Read meta.json as a raw JSON document.
pub fn read_meta_value(dir: &Path) -> Option<Value> {
    let content = std::fs::read_to_string(dir.join("meta.json")).ok()?;
    serde_json::from_str(&content).ok()
}

/// Read only the tags from an experiment's meta.json.
pub fn read_tags(dir: &Path) -> Vec<String> {
    read_meta(dir).map(|m| m.tags).unwrap_or_default()
}

/// Update only the `tags` array in an experiment's meta.json.
/// Reads the existing JSON, patches the `tags` field, writes atomically.
pub fn update_tags(dir: &Path, tags: &[String]) -> Result<()> {
    let meta_path = dir.join("meta.json");
    let mut doc: Value = if meta_path.exists() {
        let content = std::fs::read_to_string(&meta_path)?;
        serde_json::from_str(&content)?
    } else {
        serde_json::json!({})
    };
    if !doc.is_object() {
        anyhow::bail!("{} is not a JSON object", meta_path.display());
    }

    doc["tags"] = serde_json::json!(tags);

    let tmp_path = dir.join(format!("meta.json.{}.tmp", std::process::id()));
    std::fs::write(&tmp_path, serde_json::to_string_pretty(&doc)?)?;
    std::fs::rename(&tmp_path, &meta_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_read_valid_meta() {
        let dir = tempfile::tempdir().unwrap();
        let meta = r#"{
            "name": "exp-001",
            "project": "test",
            "status": "done",
            "total_steps": 1000,
            "best_metrics": {"loss": 0.05, "psnr": 28.4},
            "git_hash": "abc123"
        }"#;
        fs::write(dir.path().join("meta.json"), meta).unwrap();

        let result = read_meta(dir.path()).unwrap();
        assert_eq!(result.name.as_deref(), Some("exp-001"));
        assert_eq!(result.status.as_deref(), Some("done"));
        assert_eq!(result.total_steps, Some(1000));
        assert!((result.best_metrics["loss"] - 0.05).abs() < f64::EPSILON);
        assert_eq!(result.git_hash.as_deref(), Some("abc123"));
    }

    #[test]
    fn test_read_missing_meta() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_meta(dir.path()).is_none());
    }

    #[test]
    fn test_read_partial_meta() {
        let dir = tempfile::tempdir().unwrap();
        let meta = r#"{"name": "partial"}"#;
        fs::write(dir.path().join("meta.json"), meta).unwrap();

        let result = read_meta(dir.path()).unwrap();
        assert_eq!(result.name.as_deref(), Some("partial"));
        assert!(result.project.is_none());
        assert!(result.best_metrics.is_empty());
    }

    #[test]
    fn test_one_bad_field_does_not_hide_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        let meta = r#"{
            "name": "exp",
            "status": "done",
            "seed": "not-a-number",
            "total_steps": 12.0,
            "best_metrics": {"loss": 0.1, "broken": null},
            "tags": ["a", 7, "b"],
            "metric_goals": {"regret": "min", "weird": "sideways"}
        }"#;
        fs::write(dir.path().join("meta.json"), meta).unwrap();

        let m = read_meta(dir.path()).unwrap();
        assert_eq!(m.name.as_deref(), Some("exp"));
        assert_eq!(m.seed, None);
        assert_eq!(m.total_steps, Some(12));
        assert_eq!(m.best_metrics.len(), 1);
        assert_eq!(m.tags, vec!["a", "b"]);
        assert_eq!(m.metric_goals.len(), 1);
        assert_eq!(m.metric_goals["regret"], Goal::Min);
    }

    #[test]
    fn test_new_contract_fields() {
        let dir = tempfile::tempdir().unwrap();
        let meta = r#"{
            "schema_version": 1, "run_id": "abcdef012345", "max_steps": 30000,
            "seed": -3, "config": {"lr": 0.001},
            "best_steps": {"loss": 7}, "final_metrics": {"loss": 0.2},
            "metric_summary": {"loss": {"count":3,"min":0.1,"min_step":7,"max":1.0,"max_step":0,"last":0.2,"last_step":9}}
        }"#;
        fs::write(dir.path().join("meta.json"), meta).unwrap();

        let m = read_meta(dir.path()).unwrap();
        assert_eq!(m.schema_version, Some(1));
        assert_eq!(m.run_id.as_deref(), Some("abcdef012345"));
        assert_eq!(m.max_steps, Some(30000));
        assert_eq!(m.seed, Some(-3));
        assert_eq!(m.best_steps["loss"], 7);
        assert_eq!(m.final_metrics["loss"], 0.2);
        assert_eq!(m.metric_summary.unwrap()["loss"].last_step, 9);
        assert_eq!(m.config.unwrap()["lr"], serde_json::json!(0.001));
    }

    #[test]
    fn test_update_tags_creates_meta() {
        let dir = tempfile::tempdir().unwrap();
        let tags = vec!["best".to_string(), "baseline".to_string()];
        update_tags(dir.path(), &tags).unwrap();

        let read_back = read_tags(dir.path());
        assert_eq!(read_back, tags);
    }

    #[test]
    fn test_update_tags_preserves_existing_fields() {
        let dir = tempfile::tempdir().unwrap();
        let meta = r#"{"name": "exp-001", "custom_field": 42}"#;
        fs::write(dir.path().join("meta.json"), meta).unwrap();

        update_tags(dir.path(), &["v1".to_string()]).unwrap();

        // Verify tags were written
        let read_back = read_tags(dir.path());
        assert_eq!(read_back, vec!["v1".to_string()]);

        // Verify custom_field is preserved (read as raw Value)
        let content = fs::read_to_string(dir.path().join("meta.json")).unwrap();
        let doc: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(doc["custom_field"], serde_json::json!(42));
        assert_eq!(doc["name"], serde_json::json!("exp-001"));
    }

    #[test]
    fn test_read_tags_missing_meta() {
        let dir = tempfile::tempdir().unwrap();
        let tags = read_tags(dir.path());
        assert!(tags.is_empty());
    }
}
