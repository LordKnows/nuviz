use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Experiment metadata from meta.json.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExperimentMeta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub start_time: Option<String>,
    #[serde(default)]
    pub end_time: Option<String>,
    #[serde(default)]
    pub total_steps: Option<u64>,
    #[serde(default)]
    pub best_metrics: HashMap<String, f64>,

    // Phase 2: multi-seed and ablation fields
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub config_hash: Option<String>,
    #[serde(default)]
    pub config: Option<serde_json::Value>,

    // Phase 4: tags for experiment organization
    #[serde(default)]
    pub tags: Vec<String>,

    // Snapshot fields (flattened from Python's snapshot) — used in Phase 3+ commands
    #[serde(default)]
    pub git_hash: Option<String>,
    #[serde(default)]
    pub git_dirty: Option<bool>,
    #[serde(default)]
    pub hostname: Option<String>,
    #[serde(default)]
    pub gpu_model: Option<String>,
    #[serde(default)]
    pub python_version: Option<String>,
    #[serde(default)]
    pub pytorch_version: Option<String>,
}

/// Read meta.json from an experiment directory.
pub fn read_meta(dir: &Path) -> Option<ExperimentMeta> {
    let meta_path = dir.join("meta.json");
    let content = std::fs::read_to_string(&meta_path).ok()?;
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
    let mut doc: serde_json::Value = if meta_path.exists() {
        let content = std::fs::read_to_string(&meta_path)?;
        serde_json::from_str(&content)?
    } else {
        serde_json::json!({})
    };

    doc["tags"] = serde_json::json!(tags);

    let tmp_path = dir.join("meta.json.tmp");
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
