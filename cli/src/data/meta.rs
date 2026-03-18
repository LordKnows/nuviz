use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Experiment metadata from meta.json.
#[derive(Debug, Clone, Deserialize, Default)]
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

    // Snapshot fields (flattened from Python's snapshot)
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
}
