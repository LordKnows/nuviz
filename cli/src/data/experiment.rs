use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::data::meta;
use crate::data::metrics;

/// A discovered experiment with its metadata and location.
#[derive(Debug, Clone)]
pub struct Experiment {
    pub name: String,
    pub project: Option<String>,
    pub dir: PathBuf,
    pub status: String,
    pub total_steps: Option<u64>,
    pub best_metrics: HashMap<String, f64>,
    pub start_time: Option<String>,
    pub end_time: Option<String>,
}

/// Resolve the base directory for experiments.
pub fn resolve_base_dir(dir_override: Option<&str>) -> PathBuf {
    if let Some(d) = dir_override {
        return PathBuf::from(d);
    }

    if let Ok(env_dir) = std::env::var("NUVIZ_DIR") {
        return PathBuf::from(env_dir);
    }

    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".nuviz")
        .join("experiments")
}

/// Discover all experiments under the base directory.
///
/// An experiment directory is any directory that contains `metrics.jsonl`
/// or `meta.json`.
pub fn discover_experiments(base_dir: &Path) -> Vec<Experiment> {
    let mut experiments = Vec::new();

    if !base_dir.exists() {
        return experiments;
    }

    // Walk up to 2 levels deep: base_dir/experiment or base_dir/project/experiment
    discover_in_dir(base_dir, None, &mut experiments);

    // Sort by start_time descending (most recent first)
    experiments.sort_by(|a, b| b.start_time.cmp(&a.start_time));

    experiments
}

fn discover_in_dir(
    dir: &Path,
    project: Option<&str>,
    experiments: &mut Vec<Experiment>,
) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let has_metrics = path.join("metrics.jsonl").exists();
        let has_meta = path.join("meta.json").exists();

        if has_metrics || has_meta {
            // This is an experiment directory
            let exp = load_experiment(&path, project);
            experiments.push(exp);
        } else if project.is_none() {
            // Could be a project directory — recurse one level
            let project_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");
            discover_in_dir(&path, Some(project_name), experiments);
        }
    }
}

fn load_experiment(dir: &Path, project: Option<&str>) -> Experiment {
    let dir_name = dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    let meta_data = meta::read_meta(dir);

    // If meta.json has best_metrics, use those. Otherwise compute from JSONL.
    let (total_steps, best, status) = if let Some(ref m) = meta_data {
        (
            m.total_steps,
            if m.best_metrics.is_empty() {
                compute_metrics_from_jsonl(dir)
            } else {
                m.best_metrics.clone()
            },
            m.status.clone().unwrap_or_else(|| "unknown".into()),
        )
    } else {
        let (steps, best) = compute_steps_and_metrics(dir);
        (steps, best, "running".into())
    };

    Experiment {
        name: meta_data
            .as_ref()
            .and_then(|m| m.name.clone())
            .unwrap_or(dir_name),
        project: project
            .map(String::from)
            .or_else(|| meta_data.as_ref().and_then(|m| m.project.clone())),
        dir: dir.to_path_buf(),
        status,
        total_steps,
        best_metrics: best,
        start_time: meta_data.as_ref().and_then(|m| m.start_time.clone()),
        end_time: meta_data.as_ref().and_then(|m| m.end_time.clone()),
    }
}

fn compute_metrics_from_jsonl(dir: &Path) -> HashMap<String, f64> {
    let records = metrics::read_metrics(&dir.join("metrics.jsonl"));
    metrics::best_metrics(&records)
}

fn compute_steps_and_metrics(dir: &Path) -> (Option<u64>, HashMap<String, f64>) {
    let records = metrics::read_metrics(&dir.join("metrics.jsonl"));
    let steps = records.last().map(|r| r.step);
    let best = metrics::best_metrics(&records);
    (steps, best)
}

/// Filter experiments by project name.
pub fn filter_by_project(experiments: &[Experiment], project: &str) -> Vec<Experiment> {
    experiments
        .iter()
        .filter(|e| e.project.as_deref() == Some(project))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn create_experiment(base: &Path, project: Option<&str>, name: &str, steps: u32) {
        let dir = if let Some(p) = project {
            base.join(p).join(name)
        } else {
            base.join(name)
        };
        fs::create_dir_all(&dir).unwrap();

        // Write metrics.jsonl
        let mut jsonl = String::new();
        for i in 0..steps {
            jsonl.push_str(&format!(
                r#"{{"step":{},"timestamp":{},"metrics":{{"loss":{}}}}}"#,
                i,
                i as f64,
                1.0 - (i as f64 / steps as f64)
            ));
            jsonl.push('\n');
        }
        fs::write(dir.join("metrics.jsonl"), jsonl).unwrap();

        // Write meta.json
        let meta = format!(
            r#"{{"name":"{}","project":{},"status":"done","total_steps":{},"best_metrics":{{"loss":0.01}}}}"#,
            name,
            project.map(|p| format!(r#""{p}""#)).unwrap_or("null".into()),
            steps,
        );
        fs::write(dir.join("meta.json"), meta).unwrap();
    }

    #[test]
    fn test_discover_flat_experiments() {
        let dir = tempfile::tempdir().unwrap();
        create_experiment(dir.path(), None, "exp-001", 10);
        create_experiment(dir.path(), None, "exp-002", 20);

        let exps = discover_experiments(dir.path());
        assert_eq!(exps.len(), 2);
    }

    #[test]
    fn test_discover_project_nested() {
        let dir = tempfile::tempdir().unwrap();
        create_experiment(dir.path(), Some("project_a"), "exp-001", 10);
        create_experiment(dir.path(), Some("project_a"), "exp-002", 20);
        create_experiment(dir.path(), Some("project_b"), "exp-003", 5);

        let exps = discover_experiments(dir.path());
        assert_eq!(exps.len(), 3);
    }

    #[test]
    fn test_discover_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let exps = discover_experiments(dir.path());
        assert!(exps.is_empty());
    }

    #[test]
    fn test_discover_nonexistent_dir() {
        let exps = discover_experiments(Path::new("/nonexistent/path"));
        assert!(exps.is_empty());
    }

    #[test]
    fn test_filter_by_project() {
        let dir = tempfile::tempdir().unwrap();
        create_experiment(dir.path(), Some("alpha"), "exp-1", 10);
        create_experiment(dir.path(), Some("beta"), "exp-2", 10);

        let all = discover_experiments(dir.path());
        let filtered = filter_by_project(&all, "alpha");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "exp-1");
    }

    #[test]
    fn test_resolve_base_dir_default() {
        let dir = resolve_base_dir(None);
        assert!(dir.to_str().unwrap().contains(".nuviz"));
    }

    #[test]
    fn test_resolve_base_dir_override() {
        let dir = resolve_base_dir(Some("/custom/path"));
        assert_eq!(dir, PathBuf::from("/custom/path"));
    }
}
