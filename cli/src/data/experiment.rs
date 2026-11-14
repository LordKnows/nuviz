use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::data::goal::Goals;
use crate::data::meta::{self, ExperimentMeta};
use crate::data::status::{self, Status};
use crate::data::summary::{self, RunSummary};

/// A discovered experiment with its metadata and location.
///
/// Discovery is cheap: it reads `meta.json` and `heartbeat.json` only.
/// Metric summaries are computed on demand via [`Experiment::summary`].
#[derive(Debug, Clone)]
pub struct Experiment {
    /// Display name: `meta.name`, falling back to the directory name.
    pub name: String,
    /// Directory name; unique within its project.
    pub dir_name: String,
    pub project: Option<String>,
    pub dir: PathBuf,
    pub status: Status,
    pub meta: ExperimentMeta,
    /// Start time as seconds since the Unix epoch, when it can be determined.
    pub started: Option<f64>,
    /// Latest step reported by the heartbeat, if any.
    pub heartbeat_step: Option<u64>,
}

impl Experiment {
    /// Load an experiment from its directory.
    pub fn load(dir: &Path, project: Option<&str>) -> Self {
        let dir_name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let meta = meta::read_meta(dir).unwrap_or_default();
        let heartbeat = status::read_heartbeat(dir);
        let now = status::now_secs();
        let status = status::resolve_status(
            meta.status.as_deref(),
            heartbeat.as_ref(),
            status::activity_age(dir, now),
            now,
        );

        let started = meta
            .start_time
            .as_deref()
            .and_then(parse_iso_seconds)
            .or_else(|| created_seconds(dir));

        Experiment {
            name: meta.name.clone().unwrap_or_else(|| dir_name.clone()),
            project: project.map(String::from).or_else(|| meta.project.clone()),
            dir: dir.to_path_buf(),
            dir_name,
            status,
            started,
            heartbeat_step: heartbeat.map(|h| h.step),
            meta,
        }
    }

    /// `project/name`, or just `name` for experiments outside a project.
    pub fn qualified_name(&self) -> String {
        match &self.project {
            Some(p) => format!("{p}/{}", self.dir_name),
            None => self.dir_name.clone(),
        }
    }

    pub fn metrics_path(&self) -> PathBuf {
        self.dir.join("metrics.jsonl")
    }

    /// Min / max / last per metric.
    ///
    /// Finished runs carry this in `meta.json`; otherwise it is computed
    /// from the JSONL files through the incremental cache.
    pub fn summary(&self) -> RunSummary {
        if matches!(self.status, Status::Done | Status::Failed) {
            if let Some(metrics) = &self.meta.metric_summary {
                return RunSummary {
                    records: None,
                    last_step: self.meta.total_steps,
                    first_timestamp: None,
                    last_timestamp: None,
                    metrics: metrics.clone(),
                };
            }
        }
        summary::scan_summary(&self.dir)
    }

    /// Metric directions declared by this experiment (heuristic fallback).
    pub fn goals(&self) -> Goals {
        Goals::new(self.meta.metric_goals.clone())
    }

    /// Start date as `YYYY-MM-DD` (UTC), if known.
    pub fn started_date(&self) -> Option<String> {
        let secs = self.started?;
        chrono::DateTime::from_timestamp(secs as i64, 0).map(|d| d.format("%Y-%m-%d").to_string())
    }
}

fn parse_iso_seconds(s: &str) -> Option<f64> {
    chrono::DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|d| d.timestamp_millis() as f64 / 1000.0)
}

fn created_seconds(dir: &Path) -> Option<f64> {
    let meta = std::fs::metadata(dir).ok()?;
    let time = meta.created().or_else(|_| meta.modified()).ok()?;
    Some(time.duration_since(UNIX_EPOCH).ok()?.as_secs_f64())
}

/// Resolve the base directory for experiments.
pub fn resolve_base_dir(dir_override: Option<&str>) -> PathBuf {
    if let Some(d) = dir_override {
        return PathBuf::from(d);
    }

    if let Ok(env_dir) = std::env::var("NUVIZ_DIR") {
        if !env_dir.is_empty() {
            return PathBuf::from(env_dir);
        }
    }

    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".nuviz")
        .join("experiments")
}

/// Whether `dir` looks like an experiment directory.
pub fn is_experiment_dir(dir: &Path) -> bool {
    dir.join("metrics.jsonl").exists() || dir.join("meta.json").exists()
}

/// Discover all experiments under the base directory, most recent first.
///
/// An experiment directory is any directory that contains `metrics.jsonl`
/// or `meta.json`, either directly under the base directory or one level
/// down inside a project directory.
pub fn discover_experiments(base_dir: &Path) -> Vec<Experiment> {
    let mut experiments = Vec::new();
    discover_in_dir(base_dir, None, &mut experiments);
    sort_recent_first(&mut experiments);
    experiments
}

/// Most recent first; experiments without a known start time last.
pub fn sort_recent_first(experiments: &mut [Experiment]) {
    experiments.sort_by(|a, b| match (a.started, b.started) {
        (Some(x), Some(y)) => y.total_cmp(&x),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a.dir_name.cmp(&b.dir_name),
    });
}

fn discover_in_dir(dir: &Path, project: Option<&str>, experiments: &mut Vec<Experiment>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }

        if is_experiment_dir(&path) {
            experiments.push(Experiment::load(&path, project));
        } else if project.is_none() {
            // Could be a project directory — recurse one level
            discover_in_dir(&path, Some(name), experiments);
        }
    }
}

/// Group experiments that differ only by seed.
///
/// Uses `config_hash` when present, otherwise the experiment name with a
/// `_seed<N>` / `_s<N>` suffix stripped. Groups are keyed deterministically.
pub fn group_by_config(experiments: &[Experiment]) -> BTreeMap<String, Vec<&Experiment>> {
    let mut groups: BTreeMap<String, Vec<&Experiment>> = BTreeMap::new();
    for exp in experiments {
        let key = match &exp.meta.config_hash {
            Some(hash) => format!("hash:{hash}"),
            None => format!("name:{}", strip_seed_suffix(&exp.dir_name)),
        };
        groups.entry(key).or_default().push(exp);
    }
    groups
}

/// Strip a seed suffix from an experiment name.
/// Matches patterns like `_seed42`, `-seed42`, `_s3`.
pub fn strip_seed_suffix(name: &str) -> String {
    for marker in ["_seed", "-seed", "_s", "-s"] {
        if let Some(pos) = name.rfind(marker) {
            let suffix = &name[pos + marker.len()..];
            if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
                return name[..pos].to_string();
            }
        }
    }
    name.to_string()
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
pub(crate) mod test_support {
    use std::fs;
    use std::path::{Path, PathBuf};

    /// Create an experiment directory with `steps` records of a decaying loss.
    pub fn create_experiment(
        base: &Path,
        project: Option<&str>,
        name: &str,
        steps: u32,
        meta_json: Option<&str>,
    ) -> PathBuf {
        let dir = match project {
            Some(p) => base.join(p).join(name),
            None => base.join(name),
        };
        fs::create_dir_all(&dir).unwrap();

        let mut jsonl = String::new();
        for i in 0..steps {
            jsonl.push_str(&format!(
                "{{\"step\":{i},\"timestamp\":{}.0,\"metrics\":{{\"loss\":{},\"psnr\":{}}}}}\n",
                1_700_000_000 + i,
                1.0 - (i as f64 / steps as f64),
                20.0 + i as f64
            ));
        }
        fs::write(dir.join("metrics.jsonl"), jsonl).unwrap();
        if let Some(meta) = meta_json {
            fs::write(dir.join("meta.json"), meta).unwrap();
        }
        dir
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::create_experiment;
    use super::*;
    use crate::data::goal::Goal;
    use std::fs;

    const DONE: &str =
        r#"{"status":"done","total_steps":9,"start_time":"2026-01-02T03:04:05+00:00"}"#;

    #[test]
    fn test_discover_flat_experiments() {
        let dir = tempfile::tempdir().unwrap();
        create_experiment(dir.path(), None, "exp-001", 10, Some(DONE));
        create_experiment(dir.path(), None, "exp-002", 20, Some(DONE));

        let exps = discover_experiments(dir.path());
        assert_eq!(exps.len(), 2);
    }

    #[test]
    fn test_discover_project_nested() {
        let dir = tempfile::tempdir().unwrap();
        create_experiment(dir.path(), Some("project_a"), "exp-001", 10, Some(DONE));
        create_experiment(dir.path(), Some("project_a"), "exp-002", 20, Some(DONE));
        create_experiment(dir.path(), Some("project_b"), "exp-003", 5, Some(DONE));

        let exps = discover_experiments(dir.path());
        assert_eq!(exps.len(), 3);
        let e = exps.iter().find(|e| e.dir_name == "exp-003").unwrap();
        assert_eq!(e.project.as_deref(), Some("project_b"));
        assert_eq!(e.qualified_name(), "project_b/exp-003");
    }

    #[test]
    fn test_discover_empty_and_missing_dir() {
        let dir = tempfile::tempdir().unwrap();
        assert!(discover_experiments(dir.path()).is_empty());
        assert!(discover_experiments(Path::new("/nonexistent/path")).is_empty());
    }

    #[test]
    fn test_hidden_directories_are_skipped() {
        let dir = tempfile::tempdir().unwrap();
        create_experiment(dir.path(), None, ".cache", 3, None);
        create_experiment(dir.path(), None, "real", 3, None);
        let exps = discover_experiments(dir.path());
        assert_eq!(exps.len(), 1);
        assert_eq!(exps[0].dir_name, "real");
    }

    #[test]
    fn test_filter_by_project() {
        let dir = tempfile::tempdir().unwrap();
        create_experiment(dir.path(), Some("alpha"), "exp-1", 10, Some(DONE));
        create_experiment(dir.path(), Some("beta"), "exp-2", 10, Some(DONE));

        let all = discover_experiments(dir.path());
        let filtered = filter_by_project(&all, "alpha");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "exp-1");
    }

    #[test]
    fn test_sorted_most_recent_first() {
        let dir = tempfile::tempdir().unwrap();
        create_experiment(
            dir.path(),
            None,
            "old",
            1,
            Some(r#"{"status":"done","start_time":"2025-01-01T00:00:00+00:00"}"#),
        );
        create_experiment(
            dir.path(),
            None,
            "new",
            1,
            Some(r#"{"status":"done","start_time":"2026-06-01T00:00:00.123456+00:00"}"#),
        );
        let exps = discover_experiments(dir.path());
        assert_eq!(exps[0].dir_name, "new");
        assert_eq!(exps[0].started_date().as_deref(), Some("2026-06-01"));
    }

    #[test]
    fn test_summary_scanned_for_run_without_meta_summary() {
        let dir = tempfile::tempdir().unwrap();
        let exp_dir = create_experiment(dir.path(), None, "exp", 10, None);
        let exp = Experiment::load(&exp_dir, None);
        assert_eq!(exp.status, Status::Running); // just written, no meta
        let s = summary::scan_summary_with(&exp.dir, None);
        assert_eq!(s.last_step, Some(9));
        assert_eq!(s.metrics["psnr"].best(Goal::Max), (29.0, 9));
    }

    #[test]
    fn test_summary_from_meta_for_finished_run() {
        let dir = tempfile::tempdir().unwrap();
        let meta = r#"{"status":"done","total_steps":99,
            "metric_summary":{"loss":{"count":5,"min":0.1,"min_step":40,"max":2.0,"max_step":0,"last":0.3,"last_step":99}}}"#;
        // The JSONL deliberately disagrees: finished runs must not be rescanned.
        let exp_dir = create_experiment(dir.path(), None, "exp", 3, Some(meta));
        let s = Experiment::load(&exp_dir, None).summary();
        assert_eq!(s.last_step, Some(99));
        assert_eq!(s.metrics["loss"].best(Goal::Min), (0.1, 40));
    }

    #[test]
    fn test_status_from_heartbeat() {
        let dir = tempfile::tempdir().unwrap();
        let exp_dir =
            create_experiment(dir.path(), None, "exp", 3, Some(r#"{"status":"running"}"#));

        let now = status::now_secs();
        fs::write(
            exp_dir.join("heartbeat.json"),
            format!("{{\"timestamp\":{now},\"step\":2,\"interval\":10.0}}"),
        )
        .unwrap();
        let exp = Experiment::load(&exp_dir, None);
        assert_eq!(exp.status, Status::Running);
        assert_eq!(exp.heartbeat_step, Some(2));

        fs::write(
            exp_dir.join("heartbeat.json"),
            format!(
                "{{\"timestamp\":{},\"step\":2,\"interval\":10.0}}",
                now - 3600.0
            ),
        )
        .unwrap();
        assert_eq!(Experiment::load(&exp_dir, None).status, Status::Crashed);
    }

    #[test]
    fn test_group_by_config_hash_then_name() {
        let dir = tempfile::tempdir().unwrap();
        create_experiment(
            dir.path(),
            None,
            "a_seed1",
            1,
            Some(r#"{"config_hash":"h1"}"#),
        );
        create_experiment(dir.path(), None, "zzz", 1, Some(r#"{"config_hash":"h1"}"#));
        create_experiment(dir.path(), None, "b_seed1", 1, None);
        create_experiment(dir.path(), None, "b_seed2", 1, None);
        create_experiment(dir.path(), None, "solo", 1, None);

        let exps = discover_experiments(dir.path());
        let groups = group_by_config(&exps);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups["hash:h1"].len(), 2);
        assert_eq!(groups["name:b"].len(), 2);
        assert_eq!(groups["name:solo"].len(), 1);
    }

    #[test]
    fn test_strip_seed_suffix() {
        assert_eq!(strip_seed_suffix("exp_seed42"), "exp");
        assert_eq!(strip_seed_suffix("exp-seed0"), "exp");
        assert_eq!(strip_seed_suffix("exp_s3"), "exp");
        assert_eq!(strip_seed_suffix("exp_small"), "exp_small");
        assert_eq!(strip_seed_suffix("exp_seed"), "exp_seed");
    }

    #[test]
    fn test_resolve_base_dir() {
        assert_eq!(
            resolve_base_dir(Some("/custom/path")),
            PathBuf::from("/custom/path")
        );
    }
}
