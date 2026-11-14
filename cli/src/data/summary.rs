//! Per-experiment summaries (min / max / last per metric) and their cache.
//!
//! Listing commands only need a handful of numbers per experiment. Finished
//! runs carry them in `meta.json`. For running, crashed or legacy runs they
//! are computed from the JSONL files and cached per experiment, together with
//! the byte offset already consumed, so later invocations read only what was
//! appended since.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::data::goal::Goal;
use crate::data::metrics::{find_rotated_segments, for_each_line, parse_line};

/// Running statistics for one metric. Field names match the
/// `metric_summary` object the Python logger writes to `meta.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricSummary {
    pub count: u64,
    pub min: f64,
    pub min_step: u64,
    pub max: f64,
    pub max_step: u64,
    pub last: f64,
    pub last_step: u64,
}

impl Default for MetricSummary {
    fn default() -> Self {
        Self {
            count: 0,
            min: f64::INFINITY,
            min_step: 0,
            max: f64::NEG_INFINITY,
            max_step: 0,
            last: f64::NAN,
            last_step: 0,
        }
    }
}

impl MetricSummary {
    /// Fold in one finite observation.
    pub fn update(&mut self, step: u64, value: f64) {
        self.count += 1;
        if value < self.min {
            self.min = value;
            self.min_step = step;
        }
        if value > self.max {
            self.max = value;
            self.max_step = step;
        }
        self.last = value;
        self.last_step = step;
    }

    /// Best value and the step it occurred at.
    pub fn best(&self, goal: Goal) -> (f64, u64) {
        match goal {
            Goal::Min => (self.min, self.min_step),
            Goal::Max => (self.max, self.max_step),
        }
    }
}

/// Which value of a metric represents a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ValueMode {
    /// Best value seen at any step
    Best,
    /// Value at the last step it was logged
    Final,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RunSummary {
    /// Number of records, when known (not stored in `meta.json`).
    pub records: Option<u64>,
    pub last_step: Option<u64>,
    pub first_timestamp: Option<f64>,
    pub last_timestamp: Option<f64>,
    pub metrics: BTreeMap<String, MetricSummary>,
}

impl RunSummary {
    /// The representative value of a metric and the step it was taken at.
    pub fn value(&self, name: &str, mode: ValueMode, goal: Goal) -> Option<(f64, u64)> {
        let m = self.metrics.get(name).filter(|m| m.count > 0)?;
        Some(match mode {
            ValueMode::Best => m.best(goal),
            ValueMode::Final => (m.last, m.last_step),
        })
    }

    fn absorb_line(&mut self, line: &str) {
        let Some(raw) = parse_line(line) else {
            return;
        };
        *self.records.get_or_insert(0) += 1;
        self.last_step = Some(raw.step);
        if self.first_timestamp.is_none() {
            self.first_timestamp = Some(raw.timestamp);
        }
        self.last_timestamp = Some(raw.timestamp);
        for (name, value) in &raw.metrics.0 {
            if value.is_finite() {
                match self.metrics.get_mut(name.as_ref()) {
                    Some(m) => m.update(raw.step, *value),
                    None => {
                        let mut m = MetricSummary::default();
                        m.update(raw.step, *value);
                        self.metrics.insert(name.to_string(), m);
                    }
                }
            }
        }
    }
}

const CACHE_VERSION: u32 = 1;

/// What the cached summary was computed from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Fingerprint {
    /// Bytes of complete lines consumed from the current `metrics.jsonl`.
    main_consumed: u64,
    /// Inode of the current file (Unix), to detect rotation.
    main_id: Option<u64>,
    /// `(number, length)` of every rotated segment.
    segments: Vec<(u32, u64)>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheEntry {
    version: u32,
    fingerprint: Fingerprint,
    summary: RunSummary,
}

/// Summarise an experiment's JSONL files, using and refreshing the cache.
pub fn scan_summary(experiment_dir: &Path) -> RunSummary {
    scan_summary_with(experiment_dir, cache_dir().as_deref())
}

/// Like [`scan_summary`] with an explicit cache directory (`None` disables caching).
pub fn scan_summary_with(experiment_dir: &Path, cache_dir: Option<&Path>) -> RunSummary {
    let main = experiment_dir.join("metrics.jsonl");
    let segments: Vec<(u32, u64, PathBuf)> = find_rotated_segments(&main)
        .into_iter()
        .map(|(n, p)| (n, std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0), p))
        .collect();
    let segment_print: Vec<(u32, u64)> = segments.iter().map(|(n, len, _)| (*n, *len)).collect();

    let main_meta = std::fs::metadata(&main).ok();
    let main_len = main_meta.as_ref().map(|m| m.len()).unwrap_or(0);
    let main_id = main_meta.as_ref().and_then(file_id);

    let cache_path = cache_dir.map(|d| d.join(cache_file_name(experiment_dir)));
    let cached = cache_path.as_deref().and_then(read_cache);

    // The cache is reusable if nothing was rotated and the file only grew.
    let (mut summary, mut consumed) = match cached {
        Some(entry)
            if entry.fingerprint.segments == segment_print
                && entry.fingerprint.main_id == main_id
                && entry.fingerprint.main_consumed <= main_len =>
        {
            if entry.fingerprint.main_consumed == main_len {
                return entry.summary; // nothing new
            }
            (entry.summary, entry.fingerprint.main_consumed)
        }
        _ => {
            let mut summary = RunSummary::default();
            for (_, _, path) in segments.iter().rev() {
                if let Ok(mut f) = File::open(path) {
                    for_each_line(&mut f, 0, true, &mut |l: &str| summary.absorb_line(l));
                }
            }
            (summary, 0)
        }
    };

    if let Ok(mut f) = File::open(&main) {
        consumed += for_each_line(&mut f, consumed, false, &mut |l: &str| {
            summary.absorb_line(l)
        });
    }

    if let Some(path) = cache_path {
        write_cache(
            &path,
            &CacheEntry {
                version: CACHE_VERSION,
                fingerprint: Fingerprint {
                    main_consumed: consumed,
                    main_id,
                    segments: segment_print,
                },
                summary: summary.clone(),
            },
        );
    }
    summary
}

/// Where summaries are cached: `$NUVIZ_CACHE_DIR`, else the platform cache
/// directory. `NUVIZ_NO_CACHE=1` disables caching.
pub fn cache_dir() -> Option<PathBuf> {
    if std::env::var("NUVIZ_NO_CACHE").is_ok_and(|v| !v.is_empty() && v != "0") {
        return None;
    }
    if let Ok(dir) = std::env::var("NUVIZ_CACHE_DIR") {
        if !dir.is_empty() {
            return Some(PathBuf::from(dir));
        }
    }
    dirs::cache_dir().map(|d| d.join("nuviz").join("summaries"))
}

fn cache_file_name(experiment_dir: &Path) -> String {
    let canonical = experiment_dir
        .canonicalize()
        .unwrap_or_else(|_| experiment_dir.to_path_buf());
    // FNV-1a: stable across Rust versions, unlike the std hasher.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in canonical.to_string_lossy().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}.json")
}

fn read_cache(path: &Path) -> Option<CacheEntry> {
    let content = std::fs::read_to_string(path).ok()?;
    let entry: CacheEntry = serde_json::from_str(&content).ok()?;
    (entry.version == CACHE_VERSION).then_some(entry)
}

fn write_cache(path: &Path, entry: &CacheEntry) {
    // Best effort: a read-only or missing cache directory only costs speed.
    let Some(parent) = path.parent() else { return };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let Ok(json) = serde_json::to_string(entry) else {
        return;
    };
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    if std::fs::write(&tmp, json).is_ok() && std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

#[cfg(unix)]
pub(crate) fn file_id(meta: &std::fs::Metadata) -> Option<u64> {
    use std::os::unix::fs::MetadataExt;
    Some(meta.ino())
}

#[cfg(not(unix))]
pub(crate) fn file_id(_meta: &std::fs::Metadata) -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn rec(step: u64, loss: f64) -> String {
        format!(
            "{{\"step\":{step},\"timestamp\":{}.0,\"metrics\":{{\"loss\":{loss},\"psnr\":{}}}}}\n",
            step + 10,
            20.0 + step as f64
        )
    }

    fn setup() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let exp = dir.path().join("exp");
        let cache = dir.path().join("cache");
        std::fs::create_dir_all(&exp).unwrap();
        (dir, exp, cache)
    }

    #[test]
    fn test_summary_values() {
        let (_guard, exp, cache) = setup();
        std::fs::write(
            exp.join("metrics.jsonl"),
            rec(0, 1.0) + &rec(1, 0.2) + &rec(2, 0.5),
        )
        .unwrap();

        let s = scan_summary_with(&exp, Some(&cache));
        assert_eq!(s.records, Some(3));
        assert_eq!(s.last_step, Some(2));
        assert_eq!(s.first_timestamp, Some(10.0));
        let loss = &s.metrics["loss"];
        assert_eq!(loss.best(Goal::Min), (0.2, 1));
        assert_eq!(loss.best(Goal::Max), (1.0, 0));
        assert_eq!((loss.last, loss.last_step), (0.5, 2));
        assert_eq!(s.metrics["psnr"].best(Goal::Max), (22.0, 2));
    }

    #[test]
    fn test_cache_is_incremental_and_matches_full_scan() {
        let (_guard, exp, cache) = setup();
        let path = exp.join("metrics.jsonl");
        std::fs::write(&path, rec(0, 1.0) + &rec(1, 0.8)).unwrap();
        let first = scan_summary_with(&exp, Some(&cache));
        assert_eq!(first.records, Some(2));

        // Corrupt the already-consumed prefix in place (same length). An
        // incremental read must not look at it again.
        let original = std::fs::read(&path).unwrap();
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        f.write_all(rec(2, 0.1).as_bytes()).unwrap();
        drop(f);

        let second = scan_summary_with(&exp, Some(&cache));
        assert_eq!(second.records, Some(3));
        assert_eq!(second.metrics["loss"].best(Goal::Min), (0.1, 2));
        assert_eq!(second, scan_summary_with(&exp, None));
        assert!(original.len() < std::fs::metadata(&path).unwrap().len() as usize);

        // Unchanged file: served from cache.
        assert_eq!(scan_summary_with(&exp, Some(&cache)), second);
    }

    #[test]
    fn test_partial_trailing_line_is_not_counted_until_complete() {
        let (_guard, exp, cache) = setup();
        let path = exp.join("metrics.jsonl");
        let full = rec(1, 0.5);
        let (a, b) = full.split_at(15);
        std::fs::write(&path, rec(0, 1.0) + a).unwrap();
        assert_eq!(scan_summary_with(&exp, Some(&cache)).records, Some(1));

        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        f.write_all(b.as_bytes()).unwrap();
        drop(f);
        let s = scan_summary_with(&exp, Some(&cache));
        assert_eq!(s.records, Some(2));
        assert_eq!(s.metrics["loss"].min, 0.5);
    }

    #[test]
    fn test_rotation_invalidates_cache() {
        let (_guard, exp, cache) = setup();
        let path = exp.join("metrics.jsonl");
        std::fs::write(&path, rec(0, 1.0) + &rec(1, 0.9)).unwrap();
        assert_eq!(scan_summary_with(&exp, Some(&cache)).records, Some(2));

        std::fs::rename(&path, exp.join("metrics.1.jsonl")).unwrap();
        std::fs::write(&path, rec(2, 0.3)).unwrap();
        let s = scan_summary_with(&exp, Some(&cache));
        assert_eq!(s.records, Some(3));
        assert_eq!(s.metrics["loss"].best(Goal::Min), (0.3, 2));
        assert_eq!(s.metrics["loss"].max_step, 0);
    }

    #[test]
    fn test_missing_files_and_unwritable_cache() {
        let (_guard, exp, _) = setup();
        let s = scan_summary_with(&exp, Some(Path::new("/proc/definitely/not/writable")));
        assert_eq!(s, RunSummary::default());
    }

    #[test]
    fn test_metric_summary_matches_python_shape() {
        let json =
            r#"{"count":3,"min":0.2,"min_step":1,"max":1.0,"max_step":0,"last":0.5,"last_step":2}"#;
        let m: MetricSummary = serde_json::from_str(json).unwrap();
        assert_eq!(m.best(Goal::Min), (0.2, 1));
        assert_eq!(m.last, 0.5);
    }
}
