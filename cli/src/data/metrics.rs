use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// A single metrics record from a JSONL file.
#[derive(Debug, Clone, Deserialize)]
pub struct MetricRecord {
    pub step: u64,
    pub timestamp: f64,
    /// Metric values. A JSON `null` (how the Python logger encodes NaN/Inf)
    /// is read as `f64::NAN` so the rest of the record is preserved.
    #[serde(deserialize_with = "de_nullable_map")]
    pub metrics: HashMap<String, f64>,
    #[serde(default, deserialize_with = "de_opt_nullable_map")]
    #[allow(dead_code)]
    pub gpu: Option<HashMap<String, f64>>,
}

fn de_nullable_map<'de, D>(d: D) -> Result<HashMap<String, f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw: HashMap<String, Option<f64>> = HashMap::deserialize(d)?;
    Ok(raw
        .into_iter()
        .map(|(k, v)| (k, v.unwrap_or(f64::NAN)))
        .collect())
}

fn de_opt_nullable_map<'de, D>(d: D) -> Result<Option<HashMap<String, f64>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw: Option<HashMap<String, Option<f64>>> = Option::deserialize(d)?;
    Ok(raw.map(|m| {
        m.into_iter()
            .map(|(k, v)| (k, v.unwrap_or(f64::NAN)))
            .collect()
    }))
}

/// Read all metric records from a JSONL file and its rotated segments.
///
/// When rotation is in effect, files are named `metrics.jsonl`, `metrics.1.jsonl`,
/// `metrics.2.jsonl`, etc. Higher-numbered files contain older data.
/// Records are returned in chronological order (oldest first).
pub fn read_metrics(path: &Path) -> Vec<MetricRecord> {
    let mut all_records = Vec::new();

    // Find rotated segments (oldest first)
    let segments = find_rotated_segments(path);
    for (_, seg_path) in segments.iter().rev() {
        all_records.extend(read_single_jsonl(seg_path));
    }

    // Read the main file (newest data)
    all_records.extend(read_single_jsonl(path));

    all_records
}

/// Read records from a single JSONL file, skipping malformed lines.
fn read_single_jsonl(path: &Path) -> Vec<MetricRecord> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };

    let reader = BufReader::new(file);
    let mut records = Vec::new();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        match serde_json::from_str::<MetricRecord>(trimmed) {
            Ok(record) => records.push(record),
            Err(e) => {
                eprintln!("[nuviz] Warning: skipping malformed JSONL line: {e}");
            }
        }
    }

    records
}

/// Find rotated JSONL segments (e.g., metrics.1.jsonl, metrics.2.jsonl).
/// Returns `(number, path)` sorted by number ascending (1, 2, 3, ...).
/// Higher numbers hold older data.
pub fn find_rotated_segments(main_path: &Path) -> Vec<(u32, std::path::PathBuf)> {
    let parent = match main_path.parent() {
        Some(p) => p,
        None => return Vec::new(),
    };

    let stem = main_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("metrics");
    let ext = main_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("jsonl");

    let mut segments: Vec<(u32, std::path::PathBuf)> = Vec::new();

    let entries = match std::fs::read_dir(parent) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Match pattern: stem.N.ext (e.g., metrics.1.jsonl)
        let prefix = format!("{stem}.");
        let suffix = format!(".{ext}");

        if let Some(stripped) = name_str.strip_prefix(&prefix) {
            if let Some(num_str) = stripped.strip_suffix(&suffix) {
                if let Ok(num) = num_str.parse::<u32>() {
                    segments.push((num, entry.path()));
                }
            }
        }
    }

    segments.sort_by_key(|(n, _)| *n);
    segments
}

/// Read only the last record from a JSONL file (efficient for large files).
#[allow(dead_code)]
pub fn read_last_record(path: &Path) -> Option<MetricRecord> {
    let file = File::open(path).ok()?;
    let reader = BufReader::new(file);
    let mut last: Option<MetricRecord> = None;

    for line in reader.lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            if let Ok(record) = serde_json::from_str::<MetricRecord>(trimmed) {
                last = Some(record);
            }
        }
    }

    last
}

/// Extract the best value for each metric across all records.
/// "loss", "lpips", "error", "mse", "mae" are minimized; others are maximized.
pub fn best_metrics(records: &[MetricRecord]) -> HashMap<String, f64> {
    let mut best: HashMap<String, f64> = HashMap::new();

    for record in records {
        for (name, &value) in &record.metrics {
            if value.is_nan() || value.is_infinite() {
                continue;
            }

            let is_better = match best.get(name) {
                None => true,
                Some(&current) => {
                    if is_minimize_metric(name) {
                        value < current
                    } else {
                        value > current
                    }
                }
            };

            if is_better {
                best.insert(name.clone(), value);
            }
        }
    }

    best
}

/// Alignment mode for multi-experiment comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignMode {
    /// Align by step number (default)
    Step,
    /// Align by wall-clock time relative to first record
    WallTime,
}

/// Extract an aligned series for a given metric.
///
/// Returns (x_values, y_values) where x depends on the alignment mode.
pub fn align_series(
    records: &[MetricRecord],
    metric: &str,
    mode: AlignMode,
) -> (Vec<f64>, Vec<f64>) {
    let start_time = records.first().map(|r| r.timestamp).unwrap_or(0.0);

    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for record in records {
        if let Some(&value) = record.metrics.get(metric) {
            if !value.is_finite() {
                continue;
            }
            let x = match mode {
                AlignMode::Step => record.step as f64,
                AlignMode::WallTime => record.timestamp - start_time,
            };
            xs.push(x);
            ys.push(value);
        }
    }

    (xs, ys)
}

/// Total order that ranks the best value first and missing / non-finite
/// values last, regardless of direction.
pub fn cmp_best_first(a: Option<f64>, b: Option<f64>, minimize: bool) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let a = a.filter(|v| v.is_finite());
    let b = b.filter(|v| v.is_finite());
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(x), Some(y)) => {
            if minimize {
                x.total_cmp(&y)
            } else {
                y.total_cmp(&x)
            }
        }
    }
}

pub fn is_minimize_metric(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("loss")
        || lower.contains("lpips")
        || lower.contains("error")
        || lower.contains("mse")
        || lower.contains("mae")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_jsonl(lines: &[&str]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(f, "{line}").unwrap();
        }
        f
    }

    #[test]
    fn test_read_valid_jsonl() {
        let f = write_jsonl(&[
            r#"{"step":0,"timestamp":1.0,"metrics":{"loss":1.0,"psnr":20.0}}"#,
            r#"{"step":1,"timestamp":2.0,"metrics":{"loss":0.5,"psnr":25.0}}"#,
        ]);
        let records = read_metrics(f.path());
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].step, 0);
        assert_eq!(records[1].step, 1);
        assert!((records[0].metrics["loss"] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_skip_malformed_lines() {
        let f = write_jsonl(&[
            r#"{"step":0,"timestamp":1.0,"metrics":{"loss":1.0}}"#,
            "not json at all",
            r#"{"step":2,"timestamp":3.0,"metrics":{"loss":0.3}}"#,
        ]);
        let records = read_metrics(f.path());
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].step, 0);
        assert_eq!(records[1].step, 2);
    }

    #[test]
    fn test_skip_empty_lines() {
        let f = write_jsonl(&[
            r#"{"step":0,"timestamp":1.0,"metrics":{"loss":1.0}}"#,
            "",
            r#"{"step":1,"timestamp":2.0,"metrics":{"loss":0.5}}"#,
        ]);
        let records = read_metrics(f.path());
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_read_missing_file() {
        let records = read_metrics(Path::new("/nonexistent/path/metrics.jsonl"));
        assert!(records.is_empty());
    }

    #[test]
    fn test_gpu_field_optional() {
        let f = write_jsonl(&[
            r#"{"step":0,"timestamp":1.0,"metrics":{"loss":1.0}}"#,
            r#"{"step":1,"timestamp":2.0,"metrics":{"loss":0.5},"gpu":{"util":87,"mem_used":10240}}"#,
        ]);
        let records = read_metrics(f.path());
        assert_eq!(records.len(), 2);
        assert!(records[0].gpu.is_none());
        assert!(records[1].gpu.is_some());
    }

    #[test]
    fn test_read_last_record() {
        let f = write_jsonl(&[
            r#"{"step":0,"timestamp":1.0,"metrics":{"loss":1.0}}"#,
            r#"{"step":99,"timestamp":100.0,"metrics":{"loss":0.01}}"#,
        ]);
        let last = read_last_record(f.path()).unwrap();
        assert_eq!(last.step, 99);
    }

    #[test]
    fn test_best_metrics_minimize_loss() {
        let records = vec![
            MetricRecord {
                step: 0,
                timestamp: 1.0,
                metrics: HashMap::from([("loss".into(), 1.0), ("psnr".into(), 20.0)]),
                gpu: None,
            },
            MetricRecord {
                step: 1,
                timestamp: 2.0,
                metrics: HashMap::from([("loss".into(), 0.5), ("psnr".into(), 25.0)]),
                gpu: None,
            },
            MetricRecord {
                step: 2,
                timestamp: 3.0,
                metrics: HashMap::from([("loss".into(), 0.8), ("psnr".into(), 23.0)]),
                gpu: None,
            },
        ];
        let best = best_metrics(&records);
        assert!((best["loss"] - 0.5).abs() < f64::EPSILON);
        assert!((best["psnr"] - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_null_metric_is_nan_and_record_kept() {
        let f = write_jsonl(&[
            r#"{"step":0,"timestamp":1.0,"metrics":{"loss":null,"psnr":30.0},"gpu":{"gpu_util":null}}"#,
        ]);
        let records = read_metrics(f.path());
        assert_eq!(records.len(), 1);
        assert!(records[0].metrics["loss"].is_nan());
        assert_eq!(records[0].metrics["psnr"], 30.0);
        assert!(records[0].gpu.as_ref().unwrap()["gpu_util"].is_nan());
    }

    #[test]
    fn test_cmp_best_first_is_total_and_puts_missing_last() {
        let mut vals = [Some(0.3), None, Some(0.1), Some(f64::NAN), Some(0.2), None];
        vals.sort_by(|a, b| cmp_best_first(*a, *b, true));
        assert_eq!(vals[0], Some(0.1));
        assert_eq!(vals[1], Some(0.2));
        assert_eq!(vals[2], Some(0.3));
        assert!(vals[3..].iter().all(|v| v.is_none_or(|x| x.is_nan())));

        let mut vals = [Some(1.0), None, Some(3.0), Some(2.0)];
        vals.sort_by(|a, b| cmp_best_first(*a, *b, false));
        assert_eq!(vals, [Some(3.0), Some(2.0), Some(1.0), None]);
    }

    #[test]
    fn test_best_metrics_skips_nan() {
        let records = vec![
            MetricRecord {
                step: 0,
                timestamp: 1.0,
                metrics: HashMap::from([("loss".into(), 0.5)]),
                gpu: None,
            },
            MetricRecord {
                step: 1,
                timestamp: 2.0,
                metrics: HashMap::from([("loss".into(), f64::NAN)]),
                gpu: None,
            },
        ];
        let best = best_metrics(&records);
        assert!((best["loss"] - 0.5).abs() < f64::EPSILON);
    }
}
