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
    pub metrics: HashMap<String, f64>,
    #[serde(default)]
    pub gpu: Option<HashMap<String, f64>>,
}

/// Read all metric records from a JSONL file, skipping malformed lines.
pub fn read_metrics(path: &Path) -> Vec<MetricRecord> {
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

/// Read only the last record from a JSONL file (efficient for large files).
pub fn read_last_record(path: &Path) -> Option<MetricRecord> {
    let file = File::open(path).ok()?;
    let reader = BufReader::new(file);
    let mut last: Option<MetricRecord> = None;

    for line in reader.lines() {
        if let Ok(line) = line {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                if let Ok(record) = serde_json::from_str::<MetricRecord>(trimmed) {
                    last = Some(record);
                }
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

fn is_minimize_metric(name: &str) -> bool {
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
