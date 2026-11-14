//! Columnar in-memory storage for one experiment's metrics.
//!
//! Each metric is a dense `Vec<f64>` aligned with the `steps` / `timestamps`
//! vectors (NaN marks "not logged at this step" as well as a logged NaN).
//! Compared with one `HashMap<String, f64>` per record this is roughly an
//! order of magnitude smaller and lets charts scan a metric linearly.

use std::collections::HashMap;
use std::path::Path;

use crate::data::metrics::{for_each_line_with_history, parse_line, Pairs, RawLine};
use crate::data::summary::MetricSummary;

/// Most recent GPU sample attached to a record.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuReading {
    pub util: Option<f64>,
    pub mem_used_mb: Option<f64>,
    pub mem_total_mb: Option<f64>,
    pub temperature_c: Option<f64>,
}

impl GpuReading {
    fn from_pairs(pairs: &Pairs<'_>) -> Self {
        let get = |key: &str| {
            pairs
                .0
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| *v)
                .filter(|v| v.is_finite())
        };
        GpuReading {
            util: get("gpu_util"),
            mem_used_mb: get("mem_used_mb"),
            mem_total_mb: get("mem_total_mb"),
            temperature_c: get("temperature_c"),
        }
    }
}

/// Which quantity to use as the x coordinate of a series.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XAxis {
    Step,
    /// Seconds since the experiment's first record.
    WallTime,
}

#[derive(Debug, Default)]
pub struct MetricStore {
    steps: Vec<u64>,
    timestamps: Vec<f64>,
    names: Vec<String>,
    index: HashMap<String, usize>,
    columns: Vec<Vec<f64>>,
    gpu: Option<GpuReading>,
}

impl MetricStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load an experiment's full history (rotated segments + current file).
    pub fn load(metrics_path: &Path) -> Self {
        let mut store = Self::new();
        for_each_line_with_history(metrics_path, |line| {
            store.push_line(line);
        });
        store
    }

    /// Parse and append one JSONL line. Returns `false` if it was malformed.
    pub fn push_line(&mut self, line: &str) -> bool {
        match parse_line(line) {
            Some(raw) => {
                self.push_raw(&raw);
                true
            }
            None => false,
        }
    }

    pub fn push_raw(&mut self, raw: &RawLine<'_>) {
        self.push(
            raw.step,
            raw.timestamp,
            raw.metrics.0.iter().map(|(k, v)| (k.as_ref(), *v)),
        );
        if let Some(gpu) = &raw.gpu {
            self.gpu = Some(GpuReading::from_pairs(gpu));
        }
    }

    pub fn push<'a>(
        &mut self,
        step: u64,
        timestamp: f64,
        metrics: impl IntoIterator<Item = (&'a str, f64)>,
    ) {
        let row = self.steps.len();
        self.steps.push(step);
        self.timestamps.push(timestamp);
        for column in &mut self.columns {
            column.push(f64::NAN);
        }
        for (name, value) in metrics {
            let idx = match self.index.get(name) {
                Some(&idx) => idx,
                None => {
                    let idx = self.columns.len();
                    self.index.insert(name.to_string(), idx);
                    self.names.push(name.to_string());
                    self.columns.push(vec![f64::NAN; row + 1]);
                    idx
                }
            };
            self.columns[idx][row] = value;
        }
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    pub fn steps(&self) -> &[u64] {
        &self.steps
    }

    pub fn timestamps(&self) -> &[f64] {
        &self.timestamps
    }

    pub fn last_step(&self) -> Option<u64> {
        self.steps.last().copied()
    }

    pub fn gpu(&self) -> Option<&GpuReading> {
        self.gpu.as_ref()
    }

    /// Metric names, sorted.
    pub fn metric_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.names.iter().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    pub fn has_metric(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// The raw column for a metric (aligned with `steps()`), if it exists.
    pub fn column(&self, name: &str) -> Option<&[f64]> {
        self.index.get(name).map(|&i| self.columns[i].as_slice())
    }

    /// Value of `name` in row `row` (`None` if missing or non-finite).
    pub fn value_at(&self, name: &str, row: usize) -> Option<f64> {
        self.column(name)?
            .get(row)
            .copied()
            .filter(|v| v.is_finite())
    }

    /// Finite `(x, y)` points of a metric.
    pub fn series<'a>(&'a self, name: &str, x: XAxis) -> impl Iterator<Item = (f64, f64)> + 'a {
        let t0 = self.timestamps.first().copied().unwrap_or(0.0);
        let column = self.column(name).unwrap_or(&[]);
        column
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(move |(i, &v)| {
                let xv = match x {
                    XAxis::Step => self.steps[i] as f64,
                    XAxis::WallTime => self.timestamps[i] - t0,
                };
                (xv, v)
            })
    }

    /// Min / max / last of a metric over finite values.
    pub fn summary(&self, name: &str) -> Option<MetricSummary> {
        let column = self.column(name)?;
        let mut summary = MetricSummary::default();
        for (i, &v) in column.iter().enumerate() {
            if v.is_finite() {
                summary.update(self.steps[i], v);
            }
        }
        (summary.count > 0).then_some(summary)
    }

    /// Recent throughput in steps per second, from up to the last `window` records.
    pub fn steps_per_second(&self, window: usize) -> Option<f64> {
        let n = self.len();
        if n < 2 {
            return None;
        }
        let start = n.saturating_sub(window.max(2));
        let dt = self.timestamps[n - 1] - self.timestamps[start];
        let ds = self.steps[n - 1].checked_sub(self.steps[start])? as f64;
        (dt > 0.0 && ds > 0.0).then(|| ds / dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(step: u64, body: &str) -> String {
        format!(
            "{{\"step\":{step},\"timestamp\":{}.0,\"metrics\":{{{body}}}}}",
            step + 100
        )
    }

    #[test]
    fn test_columns_stay_aligned_when_metrics_appear_late() {
        let mut store = MetricStore::new();
        assert!(store.push_line(&line(0, "\"loss\":1.0")));
        assert!(store.push_line(&line(1, "\"loss\":0.5,\"psnr\":20.0")));
        assert!(store.push_line(&line(2, "\"psnr\":21.0")));

        assert_eq!(store.len(), 3);
        assert_eq!(store.steps(), &[0, 1, 2]);
        let loss = store.column("loss").unwrap();
        let psnr = store.column("psnr").unwrap();
        assert_eq!(loss.len(), 3);
        assert_eq!(psnr.len(), 3);
        assert!(psnr[0].is_nan());
        assert!(loss[2].is_nan());
        assert_eq!(store.value_at("psnr", 1), Some(20.0));
        assert_eq!(store.value_at("psnr", 0), None);
        assert_eq!(store.metric_names(), vec!["loss", "psnr"]);
    }

    #[test]
    fn test_null_is_nan_and_excluded_from_series_and_summary() {
        let mut store = MetricStore::new();
        store.push_line(&line(0, "\"loss\":1.0"));
        store.push_line(&line(1, "\"loss\":null"));
        store.push_line(&line(2, "\"loss\":0.25"));

        let pts: Vec<(f64, f64)> = store.series("loss", XAxis::Step).collect();
        assert_eq!(pts, vec![(0.0, 1.0), (2.0, 0.25)]);

        let s = store.summary("loss").unwrap();
        assert_eq!((s.count, s.min, s.min_step), (2, 0.25, 2));
        assert_eq!((s.max, s.max_step), (1.0, 0));
        assert_eq!((s.last, s.last_step), (0.25, 2));
    }

    #[test]
    fn test_wall_time_axis_is_relative_to_first_record() {
        let mut store = MetricStore::new();
        store.push(0, 1000.0, [("a", 1.0)]);
        store.push(5, 1060.0, [("a", 2.0)]);
        let pts: Vec<(f64, f64)> = store.series("a", XAxis::WallTime).collect();
        assert_eq!(pts, vec![(0.0, 1.0), (60.0, 2.0)]);
        assert_eq!(store.steps_per_second(100), Some(5.0 / 60.0));
    }

    #[test]
    fn test_malformed_line_rejected() {
        let mut store = MetricStore::new();
        assert!(!store.push_line("not json"));
        assert!(!store.push_line("{\"step\":1}"));
        assert!(store.is_empty());
        assert!(store.summary("x").is_none());
        assert_eq!(store.series("x", XAxis::Step).count(), 0);
    }

    #[test]
    fn test_gpu_latest_reading() {
        let mut store = MetricStore::new();
        store.push_line(
            r#"{"step":0,"timestamp":1.0,"metrics":{},"gpu":{"gpu_util":87,"mem_used_mb":1024,"mem_total_mb":24576,"temperature_c":null}}"#,
        );
        let gpu = store.gpu().unwrap();
        assert_eq!(gpu.util, Some(87.0));
        assert_eq!(gpu.mem_total_mb, Some(24576.0));
        assert_eq!(gpu.temperature_c, None);
    }

    #[test]
    fn test_load_reads_rotated_history_in_order() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("metrics.1.jsonl"),
            line(0, "\"a\":1.0") + "\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("metrics.jsonl"),
            line(1, "\"a\":2.0") + "\n",
        )
        .unwrap();
        let store = MetricStore::load(&dir.path().join("metrics.jsonl"));
        assert_eq!(store.steps(), &[0, 1]);
    }
}
