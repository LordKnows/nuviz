//! Wire format of `metrics.jsonl` and row-oriented access to it.
//!
//! One JSON object per line:
//! `{"step": 12, "timestamp": 1700000000.5, "metrics": {"loss": 0.1}, "gpu": {...}}`
//!
//! A metric value of `null` is how the Python logger encodes NaN/Inf; it is
//! read as `f64::NAN` so the rest of the record is preserved. Unknown keys
//! are ignored for forward compatibility.

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// `name -> value` pairs in file order, with JSON `null` mapped to NaN.
#[derive(Debug, Default)]
pub struct Pairs<'a>(pub Vec<(Cow<'a, str>, f64)>);

impl<'de: 'a, 'a> Deserialize<'de> for Pairs<'a> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct PairsVisitor<'a>(std::marker::PhantomData<&'a ()>);

        impl<'de: 'a, 'a> Visitor<'de> for PairsVisitor<'a> {
            type Value = Pairs<'a>;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an object mapping names to numbers or null")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut out = Vec::with_capacity(map.size_hint().unwrap_or(8));
                while let Some(key) = map.next_key::<Cow<'a, str>>()? {
                    let value = map.next_value::<Option<f64>>()?;
                    out.push((key, value.unwrap_or(f64::NAN)));
                }
                Ok(Pairs(out))
            }
        }

        d.deserialize_map(PairsVisitor(std::marker::PhantomData))
    }
}

/// Accept `12` as well as `12.0` for the step.
fn de_step<'de, D: Deserializer<'de>>(d: D) -> Result<u64, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Num {
        Int(u64),
        Float(f64),
    }
    match Num::deserialize(d)? {
        Num::Int(v) => Ok(v),
        Num::Float(v) if v >= 0.0 && v.fract() == 0.0 && v < u64::MAX as f64 => Ok(v as u64),
        Num::Float(v) => Err(de::Error::custom(format!("invalid step {v}"))),
    }
}

/// One parsed line, borrowing metric names from the input where possible.
#[derive(Debug, Deserialize)]
pub struct RawLine<'a> {
    #[serde(deserialize_with = "de_step")]
    pub step: u64,
    pub timestamp: f64,
    #[serde(borrow)]
    pub metrics: Pairs<'a>,
    #[serde(borrow, default)]
    pub gpu: Option<Pairs<'a>>,
}

/// Parse one line of `metrics.jsonl`. Returns `None` for malformed lines.
pub fn parse_line(line: &str) -> Option<RawLine<'_>> {
    serde_json::from_str(line).ok()
}

/// A single metrics record (owned, row-oriented).
#[derive(Debug, Clone)]
pub struct MetricRecord {
    pub step: u64,
    pub timestamp: f64,
    pub metrics: HashMap<String, f64>,
    pub gpu: Option<HashMap<String, f64>>,
}

impl From<RawLine<'_>> for MetricRecord {
    fn from(raw: RawLine<'_>) -> Self {
        let own = |p: Pairs<'_>| p.0.into_iter().map(|(k, v)| (k.into_owned(), v)).collect();
        MetricRecord {
            step: raw.step,
            timestamp: raw.timestamp,
            metrics: own(raw.metrics),
            gpu: raw.gpu.map(own),
        }
    }
}

/// Read all metric records from a JSONL file and its rotated segments.
///
/// When rotation is in effect, files are named `metrics.jsonl`, `metrics.1.jsonl`,
/// `metrics.2.jsonl`, etc. Higher-numbered files contain older data.
/// Records are returned in chronological order (oldest first). Malformed
/// lines are skipped.
pub fn read_metrics(path: &Path) -> Vec<MetricRecord> {
    let mut records = Vec::new();
    for_each_line_with_history(path, |line| {
        if let Some(raw) = parse_line(line) {
            records.push(raw.into());
        }
    });
    records
}

/// Feed every complete line of the rotated segments (oldest first) and then
/// the main file to `on_line`.
pub fn for_each_line_with_history(path: &Path, mut on_line: impl FnMut(&str)) {
    for (_, seg_path) in find_rotated_segments(path).iter().rev() {
        if let Ok(mut f) = File::open(seg_path) {
            for_each_line(&mut f, 0, true, &mut on_line);
        }
    }
    if let Ok(mut f) = File::open(path) {
        for_each_line(&mut f, 0, true, &mut on_line);
    }
}

/// Stream lines starting at byte offset `from`, calling `on_line` for every
/// non-empty one. Returns the number of bytes consumed.
///
/// Unless `whole_file` is set, a trailing line without a newline is left
/// unconsumed: a writer may still be in the middle of it.
pub fn for_each_line(
    file: &mut File,
    from: u64,
    whole_file: bool,
    on_line: &mut impl FnMut(&str),
) -> u64 {
    if file.seek(SeekFrom::Start(from)).is_err() {
        return 0;
    }
    let mut reader = BufReader::with_capacity(1 << 16, file);
    let mut buf: Vec<u8> = Vec::with_capacity(512);
    let mut consumed = 0u64;

    loop {
        buf.clear();
        let n = match reader.read_until(b'\n', &mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        let complete = buf.last() == Some(&b'\n');
        if !complete && !whole_file {
            break;
        }
        consumed += n as u64;
        if let Ok(line) = std::str::from_utf8(&buf) {
            let line = line.trim();
            if !line.is_empty() {
                on_line(line);
            }
        }
    }
    consumed
}

/// Find rotated JSONL segments (e.g., metrics.1.jsonl, metrics.2.jsonl).
/// Returns `(number, path)` sorted by number ascending (1, 2, 3, ...).
/// Higher numbers hold older data.
pub fn find_rotated_segments(main_path: &Path) -> Vec<(u32, PathBuf)> {
    let Some(parent) = main_path.parent() else {
        return Vec::new();
    };
    let stem = main_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("metrics");
    let ext = main_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("jsonl");
    let prefix = format!("{stem}.");
    let suffix = format!(".{ext}");

    let Ok(entries) = std::fs::read_dir(parent) else {
        return Vec::new();
    };

    let mut segments: Vec<(u32, PathBuf)> = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name();
            let num = name
                .to_str()?
                .strip_prefix(&prefix)?
                .strip_suffix(&suffix)?
                .parse::<u32>()
                .ok()?;
            Some((num, entry.path()))
        })
        .collect();
    segments.sort_by_key(|(n, _)| *n);
    segments
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
    fn test_float_step_accepted_and_unknown_keys_ignored() {
        let f = write_jsonl(&[
            r#"{"step":3.0,"timestamp":1.0,"metrics":{"loss":1.0},"future_field":[1,2]}"#,
            r#"{"step":3.5,"timestamp":1.0,"metrics":{"loss":1.0}}"#,
            r#"{"step":-1,"timestamp":1.0,"metrics":{"loss":1.0}}"#,
        ]);
        let records = read_metrics(f.path());
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].step, 3);
    }

    #[test]
    fn test_escaped_metric_names() {
        let f =
            write_jsonl(&[r#"{"step":0,"timestamp":1.0,"metrics":{"a \"b\"":1.5,"\u5b66":2.0}}"#]);
        let records = read_metrics(f.path());
        assert_eq!(records[0].metrics["a \"b\""], 1.5);
        assert_eq!(records[0].metrics["学"], 2.0);
    }

    #[test]
    fn test_for_each_line_leaves_partial_tail() {
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "one\ntwo\npart").unwrap();
        let mut file = File::open(f.path()).unwrap();
        let mut seen = Vec::new();
        let consumed = for_each_line(&mut file, 0, false, &mut |l: &str| seen.push(l.to_string()));
        assert_eq!(seen, ["one", "two"]);
        assert_eq!(consumed, 8);

        let mut seen = Vec::new();
        let consumed = for_each_line(&mut file, 0, true, &mut |l: &str| seen.push(l.to_string()));
        assert_eq!(seen, ["one", "two", "part"]);
        assert_eq!(consumed, 12);
    }
}
