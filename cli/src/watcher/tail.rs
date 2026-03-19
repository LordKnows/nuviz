use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::data::metrics::MetricRecord;

/// Incrementally reads new lines from a JSONL file.
///
/// Tracks the file position so that subsequent calls to `read_new()`
/// return only lines appended since the last read.
pub struct TailReader {
    path: PathBuf,
    position: u64,
}

impl TailReader {
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            position: 0,
        }
    }

    /// Create a TailReader starting from the end of the current file.
    #[allow(dead_code)]
    pub fn from_end(path: &Path) -> Self {
        let position = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        Self {
            path: path.to_path_buf(),
            position,
        }
    }

    /// Read all new lines since the last read.
    pub fn read_new(&mut self) -> Vec<MetricRecord> {
        let file = match File::open(&self.path) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };

        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);

        // File was truncated — reset position
        if file_len < self.position {
            self.position = 0;
        }

        // No new data
        if file_len == self.position {
            return Vec::new();
        }

        let mut reader = BufReader::new(file);
        if self.position > 0 && reader.seek(SeekFrom::Start(self.position)).is_err() {
            return Vec::new();
        }

        let mut records = Vec::new();
        let mut line = String::new();

        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break, // EOF
                Ok(n) => {
                    self.position += n as u64;
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        if let Ok(record) = serde_json::from_str::<MetricRecord>(trimmed) {
                            records.push(record);
                        }
                    }
                }
                Err(_) => break,
            }
        }

        records
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_read_new_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metrics.jsonl");

        // Write initial data
        {
            let mut f = File::create(&path).unwrap();
            writeln!(
                f,
                r#"{{"step":0,"timestamp":1.0,"metrics":{{"loss":1.0}}}}"#
            )
            .unwrap();
        }

        let mut reader = TailReader::new(&path);
        let records = reader.read_new();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].step, 0);

        // Read again — no new data
        let records = reader.read_new();
        assert!(records.is_empty());

        // Append more data
        {
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            writeln!(
                f,
                r#"{{"step":1,"timestamp":2.0,"metrics":{{"loss":0.5}}}}"#
            )
            .unwrap();
            writeln!(
                f,
                r#"{{"step":2,"timestamp":3.0,"metrics":{{"loss":0.3}}}}"#
            )
            .unwrap();
        }

        let records = reader.read_new();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].step, 1);
        assert_eq!(records[1].step, 2);
    }

    #[test]
    fn test_from_end_skips_existing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metrics.jsonl");

        {
            let mut f = File::create(&path).unwrap();
            writeln!(
                f,
                r#"{{"step":0,"timestamp":1.0,"metrics":{{"loss":1.0}}}}"#
            )
            .unwrap();
        }

        let mut reader = TailReader::from_end(&path);
        let records = reader.read_new();
        assert!(records.is_empty());

        // Append new data
        {
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            writeln!(
                f,
                r#"{{"step":1,"timestamp":2.0,"metrics":{{"loss":0.5}}}}"#
            )
            .unwrap();
        }

        let records = reader.read_new();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].step, 1);
    }

    #[test]
    fn test_missing_file() {
        let mut reader = TailReader::new(Path::new("/nonexistent/metrics.jsonl"));
        let records = reader.read_new();
        assert!(records.is_empty());
    }

    #[test]
    fn test_truncated_file_resets() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metrics.jsonl");

        {
            let mut f = File::create(&path).unwrap();
            writeln!(
                f,
                r#"{{"step":0,"timestamp":1.0,"metrics":{{"loss":1.0}}}}"#
            )
            .unwrap();
            writeln!(
                f,
                r#"{{"step":1,"timestamp":2.0,"metrics":{{"loss":0.5}}}}"#
            )
            .unwrap();
        }

        let mut reader = TailReader::new(&path);
        reader.read_new(); // Read all

        // Truncate file (simulate rotation)
        File::create(&path).unwrap();
        {
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            writeln!(
                f,
                r#"{{"step":0,"timestamp":3.0,"metrics":{{"loss":0.9}}}}"#
            )
            .unwrap();
        }

        let records = reader.read_new();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].step, 0);
    }

    #[test]
    fn test_malformed_lines_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metrics.jsonl");

        {
            let mut f = File::create(&path).unwrap();
            writeln!(
                f,
                r#"{{"step":0,"timestamp":1.0,"metrics":{{"loss":1.0}}}}"#
            )
            .unwrap();
            writeln!(f, "this is garbage").unwrap();
            writeln!(
                f,
                r#"{{"step":2,"timestamp":3.0,"metrics":{{"loss":0.3}}}}"#
            )
            .unwrap();
        }

        let mut reader = TailReader::new(&path);
        let records = reader.read_new();
        assert_eq!(records.len(), 2);
    }
}
