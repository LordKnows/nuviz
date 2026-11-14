use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::data::metrics::{find_rotated_segments, MetricRecord};

/// Incrementally reads new lines from a JSONL file.
///
/// Tracks the file position so that subsequent calls return only lines
/// appended since the last read. Three properties matter for correctness
/// while a training run is writing concurrently:
///
/// * **Complete lines only** — a line is consumed once its trailing newline
///   is on disk. A half-written line is left for the next call instead of
///   being parsed (and lost) early.
/// * **Rotation aware** — when `metrics.jsonl` is renamed to `metrics.1.jsonl`
///   and a new file is started, the unread tail of the old file is drained
///   before reading the new one (detected by inode on Unix).
/// * **Single source of history** — the first read optionally includes the
///   rotated segments, so callers never need a separate "initial load" that
///   could double-count records.
pub struct TailReader {
    path: PathBuf,
    position: u64,
    file_id: Option<u64>,
    history_pending: bool,
}

impl TailReader {
    /// Tail a file from its beginning, including rotated history segments.
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            position: 0,
            file_id: None,
            history_pending: true,
        }
    }

    /// Create a TailReader starting from the end of the current file.
    #[allow(dead_code)]
    pub fn from_end(path: &Path) -> Self {
        let meta = std::fs::metadata(path).ok();
        Self {
            path: path.to_path_buf(),
            position: meta.as_ref().map(|m| m.len()).unwrap_or(0),
            file_id: meta.as_ref().and_then(file_id),
            history_pending: false,
        }
    }

    /// Read all new metric records since the last read, skipping malformed lines.
    pub fn read_new(&mut self) -> Vec<MetricRecord> {
        let mut records = Vec::new();
        self.read_lines(|line| {
            if let Ok(record) = serde_json::from_str::<MetricRecord>(line) {
                records.push(record);
            }
        });
        records
    }

    /// Feed every new complete, non-empty line to `on_line`.
    pub fn read_lines(&mut self, mut on_line: impl FnMut(&str)) {
        if self.history_pending {
            self.history_pending = false;
            // Oldest data lives in the highest-numbered segment.
            for (_, seg) in find_rotated_segments(&self.path).iter().rev() {
                read_from(seg, 0, true, &mut on_line);
            }
        }

        let mut file = match File::open(&self.path) {
            Ok(f) => f,
            Err(_) => return, // not created yet, or mid-rotation
        };
        let meta = match file.metadata() {
            Ok(m) => m,
            Err(_) => return,
        };
        let current_id = file_id(&meta);

        if self.file_id.is_some() && current_id != self.file_id {
            // The file we were tailing was rotated away. Drain what we had not
            // read from it, plus any segments created since.
            self.drain_rotated(&mut on_line);
            self.position = 0;
        } else if meta.len() < self.position {
            // Truncated in place (or rotation on a platform without inodes).
            self.position = 0;
        }
        self.file_id = current_id;

        if meta.len() == self.position {
            return;
        }
        self.position += read_file_from(&mut file, self.position, false, &mut on_line);
    }

    fn drain_rotated(&self, on_line: &mut impl FnMut(&str)) {
        let segments = find_rotated_segments(&self.path);
        let old_index = segments.iter().position(|(_, p)| {
            std::fs::metadata(p)
                .ok()
                .and_then(|m| file_id(&m))
                .is_some_and(|id| Some(id) == self.file_id)
        });
        let Some(old_index) = old_index else {
            return; // old file is gone; nothing more to recover
        };
        read_from(&segments[old_index].1, self.position, true, on_line);
        // Segments with a lower number are newer and entirely unread.
        for (_, seg) in segments[..old_index].iter().rev() {
            read_from(seg, 0, true, on_line);
        }
    }
}

#[cfg(unix)]
fn file_id(meta: &std::fs::Metadata) -> Option<u64> {
    use std::os::unix::fs::MetadataExt;
    Some(meta.ino())
}

#[cfg(not(unix))]
fn file_id(_meta: &std::fs::Metadata) -> Option<u64> {
    None
}

fn read_from(path: &Path, from: u64, whole_file: bool, on_line: &mut impl FnMut(&str)) -> u64 {
    match File::open(path) {
        Ok(mut f) => read_file_from(&mut f, from, whole_file, on_line),
        Err(_) => 0,
    }
}

/// Read lines starting at byte offset `from`. Returns the number of bytes
/// consumed. Unless `whole_file` is set, a trailing line without a newline is
/// left unconsumed.
fn read_file_from(
    file: &mut File,
    from: u64,
    whole_file: bool,
    on_line: &mut impl FnMut(&str),
) -> u64 {
    if file.seek(SeekFrom::Start(from)).is_err() {
        return 0;
    }
    let mut buf = Vec::new();
    if file.read_to_end(&mut buf).is_err() {
        return 0;
    }

    let consumed = if whole_file {
        buf.len()
    } else {
        match buf.iter().rposition(|&b| b == b'\n') {
            Some(last_newline) => last_newline + 1,
            None => return 0,
        }
    };

    for raw in buf[..consumed].split(|&b| b == b'\n') {
        if let Ok(line) = std::str::from_utf8(raw) {
            let line = line.trim();
            if !line.is_empty() {
                on_line(line);
            }
        }
    }

    consumed as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
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

    fn rec(i: u64) -> String {
        format!("{{\"step\":{i},\"timestamp\":{i}.0,\"metrics\":{{\"loss\":1.0}}}}\n")
    }

    #[test]
    fn test_partial_line_is_not_lost() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metrics.jsonl");
        let line = rec(1);
        let (a, b) = line.split_at(20);

        let mut reader = TailReader::new(&path);
        let mut f = File::create(&path).unwrap();
        f.write_all(a.as_bytes()).unwrap();
        f.flush().unwrap();
        assert!(reader.read_new().is_empty());

        f.write_all(b.as_bytes()).unwrap();
        f.flush().unwrap();
        let records = reader.read_new();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].step, 1);
    }

    #[test]
    fn test_null_metric_keeps_record() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metrics.jsonl");
        fs::write(
            &path,
            "{\"step\":3,\"timestamp\":1.0,\"metrics\":{\"loss\":null,\"psnr\":30.0}}\n",
        )
        .unwrap();
        let records = TailReader::new(&path).read_new();
        assert_eq!(records.len(), 1);
        assert!(records[0].metrics["loss"].is_nan());
        assert_eq!(records[0].metrics["psnr"], 30.0);
    }

    #[test]
    fn test_first_read_includes_rotated_history_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metrics.jsonl");
        fs::write(dir.path().join("metrics.2.jsonl"), rec(0) + &rec(1)).unwrap();
        fs::write(dir.path().join("metrics.1.jsonl"), rec(2) + &rec(3)).unwrap();
        fs::write(&path, rec(4)).unwrap();

        let mut reader = TailReader::new(&path);
        let steps: Vec<u64> = reader.read_new().iter().map(|r| r.step).collect();
        assert_eq!(steps, vec![0, 1, 2, 3, 4]);
        assert!(reader.read_new().is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn test_rotation_loses_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metrics.jsonl");
        let mut reader = TailReader::new(&path);

        fs::write(&path, (0..10).map(rec).collect::<String>()).unwrap();
        let mut steps: Vec<u64> = reader.read_new().iter().map(|r| r.step).collect();

        // Writer appends 5 more, rotates, then writes a larger new file
        // before the reader polls again.
        let mut f = fs::OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all((10..15).map(rec).collect::<String>().as_bytes())
            .unwrap();
        drop(f);
        fs::rename(&path, dir.path().join("metrics.1.jsonl")).unwrap();
        fs::write(&path, (15..35).map(rec).collect::<String>()).unwrap();

        steps.extend(reader.read_new().iter().map(|r| r.step));
        assert_eq!(steps, (0..35).collect::<Vec<u64>>());
    }

    #[cfg(unix)]
    #[test]
    fn test_double_rotation_between_polls() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metrics.jsonl");
        let mut reader = TailReader::new(&path);

        fs::write(&path, (0..4).map(rec).collect::<String>()).unwrap();
        let mut steps: Vec<u64> = reader.read_new().iter().map(|r| r.step).collect();

        let mut f = fs::OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(rec(4).as_bytes()).unwrap();
        drop(f);
        fs::rename(&path, dir.path().join("metrics.1.jsonl")).unwrap();
        fs::write(&path, (5..8).map(rec).collect::<String>()).unwrap();
        fs::rename(
            dir.path().join("metrics.1.jsonl"),
            dir.path().join("metrics.2.jsonl"),
        )
        .unwrap();
        fs::rename(&path, dir.path().join("metrics.1.jsonl")).unwrap();
        fs::write(&path, (8..10).map(rec).collect::<String>()).unwrap();

        steps.extend(reader.read_new().iter().map(|r| r.step));
        assert_eq!(steps, (0..10).collect::<Vec<u64>>());
    }
}
