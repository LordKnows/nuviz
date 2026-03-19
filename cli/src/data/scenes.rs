use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// A per-scene metrics record from scenes.jsonl.
#[derive(Debug, Clone, Deserialize)]
pub struct SceneRecord {
    pub scene: String,
    pub metrics: HashMap<String, f64>,
    pub timestamp: f64,
}

/// Read all scene records from a scenes.jsonl file, skipping malformed lines.
pub fn read_scenes(path: &Path) -> Vec<SceneRecord> {
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

        match serde_json::from_str::<SceneRecord>(trimmed) {
            Ok(record) => records.push(record),
            Err(e) => {
                eprintln!("[nuviz] Warning: skipping malformed scene line: {e}");
            }
        }
    }

    records
}

/// Group scene records by scene name, keeping the latest record per scene.
pub fn scenes_by_name(records: &[SceneRecord]) -> HashMap<String, &SceneRecord> {
    let mut map: HashMap<String, &SceneRecord> = HashMap::new();
    for record in records {
        map.entry(record.scene.clone())
            .and_modify(|existing| {
                if record.timestamp > existing.timestamp {
                    *existing = record;
                }
            })
            .or_insert(record);
    }
    map
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
    fn test_read_valid_scenes() {
        let f = write_jsonl(&[
            r#"{"scene":"garden","metrics":{"psnr":27.41,"ssim":0.945},"timestamp":1.0}"#,
            r#"{"scene":"bicycle","metrics":{"psnr":25.12,"ssim":0.912},"timestamp":2.0}"#,
        ]);
        let records = read_scenes(f.path());
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].scene, "garden");
        assert!((records[0].metrics["psnr"] - 27.41).abs() < f64::EPSILON);
    }

    #[test]
    fn test_skip_malformed() {
        let f = write_jsonl(&[
            r#"{"scene":"garden","metrics":{"psnr":27.0},"timestamp":1.0}"#,
            "invalid json",
            r#"{"scene":"stump","metrics":{"psnr":26.0},"timestamp":3.0}"#,
        ]);
        let records = read_scenes(f.path());
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_missing_file() {
        let records = read_scenes(Path::new("/nonexistent/scenes.jsonl"));
        assert!(records.is_empty());
    }

    #[test]
    fn test_scenes_by_name() {
        let records = vec![
            SceneRecord {
                scene: "garden".into(),
                metrics: HashMap::from([("psnr".into(), 25.0)]),
                timestamp: 1.0,
            },
            SceneRecord {
                scene: "garden".into(),
                metrics: HashMap::from([("psnr".into(), 27.0)]),
                timestamp: 2.0,
            },
            SceneRecord {
                scene: "bicycle".into(),
                metrics: HashMap::from([("psnr".into(), 24.0)]),
                timestamp: 1.0,
            },
        ];
        let by_name = scenes_by_name(&records);
        assert_eq!(by_name.len(), 2);
        // Should have the latest garden record
        assert!((by_name["garden"].metrics["psnr"] - 27.0).abs() < f64::EPSILON);
    }
}
