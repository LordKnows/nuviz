//! Experiment liveness: combining the declared status with the heartbeat.
//!
//! The Python logger sets `status` in `meta.json` when a run starts
//! (`running`) and ends (`done` / `failed`), and refreshes `heartbeat.json`
//! while it is alive. A run that still says `running` but whose heartbeat
//! has stopped was killed without a chance to clean up.

use std::fmt;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Status {
    Running,
    Done,
    Failed,
    /// Declared running, but the process is no longer writing.
    Crashed,
    Unknown,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Running => "running",
            Status::Done => "done",
            Status::Failed => "failed",
            Status::Crashed => "crashed",
            Status::Unknown => "unknown",
        }
    }

    pub fn is_running(self) -> bool {
        self == Status::Running
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Heartbeat {
    pub timestamp: f64,
    #[serde(default)]
    pub step: u64,
    /// Seconds between heartbeats promised by the writer.
    #[serde(default = "default_interval")]
    pub interval: f64,
}

fn default_interval() -> f64 {
    10.0
}

/// A heartbeat this many intervals late (and at least `MIN_GRACE_SECS`)
/// means the process is gone. Generous, to tolerate stalled filesystems and
/// modest clock skew between the training node and the machine running the CLI.
const GRACE_INTERVALS: f64 = 6.0;
const MIN_GRACE_SECS: f64 = 60.0;
/// Without a heartbeat file, fall back to how recently metrics were written.
const ACTIVITY_GRACE_SECS: f64 = 120.0;

pub fn read_heartbeat(dir: &Path) -> Option<Heartbeat> {
    let content = std::fs::read_to_string(dir.join("heartbeat.json")).ok()?;
    serde_json::from_str(&content).ok()
}

pub fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Seconds since the experiment's files were last modified.
pub fn activity_age(dir: &Path, now: f64) -> Option<f64> {
    ["metrics.jsonl", "meta.json"]
        .iter()
        .filter_map(|f| std::fs::metadata(dir.join(f)).ok()?.modified().ok())
        .filter_map(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| now - d.as_secs_f64())
        .min_by(|a, b| a.total_cmp(b))
}

pub fn resolve_status(
    declared: Option<&str>,
    heartbeat: Option<&Heartbeat>,
    activity_age: Option<f64>,
    now: f64,
) -> Status {
    let recently_active = activity_age.is_some_and(|age| age < ACTIVITY_GRACE_SECS);
    match declared {
        Some("done") => Status::Done,
        Some("failed") => Status::Failed,
        Some("crashed") => Status::Crashed,
        Some("running") => match heartbeat {
            Some(hb) => {
                let grace = (hb.interval * GRACE_INTERVALS).max(MIN_GRACE_SECS);
                if now - hb.timestamp <= grace {
                    Status::Running
                } else {
                    Status::Crashed
                }
            }
            None if recently_active => Status::Running,
            None => Status::Crashed,
        },
        // Data written without a status (older logger versions, or no meta.json at all).
        None if recently_active => Status::Running,
        _ => Status::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hb(timestamp: f64, interval: f64) -> Heartbeat {
        Heartbeat {
            timestamp,
            step: 0,
            interval,
        }
    }

    #[test]
    fn test_terminal_states_ignore_heartbeat() {
        assert_eq!(
            resolve_status(Some("done"), Some(&hb(0.0, 10.0)), None, 1e9),
            Status::Done
        );
        assert_eq!(
            resolve_status(Some("failed"), None, None, 1e9),
            Status::Failed
        );
    }

    #[test]
    fn test_running_with_fresh_heartbeat() {
        assert_eq!(
            resolve_status(Some("running"), Some(&hb(1000.0, 10.0)), None, 1030.0),
            Status::Running
        );
        // Reader's clock behind the writer's: still alive.
        assert_eq!(
            resolve_status(Some("running"), Some(&hb(1000.0, 10.0)), None, 990.0),
            Status::Running
        );
    }

    #[test]
    fn test_running_with_stale_heartbeat_is_crashed() {
        assert_eq!(
            resolve_status(Some("running"), Some(&hb(1000.0, 10.0)), Some(1.0), 1061.0),
            Status::Crashed
        );
        // A long declared interval widens the grace period.
        assert_eq!(
            resolve_status(Some("running"), Some(&hb(1000.0, 100.0)), None, 1500.0),
            Status::Running
        );
    }

    #[test]
    fn test_running_without_heartbeat_uses_file_activity() {
        assert_eq!(
            resolve_status(Some("running"), None, Some(5.0), 0.0),
            Status::Running
        );
        assert_eq!(
            resolve_status(Some("running"), None, Some(5000.0), 0.0),
            Status::Crashed
        );
    }

    #[test]
    fn test_no_declared_status() {
        assert_eq!(resolve_status(None, None, Some(5.0), 0.0), Status::Running);
        assert_eq!(
            resolve_status(None, None, Some(5000.0), 0.0),
            Status::Unknown
        );
        assert_eq!(
            resolve_status(Some("weird"), None, Some(1.0), 0.0),
            Status::Unknown
        );
    }
}
