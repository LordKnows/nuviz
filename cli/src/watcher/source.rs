//! A live view of one experiment: its metrics, alerts and status.

use std::time::{Duration, Instant};

use crate::data::experiment::Experiment;
use crate::data::status::Status;
use crate::data::store::MetricStore;
use crate::watcher::tail::TailReader;

/// How often to re-read `meta.json` / `heartbeat.json` for status changes.
const STATUS_REFRESH: Duration = Duration::from_secs(2);
/// Keep only the most recent alerts.
const MAX_ALERTS: usize = 200;

#[derive(Debug, Clone, PartialEq)]
pub struct Alert {
    pub step: Option<u64>,
    /// `nan`, `inf` or `spike`.
    pub kind: String,
    pub metric: String,
    pub message: String,
}

impl Alert {
    /// Parse one line of `events.jsonl`; `None` unless it is an alert.
    fn from_event_line(line: &str) -> Option<Self> {
        let v: serde_json::Value = serde_json::from_str(line).ok()?;
        if v.get("type")?.as_str()? != "alert" {
            return None;
        }
        let text = |key: &str| {
            v.get(key)
                .and_then(|x| x.as_str())
                .unwrap_or_default()
                .to_string()
        };
        Some(Alert {
            step: v.get("step").and_then(|s| s.as_u64()),
            kind: text("alert_type"),
            metric: text("metric"),
            message: text("message"),
        })
    }
}

pub struct Source {
    pub exp: Experiment,
    /// Name shown in the UI (unique among the watched experiments).
    pub label: String,
    pub store: MetricStore,
    pub alerts: Vec<Alert>,
    metrics_tail: TailReader,
    events_tail: TailReader,
    status_checked: Instant,
}

impl Source {
    pub fn new(exp: Experiment, label: String) -> Self {
        let metrics_tail = TailReader::new(&exp.metrics_path());
        let events_tail = TailReader::new(&exp.dir.join("events.jsonl"));
        Self {
            exp,
            label,
            store: MetricStore::new(),
            alerts: Vec::new(),
            metrics_tail,
            events_tail,
            status_checked: Instant::now(),
        }
    }

    /// Read whatever was appended since the last call and refresh the run
    /// status when due. Returns `true` if anything visible changed.
    pub fn poll(&mut self) -> bool {
        let mut changed = false;

        let store = &mut self.store;
        self.metrics_tail.read_lines(|line| {
            changed |= store.push_line(line);
        });

        let alerts = &mut self.alerts;
        self.events_tail.read_lines(|line| {
            if let Some(alert) = Alert::from_event_line(line) {
                alerts.push(alert);
                changed = true;
            }
        });
        if self.alerts.len() > MAX_ALERTS {
            let excess = self.alerts.len() - MAX_ALERTS;
            self.alerts.drain(..excess);
        }

        if self.status_checked.elapsed() >= STATUS_REFRESH {
            changed |= self.refresh_status();
        }
        changed
    }

    /// Re-read metadata; returns `true` if the status or planned steps changed.
    pub fn refresh_status(&mut self) -> bool {
        self.status_checked = Instant::now();
        let fresh = Experiment::load(&self.exp.dir, self.exp.project.as_deref());
        let changed = fresh.status != self.exp.status
            || fresh.meta.max_steps != self.exp.meta.max_steps
            || fresh.meta.metric_goals != self.exp.meta.metric_goals;
        self.exp = fresh;
        changed
    }

    pub fn status(&self) -> Status {
        self.exp.status
    }

    /// Seconds until `max_steps` at the recent rate, for a running experiment.
    pub fn eta_seconds(&self) -> Option<f64> {
        if !self.exp.status.is_running() {
            return None;
        }
        let max = self.exp.meta.max_steps?;
        let step = self.store.last_step()?;
        let rate = self.store.steps_per_second(200)?;
        Some(max.saturating_sub(step) as f64 / rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    fn rec(step: u64) -> String {
        format!(
            "{{\"step\":{step},\"timestamp\":{}.0,\"metrics\":{{\"loss\":0.5}}}}\n",
            1000 + step * 2
        )
    }

    #[test]
    fn test_poll_reads_history_once_then_only_new_data() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("metrics.1.jsonl"), rec(0) + &rec(1)).unwrap();
        fs::write(dir.path().join("metrics.jsonl"), rec(2)).unwrap();
        fs::write(dir.path().join("meta.json"), r#"{"status":"done"}"#).unwrap();

        let mut source = Source::new(Experiment::load(dir.path(), None), "exp".into());
        assert!(source.poll());
        assert_eq!(source.store.steps(), &[0, 1, 2]);
        assert!(!source.poll(), "nothing new: no change, no duplicates");
        assert_eq!(source.store.len(), 3);

        let mut f = fs::OpenOptions::new()
            .append(true)
            .open(dir.path().join("metrics.jsonl"))
            .unwrap();
        f.write_all(rec(3).as_bytes()).unwrap();
        assert!(source.poll());
        assert_eq!(source.store.steps(), &[0, 1, 2, 3]);
    }

    #[test]
    fn test_alerts_come_from_events_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("metrics.jsonl"), rec(0)).unwrap();
        fs::write(
            dir.path().join("events.jsonl"),
            concat!(
                r#"{"timestamp":1.0,"type":"image","step":1,"tag":"render","path":"images/x.png"}"#,
                "\n",
                r#"{"timestamp":2.0,"type":"alert","step":7,"alert_type":"nan","metric":"loss","value":null,"message":"loss is NaN at step 7"}"#,
                "\n",
                "garbage\n",
            ),
        )
        .unwrap();

        let mut source = Source::new(Experiment::load(dir.path(), None), "exp".into());
        source.poll();
        assert_eq!(
            source.alerts,
            vec![Alert {
                step: Some(7),
                kind: "nan".into(),
                metric: "loss".into(),
                message: "loss is NaN at step 7".into(),
            }]
        );
    }

    #[test]
    fn test_status_refresh_detects_finish_and_eta() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("metrics.jsonl"), rec(0) + &rec(10)).unwrap();
        let now = crate::data::status::now_secs();
        fs::write(
            dir.path().join("meta.json"),
            r#"{"status":"running","max_steps":110}"#,
        )
        .unwrap();
        fs::write(
            dir.path().join("heartbeat.json"),
            format!("{{\"timestamp\":{now},\"step\":10,\"interval\":10.0}}"),
        )
        .unwrap();

        let mut source = Source::new(Experiment::load(dir.path(), None), "exp".into());
        source.poll();
        assert_eq!(source.status(), Status::Running);
        // 10 steps in 20 s = 0.5 it/s; 100 steps to go.
        assert_eq!(source.eta_seconds(), Some(200.0));

        fs::write(dir.path().join("meta.json"), r#"{"status":"done"}"#).unwrap();
        assert!(source.refresh_status());
        assert_eq!(source.status(), Status::Done);
        assert_eq!(source.eta_seconds(), None);
        assert!(!source.refresh_status());
    }
}
