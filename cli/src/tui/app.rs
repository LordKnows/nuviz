use std::collections::HashMap;

use crate::data::metrics::MetricRecord;

/// Panel focus state for keyboard navigation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Panel {
    Chart1,
    Chart2,
    Info,
}

impl Panel {
    pub fn next(self) -> Self {
        match self {
            Panel::Chart1 => Panel::Chart2,
            Panel::Chart2 => Panel::Info,
            Panel::Info => Panel::Chart1,
        }
    }
}

/// Application state for the watch TUI.
pub struct App {
    /// All metric records received so far, per experiment
    pub records: HashMap<String, Vec<MetricRecord>>,
    /// Which experiments we are watching
    pub experiment_names: Vec<String>,
    /// Currently focused panel
    pub focus: Panel,
    /// Time axis zoom level (1.0 = fit all data)
    pub zoom: f64,
    /// Whether the app should quit
    pub should_quit: bool,
    /// Metric names to display in chart panels
    pub chart_metrics: (String, String),
    /// Alert messages
    pub alerts: Vec<String>,
}

impl App {
    pub fn new(experiment_names: Vec<String>) -> Self {
        Self {
            records: HashMap::new(),
            experiment_names,
            focus: Panel::Chart1,
            zoom: 1.0,
            should_quit: false,
            chart_metrics: ("loss".into(), "psnr".into()),
            alerts: Vec::new(),
        }
    }

    /// Add new records for an experiment.
    pub fn push_records(&mut self, experiment: &str, new_records: Vec<MetricRecord>) {
        // Auto-detect chart metrics from first record
        if self.records.is_empty() || self.records.values().all(|v| v.is_empty()) {
            if let Some(first) = new_records.first() {
                let mut names: Vec<&String> = first.metrics.keys().collect();
                names.sort();
                if let Some(name) = names.first() {
                    self.chart_metrics.0 = (*name).clone();
                }
                if let Some(name) = names.get(1) {
                    self.chart_metrics.1 = (*name).clone();
                }
            }
        }

        let entry = self.records.entry(experiment.into()).or_default();
        entry.extend(new_records);
    }

    /// Get metric values for a specific metric across all steps.
    pub fn metric_series(&self, experiment: &str, metric: &str) -> Vec<f64> {
        self.records
            .get(experiment)
            .map(|records| {
                records
                    .iter()
                    .filter_map(|r| r.metrics.get(metric).copied())
                    .filter(|v| v.is_finite())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get current step for an experiment.
    pub fn current_step(&self, experiment: &str) -> Option<u64> {
        self.records
            .get(experiment)
            .and_then(|r| r.last())
            .map(|r| r.step)
    }

    /// Get best value for a metric.
    pub fn best_metric(&self, experiment: &str, metric: &str) -> Option<f64> {
        let series = self.metric_series(experiment, metric);
        if series.is_empty() {
            return None;
        }

        let lower = metric.to_lowercase();
        let minimize = lower.contains("loss") || lower.contains("lpips") || lower.contains("error");

        if minimize {
            series.into_iter().reduce(f64::min)
        } else {
            series.into_iter().reduce(f64::max)
        }
    }

    /// Estimate ETA based on step rate.
    pub fn eta_seconds(&self, experiment: &str, total_steps: u64) -> Option<f64> {
        let records = self.records.get(experiment)?;
        if records.len() < 2 {
            return None;
        }

        let first = records.first()?;
        let last = records.last()?;
        let elapsed = last.timestamp - first.timestamp;
        let steps_done = last.step - first.step;

        if steps_done == 0 || elapsed <= 0.0 {
            return None;
        }

        let steps_remaining = total_steps.saturating_sub(last.step);
        let rate = elapsed / steps_done as f64;
        Some(steps_remaining as f64 * rate)
    }

    pub fn handle_key(&mut self, key: crossterm::event::KeyCode) {
        use crossterm::event::KeyCode;
        match key {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Tab => self.focus = self.focus.next(),
            KeyCode::Char(']') => {
                self.zoom = (self.zoom * 1.5).min(10.0);
            }
            KeyCode::Char('[') => {
                self.zoom = (self.zoom / 1.5).max(0.1);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_record(step: u64, loss: f64, psnr: f64) -> MetricRecord {
        MetricRecord {
            step,
            timestamp: step as f64 * 1.0,
            metrics: HashMap::from([("loss".into(), loss), ("psnr".into(), psnr)]),
            gpu: None,
        }
    }

    #[test]
    fn test_push_records() {
        let mut app = App::new(vec!["exp-1".into()]);
        app.push_records("exp-1", vec![make_record(0, 1.0, 20.0)]);
        assert_eq!(app.records["exp-1"].len(), 1);

        app.push_records("exp-1", vec![make_record(1, 0.5, 25.0)]);
        assert_eq!(app.records["exp-1"].len(), 2);
    }

    #[test]
    fn test_metric_series() {
        let mut app = App::new(vec!["exp".into()]);
        app.push_records(
            "exp",
            vec![
                make_record(0, 1.0, 20.0),
                make_record(1, 0.5, 25.0),
                make_record(2, 0.3, 28.0),
            ],
        );

        let loss = app.metric_series("exp", "loss");
        assert_eq!(loss, vec![1.0, 0.5, 0.3]);

        let psnr = app.metric_series("exp", "psnr");
        assert_eq!(psnr, vec![20.0, 25.0, 28.0]);
    }

    #[test]
    fn test_metric_series_filters_nan() {
        let mut app = App::new(vec!["exp".into()]);
        app.push_records(
            "exp",
            vec![
                make_record(0, 1.0, 20.0),
                MetricRecord {
                    step: 1,
                    timestamp: 1.0,
                    metrics: HashMap::from([("loss".into(), f64::NAN)]),
                    gpu: None,
                },
                make_record(2, 0.5, 25.0),
            ],
        );

        let loss = app.metric_series("exp", "loss");
        assert_eq!(loss, vec![1.0, 0.5]);
    }

    #[test]
    fn test_best_metric_loss_minimized() {
        let mut app = App::new(vec!["exp".into()]);
        app.push_records(
            "exp",
            vec![
                make_record(0, 1.0, 20.0),
                make_record(1, 0.3, 28.0),
                make_record(2, 0.5, 25.0),
            ],
        );
        assert_eq!(app.best_metric("exp", "loss"), Some(0.3));
    }

    #[test]
    fn test_best_metric_psnr_maximized() {
        let mut app = App::new(vec!["exp".into()]);
        app.push_records(
            "exp",
            vec![
                make_record(0, 1.0, 20.0),
                make_record(1, 0.3, 28.0),
                make_record(2, 0.5, 25.0),
            ],
        );
        assert_eq!(app.best_metric("exp", "psnr"), Some(28.0));
    }

    #[test]
    fn test_current_step() {
        let mut app = App::new(vec!["exp".into()]);
        app.push_records(
            "exp",
            vec![make_record(0, 1.0, 20.0), make_record(99, 0.1, 30.0)],
        );
        assert_eq!(app.current_step("exp"), Some(99));
    }

    #[test]
    fn test_eta_seconds() {
        let mut app = App::new(vec!["exp".into()]);
        app.push_records(
            "exp",
            vec![
                MetricRecord {
                    step: 0,
                    timestamp: 0.0,
                    metrics: HashMap::from([("loss".into(), 1.0)]),
                    gpu: None,
                },
                MetricRecord {
                    step: 100,
                    timestamp: 100.0,
                    metrics: HashMap::from([("loss".into(), 0.5)]),
                    gpu: None,
                },
            ],
        );

        let eta = app.eta_seconds("exp", 200).unwrap();
        assert!((eta - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_handle_key_quit() {
        let mut app = App::new(vec![]);
        assert!(!app.should_quit);
        app.handle_key(crossterm::event::KeyCode::Char('q'));
        assert!(app.should_quit);
    }

    #[test]
    fn test_handle_key_tab_cycles() {
        let mut app = App::new(vec![]);
        assert_eq!(app.focus, Panel::Chart1);
        app.handle_key(crossterm::event::KeyCode::Tab);
        assert_eq!(app.focus, Panel::Chart2);
        app.handle_key(crossterm::event::KeyCode::Tab);
        assert_eq!(app.focus, Panel::Info);
        app.handle_key(crossterm::event::KeyCode::Tab);
        assert_eq!(app.focus, Panel::Chart1);
    }

    #[test]
    fn test_handle_key_zoom() {
        let mut app = App::new(vec![]);
        let initial_zoom = app.zoom;
        app.handle_key(crossterm::event::KeyCode::Char(']'));
        assert!(app.zoom > initial_zoom);
        app.handle_key(crossterm::event::KeyCode::Char('['));
        assert!((app.zoom - initial_zoom).abs() < 0.01);
    }
}
