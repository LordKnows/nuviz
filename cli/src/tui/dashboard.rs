//! The live dashboard behind `nuviz watch` and `nuviz compare`.
//!
//! Both commands show the same thing — metric curves for one or more
//! experiments plus run status — and differ only in their starting layout:
//! `watch` opens with two charts, `compare` with one large chart.

use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Clear, Paragraph, Row, Table},
    Frame,
};

use crate::data::goal::{Goal, Goals};
use crate::data::status::Status;
use crate::data::store::XAxis;
use crate::tui::chart::{self, series_color, ChartSpec, XFormat};
use crate::tui::terminal::TerminalGuard;
use crate::watcher::signal::FsSignal;
use crate::watcher::source::Source;

/// EMA factors cycled by the `s` key.
const SMOOTHING_LEVELS: [f64; 4] = [0.0, 0.6, 0.9, 0.97];
const MAX_ZOOM: f64 = 1024.0;
/// Poll sources at least this often, even without filesystem notifications.
const POLL_INTERVAL: Duration = Duration::from_millis(500);
const INPUT_TIMEOUT: Duration = Duration::from_millis(100);
/// Width of a "last (best@step)" column in the runs table.
const METRIC_COLUMN: u16 = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutMode {
    /// Two charts side by side.
    Dual,
    /// One large chart.
    Single,
}

#[derive(Debug, Clone, Default)]
struct Panel {
    metric: Option<String>,
    log_y: bool,
    /// The user (or `--metric`) chose this metric; don't auto-replace it.
    pinned: bool,
}

pub struct Dashboard {
    pub sources: Vec<Source>,
    metrics: Vec<String>,
    panels: [Panel; 2],
    focus: usize,
    layout: LayoutMode,
    x_axis: XAxis,
    smoothing: usize,
    zoom: f64,
    /// 0 = window ends at the newest data, 1 = window starts at the oldest.
    pan: f64,
    show_help: bool,
    dirty: bool,
    pub should_quit: bool,
}

impl Dashboard {
    pub fn new(
        sources: Vec<Source>,
        layout: LayoutMode,
        x_axis: XAxis,
        metric: Option<String>,
    ) -> Self {
        let mut panels = [Panel::default(), Panel::default()];
        if let Some(metric) = metric {
            panels[0] = Panel {
                metric: Some(metric),
                log_y: false,
                pinned: true,
            };
        }
        Self {
            sources,
            metrics: Vec::new(),
            panels,
            focus: 0,
            layout,
            x_axis,
            smoothing: 0,
            zoom: 1.0,
            pan: 0.0,
            show_help: false,
            dirty: true,
            should_quit: false,
        }
    }

    /// Pull new data from every source. Returns `true` if anything changed.
    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        for source in &mut self.sources {
            changed |= source.poll();
        }
        if changed {
            self.refresh_metrics();
            self.dirty = true;
        }
        changed
    }

    /// Whether a redraw is needed; clears the flag.
    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    fn goals(&self) -> Goals {
        let mut goals = Goals::default();
        for source in &self.sources {
            goals.merge(&source.exp.meta.metric_goals);
        }
        goals
    }

    fn visible_panels(&self) -> usize {
        match self.layout {
            LayoutMode::Dual => 2,
            LayoutMode::Single => 1,
        }
    }

    /// Rebuild the metric list and give panels without a metric a sensible one:
    /// something to minimize (ideally `loss`) on the left, something to
    /// maximize on the right.
    fn refresh_metrics(&mut self) {
        let mut names: Vec<String> = self
            .sources
            .iter()
            .flat_map(|s| s.store.metric_names())
            .map(String::from)
            .collect();
        names.sort();
        names.dedup();
        self.metrics = names;
        if self.metrics.is_empty() {
            return;
        }

        let goals = self.goals();
        let taken = |panels: &[Panel; 2], name: &str| {
            panels.iter().any(|p| p.metric.as_deref() == Some(name))
        };

        if self.panels[0].metric.is_none() {
            let pick = self
                .metrics
                .iter()
                .find(|m| m.as_str() == "loss" && !taken(&self.panels, m))
                .or_else(|| {
                    self.metrics
                        .iter()
                        .find(|m| goals.get(m) == Goal::Min && !taken(&self.panels, m))
                })
                .or_else(|| self.metrics.iter().find(|m| !taken(&self.panels, m)))
                .cloned();
            self.panels[0].metric = pick;
        }
        if self.panels[1].metric.is_none() {
            // Schedules and counters are rarely what one wants to watch.
            let bookkeeping = |m: &str| {
                matches!(
                    m.to_lowercase().as_str(),
                    "lr" | "learning_rate" | "epoch" | "step" | "iteration" | "iter" | "time"
                )
            };
            let pick = self
                .metrics
                .iter()
                .find(|m| goals.get(m) == Goal::Max && !bookkeeping(m) && !taken(&self.panels, m))
                .or_else(|| {
                    self.metrics
                        .iter()
                        .find(|m| !bookkeeping(m) && !taken(&self.panels, m))
                })
                .or_else(|| self.metrics.iter().find(|m| !taken(&self.panels, m)))
                .cloned();
            self.panels[1].metric = pick;
        }
    }

    fn cycle_metric(&mut self, forward: bool) {
        if self.metrics.is_empty() {
            return;
        }
        let n = self.metrics.len();
        let panel = &mut self.panels[self.focus];
        let current = panel
            .metric
            .as_ref()
            .and_then(|m| self.metrics.iter().position(|x| x == m));
        let next = match (current, forward) {
            (Some(i), true) => (i + 1) % n,
            (Some(i), false) => (i + n - 1) % n,
            (None, _) => 0,
        };
        panel.metric = Some(self.metrics[next].clone());
        panel.pinned = true;
    }

    pub fn handle_key(&mut self, key: KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }
        self.dirty = true;

        if self.show_help {
            self.show_help = false;
            if !matches!(key.code, KeyCode::Char('q')) {
                return;
            }
        }

        let ctrl_c =
            key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            _ if ctrl_c => self.should_quit = true,
            KeyCode::Char('?') | KeyCode::Char('h') => self.show_help = true,
            KeyCode::Tab | KeyCode::BackTab => {
                self.focus = (self.focus + 1) % self.visible_panels();
            }
            KeyCode::Char('m') | KeyCode::Down => self.cycle_metric(true),
            KeyCode::Char('M') | KeyCode::Up => self.cycle_metric(false),
            KeyCode::Char('a') => {
                self.x_axis = match self.x_axis {
                    XAxis::Step => XAxis::WallTime,
                    XAxis::WallTime => XAxis::Step,
                };
            }
            KeyCode::Char('l') => {
                let panel = &mut self.panels[self.focus];
                panel.log_y = !panel.log_y;
            }
            KeyCode::Char('s') => self.smoothing = (self.smoothing + 1) % SMOOTHING_LEVELS.len(),
            KeyCode::Char('S') => {
                self.smoothing =
                    (self.smoothing + SMOOTHING_LEVELS.len() - 1) % SMOOTHING_LEVELS.len();
            }
            KeyCode::Char('+') | KeyCode::Char('=') | KeyCode::Char(']') => {
                self.zoom = (self.zoom * 2.0).min(MAX_ZOOM);
            }
            KeyCode::Char('-') | KeyCode::Char('_') | KeyCode::Char('[') => {
                self.zoom = (self.zoom / 2.0).max(1.0);
                if self.zoom == 1.0 {
                    self.pan = 0.0;
                }
            }
            // Pan by a quarter of the visible window.
            KeyCode::Left => self.pan = (self.pan + self.pan_step()).min(1.0),
            KeyCode::Right => self.pan = (self.pan - self.pan_step()).max(0.0),
            KeyCode::Char('r') | KeyCode::Char('0') => {
                self.zoom = 1.0;
                self.pan = 0.0;
            }
            KeyCode::Char('v') => {
                self.layout = match self.layout {
                    LayoutMode::Dual => LayoutMode::Single,
                    LayoutMode::Single => LayoutMode::Dual,
                };
                self.focus = self.focus.min(self.visible_panels() - 1);
            }
            _ => {}
        }
    }

    fn label_width(&self) -> u16 {
        self.sources
            .iter()
            .map(|s| s.label.chars().count())
            .max()
            .unwrap_or(3)
            .clamp(3, 28) as u16
    }

    fn runs_table_width(&self) -> u16 {
        let fixed = 2 + self.label_width() + 7 + 15 + 9 + 7;
        let metrics = self.visible_panels() as u16 * METRIC_COLUMN;
        let columns = 6 + self.visible_panels() as u16;
        fixed + metrics + (columns - 1) + 2 // column spacing + borders
    }

    fn pan_step(&self) -> f64 {
        if self.zoom <= 1.0 {
            0.0
        } else {
            0.25 / (self.zoom - 1.0)
        }
    }

    /// The x interval currently shown, after zoom and pan.
    fn x_window(&self) -> Option<(f64, f64)> {
        let (lo, hi) = self
            .sources
            .iter()
            .filter_map(|s| s.store.x_bounds(self.x_axis))
            .reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)))?;
        let full = hi - lo;
        let visible = full / self.zoom;
        let right = hi - self.pan * (full - visible);
        Some((right - visible, right))
    }

    // ------------------------------------------------------------ drawing

    pub fn render(&self, frame: &mut Frame) {
        let area = frame.area();
        if area.width < 40 || area.height < 12 {
            frame.render_widget(Paragraph::new("Terminal too small"), area);
            return;
        }

        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(1)])
            .split(area);
        let (body, footer) = (rows[0], rows[1]);

        let bottom_height = (self.sources.len() as u16 + 3).clamp(6, body.height / 2);
        let parts = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(bottom_height)])
            .split(body);

        let chart_areas = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(vec![
                Constraint::Ratio(1, self.visible_panels() as u32);
                self.visible_panels()
            ])
            .split(parts[0]);
        for (i, chart_area) in chart_areas.iter().enumerate() {
            self.render_chart(frame, *chart_area, i);
        }

        // Give the runs table what it needs and the activity panel the rest.
        let runs_width = self
            .runs_table_width()
            .min(parts[1].width.saturating_sub(30));
        let bottom = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(runs_width), Constraint::Min(0)])
            .split(parts[1]);
        self.render_runs(frame, bottom[0]);
        self.render_activity(frame, bottom[1]);

        frame.render_widget(
            Paragraph::new(Span::styled(
                " q quit │ Tab focus │ m/M metric │ a x-axis │ l log │ s smooth │ +/- zoom │ ←/→ pan │ r reset │ v layout │ ? help",
                Style::new().fg(Color::DarkGray),
            )),
            footer,
        );

        if self.show_help {
            self.render_help(frame, area);
        }
    }

    fn render_chart(&self, frame: &mut Frame, area: Rect, index: usize) {
        let panel = &self.panels[index];
        let focused = index == self.focus;
        let border = if focused {
            Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(Color::DarkGray)
        };

        let Some(metric) = panel.metric.as_deref() else {
            let block = Block::default().borders(Borders::ALL).border_style(border);
            frame.render_widget(Paragraph::new("Waiting for data…").block(block), area);
            return;
        };

        let smoothing = SMOOTHING_LEVELS[self.smoothing];
        let mut title = format!(" {metric} {} ", self.goals().get(metric).arrow());
        if panel.log_y {
            title.push_str("· log ");
        }
        if smoothing > 0.0 {
            title.push_str(&format!("· smooth {smoothing} "));
        }
        if self.zoom > 1.0 {
            title.push_str(&format!("· zoom {}× ", self.zoom));
        }
        let x_label = match self.x_axis {
            XAxis::Step => " step ",
            XAxis::WallTime => " time ",
        };
        let block = Block::default()
            .title(title)
            .title_bottom(Line::from(x_label).right_aligned())
            .borders(Borders::ALL)
            .border_style(border);
        let inner = block.inner(area);

        let lines = match self.x_window() {
            Some(x_range) => {
                let series = self
                    .sources
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (series_color(i), s.store.series(metric, self.x_axis)))
                    .collect();
                chart::plot(
                    series,
                    &ChartSpec {
                        width: inner.width as usize,
                        height: inner.height as usize,
                        x_range,
                        x_format: match self.x_axis {
                            XAxis::Step => XFormat::Count,
                            XAxis::WallTime => XFormat::Duration,
                        },
                        log_y: panel.log_y,
                        smoothing,
                    },
                )
            }
            None => vec![Line::from("Waiting for data…")],
        };
        frame.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn render_runs(&self, frame: &mut Frame, area: Rect) {
        let goals = self.goals();
        let shown: Vec<&str> = self.panels[..self.visible_panels()]
            .iter()
            .filter_map(|p| p.metric.as_deref())
            .collect();

        let mut header = vec![
            Cell::from(""),
            Cell::from("Run"),
            Cell::from("Status"),
            Cell::from("Step"),
            Cell::from("Speed"),
            Cell::from("ETA"),
        ];
        let mut widths = vec![
            Constraint::Length(2),
            Constraint::Length(self.label_width()),
            Constraint::Length(7),
            Constraint::Length(15),
            Constraint::Length(9),
            Constraint::Length(7),
        ];
        for metric in &shown {
            header.push(Cell::from(format!("{metric} (best)")));
            widths.push(Constraint::Length(METRIC_COLUMN));
        }

        let rows: Vec<Row> = self
            .sources
            .iter()
            .enumerate()
            .map(|(i, source)| {
                let color = series_color(i);
                let status = source.status();
                let step = match (source.store.last_step(), source.exp.meta.max_steps) {
                    (Some(s), Some(max)) if max > 0 => {
                        format!(
                            "{}/{} {:.0}%",
                            chart::format_count(s as f64),
                            chart::format_count(max as f64),
                            100.0 * s as f64 / max as f64
                        )
                    }
                    (Some(s), _) => s.to_string(),
                    (None, _) => "-".into(),
                };
                let speed = match source.store.steps_per_second(200) {
                    Some(rate) if status.is_running() => format_rate(rate),
                    _ => "-".into(),
                };
                let eta = source
                    .eta_seconds()
                    .map(chart::format_duration)
                    .unwrap_or_else(|| "-".into());

                let mut cells = vec![
                    Cell::from(Span::styled("██", Style::new().fg(color))),
                    Cell::from(Span::styled(
                        source.label.clone(),
                        Style::new().fg(color).add_modifier(Modifier::BOLD),
                    )),
                    Cell::from(Span::styled(status.as_str(), status_style(status))),
                    Cell::from(step),
                    Cell::from(speed),
                    Cell::from(eta),
                ];
                for metric in &shown {
                    cells.push(Cell::from(match source.store.summary(metric) {
                        Some(s) => {
                            let (best, at) = s.best(goals.get(metric));
                            format!(
                                "{} ({}@{})",
                                chart::format_number(s.last),
                                chart::format_number(best),
                                chart::format_count(at as f64)
                            )
                        }
                        None => "-".into(),
                    }));
                }
                Row::new(cells)
            })
            .collect();

        let table = Table::new(rows, widths)
            .header(Row::new(header).style(Style::new().fg(Color::DarkGray)))
            .column_spacing(1)
            .block(
                Block::default()
                    .title(" Runs ")
                    .borders(Borders::ALL)
                    .border_style(Style::new().fg(Color::DarkGray)),
            );
        frame.render_widget(table, area);
    }

    fn render_activity(&self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .title(" GPU / Alerts ")
            .borders(Borders::ALL)
            .border_style(Style::new().fg(Color::DarkGray));
        let capacity = block.inner(area).height as usize;
        let mut lines: Vec<Line> = Vec::new();

        for (i, source) in self.sources.iter().enumerate() {
            let Some(gpu) = source.store.gpu() else {
                continue;
            };
            let mut spans = vec![Span::styled("■ ", Style::new().fg(series_color(i)))];
            if let Some(util) = gpu.util {
                spans.push(Span::raw(format!(
                    "GPU {} {util:.0}%  ",
                    bar(util / 100.0, 8)
                )));
            }
            if let (Some(used), Some(total)) = (gpu.mem_used_mb, gpu.mem_total_mb) {
                let style = if used / total.max(1.0) >= 0.95 {
                    Style::new().fg(Color::Red)
                } else {
                    Style::new()
                };
                spans.push(Span::styled(
                    format!("{:.1}/{:.1}G  ", used / 1024.0, total / 1024.0),
                    style,
                ));
            }
            if let Some(temp) = gpu.temperature_c {
                let style = if temp >= 85.0 {
                    Style::new().fg(Color::Red)
                } else {
                    Style::new()
                };
                spans.push(Span::styled(format!("{temp:.0}°C"), style));
            }
            lines.push(Line::from(spans));
        }

        // Newest alerts first, across all runs.
        let mut alerts: Vec<(usize, &crate::watcher::source::Alert)> = self
            .sources
            .iter()
            .enumerate()
            .flat_map(|(i, s)| s.alerts.iter().map(move |a| (i, a)))
            .collect();
        alerts.sort_by_key(|(_, a)| std::cmp::Reverse(a.step));
        let total_alerts = alerts.len();

        let room = capacity.saturating_sub(lines.len());
        for (i, alert) in alerts.into_iter().take(room) {
            let style = match alert.kind.as_str() {
                "nan" | "inf" => Style::new().fg(Color::Red),
                _ => Style::new().fg(Color::Yellow),
            };
            let text = if self.sources.len() > 1 {
                format!("⚠ {}: {}", self.sources[i].label, alert.message)
            } else {
                format!("⚠ {}", alert.message)
            };
            lines.push(Line::from(Span::styled(text, style)));
        }
        if lines.is_empty() {
            lines.push(Line::from(Span::styled(
                "No GPU data or alerts",
                Style::new().fg(Color::DarkGray),
            )));
        }

        let title = if total_alerts > 0 {
            format!(" GPU / Alerts ({total_alerts}) ")
        } else {
            " GPU / Alerts ".to_string()
        };
        frame.render_widget(Paragraph::new(lines).block(block.title(title)), area);
    }

    fn render_help(&self, frame: &mut Frame, area: Rect) {
        const HELP: [(&str, &str); 11] = [
            ("q, Esc", "quit"),
            ("Tab", "focus the other chart"),
            (
                "m / M, ↓ / ↑",
                "next / previous metric in the focused chart",
            ),
            ("a", "x axis: step ↔ wall time"),
            ("l", "linear ↔ log y axis"),
            ("s / S", "more / less smoothing"),
            ("+ / -", "zoom the x axis in / out"),
            ("← / →", "pan when zoomed in"),
            ("r", "reset zoom and pan"),
            ("v", "one chart ↔ two charts"),
            ("?", "toggle this help"),
        ];
        let width = 62.min(area.width.saturating_sub(4));
        let height = (HELP.len() as u16 + 2).min(area.height.saturating_sub(2));
        let popup = Rect::new(
            area.x + (area.width - width) / 2,
            area.y + (area.height - height) / 2,
            width,
            height,
        );
        let lines: Vec<Line> = HELP
            .iter()
            .map(|(keys, what)| {
                Line::from(vec![
                    Span::styled(format!(" {keys:<14}"), Style::new().fg(Color::Cyan)),
                    Span::raw(*what),
                ])
            })
            .collect();
        frame.render_widget(Clear, popup);
        frame.render_widget(
            Paragraph::new(lines).block(Block::default().title(" Keys ").borders(Borders::ALL)),
            popup,
        );
    }
}

fn status_style(status: Status) -> Style {
    match status {
        Status::Running => Style::new().fg(Color::Yellow),
        Status::Done => Style::new().fg(Color::Green),
        Status::Failed | Status::Crashed => Style::new().fg(Color::Red),
        Status::Unknown => Style::new().fg(Color::DarkGray),
    }
}

fn format_rate(steps_per_second: f64) -> String {
    if steps_per_second >= 100.0 {
        format!("{steps_per_second:.0} it/s")
    } else if steps_per_second >= 1.0 {
        format!("{steps_per_second:.1} it/s")
    } else {
        format!("{:.1} s/it", 1.0 / steps_per_second)
    }
}

/// A text progress bar, e.g. `████░░░░`.
fn bar(fraction: f64, width: usize) -> String {
    let filled = (fraction.clamp(0.0, 1.0) * width as f64).round() as usize;
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

/// Run the dashboard until the user quits.
pub fn run(dashboard: &mut Dashboard, signal: &FsSignal) -> Result<()> {
    let mut guard = TerminalGuard::enter()?;
    let mut last_poll: Option<Instant> = None;

    loop {
        if signal.changed() || last_poll.is_none_or(|t| t.elapsed() >= POLL_INTERVAL) {
            dashboard.poll();
            last_poll = Some(Instant::now());
        }

        // Redraw only when data, input or the window size changed.
        if dashboard.take_dirty() {
            guard.terminal.draw(|frame| dashboard.render(frame))?;
        }

        if event::poll(INPUT_TIMEOUT)? {
            match event::read()? {
                Event::Key(key) => dashboard.handle_key(key),
                Event::Resize(..) => dashboard.mark_dirty(),
                _ => {}
            }
        }

        if dashboard.should_quit {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::experiment::Experiment;
    use ratatui::{backend::TestBackend, Terminal};
    use std::fs;
    use std::path::Path;

    fn write_run(base: &Path, name: &str, meta: &str, steps: u64, gpu: bool) -> Source {
        let dir = base.join(name);
        fs::create_dir_all(&dir).unwrap();
        let mut jsonl = String::new();
        for i in 0..steps {
            let gpu_field = if gpu {
                ",\"gpu\":{\"gpu_util\":87,\"mem_used_mb\":10240,\"mem_total_mb\":24576,\"temperature_c\":72}"
            } else {
                ""
            };
            jsonl.push_str(&format!(
                "{{\"step\":{},\"timestamp\":{}.0,\"metrics\":{{\"loss\":{},\"psnr\":{},\"lr\":0.001}}{gpu_field}}}\n",
                i * 100,
                1_700_000_000 + i * 10,
                1.0 / (i + 1) as f64,
                20.0 + i as f64
            ));
        }
        fs::write(dir.join("metrics.jsonl"), jsonl).unwrap();
        fs::write(dir.join("meta.json"), meta).unwrap();
        Source::new(Experiment::load(&dir, None), name.to_string())
    }

    fn screen(dashboard: &Dashboard, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|f| dashboard.render(f)).unwrap();
        let buffer = terminal.backend().buffer().clone();
        (0..height)
            .map(|y| {
                (0..width)
                    .map(|x| buffer[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn two_runs(dir: &Path) -> Dashboard {
        let a = write_run(
            dir,
            "wide",
            r#"{"status":"done","max_steps":5000}"#,
            50,
            true,
        );
        let b = write_run(dir, "narrow", r#"{"status":"failed"}"#, 30, false);
        fs::write(
            dir.join("narrow").join("events.jsonl"),
            r#"{"timestamp":1.0,"type":"alert","step":2900,"alert_type":"nan","metric":"loss","message":"loss is NaN at step 2900"}"#
                .to_string()
                + "\n",
        )
        .unwrap();
        let mut d = Dashboard::new(vec![a, b], LayoutMode::Dual, XAxis::Step, None);
        d.poll();
        d
    }

    #[test]
    fn test_default_panels_pick_loss_and_a_metric_to_maximize() {
        let dir = tempfile::tempdir().unwrap();
        let d = two_runs(dir.path());
        assert_eq!(d.metrics, ["loss", "lr", "psnr"]);
        assert_eq!(d.panels[0].metric.as_deref(), Some("loss"));
        assert_eq!(
            d.panels[1].metric.as_deref(),
            Some("psnr"),
            "not the lr schedule"
        );
    }

    #[test]
    fn test_render_shows_every_run_status_gpu_and_alerts() {
        let dir = tempfile::tempdir().unwrap();
        let d = two_runs(dir.path());
        let s = screen(&d, 150, 40);

        assert!(s.contains(" loss ↓ "), "{s}");
        assert!(s.contains(" psnr ↑ "), "{s}");
        // Both experiments are listed and plotted, not just the first one.
        assert!(s.contains("wide") && s.contains("narrow"), "{s}");
        assert!(s.contains("done") && s.contains("failed"), "{s}");
        assert!(s.contains("4.9k/5k 98%"), "{s}");
        assert!(
            s.contains("GPU ") && s.contains("87%") && s.contains("72°C"),
            "{s}"
        );
        assert!(s.contains("10.0/24.0G"), "{s}");
        assert!(s.contains("narrow: loss is NaN at step 2900"), "{s}");
        assert!(s.contains("GPU / Alerts (1)"), "{s}");
        // Last and best value of the charted metric: loss ends at its minimum.
        assert!(s.contains("0.0200 (0.0200@4.9k)"), "{s}");
        assert!(s.contains("69.00 (69.00@4.9k)"), "{s}");
        // x axis runs from step 0 to the last step.
        assert!(s.contains("4.9k"), "{s}");
        assert!(
            s.chars().any(|c| ('\u{2801}'..='\u{28ff}').contains(&c)),
            "curves drawn"
        );
    }

    #[test]
    fn test_single_layout_and_requested_metric() {
        let dir = tempfile::tempdir().unwrap();
        let a = write_run(dir.path(), "a", r#"{"status":"done"}"#, 10, false);
        let mut d = Dashboard::new(
            vec![a],
            LayoutMode::Single,
            XAxis::WallTime,
            Some("psnr".into()),
        );
        d.poll();
        let s = screen(&d, 100, 30);
        assert!(s.contains(" psnr ↑ "), "{s}");
        assert!(!s.contains(" loss ↓ "), "{s}");
        assert!(s.contains(" time "), "{s}");
        assert!(s.contains("1m30s"), "x axis in wall time: {s}");
    }

    #[test]
    fn test_keys_change_view_state() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = two_runs(dir.path());
        d.take_dirty();

        d.handle_key(key(KeyCode::Char('m')));
        assert_eq!(d.panels[0].metric.as_deref(), Some("lr"));
        d.handle_key(key(KeyCode::Char('M')));
        assert_eq!(d.panels[0].metric.as_deref(), Some("loss"));
        d.handle_key(key(KeyCode::Up));
        assert_eq!(d.panels[0].metric.as_deref(), Some("psnr"), "wraps around");

        d.handle_key(key(KeyCode::Tab));
        assert_eq!(d.focus, 1);
        d.handle_key(key(KeyCode::Char('l')));
        assert!(d.panels[1].log_y && !d.panels[0].log_y);

        d.handle_key(key(KeyCode::Char('a')));
        assert_eq!(d.x_axis, XAxis::WallTime);
        d.handle_key(key(KeyCode::Char('s')));
        assert_eq!(SMOOTHING_LEVELS[d.smoothing], 0.6);

        d.handle_key(key(KeyCode::Char('v')));
        assert_eq!(d.layout, LayoutMode::Single);
        assert_eq!(d.focus, 0, "focus stays on a visible chart");

        assert!(d.take_dirty(), "input requests a redraw");
        assert!(!d.should_quit);
        d.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(d.should_quit);
    }

    #[test]
    fn test_zoom_and_pan_window() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = two_runs(dir.path());
        assert_eq!(d.x_window(), Some((0.0, 4900.0)));

        d.handle_key(key(KeyCode::Char('+')));
        assert_eq!(
            d.x_window(),
            Some((2450.0, 4900.0)),
            "zoom keeps the newest data"
        );
        d.handle_key(key(KeyCode::Left));
        assert_eq!(d.x_window(), Some((1837.5, 4287.5)));
        for _ in 0..10 {
            d.handle_key(key(KeyCode::Left));
        }
        assert_eq!(
            d.x_window(),
            Some((0.0, 2450.0)),
            "pan stops at the oldest data"
        );
        d.handle_key(key(KeyCode::Char('r')));
        assert_eq!(d.x_window(), Some((0.0, 4900.0)));
        d.handle_key(key(KeyCode::Char('-')));
        assert_eq!(d.zoom, 1.0, "cannot zoom out past the full range");
    }

    #[test]
    fn test_help_overlay_and_small_terminal() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = two_runs(dir.path());
        d.handle_key(key(KeyCode::Char('?')));
        assert!(screen(&d, 120, 40).contains("linear ↔ log y axis"));
        d.handle_key(key(KeyCode::Char('x')));
        assert!(!d.show_help, "any key closes help");
        assert!(screen(&d, 30, 8).contains("Terminal too small"));
    }

    #[test]
    fn test_no_data_yet() {
        let dir = tempfile::tempdir().unwrap();
        let exp_dir = dir.path().join("empty");
        fs::create_dir_all(&exp_dir).unwrap();
        fs::write(exp_dir.join("meta.json"), r#"{"status":"running"}"#).unwrap();
        let source = Source::new(Experiment::load(&exp_dir, None), "empty".into());
        let mut d = Dashboard::new(vec![source], LayoutMode::Dual, XAxis::Step, None);
        d.poll();
        let s = screen(&d, 100, 30);
        assert!(s.contains("Waiting for data"), "{s}");
        assert!(s.contains("empty"), "{s}");
    }

    #[test]
    fn test_redraw_only_when_something_changed() {
        let dir = tempfile::tempdir().unwrap();
        let mut d = two_runs(dir.path());
        assert!(d.take_dirty());
        assert!(!d.poll(), "no new data");
        assert!(!d.take_dirty(), "nothing changed: no redraw");
    }
}
