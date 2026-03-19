use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};
use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::time::Duration;

use crate::cli::CompareArgs;
use crate::data::experiment::{discover_experiments, filter_by_project};
use crate::data::metrics::{self, AlignMode, MetricRecord};
use crate::tui::chart::{self, BrailleCanvas};
use crate::watcher::file_watcher;

/// Fixed color palette for experiments (up to 8).
const EXPERIMENT_COLORS: [Color; 8] = [
    Color::Green,
    Color::Yellow,
    Color::Cyan,
    Color::Magenta,
    Color::Red,
    Color::Blue,
    Color::LightGreen,
    Color::LightYellow,
];

/// Application state for compare TUI.
struct CompareApp {
    records: HashMap<String, Vec<MetricRecord>>,
    experiment_names: Vec<String>,
    metric_names: Vec<String>,
    selected_metric: usize,
    align_mode: AlignMode,
    cursor_pos: usize,
    should_quit: bool,
}

impl CompareApp {
    fn new(experiment_names: Vec<String>, align_mode: AlignMode) -> Self {
        Self {
            records: HashMap::new(),
            experiment_names,
            metric_names: Vec::new(),
            selected_metric: 0,
            align_mode,
            cursor_pos: 0,
            should_quit: false,
        }
    }

    fn push_records(&mut self, experiment: &str, new_records: Vec<MetricRecord>) {
        // Auto-detect metric names from first records
        if self.metric_names.is_empty() {
            if let Some(first) = new_records.first() {
                let mut names: Vec<String> = first.metrics.keys().cloned().collect();
                names.sort();
                self.metric_names = names;
            }
        }
        let entry = self.records.entry(experiment.into()).or_default();
        entry.extend(new_records);
    }

    fn current_metric(&self) -> &str {
        self.metric_names
            .get(self.selected_metric)
            .map(|s| s.as_str())
            .unwrap_or("loss")
    }

    fn handle_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('m') | KeyCode::Tab => {
                if !self.metric_names.is_empty() {
                    self.selected_metric = (self.selected_metric + 1) % self.metric_names.len();
                }
            }
            KeyCode::Char('a') => {
                self.align_mode = match self.align_mode {
                    AlignMode::Step => AlignMode::WallTime,
                    AlignMode::WallTime => AlignMode::Step,
                };
            }
            KeyCode::Left => {
                self.cursor_pos = self.cursor_pos.saturating_sub(1);
            }
            KeyCode::Right => {
                self.cursor_pos = self.cursor_pos.saturating_add(1);
            }
            _ => {}
        }
    }
}

pub fn run(args: CompareArgs, base_dir: &Path) -> Result<()> {
    let experiment_dirs = resolve_experiments(&args, base_dir)?;

    if experiment_dirs.len() < 2 {
        anyhow::bail!(
            "Compare requires at least 2 experiments. Found {}",
            experiment_dirs.len()
        );
    }

    let experiment_names: Vec<String> = experiment_dirs
        .iter()
        .map(|d| {
            d.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string()
        })
        .collect();

    let align_mode = match args.align.as_str() {
        "wall_time" => AlignMode::WallTime,
        _ => AlignMode::Step,
    };

    let mut app = CompareApp::new(experiment_names.clone(), align_mode);

    // Set initial metric if specified
    if let Some(ref metric) = args.metric {
        // Will be adjusted after records are loaded
        app.metric_names = vec![metric.clone()];
    }

    // Load existing data
    for (name, dir) in experiment_names.iter().zip(&experiment_dirs) {
        let records = metrics::read_metrics(&dir.join("metrics.jsonl"));
        if !records.is_empty() {
            app.push_records(name, records);
        }
    }

    // If user specified a metric, find its index
    if let Some(ref metric) = args.metric {
        if let Some(pos) = app.metric_names.iter().position(|m| m == metric) {
            app.selected_metric = pos;
        }
    }

    // Set up file watchers
    let watchers: Vec<_> = experiment_dirs
        .iter()
        .map(|dir| file_watcher::watch_metrics(&dir.join("metrics.jsonl"), args.poll))
        .collect::<Result<Vec<_>, _>>()
        .context("Failed to set up file watchers")?;

    // Set up terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_event_loop(&mut terminal, &mut app, &watchers, &experiment_names);

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;

    result
}

fn run_event_loop(
    terminal: &mut Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    app: &mut CompareApp,
    watchers: &[std::sync::mpsc::Receiver<Vec<MetricRecord>>],
    experiment_names: &[String],
) -> Result<()> {
    loop {
        // Check for new metrics
        for (i, watcher) in watchers.iter().enumerate() {
            while let Ok(records) = watcher.try_recv() {
                if let Some(name) = experiment_names.get(i) {
                    app.push_records(name, records);
                }
            }
        }

        // Draw
        terminal.draw(|frame| {
            let size = frame.area();

            let layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(8), Constraint::Length(10)])
                .split(size);

            // Chart area
            render_compare_chart(frame, app, layout[0]);

            // Legend + info panel
            render_legend(frame, app, layout[1]);
        })?;

        // Handle input
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    app.handle_key(key.code);
                }
            }
        }

        if app.should_quit {
            return Ok(());
        }
    }
}

fn render_compare_chart(frame: &mut ratatui::Frame, app: &CompareApp, area: ratatui::layout::Rect) {
    let metric = app.current_metric();
    let align_label = match app.align_mode {
        AlignMode::Step => "step",
        AlignMode::WallTime => "wall time",
    };
    let title = format!(" {metric} (align: {align_label}) ");

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let inner_width = area.width.saturating_sub(2) as usize;
    let inner_height = area.height.saturating_sub(2) as usize;

    if inner_width < 10 || inner_height < 2 {
        frame.render_widget(Paragraph::new("(too small)").block(block), area);
        return;
    }

    // Collect aligned series for all experiments
    let mut all_series: Vec<(String, Vec<f64>, Vec<f64>)> = Vec::new();
    for name in &app.experiment_names {
        if let Some(records) = app.records.get(name) {
            let (xs, ys) = metrics::align_series(records, metric, app.align_mode);
            if !ys.is_empty() {
                all_series.push((name.clone(), xs, ys));
            }
        }
    }

    if all_series.is_empty() {
        frame.render_widget(Paragraph::new("No data").block(block), area);
        return;
    }

    // Find global Y range
    let all_ys: Vec<f64> = all_series
        .iter()
        .flat_map(|(_, _, ys)| ys.iter().copied())
        .collect();
    let y_min = all_ys.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = all_ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_range = if (y_max - y_min).abs() < f64::EPSILON {
        1.0
    } else {
        y_max - y_min
    };

    // Find global X range
    let all_xs: Vec<f64> = all_series
        .iter()
        .flat_map(|(_, xs, _)| xs.iter().copied())
        .collect();
    let x_min = all_xs.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = all_xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let x_range = if (x_max - x_min).abs() < f64::EPSILON {
        1.0
    } else {
        x_max - x_min
    };

    let label_width = 8;
    let chart_width = inner_width.saturating_sub(label_width);
    if chart_width == 0 {
        frame.render_widget(Paragraph::new("(too narrow)").block(block), area);
        return;
    }

    // Render each series with its own color
    // We render all onto separate canvases and compose the output
    let mut canvas = BrailleCanvas::new(chart_width, inner_height);
    let pw = canvas.pixel_width();
    let ph = canvas.pixel_height();

    // For each series, draw lines on a shared canvas
    for (_, xs, ys) in &all_series {
        let points: Vec<(usize, usize)> = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| {
                let px = if x_range > 0.0 {
                    ((x - x_min) / x_range * (pw - 1) as f64) as usize
                } else {
                    pw / 2
                };
                let py = ((y_max - y) / y_range * (ph - 1) as f64) as usize;
                (px.min(pw - 1), py.min(ph - 1))
            })
            .collect();

        for window in points.windows(2) {
            canvas.draw_line(window[0].0, window[0].1, window[1].0, window[1].1);
        }
        if points.len() == 1 {
            canvas.set_pixel(points[0].0, points[0].1);
        }
    }

    // Render with Y-axis labels
    let rendered = canvas.render();
    let mut lines: Vec<Line> = Vec::new();

    for (i, line) in rendered.iter().enumerate() {
        let y_val = if i == 0 {
            y_max
        } else if i == inner_height - 1 {
            y_min
        } else {
            y_max - (i as f64 / (inner_height - 1) as f64) * y_range
        };
        let y_label = chart::format_number(y_val);
        lines.push(Line::from(format!("{y_label:>label_width$}┤{line}")));
    }

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, area);
}

fn render_legend(frame: &mut ratatui::Frame, app: &CompareApp, area: ratatui::layout::Rect) {
    let block = Block::default()
        .title(" Legend ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    let metric = app.current_metric();
    let minimize = metric.to_lowercase().contains("loss")
        || metric.to_lowercase().contains("lpips")
        || metric.to_lowercase().contains("error")
        || metric.to_lowercase().contains("mse")
        || metric.to_lowercase().contains("mae");

    let mut lines: Vec<Line> = Vec::new();

    for (i, name) in app.experiment_names.iter().enumerate() {
        let color = EXPERIMENT_COLORS[i % EXPERIMENT_COLORS.len()];

        let best = if let Some(records) = app.records.get(name) {
            let (_, ys) = metrics::align_series(records, metric, app.align_mode);
            if minimize {
                ys.iter().copied().fold(f64::INFINITY, f64::min)
            } else {
                ys.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            }
        } else {
            f64::NAN
        };

        let best_str = if best.is_finite() {
            format!("{best:.4}")
        } else {
            "-".into()
        };

        lines.push(Line::from(vec![
            Span::styled("██ ", Style::default().fg(color)),
            Span::styled(
                name.to_string(),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("  best {metric}: {best_str}")),
        ]));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "q:quit  m/Tab:metric  a:align  ←/→:cursor",
        Style::default().fg(Color::DarkGray),
    )));

    let paragraph = Paragraph::new(lines).block(block);
    frame.render_widget(paragraph, area);
}

fn resolve_experiments(args: &CompareArgs, base_dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    if !args.experiments.is_empty() {
        let mut dirs = Vec::new();
        let all_experiments = discover_experiments(base_dir);

        for name in &args.experiments {
            if let Some(exp) = all_experiments.iter().find(|e| e.name == *name) {
                dirs.push(exp.dir.clone());
            } else {
                let direct = base_dir.join(name);
                if direct.exists() {
                    dirs.push(direct);
                } else {
                    eprintln!("[nuviz] Warning: experiment '{name}' not found");
                }
            }
        }
        Ok(dirs)
    } else if let Some(ref project) = args.project {
        let experiments = discover_experiments(base_dir);
        let filtered = filter_by_project(&experiments, project);
        Ok(filtered.into_iter().map(|e| e.dir).collect())
    } else {
        anyhow::bail!("Specify experiment names or use --project to compare all in a project");
    }
}
