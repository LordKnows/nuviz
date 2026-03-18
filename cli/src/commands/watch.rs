use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    layout::{Constraint, Direction, Layout},
    Terminal,
};
use std::io;
use std::path::Path;
use std::time::Duration;

use crate::cli::WatchArgs;
use crate::data::experiment::{discover_experiments, filter_by_project};
use crate::data::metrics;
use crate::tui::app::{App, Panel};
use crate::tui::widgets::{self, BrailleChart};
use crate::watcher::file_watcher;

pub fn run(args: WatchArgs, base_dir: &Path) -> Result<()> {
    // Resolve which experiments to watch
    let experiment_dirs = resolve_experiments(&args, base_dir)?;

    if experiment_dirs.is_empty() {
        anyhow::bail!("No experiments found to watch");
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

    // Load existing data
    let mut app = App::new(experiment_names.clone());
    for (name, dir) in experiment_names.iter().zip(&experiment_dirs) {
        let jsonl_path = dir.join("metrics.jsonl");
        let records = metrics::read_metrics(&jsonl_path);
        if !records.is_empty() {
            app.push_records(name, records);
        }
    }

    // Set up file watchers
    let watchers: Vec<_> = experiment_dirs
        .iter()
        .map(|dir| {
            let jsonl_path = dir.join("metrics.jsonl");
            file_watcher::watch_metrics(&jsonl_path, args.poll)
        })
        .collect::<Result<Vec<_>, _>>()
        .context("Failed to set up file watchers")?;

    // Set up terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Main event loop
    let result = run_event_loop(&mut terminal, &mut app, &watchers, &experiment_names);

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;

    result
}

fn run_event_loop(
    terminal: &mut Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    app: &mut App,
    watchers: &[std::sync::mpsc::Receiver<Vec<crate::data::metrics::MetricRecord>>],
    experiment_names: &[String],
) -> Result<()> {
    loop {
        // Check for new metrics from watchers
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

            // Layout: two charts on top, info panel at bottom
            let main_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
                .split(size);

            let chart_layout = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(main_layout[0]);

            // Chart 1 (e.g., loss)
            let chart1 = BrailleChart {
                app,
                metric: &app.chart_metrics.0.clone(),
                focused: app.focus == Panel::Chart1,
            };
            frame.render_widget(chart1.render_to_paragraph(chart_layout[0]), chart_layout[0]);

            // Chart 2 (e.g., psnr)
            let chart2 = BrailleChart {
                app,
                metric: &app.chart_metrics.1.clone(),
                focused: app.focus == Panel::Chart2,
            };
            frame.render_widget(chart2.render_to_paragraph(chart_layout[1]), chart_layout[1]);

            // Info panel
            let info = widgets::info_panel(app, app.focus == Panel::Info);
            frame.render_widget(info, main_layout[1]);
        })?;

        // Handle input (with timeout for non-blocking)
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

fn resolve_experiments(
    args: &WatchArgs,
    base_dir: &Path,
) -> Result<Vec<std::path::PathBuf>> {
    if !args.experiments.is_empty() {
        // Explicit experiment names
        let mut dirs = Vec::new();
        let all_experiments = discover_experiments(base_dir);

        for name in &args.experiments {
            if let Some(exp) = all_experiments.iter().find(|e| e.name == *name) {
                dirs.push(exp.dir.clone());
            } else {
                // Try as a direct path under base_dir
                let direct = base_dir.join(name);
                if direct.exists() {
                    dirs.push(direct);
                } else {
                    eprintln!("[nuviz] Warning: experiment '{name}' not found");
                }
            }
        }
        Ok(dirs)
    } else {
        // Use --project + --latest or default to latest 1
        let mut experiments = discover_experiments(base_dir);

        if let Some(ref project) = args.project {
            experiments = filter_by_project(&experiments, project);
        }

        let n = args.latest.unwrap_or(1);
        experiments.truncate(n);

        Ok(experiments.into_iter().map(|e| e.dir).collect())
    }
}
