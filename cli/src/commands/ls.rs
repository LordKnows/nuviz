use anyhow::Result;
use comfy_table::{modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL, Cell, Color, Table};
use std::path::Path;

use crate::cli::{LsArgs, LsSort};
use crate::commands::common::{display_labels, load_scope, print_no_experiments};
use crate::data::experiment::{sort_recent_first, Experiment};
use crate::data::status::Status;
use crate::data::summary::{RunSummary, ValueMode};

pub fn run(args: LsArgs, base_dir: &Path) -> Result<()> {
    let mut experiments = load_scope(base_dir, args.project.as_deref());

    if let Some(status) = &args.status {
        experiments.retain(|e| e.status.as_str() == status.to_lowercase());
    }
    if let Some(tag) = &args.tag {
        experiments.retain(|e| e.meta.tags.contains(tag));
    }

    if experiments.is_empty() {
        print_no_experiments(base_dir);
        return Ok(());
    }

    let mut rows: Vec<(Experiment, RunSummary)> = experiments
        .into_iter()
        .map(|e| {
            let summary = e.summary();
            (e, summary)
        })
        .collect();
    sort_rows(&mut rows, args.sort);

    println!("{}", render(&rows, &args.metric));
    Ok(())
}

fn sort_rows(rows: &mut [(Experiment, RunSummary)], sort: LsSort) {
    match sort {
        LsSort::Name => rows.sort_by_key(|a| a.0.qualified_name()),
        LsSort::Steps => rows.sort_by_key(|(e, s)| std::cmp::Reverse(current_step(e, s))),
        LsSort::Date => {
            // Sort the experiments, then carry their summaries along.
            let mut order: Vec<Experiment> = rows.iter().map(|(e, _)| e.clone()).collect();
            sort_recent_first(&mut order);
            rows.sort_by_key(|(e, _)| order.iter().position(|o| o.dir == e.dir));
        }
    }
}

fn current_step(exp: &Experiment, summary: &RunSummary) -> Option<u64> {
    summary
        .last_step
        .or(exp.meta.total_steps)
        .or(exp.heartbeat_step)
}

pub fn status_cell(status: Status) -> Cell {
    let cell = Cell::new(status.as_str());
    match status {
        Status::Done => cell.fg(Color::Green),
        Status::Running => cell.fg(Color::Yellow),
        Status::Failed | Status::Crashed => cell.fg(Color::Red),
        Status::Unknown => cell,
    }
}

fn render(rows: &[(Experiment, RunSummary)], metric: &str) -> Table {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS);

    table.set_header(vec![
        "Name".to_string(),
        "Project".to_string(),
        "Status".to_string(),
        "Steps".to_string(),
        format!("Best {metric}"),
        "Started".to_string(),
        "Tags".to_string(),
    ]);

    let refs: Vec<&Experiment> = rows.iter().map(|(e, _)| e).collect();
    let labels = display_labels(&refs);

    for ((exp, summary), label) in rows.iter().zip(labels) {
        let steps = match (current_step(exp, summary), exp.meta.max_steps) {
            (Some(s), Some(max)) if exp.status.is_running() => format!("{s}/{max}"),
            (Some(s), _) => s.to_string(),
            (None, _) => "-".into(),
        };

        let goal = exp.goals().get(metric);
        let best = summary
            .value(metric, ValueMode::Best, goal)
            .map(|(v, _)| format!("{v:.4}"))
            .unwrap_or_else(|| "-".into());

        table.add_row(vec![
            Cell::new(label),
            Cell::new(exp.project.as_deref().unwrap_or("-")),
            status_cell(exp.status),
            Cell::new(steps),
            Cell::new(best),
            Cell::new(exp.started_date().unwrap_or_else(|| "-".into())),
            Cell::new(exp.meta.tags.join(", ")),
        ]);
    }
    table
}
