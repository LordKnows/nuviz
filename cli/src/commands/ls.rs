use anyhow::Result;
use comfy_table::{modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL, Cell, Color, Table};
use std::path::Path;

use crate::cli::LsArgs;
use crate::data::experiment::{discover_experiments, filter_by_project, Experiment};

pub fn run(args: LsArgs, base_dir: &Path) -> Result<()> {
    let mut experiments = discover_experiments(base_dir);

    if let Some(ref project) = args.project {
        experiments = filter_by_project(&experiments, project);
    }

    sort_experiments(&mut experiments, &args.sort);

    if experiments.is_empty() {
        println!("No experiments found in {}", base_dir.display());
        return Ok(());
    }

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS);

    table.set_header(vec![
        "Name",
        "Project",
        "Status",
        "Steps",
        "Best Loss",
        "Started",
    ]);

    for exp in &experiments {
        let status_cell = match exp.status.as_str() {
            "done" => Cell::new("done").fg(Color::Green),
            "running" => Cell::new("running").fg(Color::Yellow),
            _ => Cell::new(&exp.status),
        };

        let steps = exp
            .total_steps
            .map(|s| s.to_string())
            .unwrap_or_else(|| "-".into());

        let best_loss = exp
            .best_metrics
            .get("loss")
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "-".into());

        let started = exp
            .start_time
            .as_deref()
            .and_then(|s| s.split('T').next())
            .unwrap_or("-");

        table.add_row(vec![
            Cell::new(&exp.name),
            Cell::new(exp.project.as_deref().unwrap_or("-")),
            status_cell,
            Cell::new(&steps),
            Cell::new(&best_loss),
            Cell::new(started),
        ]);
    }

    println!("{table}");
    Ok(())
}

fn sort_experiments(experiments: &mut [Experiment], sort_by: &str) {
    match sort_by {
        "name" => experiments.sort_by(|a, b| a.name.cmp(&b.name)),
        "steps" => experiments.sort_by(|a, b| b.total_steps.cmp(&a.total_steps)),
        _ => experiments.sort_by(|a, b| b.start_time.cmp(&a.start_time)),
    }
}
