use anyhow::Result;
use comfy_table::{modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL, Cell, Color, Table};
use std::collections::BTreeSet;
use std::path::Path;

use crate::cli::LeaderboardArgs;
use crate::data::experiment::{discover_experiments, filter_by_project, Experiment};

pub fn run(args: LeaderboardArgs, base_dir: &Path) -> Result<()> {
    let mut experiments = discover_experiments(base_dir);

    if let Some(ref project) = args.project {
        experiments = filter_by_project(&experiments, project);
    }

    if experiments.is_empty() {
        println!("No experiments found in {}", base_dir.display());
        return Ok(());
    }

    // Collect all metric names across experiments
    let metric_names: BTreeSet<String> = experiments
        .iter()
        .flat_map(|e| e.best_metrics.keys().cloned())
        .collect();

    // Determine sort metric
    let sort_metric = args
        .sort
        .clone()
        .unwrap_or_else(|| metric_names.iter().next().cloned().unwrap_or_default());

    // Sort experiments by the chosen metric
    experiments.sort_by(|a, b| {
        let va = a
            .best_metrics
            .get(&sort_metric)
            .copied()
            .unwrap_or(f64::NAN);
        let vb = b
            .best_metrics
            .get(&sort_metric)
            .copied()
            .unwrap_or(f64::NAN);

        if is_minimize_metric(&sort_metric) ^ args.asc {
            // minimize + desc => sort ascending (best=lowest first)
            // minimize + asc => sort descending
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
        }
    });

    // Apply --top limit
    if let Some(top) = args.top {
        experiments.truncate(top);
    }

    // Find best value for each metric (for highlighting)
    let best_per_metric = find_best_per_metric(&experiments, &metric_names);
    let second_best_per_metric =
        find_second_best_per_metric(&experiments, &metric_names, &best_per_metric);

    match args.format.as_str() {
        "markdown" => print_markdown(&experiments, &metric_names, &sort_metric, &best_per_metric),
        "latex" => print_latex(
            &experiments,
            &metric_names,
            &sort_metric,
            &best_per_metric,
            &second_best_per_metric,
        ),
        "csv" => print_csv(&experiments, &metric_names),
        _ => print_table(&experiments, &metric_names, &best_per_metric, &sort_metric),
    }

    Ok(())
}

fn print_table(
    experiments: &[Experiment],
    metric_names: &BTreeSet<String>,
    best_per_metric: &std::collections::HashMap<String, f64>,
    sort_metric: &str,
) {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS);

    let mut header = vec!["Rank".to_string(), "Experiment".to_string()];
    for name in metric_names {
        let arrow = if is_minimize_metric(name) {
            " ↓"
        } else {
            " ↑"
        };
        let marker = if name == sort_metric { " *" } else { "" };
        header.push(format!("{name}{arrow}{marker}"));
    }
    header.push("Status".to_string());
    table.set_header(header);

    for (i, exp) in experiments.iter().enumerate() {
        let mut row = vec![Cell::new(i + 1), Cell::new(&exp.name)];

        for name in metric_names {
            let cell = match exp.best_metrics.get(name) {
                Some(&v) => {
                    let formatted = format!("{v:.4}");
                    if best_per_metric.get(name).copied() == Some(v) {
                        Cell::new(formatted).fg(Color::Green)
                    } else {
                        Cell::new(formatted)
                    }
                }
                None => Cell::new("-"),
            };
            row.push(cell);
        }

        let status_cell = match exp.status.as_str() {
            "done" => Cell::new("done").fg(Color::Green),
            "running" => Cell::new("running").fg(Color::Yellow),
            _ => Cell::new(&exp.status),
        };
        row.push(status_cell);

        table.add_row(row);
    }

    println!("{table}");
}

fn print_markdown(
    experiments: &[Experiment],
    metric_names: &BTreeSet<String>,
    _sort_metric: &str,
    best_per_metric: &std::collections::HashMap<String, f64>,
) {
    // Header
    let mut header = vec!["Rank".to_string(), "Experiment".to_string()];
    let mut separator = vec!["---".to_string(), "---".to_string()];
    for name in metric_names {
        let arrow = if is_minimize_metric(name) {
            " ↓"
        } else {
            " ↑"
        };
        header.push(format!("{name}{arrow}"));
        separator.push("---".to_string());
    }
    header.push("Status".to_string());
    separator.push("---".to_string());

    println!("| {} |", header.join(" | "));
    println!("| {} |", separator.join(" | "));

    for (i, exp) in experiments.iter().enumerate() {
        let mut row = vec![format!("{}", i + 1), exp.name.clone()];
        for name in metric_names {
            match exp.best_metrics.get(name) {
                Some(&v) => {
                    let formatted = format!("{v:.4}");
                    if best_per_metric.get(name).copied() == Some(v) {
                        row.push(format!("**{formatted}**"));
                    } else {
                        row.push(formatted);
                    }
                }
                None => row.push("-".into()),
            }
        }
        row.push(exp.status.clone());
        println!("| {} |", row.join(" | "));
    }
}

fn print_latex(
    experiments: &[Experiment],
    metric_names: &BTreeSet<String>,
    _sort_metric: &str,
    best_per_metric: &std::collections::HashMap<String, f64>,
    second_best_per_metric: &std::collections::HashMap<String, f64>,
) {
    let cols = "l".to_string() + &"c".repeat(metric_names.len() + 1);
    println!("\\begin{{table}}[t]");
    println!("\\centering");
    println!("\\caption{{Experiment leaderboard.}}");
    println!("\\begin{{tabular}}{{{cols}}}");
    println!("\\toprule");

    let mut header = vec!["Experiment".to_string()];
    for name in metric_names {
        let arrow = if is_minimize_metric(name) {
            "$\\downarrow$"
        } else {
            "$\\uparrow$"
        };
        header.push(format!("{name} {arrow}"));
    }
    header.push("Status".into());
    println!("{} \\\\", header.join(" & "));
    println!("\\midrule");

    for exp in experiments {
        let mut row = vec![exp.name.replace('_', "\\_")];
        for name in metric_names {
            match exp.best_metrics.get(name) {
                Some(&v) => {
                    let formatted = format!("{v:.4}");
                    if best_per_metric.get(name).copied() == Some(v) {
                        row.push(format!("\\textbf{{{formatted}}}"));
                    } else if second_best_per_metric.get(name).copied() == Some(v) {
                        row.push(format!("\\underline{{{formatted}}}"));
                    } else {
                        row.push(formatted);
                    }
                }
                None => row.push("-".into()),
            }
        }
        row.push(exp.status.clone());
        println!("{} \\\\", row.join(" & "));
    }

    println!("\\bottomrule");
    println!("\\end{{tabular}}");
    println!("\\end{{table}}");
}

fn print_csv(experiments: &[Experiment], metric_names: &BTreeSet<String>) {
    let mut header = vec!["rank".to_string(), "experiment".to_string()];
    for name in metric_names {
        header.push(name.clone());
    }
    header.push("status".to_string());
    println!("{}", header.join(","));

    for (i, exp) in experiments.iter().enumerate() {
        let mut row = vec![format!("{}", i + 1), exp.name.clone()];
        for name in metric_names {
            match exp.best_metrics.get(name) {
                Some(v) => row.push(format!("{v:.6}")),
                None => row.push(String::new()),
            }
        }
        row.push(exp.status.clone());
        println!("{}", row.join(","));
    }
}

fn find_best_per_metric(
    experiments: &[Experiment],
    metric_names: &BTreeSet<String>,
) -> std::collections::HashMap<String, f64> {
    let mut best = std::collections::HashMap::new();

    for name in metric_names {
        let minimize = is_minimize_metric(name);
        let mut best_val: Option<f64> = None;

        for exp in experiments {
            if let Some(&v) = exp.best_metrics.get(name) {
                if v.is_nan() || v.is_infinite() {
                    continue;
                }
                let is_better = match best_val {
                    None => true,
                    Some(current) => {
                        if minimize {
                            v < current
                        } else {
                            v > current
                        }
                    }
                };
                if is_better {
                    best_val = Some(v);
                }
            }
        }

        if let Some(v) = best_val {
            best.insert(name.clone(), v);
        }
    }

    best
}

fn find_second_best_per_metric(
    experiments: &[Experiment],
    metric_names: &BTreeSet<String>,
    best_per_metric: &std::collections::HashMap<String, f64>,
) -> std::collections::HashMap<String, f64> {
    let mut second = std::collections::HashMap::new();

    for name in metric_names {
        let minimize = is_minimize_metric(name);
        let best_val = best_per_metric.get(name).copied();
        let mut second_val: Option<f64> = None;

        for exp in experiments {
            if let Some(&v) = exp.best_metrics.get(name) {
                if v.is_nan() || v.is_infinite() {
                    continue;
                }
                if Some(v) == best_val {
                    continue;
                }
                let is_better = match second_val {
                    None => true,
                    Some(current) => {
                        if minimize {
                            v < current
                        } else {
                            v > current
                        }
                    }
                };
                if is_better {
                    second_val = Some(v);
                }
            }
        }

        if let Some(v) = second_val {
            second.insert(name.clone(), v);
        }
    }

    second
}

fn is_minimize_metric(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("loss")
        || lower.contains("lpips")
        || lower.contains("error")
        || lower.contains("mse")
        || lower.contains("mae")
}
