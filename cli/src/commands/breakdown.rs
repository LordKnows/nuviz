use anyhow::{Context, Result};
use comfy_table::{modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL, Cell, Color, Table};
use std::collections::BTreeSet;
use std::path::Path;

use crate::cli::BreakdownArgs;
use crate::data::experiment::{discover_experiments, filter_by_project, Experiment};
use crate::data::metrics::is_minimize_metric;
use crate::data::scenes::{read_scenes, scenes_by_name, SceneRecord};

pub fn run(args: BreakdownArgs, base_dir: &Path) -> Result<()> {
    let experiments = discover_experiments(base_dir);

    let exp = find_experiment(&experiments, &args.experiment, args.project.as_deref())
        .context(format!("Experiment '{}' not found", args.experiment))?;

    let scenes_path = exp.dir.join("scenes.jsonl");
    let records = read_scenes(&scenes_path);
    if records.is_empty() {
        println!(
            "No scene data found for '{}'. Use log.scene() in your training script.",
            args.experiment
        );
        return Ok(());
    }

    let by_name = scenes_by_name(&records);

    // Collect all metric names across scenes
    let metric_names: BTreeSet<String> = records
        .iter()
        .flat_map(|r| r.metrics.keys().cloned())
        .collect();

    if let Some(ref diff_exp_name) = args.diff {
        let exp2 = find_experiment(&experiments, diff_exp_name, args.project.as_deref())
            .context(format!("Experiment '{diff_exp_name}' not found"))?;
        let records2 = read_scenes(&exp2.dir.join("scenes.jsonl"));
        let by_name2 = scenes_by_name(&records2);
        print_diff(
            &args,
            &exp.name,
            &by_name,
            diff_exp_name,
            &by_name2,
            &metric_names,
        );
    } else if args.latex {
        print_latex(&exp.name, &by_name, &metric_names);
    } else if args.markdown {
        print_markdown(&exp.name, &by_name, &metric_names);
    } else {
        print_table(&exp.name, &by_name, &metric_names);
    }

    Ok(())
}

fn print_table(
    exp_name: &str,
    by_name: &std::collections::HashMap<String, &SceneRecord>,
    metric_names: &BTreeSet<String>,
) {
    println!("{exp_name}: Per-Scene Breakdown\n");

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS);

    let mut header = vec!["Scene".to_string()];
    for name in metric_names {
        let arrow = if is_minimize_metric(name) {
            " ↓"
        } else {
            " ↑"
        };
        header.push(format!("{name}{arrow}"));
    }
    table.set_header(header);

    // Find best per metric for highlighting
    let best = find_best_per_metric(by_name, metric_names);

    let mut scene_names: Vec<&String> = by_name.keys().collect();
    scene_names.sort();

    let mut sums: std::collections::HashMap<String, (f64, usize)> =
        std::collections::HashMap::new();

    for scene_name in &scene_names {
        let record = by_name[*scene_name];
        let mut row = vec![Cell::new(*scene_name)];

        for name in metric_names {
            let cell = match record.metrics.get(name) {
                Some(&v) => {
                    sums.entry(name.clone()).or_insert((0.0, 0));
                    let entry = sums.get_mut(name).unwrap();
                    entry.0 += v;
                    entry.1 += 1;

                    let formatted = format!("{v:.4}");
                    if best.get(name).copied() == Some(v) {
                        Cell::new(formatted).fg(Color::Green)
                    } else {
                        Cell::new(formatted)
                    }
                }
                None => Cell::new("-"),
            };
            row.push(cell);
        }
        table.add_row(row);
    }

    // Mean row
    let mut mean_row = vec![Cell::new("Mean")];
    for name in metric_names {
        if let Some(&(sum, count)) = sums.get(name) {
            mean_row.push(Cell::new(format!("{:.4}", sum / count as f64)));
        } else {
            mean_row.push(Cell::new("-"));
        }
    }
    table.add_row(mean_row);

    println!("{table}");
}

fn print_latex(
    exp_name: &str,
    by_name: &std::collections::HashMap<String, &SceneRecord>,
    metric_names: &BTreeSet<String>,
) {
    let best = find_best_per_metric(by_name, metric_names);
    let second_best = find_second_best_per_metric(by_name, metric_names);

    let cols = "l".to_string() + &"c".repeat(metric_names.len());
    println!("\\begin{{table}}[t]");
    println!("\\centering");
    println!(
        "\\caption{{Per-scene quantitative results for {}.}}",
        exp_name.replace('_', "\\_")
    );
    println!("\\label{{tab:per_scene}}");
    println!("\\begin{{tabular}}{{{cols}}}");
    println!("\\toprule");

    let mut header = vec!["Scene".to_string()];
    for name in metric_names {
        let arrow = if is_minimize_metric(name) {
            "$\\downarrow$"
        } else {
            "$\\uparrow$"
        };
        header.push(format!("{name} {arrow}"));
    }
    println!("{} \\\\", header.join(" & "));
    println!("\\midrule");

    let mut scene_names: Vec<&String> = by_name.keys().collect();
    scene_names.sort();

    let mut sums: std::collections::HashMap<String, (f64, usize)> =
        std::collections::HashMap::new();

    for scene_name in &scene_names {
        let record = by_name[*scene_name];
        let display_name = capitalize(scene_name);
        let mut row = vec![display_name];

        for name in metric_names {
            match record.metrics.get(name) {
                Some(&v) => {
                    sums.entry(name.clone()).or_insert((0.0, 0));
                    let entry = sums.get_mut(name).unwrap();
                    entry.0 += v;
                    entry.1 += 1;

                    let formatted = format!("{v:.4}");
                    if best.get(name).copied() == Some(v) {
                        row.push(format!("\\textbf{{{formatted}}}"));
                    } else if second_best.get(name).copied() == Some(v) {
                        row.push(format!("\\underline{{{formatted}}}"));
                    } else {
                        row.push(formatted);
                    }
                }
                None => row.push("-".into()),
            }
        }
        println!("{} \\\\", row.join(" & "));
    }

    println!("\\midrule");

    // Mean row
    let mut mean_row = vec!["Mean".to_string()];
    for name in metric_names {
        if let Some(&(sum, count)) = sums.get(name) {
            mean_row.push(format!("{:.4}", sum / count as f64));
        } else {
            mean_row.push("-".into());
        }
    }
    println!("{} \\\\", mean_row.join(" & "));

    println!("\\bottomrule");
    println!("\\end{{tabular}}");
    println!("\\end{{table}}");
}

fn print_markdown(
    exp_name: &str,
    by_name: &std::collections::HashMap<String, &SceneRecord>,
    metric_names: &BTreeSet<String>,
) {
    let best = find_best_per_metric(by_name, metric_names);

    println!("### {exp_name}: Per-Scene Breakdown\n");

    let mut header = vec!["Scene".to_string()];
    let mut separator = vec!["---".to_string()];
    for name in metric_names {
        let arrow = if is_minimize_metric(name) {
            " ↓"
        } else {
            " ↑"
        };
        header.push(format!("{name}{arrow}"));
        separator.push("---".to_string());
    }
    println!("| {} |", header.join(" | "));
    println!("| {} |", separator.join(" | "));

    let mut scene_names: Vec<&String> = by_name.keys().collect();
    scene_names.sort();

    let mut sums: std::collections::HashMap<String, (f64, usize)> =
        std::collections::HashMap::new();

    for scene_name in &scene_names {
        let record = by_name[*scene_name];
        let mut row = vec![(*scene_name).clone()];

        for name in metric_names {
            match record.metrics.get(name) {
                Some(&v) => {
                    sums.entry(name.clone()).or_insert((0.0, 0));
                    let entry = sums.get_mut(name).unwrap();
                    entry.0 += v;
                    entry.1 += 1;

                    let formatted = format!("{v:.4}");
                    if best.get(name).copied() == Some(v) {
                        row.push(format!("**{formatted}**"));
                    } else {
                        row.push(formatted);
                    }
                }
                None => row.push("-".into()),
            }
        }
        println!("| {} |", row.join(" | "));
    }

    // Mean row
    let mut mean_row = vec!["**Mean**".to_string()];
    for name in metric_names {
        if let Some(&(sum, count)) = sums.get(name) {
            mean_row.push(format!("{:.4}", sum / count as f64));
        } else {
            mean_row.push("-".into());
        }
    }
    println!("| {} |", mean_row.join(" | "));
}

fn print_diff(
    args: &BreakdownArgs,
    name1: &str,
    by_name1: &std::collections::HashMap<String, &SceneRecord>,
    name2: &str,
    by_name2: &std::collections::HashMap<String, &SceneRecord>,
    metric_names: &BTreeSet<String>,
) {
    println!("Diff: {name1} vs {name2}\n");

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS);

    let mut header = vec!["Scene".to_string()];
    for name in metric_names {
        let arrow = if is_minimize_metric(name) {
            " ↓"
        } else {
            " ↑"
        };
        header.push(format!("{name}{arrow} ({name1})"));
        header.push(format!("{name}{arrow} ({name2})"));
        header.push(format!("Δ{name}"));
    }
    table.set_header(header);

    let all_scenes: BTreeSet<String> = by_name1.keys().chain(by_name2.keys()).cloned().collect();

    for scene_name in &all_scenes {
        let mut row = vec![Cell::new(scene_name)];

        for name in metric_names {
            let v1 = by_name1
                .get(scene_name)
                .and_then(|r| r.metrics.get(name).copied());
            let v2 = by_name2
                .get(scene_name)
                .and_then(|r| r.metrics.get(name).copied());

            row.push(Cell::new(
                v1.map(|v| format!("{v:.4}")).unwrap_or("-".into()),
            ));
            row.push(Cell::new(
                v2.map(|v| format!("{v:.4}")).unwrap_or("-".into()),
            ));

            match (v1, v2) {
                (Some(a), Some(b)) => {
                    let delta = a - b;
                    let formatted = format!("{delta:+.4}");
                    let minimize = is_minimize_metric(name);
                    let color = if (minimize && delta < 0.0) || (!minimize && delta > 0.0) {
                        Color::Green
                    } else if delta.abs() < f64::EPSILON {
                        Color::White
                    } else {
                        Color::Red
                    };
                    row.push(Cell::new(formatted).fg(color));
                }
                _ => row.push(Cell::new("-")),
            }
        }
        table.add_row(row);
    }

    let _ = args; // suppress unused warning
    println!("{table}");
}

fn find_best_per_metric(
    by_name: &std::collections::HashMap<String, &SceneRecord>,
    metric_names: &BTreeSet<String>,
) -> std::collections::HashMap<String, f64> {
    let mut best = std::collections::HashMap::new();
    for name in metric_names {
        let minimize = is_minimize_metric(name);
        let mut best_val: Option<f64> = None;
        for record in by_name.values() {
            if let Some(&v) = record.metrics.get(name) {
                if !v.is_finite() {
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
    by_name: &std::collections::HashMap<String, &SceneRecord>,
    metric_names: &BTreeSet<String>,
) -> std::collections::HashMap<String, f64> {
    let mut second = std::collections::HashMap::new();
    let best = find_best_per_metric(by_name, metric_names);

    for name in metric_names {
        let minimize = is_minimize_metric(name);
        let best_val = best.get(name).copied();
        let mut second_val: Option<f64> = None;

        for record in by_name.values() {
            if let Some(&v) = record.metrics.get(name) {
                if !v.is_finite() {
                    continue;
                }
                if best_val == Some(v) {
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

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
    }
}

fn find_experiment<'a>(
    experiments: &'a [Experiment],
    name: &str,
    project: Option<&str>,
) -> Option<&'a Experiment> {
    if let Some(p) = project {
        let filtered = filter_by_project(experiments, p);
        if let Some(exp) = filtered.iter().find(|e| e.name == name) {
            // Find the matching experiment in the original slice to return a reference with 'a lifetime
            return experiments
                .iter()
                .find(|e| e.name == exp.name && e.dir == exp.dir);
        }
    }
    experiments.iter().find(|e| e.name == name)
}
