use anyhow::Result;
use comfy_table::{modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL, Cell, Color, Table};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use crate::cli::MatrixArgs;
use crate::data::aggregation::format_mean_std;
use crate::data::experiment::{
    discover_experiments, filter_by_project, group_by_config, Experiment,
};
use crate::data::metrics::is_minimize_metric;

pub fn run(args: MatrixArgs, base_dir: &Path) -> Result<()> {
    let mut experiments = discover_experiments(base_dir);

    if let Some(ref project) = args.project {
        experiments = filter_by_project(&experiments, project);
    }

    if experiments.is_empty() {
        println!("No experiments found in {}", base_dir.display());
        return Ok(());
    }

    // Extract parameter values from experiment configs or names
    let rows_param = &args.rows;
    let cols_param = &args.cols;
    let metric = &args.metric;

    // Build the matrix: (row_val, col_val) -> metric_value or aggregated
    let mut matrix: HashMap<(String, String), Vec<f64>> = HashMap::new();
    let mut row_values = BTreeSet::new();
    let mut col_values = BTreeSet::new();

    // Try to group by config for multi-seed aggregation
    let groups = group_by_config(&experiments);

    for group_exps in groups.values() {
        for exp in group_exps {
            let row_val = extract_param(exp, rows_param);
            let col_val = extract_param(exp, cols_param);

            if let (Some(rv), Some(cv)) = (row_val, col_val) {
                if let Some(&metric_val) = exp.best_metrics.get(metric) {
                    if metric_val.is_finite() {
                        row_values.insert(rv.clone());
                        col_values.insert(cv.clone());
                        matrix.entry((rv, cv)).or_default().push(metric_val);
                    }
                }
            }
        }
    }

    if matrix.is_empty() {
        println!("Could not build matrix. Ensure experiments have config metadata");
        println!("with parameters '{}' and '{}'.", rows_param, cols_param);
        println!("\nTip: Use Ablation.export() to generate configs with parameter metadata,");
        println!("or name experiments with parameter values (e.g., 'lr_1e-4_sh_2').");
        return Ok(());
    }

    let minimize = is_minimize_metric(metric);

    match args.format.as_str() {
        "latex" => print_latex(
            &matrix,
            &row_values,
            &col_values,
            rows_param,
            cols_param,
            metric,
            minimize,
        ),
        "markdown" => print_markdown(
            &matrix,
            &row_values,
            &col_values,
            rows_param,
            cols_param,
            metric,
            minimize,
        ),
        "csv" => print_csv(&matrix, &row_values, &col_values, rows_param, cols_param),
        _ => print_table(
            &matrix,
            &row_values,
            &col_values,
            rows_param,
            cols_param,
            metric,
            minimize,
        ),
    }

    // Print key findings
    if args.format == "table" {
        print_findings(
            &matrix,
            &row_values,
            &col_values,
            rows_param,
            cols_param,
            metric,
            minimize,
        );
    }

    Ok(())
}

fn extract_param(exp: &Experiment, param: &str) -> Option<String> {
    // Try config JSON first
    if let Some(ref config) = exp.config {
        let keys: Vec<&str> = param.split('.').collect();
        let mut current = config;
        for key in &keys {
            current = current.get(key)?;
        }
        return Some(json_value_to_string(current));
    }

    // Fallback: parse experiment name for param_value pattern
    let name = &exp.name;
    let param_short = param.split('.').next_back().unwrap_or(param);

    // Look for patterns like "lr_1e-4" or "lr1e-4" or "lr-1e-4"
    if let Some(pos) = name.find(param_short) {
        let after = &name[pos + param_short.len()..];
        // Skip separator characters
        let value_start = after.trim_start_matches(['_', '-', '=']);
        // Take until next separator
        let value: String = value_start
            .chars()
            .take_while(|c| !matches!(c, '_' | ' '))
            .collect();
        if !value.is_empty() {
            return Some(value);
        }
    }

    None
}

fn json_value_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                format!("{f}")
            } else {
                n.to_string()
            }
        }
        serde_json::Value::Bool(b) => b.to_string(),
        other => other.to_string(),
    }
}

fn cell_value(values: &[f64]) -> (f64, String) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() > 1 {
        let variance =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        let std = variance.sqrt();
        (mean, format_mean_std(mean, std))
    } else {
        (mean, format!("{mean:.4}"))
    }
}

fn print_table(
    matrix: &HashMap<(String, String), Vec<f64>>,
    row_values: &BTreeSet<String>,
    col_values: &BTreeSet<String>,
    rows_param: &str,
    _cols_param: &str,
    metric: &str,
    minimize: bool,
) {
    let arrow = if minimize { "↓" } else { "↑" };
    println!("{metric} {arrow}\n");

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS);

    let mut header = vec![rows_param.to_string()];
    for cv in col_values {
        header.push(cv.clone());
    }
    table.set_header(header);

    // Find global best for highlighting
    let global_best = find_global_best(matrix, minimize);

    for rv in row_values {
        let mut row = vec![Cell::new(rv)];
        for cv in col_values {
            if let Some(values) = matrix.get(&(rv.clone(), cv.clone())) {
                let (mean, display) = cell_value(values);
                let cell = if Some(mean) == global_best {
                    Cell::new(format!("{display} *")).fg(Color::Green)
                } else {
                    Cell::new(display)
                };
                row.push(cell);
            } else {
                row.push(Cell::new("-"));
            }
        }
        table.add_row(row);
    }

    println!("{table}");
}

fn print_latex(
    matrix: &HashMap<(String, String), Vec<f64>>,
    row_values: &BTreeSet<String>,
    col_values: &BTreeSet<String>,
    rows_param: &str,
    _cols_param: &str,
    metric: &str,
    minimize: bool,
) {
    let global_best = find_global_best(matrix, minimize);
    let global_second = find_global_second_best(matrix, minimize, global_best);

    let arrow = if minimize {
        "$\\downarrow$"
    } else {
        "$\\uparrow$"
    };
    let cols = "l".to_string() + &"c".repeat(col_values.len());

    println!("\\begin{{table}}[t]");
    println!("\\centering");
    println!("\\caption{{Ablation matrix: {metric} {arrow}.}}");
    println!("\\begin{{tabular}}{{{cols}}}");
    println!("\\toprule");

    let mut header = vec![rows_param.replace('_', "\\_")];
    for cv in col_values {
        header.push(cv.replace('_', "\\_"));
    }
    println!("{} \\\\", header.join(" & "));
    println!("\\midrule");

    for rv in row_values {
        let mut row = vec![rv.replace('_', "\\_")];
        for cv in col_values {
            if let Some(values) = matrix.get(&(rv.clone(), cv.clone())) {
                let (mean, display) = cell_value(values);
                if Some(mean) == global_best {
                    row.push(format!("\\textbf{{{display}}}"));
                } else if Some(mean) == global_second {
                    row.push(format!("\\underline{{{display}}}"));
                } else {
                    row.push(display);
                }
            } else {
                row.push("-".into());
            }
        }
        println!("{} \\\\", row.join(" & "));
    }

    println!("\\bottomrule");
    println!("\\end{{tabular}}");
    println!("\\end{{table}}");
}

fn print_markdown(
    matrix: &HashMap<(String, String), Vec<f64>>,
    row_values: &BTreeSet<String>,
    col_values: &BTreeSet<String>,
    rows_param: &str,
    _cols_param: &str,
    metric: &str,
    minimize: bool,
) {
    let global_best = find_global_best(matrix, minimize);
    let arrow = if minimize { "↓" } else { "↑" };
    println!("### {metric} {arrow}\n");

    let mut header = vec![rows_param.to_string()];
    let mut separator = vec!["---".to_string()];
    for cv in col_values {
        header.push(cv.clone());
        separator.push("---".to_string());
    }
    println!("| {} |", header.join(" | "));
    println!("| {} |", separator.join(" | "));

    for rv in row_values {
        let mut row = vec![rv.clone()];
        for cv in col_values {
            if let Some(values) = matrix.get(&(rv.clone(), cv.clone())) {
                let (mean, display) = cell_value(values);
                if Some(mean) == global_best {
                    row.push(format!("**{display}**"));
                } else {
                    row.push(display);
                }
            } else {
                row.push("-".into());
            }
        }
        println!("| {} |", row.join(" | "));
    }
}

fn print_csv(
    matrix: &HashMap<(String, String), Vec<f64>>,
    row_values: &BTreeSet<String>,
    col_values: &BTreeSet<String>,
    rows_param: &str,
    _cols_param: &str,
) {
    let mut header = vec![rows_param.to_string()];
    for cv in col_values {
        header.push(cv.clone());
    }
    println!("{}", header.join(","));

    for rv in row_values {
        let mut row = vec![rv.clone()];
        for cv in col_values {
            if let Some(values) = matrix.get(&(rv.clone(), cv.clone())) {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                row.push(format!("{mean:.6}"));
            } else {
                row.push(String::new());
            }
        }
        println!("{}", row.join(","));
    }
}

fn print_findings(
    matrix: &HashMap<(String, String), Vec<f64>>,
    row_values: &BTreeSet<String>,
    col_values: &BTreeSet<String>,
    rows_param: &str,
    cols_param: &str,
    metric: &str,
    minimize: bool,
) {
    println!("\nKey findings:");

    // Compute range per row param and per col param
    let row_range = compute_param_range(matrix, row_values, col_values, true, minimize);
    let col_range = compute_param_range(matrix, row_values, col_values, false, minimize);

    if let Some((param_range, best_val)) = row_range {
        println!("  {rows_param} range: {param_range:.2} {metric} (best: {best_val})");
    }
    if let Some((param_range, best_val)) = col_range {
        println!("  {cols_param} range: {param_range:.2} {metric} (best: {best_val})");
    }
}

fn compute_param_range(
    matrix: &HashMap<(String, String), Vec<f64>>,
    row_values: &BTreeSet<String>,
    col_values: &BTreeSet<String>,
    by_row: bool,
    minimize: bool,
) -> Option<(f64, String)> {
    let outer: Vec<&String> = if by_row {
        row_values.iter().collect()
    } else {
        col_values.iter().collect()
    };
    let inner: Vec<&String> = if by_row {
        col_values.iter().collect()
    } else {
        row_values.iter().collect()
    };

    let mut means: Vec<(String, f64)> = Vec::new();
    for o in &outer {
        let mut vals = Vec::new();
        for i in &inner {
            let key = if by_row {
                ((*o).clone(), (*i).clone())
            } else {
                ((*i).clone(), (*o).clone())
            };
            if let Some(values) = matrix.get(&key) {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                vals.push(mean);
            }
        }
        if !vals.is_empty() {
            let avg = vals.iter().sum::<f64>() / vals.len() as f64;
            means.push(((*o).clone(), avg));
        }
    }

    if means.len() < 2 {
        return None;
    }

    let min = means.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
    let max = means
        .iter()
        .map(|(_, v)| *v)
        .fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    let best = if minimize {
        means.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    } else {
        means.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    };

    best.map(|(name, _)| (range, name.clone()))
}

fn find_global_best(matrix: &HashMap<(String, String), Vec<f64>>, minimize: bool) -> Option<f64> {
    let mut best: Option<f64> = None;
    for values in matrix.values() {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let is_better = match best {
            None => true,
            Some(current) => {
                if minimize {
                    mean < current
                } else {
                    mean > current
                }
            }
        };
        if is_better {
            best = Some(mean);
        }
    }
    best
}

fn find_global_second_best(
    matrix: &HashMap<(String, String), Vec<f64>>,
    minimize: bool,
    best: Option<f64>,
) -> Option<f64> {
    let mut second: Option<f64> = None;
    for values in matrix.values() {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        if Some(mean) == best {
            continue;
        }
        let is_better = match second {
            None => true,
            Some(current) => {
                if minimize {
                    mean < current
                } else {
                    mean > current
                }
            }
        };
        if is_better {
            second = Some(mean);
        }
    }
    second
}
