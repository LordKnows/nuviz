use anyhow::Result;
use comfy_table::{modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL, Cell, Color, Table};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use crate::cli::{MatrixArgs, TableFormat};
use crate::commands::common::{load_scope, print_no_experiments};
use crate::data::aggregation::{format_stat, mean_std};
use crate::data::experiment::Experiment;
use crate::data::goal::Goals;
use crate::output::{csv_line, latex_escape};

pub fn run(args: MatrixArgs, base_dir: &Path) -> Result<()> {
    let experiments = load_scope(base_dir, args.project.as_deref());
    if experiments.is_empty() {
        print_no_experiments(base_dir);
        return Ok(());
    }

    let rows_param = &args.rows;
    let cols_param = &args.cols;
    let metric = &args.metric;

    let mut goals = Goals::default();
    for exp in &experiments {
        goals.merge(&exp.meta.metric_goals);
    }
    let goal = goals.get(metric);
    let minimize = goal.is_min();

    // (row_val, col_val) -> one value per run; several runs in a cell
    // (multiple seeds) are shown as mean ± std.
    let mut matrix: HashMap<(String, String), Vec<f64>> = HashMap::new();
    let mut row_values = BTreeSet::new();
    let mut col_values = BTreeSet::new();

    for exp in &experiments {
        let (Some(rv), Some(cv)) = (
            extract_param(exp, rows_param),
            extract_param(exp, cols_param),
        ) else {
            continue;
        };
        if let Some((value, _)) = exp.summary().value(metric, args.value, goal) {
            if value.is_finite() {
                row_values.insert(rv.clone());
                col_values.insert(cv.clone());
                matrix.entry((rv, cv)).or_default().push(value);
            }
        }
    }

    if matrix.is_empty() {
        println!("Could not build matrix. Ensure experiments log '{metric}' and record");
        println!("the parameters '{rows_param}' and '{cols_param}'.");
        println!("\nTip: pass hyperparameters to the logger, e.g.");
        println!("  Logger(name, hparams={{\"{rows_param}\": ..., \"{cols_param}\": ...}})");
        println!("or name experiments with parameter values (e.g., 'lr_1e-4_sh_2').");
        return Ok(());
    }

    let row_values = natural_sorted(row_values);
    let col_values = natural_sorted(col_values);

    match args.format {
        TableFormat::Latex => print_latex(
            &matrix,
            &row_values,
            &col_values,
            rows_param,
            cols_param,
            metric,
            minimize,
        ),
        TableFormat::Markdown => print_markdown(
            &matrix,
            &row_values,
            &col_values,
            rows_param,
            cols_param,
            metric,
            minimize,
        ),
        TableFormat::Csv => print_csv(&matrix, &row_values, &col_values, rows_param, cols_param),
        TableFormat::Table => {
            print_table(
                &matrix,
                &row_values,
                &col_values,
                rows_param,
                cols_param,
                metric,
                minimize,
            );
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
    }

    Ok(())
}

/// Order parameter values the way a person would: numbers numerically
/// (so 64 < 128 and 1e-4 < 0.001), then everything else alphabetically.
fn natural_sorted(values: BTreeSet<String>) -> Vec<String> {
    let mut values: Vec<String> = values.into_iter().collect();
    values.sort_by(|a, b| match (a.parse::<f64>(), b.parse::<f64>()) {
        (Ok(x), Ok(y)) => x.total_cmp(&y).then_with(|| a.cmp(b)),
        (Ok(_), Err(_)) => std::cmp::Ordering::Less,
        (Err(_), Ok(_)) => std::cmp::Ordering::Greater,
        (Err(_), Err(_)) => a.cmp(b),
    });
    values
}

fn extract_param(exp: &Experiment, param: &str) -> Option<String> {
    // Try config JSON first
    if let Some(ref config) = exp.meta.config {
        let keys: Vec<&str> = param.split('.').collect();
        let mut current = config;
        for key in &keys {
            current = current.get(key)?;
        }
        return Some(json_value_to_string(current));
    }

    // Fallback: parse experiment name for param_value pattern
    let name = &exp.dir_name;
    let param_short = param.split('.').next_back().unwrap_or(param);

    // Look for patterns like "lr_1e-4", "lr1e-4" or "lr-1e-4". The value
    // ends at `_`, or at a `-` that starts the next word ("lr_0.001-sh_2"),
    // but not at the `-` inside a number such as "1e-4".
    if let Some(pos) = name.find(param_short) {
        let after = &name[pos + param_short.len()..];
        let rest: Vec<char> = after.trim_start_matches(['_', '-', '=']).chars().collect();
        let end = (0..rest.len())
            .find(|&i| {
                rest[i] == '_'
                    || rest[i] == ' '
                    || (rest[i] == '-' && rest.get(i + 1).is_some_and(|c| c.is_alphabetic()))
            })
            .unwrap_or(rest.len());
        let value: String = rest[..end].iter().collect();
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
    match mean_std(values) {
        Some(stat) => (stat.mean, format_stat(&stat, 4)),
        None => (f64::NAN, "-".into()),
    }
}

fn print_table(
    matrix: &HashMap<(String, String), Vec<f64>>,
    row_values: &[String],
    col_values: &[String],
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
    row_values: &[String],
    col_values: &[String],
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
    println!(
        "\\caption{{Ablation matrix: {} {arrow}.}}",
        latex_escape(metric)
    );
    println!("\\begin{{tabular}}{{{cols}}}");
    println!("\\toprule");

    let mut header = vec![latex_escape(rows_param)];
    for cv in col_values {
        header.push(latex_escape(cv));
    }
    println!("{} \\\\", header.join(" & "));
    println!("\\midrule");

    for rv in row_values {
        let mut row = vec![latex_escape(rv)];
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
    row_values: &[String],
    col_values: &[String],
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
    row_values: &[String],
    col_values: &[String],
    rows_param: &str,
    _cols_param: &str,
) {
    let mut header = vec![rows_param.to_string()];
    for cv in col_values {
        header.push(cv.clone());
    }
    println!("{}", csv_line(&header));

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
        println!("{}", csv_line(&row));
    }
}

fn print_findings(
    matrix: &HashMap<(String, String), Vec<f64>>,
    row_values: &[String],
    col_values: &[String],
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
    row_values: &[String],
    col_values: &[String],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_natural_sort_orders_numbers_numerically() {
        let values: BTreeSet<String> = ["128", "64", "1e-4", "0.001", "true", "adam", "8"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            natural_sorted(values),
            ["1e-4", "0.001", "8", "64", "128", "adam", "true"]
        );
    }

    #[test]
    fn test_param_from_name_when_no_config_was_logged() {
        use crate::data::meta::ExperimentMeta;
        use crate::data::status::Status;
        let exp = |name: &str| Experiment {
            name: name.into(),
            dir_name: name.into(),
            project: None,
            dir: name.into(),
            status: Status::Done,
            meta: ExperimentMeta::default(),
            started: None,
            heartbeat_step: None,
        };
        let e = exp("demo-lr_0.001-sh_2_seed0");
        assert_eq!(extract_param(&e, "lr").as_deref(), Some("0.001"));
        assert_eq!(extract_param(&e, "sh").as_deref(), Some("2"));
        assert_eq!(
            extract_param(&exp("lr_1e-4_d8"), "lr").as_deref(),
            Some("1e-4")
        );
        assert_eq!(extract_param(&exp("baseline"), "lr"), None);
    }

    #[test]
    fn test_json_value_to_string() {
        assert_eq!(json_value_to_string(&serde_json::json!(0.001)), "0.001");
        assert_eq!(json_value_to_string(&serde_json::json!(64)), "64");
        assert_eq!(json_value_to_string(&serde_json::json!("adam")), "adam");
        assert_eq!(json_value_to_string(&serde_json::json!(true)), "true");
    }

    #[test]
    fn test_cell_value_aggregates_seeds() {
        assert_eq!(cell_value(&[0.5]).1, "0.5000");
        assert_eq!(cell_value(&[1.0, 3.0]).1, "2.0000±1.4142");
    }
}
