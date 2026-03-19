use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::path::Path;

use crate::cli::ExportArgs;
use crate::data::experiment::{discover_experiments, filter_by_project};
use crate::data::metrics;

pub fn run(args: ExportArgs, base_dir: &Path) -> Result<()> {
    let experiments = discover_experiments(base_dir);
    let exp = find_experiment(&experiments, &args.experiment, args.project.as_deref())
        .context(format!("Experiment '{}' not found", args.experiment))?;

    let records = metrics::read_metrics(&exp.dir.join("metrics.jsonl"));
    if records.is_empty() {
        println!("No metric records found for '{}'", args.experiment);
        return Ok(());
    }

    // Collect all metric names
    let all_metrics: BTreeSet<String> = records
        .iter()
        .flat_map(|r| r.metrics.keys().cloned())
        .collect();

    // Filter to requested metrics if specified
    let metric_names: Vec<String> = if let Some(ref filter) = args.metric {
        let filter_set: BTreeSet<String> = filter.iter().cloned().collect();
        all_metrics
            .into_iter()
            .filter(|n| filter_set.contains(n))
            .collect()
    } else {
        all_metrics.into_iter().collect()
    };

    match args.format.as_str() {
        "json" => print_json(&records, &metric_names),
        _ => print_csv(&records, &metric_names),
    }

    Ok(())
}

fn print_csv(records: &[metrics::MetricRecord], metric_names: &[String]) {
    // Header
    let mut header = vec!["step".to_string(), "timestamp".to_string()];
    header.extend(metric_names.iter().cloned());
    println!("{}", header.join(","));

    // Rows
    for record in records {
        let mut row = vec![record.step.to_string(), format!("{:.6}", record.timestamp)];
        for name in metric_names {
            match record.metrics.get(name) {
                Some(v) if v.is_finite() => row.push(format!("{v:.6}")),
                _ => row.push(String::new()),
            }
        }
        println!("{}", row.join(","));
    }
}

fn print_json(records: &[metrics::MetricRecord], metric_names: &[String]) {
    println!("[");
    for (i, record) in records.iter().enumerate() {
        let mut obj = serde_json::Map::new();
        obj.insert("step".into(), serde_json::Value::from(record.step));
        obj.insert(
            "timestamp".into(),
            serde_json::Value::from(record.timestamp),
        );

        let mut metrics_obj = serde_json::Map::new();
        for name in metric_names {
            if let Some(&v) = record.metrics.get(name) {
                if v.is_finite() {
                    metrics_obj.insert(name.clone(), serde_json::Value::from(v));
                }
            }
        }
        obj.insert("metrics".into(), serde_json::Value::Object(metrics_obj));

        let json = serde_json::to_string(&serde_json::Value::Object(obj)).unwrap_or_default();
        if i < records.len() - 1 {
            println!("  {json},");
        } else {
            println!("  {json}");
        }
    }
    println!("]");
}

fn find_experiment<'a>(
    experiments: &'a [crate::data::experiment::Experiment],
    name: &str,
    project: Option<&str>,
) -> Option<&'a crate::data::experiment::Experiment> {
    if let Some(p) = project {
        let filtered = filter_by_project(experiments, p);
        if let Some(exp) = filtered.iter().find(|e| e.name == name) {
            return experiments
                .iter()
                .find(|e| e.name == exp.name && e.dir == exp.dir);
        }
    }
    experiments.iter().find(|e| e.name == name)
}
