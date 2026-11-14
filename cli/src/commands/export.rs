use anyhow::Result;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::cli::{ExportArgs, ExportFormat};
use crate::commands::common::find_experiment;
use crate::data::store::MetricStore;
use crate::output::csv_line;

pub fn run(args: ExportArgs, base_dir: &Path) -> Result<()> {
    let exp = find_experiment(base_dir, &args.experiment, args.project.as_deref())?;

    let store = MetricStore::load(&exp.metrics_path());
    if store.is_empty() {
        eprintln!("No metric records found for '{}'", args.experiment);
        return Ok(());
    }

    let names = select_metrics(&store, args.metric.as_deref());
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let result = match args.format {
        ExportFormat::Json => write_json(&store, &names, &mut out),
        ExportFormat::Csv => write_csv(&store, &names, &mut out),
    }
    .and_then(|()| out.flush());

    match result {
        // `nuviz export ... | head` closes the pipe early; that is not an error.
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        other => Ok(other?),
    }
}

/// Requested metrics in the order asked for, or every metric sorted by name.
fn select_metrics(store: &MetricStore, filter: Option<&[String]>) -> Vec<String> {
    match filter {
        Some(wanted) => wanted
            .iter()
            .filter(|n| store.has_metric(n))
            .cloned()
            .collect(),
        None => store.metric_names().into_iter().map(String::from).collect(),
    }
}

/// Shortest decimal representation that round-trips, so exported values
/// are exactly what was logged.
fn number(v: f64) -> String {
    format!("{v}")
}

fn write_csv(store: &MetricStore, names: &[String], out: &mut impl Write) -> io::Result<()> {
    let mut header = vec!["step".to_string(), "timestamp".to_string()];
    header.extend(names.iter().cloned());
    writeln!(out, "{}", csv_line(&header))?;

    let columns: Vec<&[f64]> = names.iter().filter_map(|n| store.column(n)).collect();
    for row in 0..store.len() {
        write!(out, "{},{:.6}", store.steps()[row], store.timestamps()[row])?;
        for column in &columns {
            let v = column[row];
            if v.is_finite() {
                write!(out, ",{}", number(v))?;
            } else {
                write!(out, ",")?; // missing, NaN or Inf
            }
        }
        writeln!(out)?;
    }
    Ok(())
}

fn write_json(store: &MetricStore, names: &[String], out: &mut impl Write) -> io::Result<()> {
    let columns: Vec<(&String, &[f64])> = names
        .iter()
        .filter_map(|n| Some((n, store.column(n)?)))
        .collect();

    writeln!(out, "[")?;
    for row in 0..store.len() {
        let mut metrics = serde_json::Map::new();
        for (name, column) in &columns {
            if column[row].is_finite() {
                metrics.insert((*name).clone(), serde_json::Value::from(column[row]));
            }
        }
        let record = serde_json::json!({
            "step": store.steps()[row],
            "timestamp": store.timestamps()[row],
            "metrics": metrics,
        });
        let comma = if row + 1 < store.len() { "," } else { "" };
        writeln!(out, "  {record}{comma}")?;
    }
    writeln!(out, "]")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> MetricStore {
        let mut s = MetricStore::new();
        s.push_line(r#"{"step":0,"timestamp":1.5,"metrics":{"loss":1.0,"a,b":0.1}}"#);
        s.push_line(r#"{"step":1,"timestamp":2.5,"metrics":{"loss":null,"psnr":30.123456789}}"#);
        s
    }

    #[test]
    fn test_csv_keeps_rows_with_nan_and_escapes_header() {
        let s = store();
        let mut out = Vec::new();
        write_csv(&s, &select_metrics(&s, None), &mut out).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "step,timestamp,\"a,b\",loss,psnr\n0,1.500000,0.1,1,\n1,2.500000,,,30.123456789\n"
        );
    }

    #[test]
    fn test_metric_filter_keeps_requested_order_and_ignores_unknown() {
        let s = store();
        let wanted = vec!["psnr".to_string(), "nope".to_string(), "loss".to_string()];
        assert_eq!(select_metrics(&s, Some(&wanted)), ["psnr", "loss"]);
    }

    #[test]
    fn test_json_is_valid_and_omits_non_finite() {
        let s = store();
        let mut out = Vec::new();
        write_json(&s, &select_metrics(&s, None), &mut out).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        let rows = parsed.as_array().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["metrics"]["loss"], serde_json::json!(1.0));
        assert!(rows[1]["metrics"].get("loss").is_none());
        assert_eq!(rows[1]["metrics"]["psnr"], serde_json::json!(30.123456789));
    }
}
