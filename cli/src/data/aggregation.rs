use std::collections::HashMap;

use crate::data::experiment::Experiment;

/// Aggregated metrics across multiple experiment runs (mean +/- std).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AggregatedMetrics {
    pub mean: HashMap<String, f64>,
    pub std: HashMap<String, f64>,
    pub count: usize,
}

/// Compute mean and standard deviation of best_metrics across experiments.
#[allow(dead_code)]
pub fn aggregate_experiments(experiments: &[&Experiment]) -> AggregatedMetrics {
    let n = experiments.len();
    if n == 0 {
        return AggregatedMetrics {
            mean: HashMap::new(),
            std: HashMap::new(),
            count: 0,
        };
    }

    // Collect all metric values per name
    let mut values_by_metric: HashMap<String, Vec<f64>> = HashMap::new();
    for exp in experiments {
        for (name, &value) in &exp.best_metrics {
            if value.is_finite() {
                values_by_metric
                    .entry(name.clone())
                    .or_default()
                    .push(value);
            }
        }
    }

    let mut mean = HashMap::new();
    let mut std = HashMap::new();

    for (name, values) in &values_by_metric {
        let count = values.len() as f64;
        let m = values.iter().sum::<f64>() / count;
        mean.insert(name.clone(), m);

        if values.len() > 1 {
            let variance = values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (count - 1.0);
            std.insert(name.clone(), variance.sqrt());
        } else {
            std.insert(name.clone(), 0.0);
        }
    }

    AggregatedMetrics {
        mean,
        std,
        count: n,
    }
}

/// Format a value as "mean +/- std" string.
pub fn format_mean_std(mean: f64, std: f64) -> String {
    if std.abs() < f64::EPSILON {
        format!("{mean:.4}")
    } else {
        format!("{mean:.4}±{std:.4}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_experiment(name: &str, metrics: &[(&str, f64)]) -> Experiment {
        Experiment {
            name: name.into(),
            project: None,
            dir: PathBuf::from("/tmp"),
            status: "done".into(),
            total_steps: Some(100),
            best_metrics: metrics.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            start_time: None,
            end_time: None,
            seed: None,
            config_hash: None,
            config: None,
        }
    }

    #[test]
    fn test_aggregate_single() {
        let exp = make_experiment("e1", &[("psnr", 28.0), ("loss", 0.05)]);
        let result = aggregate_experiments(&[&exp]);
        assert_eq!(result.count, 1);
        assert!((result.mean["psnr"] - 28.0).abs() < f64::EPSILON);
        assert!((result.std["psnr"]).abs() < f64::EPSILON);
    }

    #[test]
    fn test_aggregate_multiple() {
        let e1 = make_experiment("e1", &[("psnr", 28.0)]);
        let e2 = make_experiment("e2", &[("psnr", 30.0)]);
        let e3 = make_experiment("e3", &[("psnr", 29.0)]);
        let result = aggregate_experiments(&[&e1, &e2, &e3]);
        assert_eq!(result.count, 3);
        assert!((result.mean["psnr"] - 29.0).abs() < f64::EPSILON);
        assert!(result.std["psnr"] > 0.0);
    }

    #[test]
    fn test_aggregate_empty() {
        let result = aggregate_experiments(&[]);
        assert_eq!(result.count, 0);
        assert!(result.mean.is_empty());
    }

    #[test]
    fn test_format_mean_std() {
        assert_eq!(format_mean_std(28.0, 0.0), "28.0000");
        assert_eq!(format_mean_std(28.5, 1.2), "28.5000±1.2000");
    }
}
