//! Mean / standard deviation across runs (multi-seed aggregation).

/// Sample statistics over a set of runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stat {
    pub mean: f64,
    /// Sample standard deviation (n - 1); `0.0` for a single value.
    pub std: f64,
    pub n: usize,
}

/// Mean and sample standard deviation of the finite values.
pub fn mean_std(values: &[f64]) -> Option<Stat> {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let n = finite.len();
    if n == 0 {
        return None;
    }
    let mean = finite.iter().sum::<f64>() / n as f64;
    let std = if n > 1 {
        (finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
    } else {
        0.0
    };
    Some(Stat { mean, std, n })
}

/// Format as `mean±std`, or just the mean for a single run.
pub fn format_stat(stat: &Stat, precision: usize) -> String {
    if stat.n > 1 {
        format!("{:.p$}±{:.p$}", stat.mean, stat.std, p = precision)
    } else {
        format!("{:.p$}", stat.mean, p = precision)
    }
}

/// LaTeX variant of [`format_stat`].
pub fn format_stat_latex(stat: &Stat, precision: usize) -> String {
    if stat.n > 1 {
        format!("{:.p$} $\\pm$ {:.p$}", stat.mean, stat.std, p = precision)
    } else {
        format!("{:.p$}", stat.mean, p = precision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_value() {
        let s = mean_std(&[28.0]).unwrap();
        assert_eq!((s.mean, s.std, s.n), (28.0, 0.0, 1));
        assert_eq!(format_stat(&s, 4), "28.0000");
    }

    #[test]
    fn test_multiple_values() {
        let s = mean_std(&[28.0, 30.0, 29.0]).unwrap();
        assert!((s.mean - 29.0).abs() < 1e-12);
        assert!((s.std - 1.0).abs() < 1e-12);
        assert_eq!(s.n, 3);
        assert_eq!(format_stat(&s, 2), "29.00±1.00");
        assert_eq!(format_stat_latex(&s, 1), "29.0 $\\pm$ 1.0");
    }

    #[test]
    fn test_empty_and_non_finite() {
        assert!(mean_std(&[]).is_none());
        assert!(mean_std(&[f64::NAN]).is_none());
        let s = mean_std(&[1.0, f64::NAN, 3.0]).unwrap();
        assert_eq!((s.mean, s.n), (2.0, 2));
    }
}
