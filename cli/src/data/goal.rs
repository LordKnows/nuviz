//! Metric direction: is lower or higher better?
//!
//! Explicit goals come from `meta.json` (`metric_goals`, written by
//! `Logger.define_metric()`) and always win. Otherwise a name heuristic is
//! used. The Python logger implements the same heuristic
//! (`python/src/nuviz/goals.py`); both are checked against
//! `spec/fixtures/metric_goals.json`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Goal {
    Min,
    Max,
}

impl Goal {
    pub fn is_min(self) -> bool {
        self == Goal::Min
    }

    /// Unicode arrow for table headers.
    pub fn arrow(self) -> &'static str {
        match self {
            Goal::Min => "↓",
            Goal::Max => "↑",
        }
    }

    pub fn latex_arrow(self) -> &'static str {
        match self {
            Goal::Min => "$\\downarrow$",
            Goal::Max => "$\\uparrow$",
        }
    }

    /// Whether `candidate` is strictly better than `current`.
    pub fn better(self, candidate: f64, current: f64) -> bool {
        match self {
            Goal::Min => candidate < current,
            Goal::Max => candidate > current,
        }
    }
}

/// Matched anywhere in the lower-cased name (e.g. "val_loss", "l1loss").
const MIN_SUBSTRINGS: [&str; 5] = ["loss", "lpips", "error", "mse", "mae"];
/// Matched only as a whole token: too short to match safely as substrings.
const MIN_TOKENS: [&str; 11] = [
    "err",
    "rmse",
    "fid",
    "kid",
    "ppl",
    "perplexity",
    "wer",
    "cer",
    "nll",
    "kl",
    "bpd",
];

/// Heuristic goal for a metric name.
pub fn default_goal(name: &str) -> Goal {
    let lower = name.to_lowercase();
    if MIN_SUBSTRINGS.iter().any(|k| lower.contains(k)) {
        return Goal::Min;
    }
    let is_min_token = lower
        .split(|c: char| !(c.is_ascii_lowercase() || c.is_ascii_digit()))
        .any(|tok| MIN_TOKENS.contains(&tok));
    if is_min_token {
        Goal::Min
    } else {
        Goal::Max
    }
}

/// Explicit goals with the heuristic as fallback.
#[derive(Debug, Clone, Default)]
pub struct Goals {
    explicit: HashMap<String, Goal>,
}

impl Goals {
    pub fn new(explicit: HashMap<String, Goal>) -> Self {
        Self { explicit }
    }

    /// Add goals from another experiment; the first declaration wins.
    pub fn merge(&mut self, other: &HashMap<String, Goal>) {
        for (name, goal) in other {
            self.explicit.entry(name.clone()).or_insert(*goal);
        }
    }

    pub fn get(&self, name: &str) -> Goal {
        self.explicit
            .get(name)
            .copied()
            .unwrap_or_else(|| default_goal(name))
    }
}

/// Total order that ranks the best value first and missing / non-finite
/// values last, regardless of direction.
pub fn cmp_best_first(a: Option<f64>, b: Option<f64>, goal: Goal) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let a = a.filter(|v| v.is_finite());
    let b = b.filter(|v| v.is_finite());
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(x), Some(y)) => match goal {
            Goal::Min => x.total_cmp(&y),
            Goal::Max => y.total_cmp(&x),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_matches_shared_fixture() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../spec/fixtures/metric_goals.json");
        let cases: HashMap<String, Goal> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert!(cases.len() > 20);
        for (name, want) in &cases {
            assert_eq!(default_goal(name), *want, "metric {name:?}");
        }
    }

    #[test]
    fn test_explicit_goal_overrides_heuristic() {
        let goals = Goals::new(HashMap::from([("regret".to_string(), Goal::Min)]));
        assert_eq!(goals.get("regret"), Goal::Min);
        assert_eq!(goals.get("score"), Goal::Max);
        assert_eq!(goals.get("loss"), Goal::Min);
    }

    #[test]
    fn test_merge_first_declaration_wins() {
        let mut goals = Goals::new(HashMap::from([("x".to_string(), Goal::Min)]));
        goals.merge(&HashMap::from([
            ("x".to_string(), Goal::Max),
            ("y".to_string(), Goal::Min),
        ]));
        assert_eq!(goals.get("x"), Goal::Min);
        assert_eq!(goals.get("y"), Goal::Min);
    }

    #[test]
    fn test_cmp_best_first_is_total_and_puts_missing_last() {
        let mut vals = [Some(0.3), None, Some(0.1), Some(f64::NAN), Some(0.2), None];
        vals.sort_by(|a, b| cmp_best_first(*a, *b, Goal::Min));
        assert_eq!(vals[0], Some(0.1));
        assert_eq!(vals[1], Some(0.2));
        assert_eq!(vals[2], Some(0.3));
        assert!(vals[3..].iter().all(|v| v.is_none_or(|x| x.is_nan())));

        let mut vals = [Some(1.0), None, Some(3.0), Some(2.0)];
        vals.sort_by(|a, b| cmp_best_first(*a, *b, Goal::Max));
        assert_eq!(vals, [Some(3.0), Some(2.0), Some(1.0), None]);
    }

    #[test]
    fn test_better() {
        assert!(Goal::Min.better(0.1, 0.2));
        assert!(!Goal::Min.better(0.2, 0.2));
        assert!(Goal::Max.better(0.3, 0.2));
    }
}
