//! Helpers shared by command implementations.

use std::path::Path;

use anyhow::Result;

use crate::data::experiment::{discover_experiments, filter_by_project, Experiment};
use crate::data::resolve;

/// Discover experiments, optionally restricted to one project.
pub fn load_scope(base_dir: &Path, project: Option<&str>) -> Vec<Experiment> {
    let experiments = discover_experiments(base_dir);
    match project {
        Some(p) => filter_by_project(&experiments, p),
        None => experiments,
    }
}

/// Resolve one experiment from what the user typed.
pub fn find_experiment(base_dir: &Path, query: &str, project: Option<&str>) -> Result<Experiment> {
    resolve::resolve(&discover_experiments(base_dir), query, project)
}

/// Labels for a set of experiments: plain names, or `project/name` for any
/// name that would otherwise be shown twice.
pub fn display_labels(experiments: &[&Experiment]) -> Vec<String> {
    experiments
        .iter()
        .map(|e| {
            let clashes = experiments.iter().filter(|o| o.name == e.name).count() > 1;
            if clashes {
                e.qualified_name()
            } else {
                e.name.clone()
            }
        })
        .collect()
}

pub fn print_no_experiments(base_dir: &Path) {
    println!("No experiments found in {}", base_dir.display());
}
