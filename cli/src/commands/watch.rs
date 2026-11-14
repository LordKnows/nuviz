use anyhow::Result;
use std::path::Path;

use crate::cli::WatchArgs;
use crate::commands::common::{display_labels, load_scope};
use crate::data::experiment::{discover_experiments, Experiment};
use crate::data::resolve::resolve_many;
use crate::data::store::XAxis;
use crate::tui::dashboard::{self, Dashboard, LayoutMode};
use crate::watcher::signal::FsSignal;
use crate::watcher::source::Source;

pub fn run(args: WatchArgs, base_dir: &Path) -> Result<()> {
    let experiments = resolve_experiments(&args, base_dir)?;
    if experiments.is_empty() {
        anyhow::bail!(
            "No experiments found to watch in {} (run a training script first, or pass a name)",
            base_dir.display()
        );
    }
    launch(experiments, LayoutMode::Dual, XAxis::Step, None, args.poll)
}

/// Open the dashboard on a set of experiments.
pub fn launch(
    experiments: Vec<Experiment>,
    layout: LayoutMode,
    x_axis: XAxis,
    metric: Option<String>,
    poll_only: bool,
) -> Result<()> {
    let labels = display_labels(&experiments.iter().collect::<Vec<_>>());
    let dirs: Vec<_> = experiments.iter().map(|e| e.dir.clone()).collect();
    let sources: Vec<Source> = experiments
        .into_iter()
        .zip(labels)
        .map(|(exp, label)| Source::new(exp, label))
        .collect();

    let signal = if poll_only {
        FsSignal::disabled()
    } else {
        FsSignal::watch(&dirs)
    };
    let mut dashboard = Dashboard::new(sources, layout, x_axis, metric);
    dashboard::run(&mut dashboard, &signal)
}

fn resolve_experiments(args: &WatchArgs, base_dir: &Path) -> Result<Vec<Experiment>> {
    if !args.experiments.is_empty() {
        let all = discover_experiments(base_dir);
        return resolve_many(&all, &args.experiments, args.project.as_deref());
    }

    // Default: the most recent experiment(s), optionally within a project.
    let mut experiments = load_scope(base_dir, args.project.as_deref());
    experiments.truncate(args.latest.unwrap_or(1));
    Ok(experiments)
}
