use anyhow::Result;
use std::path::Path;

use crate::cli::{Align, CompareArgs};
use crate::commands::common::load_scope;
use crate::commands::watch::launch;
use crate::data::experiment::{discover_experiments, Experiment};
use crate::data::resolve::resolve_many;
use crate::data::store::XAxis;
use crate::tui::dashboard::LayoutMode;

pub fn run(args: CompareArgs, base_dir: &Path) -> Result<()> {
    let experiments = resolve_experiments(&args, base_dir)?;
    if experiments.len() < 2 {
        anyhow::bail!(
            "compare needs at least 2 experiments, found {}",
            experiments.len()
        );
    }

    let x_axis = match args.align {
        Align::Step => XAxis::Step,
        Align::WallTime => XAxis::WallTime,
    };
    launch(
        experiments,
        LayoutMode::Single,
        x_axis,
        args.metric,
        args.poll,
    )
}

fn resolve_experiments(args: &CompareArgs, base_dir: &Path) -> Result<Vec<Experiment>> {
    if !args.experiments.is_empty() {
        let all = discover_experiments(base_dir);
        resolve_many(&all, &args.experiments, args.project.as_deref())
    } else if args.project.is_some() {
        Ok(load_scope(base_dir, args.project.as_deref()))
    } else {
        anyhow::bail!("specify experiments to compare, or --project to compare a whole project")
    }
}
