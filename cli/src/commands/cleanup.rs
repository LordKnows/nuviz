use anyhow::Result;
use std::path::Path;

use crate::cli::CleanupArgs;
use crate::commands::common::load_scope;
use crate::data::experiment::Experiment;
use crate::data::goal::{cmp_best_first, Goals};
use crate::data::status::Status;
use crate::data::summary::ValueMode;

pub fn run(args: CleanupArgs, base_dir: &Path) -> Result<()> {
    let experiments = load_scope(base_dir, args.project.as_deref());
    if experiments.is_empty() {
        println!("No experiments found.");
        return Ok(());
    }

    let mut goals = Goals::default();
    for exp in &experiments {
        goals.merge(&exp.meta.metric_goals);
    }
    let goal = goals.get(&args.metric);

    let candidates: Vec<Candidate> = experiments
        .into_iter()
        .map(|exp| {
            let value = exp
                .summary()
                .value(&args.metric, ValueMode::Best, goal)
                .map(|(v, _)| v);
            Candidate { exp, value }
        })
        .collect();
    let plan = plan(candidates, args.keep_top, args.include_tagged, |a, b| {
        cmp_best_first(a, b, goal)
    });

    if plan.protected_running + plan.protected_tagged > 0 {
        println!(
            "Protected: {} running, {} tagged (never removed{}).",
            plan.protected_running,
            plan.protected_tagged,
            if args.include_tagged {
                ""
            } else {
                "; pass --include-tagged to consider tagged runs"
            }
        );
    }

    if plan.remove.is_empty() {
        println!(
            "Nothing to remove: {} removable experiment(s), keeping top {}.",
            plan.keep.len(),
            args.keep_top
        );
        return Ok(());
    }

    let mut total_bytes: u64 = 0;
    println!(
        "\nExperiments to remove (keeping top {} by {}):\n",
        args.keep_top, args.metric
    );
    println!(
        "  {:<34} {:>10} {:>12} {:>10}",
        "Name", "Status", args.metric, "Size"
    );
    println!("  {}", "-".repeat(70));

    for c in &plan.remove {
        let size = dir_size(&c.exp.dir);
        total_bytes += size;
        let value = c
            .value
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "N/A".into());
        println!(
            "  {:<34} {:>10} {:>12} {:>10}",
            c.exp.qualified_name(),
            c.exp.status,
            value,
            format_size(size)
        );
    }

    println!("\n  Total reclaimable: {}", format_size(total_bytes));
    println!(
        "  Experiments to remove: {}/{}\n",
        plan.remove.len(),
        plan.remove.len() + plan.keep.len()
    );

    if !args.force {
        println!("Dry run — no files deleted. Use --force to delete.");
        return Ok(());
    }

    let mut freed = 0u64;
    for c in &plan.remove {
        let size = dir_size(&c.exp.dir);
        match std::fs::remove_dir_all(&c.exp.dir) {
            Ok(()) => {
                freed += size;
                println!("  Deleted: {}", c.exp.qualified_name());
            }
            Err(e) => eprintln!("  Failed to delete {}: {e}", c.exp.qualified_name()),
        }
    }

    println!("\nCleanup complete. Freed {}.", format_size(freed));
    Ok(())
}

pub struct Candidate {
    pub exp: Experiment,
    pub value: Option<f64>,
}

pub struct Plan {
    pub keep: Vec<Candidate>,
    pub remove: Vec<Candidate>,
    pub protected_running: usize,
    pub protected_tagged: usize,
}

/// Decide what to remove. Running experiments are never candidates; tagged
/// ones only with `include_tagged`. Of the rest, the best `keep_top` by the
/// metric are kept; runs without the metric rank last and go first.
pub fn plan(
    candidates: Vec<Candidate>,
    keep_top: usize,
    include_tagged: bool,
    cmp: impl Fn(Option<f64>, Option<f64>) -> std::cmp::Ordering,
) -> Plan {
    let mut protected_running = 0;
    let mut protected_tagged = 0;
    let mut removable: Vec<Candidate> = Vec::new();

    for c in candidates {
        if c.exp.status == Status::Running {
            protected_running += 1;
        } else if !include_tagged && !c.exp.meta.tags.is_empty() {
            protected_tagged += 1;
        } else {
            removable.push(c);
        }
    }

    removable.sort_by(|a, b| {
        cmp(a.value, b.value).then_with(|| a.exp.qualified_name().cmp(&b.exp.qualified_name()))
    });
    let remove = removable.split_off(keep_top.min(removable.len()));

    Plan {
        keep: removable,
        remove,
        protected_running,
        protected_tagged,
    }
}

fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                total += dir_size(&entry.path());
            } else if file_type.is_file() {
                total += entry.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    total
}

fn format_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    let b = bytes as f64;
    if b < KB {
        format!("{bytes} B")
    } else if b < KB * KB {
        format!("{:.1} KB", b / KB)
    } else if b < KB * KB * KB {
        format!("{:.1} MB", b / (KB * KB))
    } else {
        format!("{:.1} GB", b / (KB * KB * KB))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::goal::Goal;
    use crate::data::meta::ExperimentMeta;
    use std::path::PathBuf;

    fn candidate(name: &str, status: Status, tags: &[&str], value: Option<f64>) -> Candidate {
        Candidate {
            exp: Experiment {
                name: name.into(),
                dir_name: name.into(),
                project: None,
                dir: PathBuf::from(name),
                status,
                meta: ExperimentMeta {
                    tags: tags.iter().map(|t| t.to_string()).collect(),
                    ..Default::default()
                },
                started: None,
                heartbeat_step: None,
            },
            value,
        }
    }

    fn names(cs: &[Candidate]) -> Vec<&str> {
        cs.iter().map(|c| c.exp.name.as_str()).collect()
    }

    fn by_loss(a: Option<f64>, b: Option<f64>) -> std::cmp::Ordering {
        cmp_best_first(a, b, Goal::Min)
    }

    #[test]
    fn test_keeps_best_and_removes_runs_without_metric_first() {
        let plan = plan(
            vec![
                candidate("mid", Status::Done, &[], Some(0.3)),
                candidate("none", Status::Crashed, &[], None),
                candidate("best", Status::Done, &[], Some(0.1)),
                candidate("worst", Status::Failed, &[], Some(0.9)),
            ],
            2,
            false,
            by_loss,
        );
        assert_eq!(names(&plan.keep), ["best", "mid"]);
        assert_eq!(names(&plan.remove), ["worst", "none"]);
    }

    #[test]
    fn test_running_is_never_removed_even_with_include_tagged() {
        let plan = plan(
            vec![
                candidate("live", Status::Running, &[], None),
                candidate("live_tagged", Status::Running, &["x"], Some(9.0)),
                candidate("old", Status::Done, &[], Some(0.5)),
            ],
            0,
            true,
            by_loss,
        );
        assert_eq!(plan.protected_running, 2);
        assert_eq!(names(&plan.remove), ["old"]);
    }

    #[test]
    fn test_tagged_kept_unless_included() {
        let make = || {
            vec![
                candidate("tagged", Status::Done, &["paper"], Some(0.9)),
                candidate("plain", Status::Done, &[], Some(0.1)),
            ]
        };
        let plan_default = plan(make(), 0, false, by_loss);
        assert_eq!(plan_default.protected_tagged, 1);
        assert_eq!(names(&plan_default.remove), ["plain"]);

        let plan_all = plan(make(), 0, true, by_loss);
        assert_eq!(plan_all.protected_tagged, 0);
        assert_eq!(names(&plan_all.remove), ["plain", "tagged"]);
    }

    #[test]
    fn test_keep_more_than_available() {
        let plan = plan(
            vec![candidate("only", Status::Done, &[], Some(0.1))],
            5,
            false,
            by_loss,
        );
        assert!(plan.remove.is_empty());
        assert_eq!(names(&plan.keep), ["only"]);
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(2048), "2.0 KB");
        assert_eq!(format_size(5 * 1024 * 1024), "5.0 MB");
    }
}
