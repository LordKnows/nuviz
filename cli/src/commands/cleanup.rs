use anyhow::Result;
use std::path::Path;

use crate::cli::CleanupArgs;
use crate::data::experiment::{discover_experiments, filter_by_project};

pub fn run(args: CleanupArgs, base_dir: &Path) -> Result<()> {
    let experiments = discover_experiments(base_dir);

    let mut targets = if let Some(ref project) = args.project {
        filter_by_project(&experiments, project)
    } else {
        experiments
    };

    if targets.is_empty() {
        println!("No experiments found.");
        return Ok(());
    }

    let metric = args.metric.as_deref().unwrap_or("loss");
    let keep = args.keep_top.unwrap_or(5);

    // Sort by metric (lower is better for loss-like, higher for others)
    let minimize = metric.contains("loss")
        || metric.contains("error")
        || metric.contains("lpips")
        || metric.contains("mse")
        || metric.contains("mae");

    targets.sort_by(|a, b| {
        let va = a.best_metrics.get(metric).copied().unwrap_or(f64::NAN);
        let vb = b.best_metrics.get(metric).copied().unwrap_or(f64::NAN);
        if minimize {
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
        }
    });

    if targets.len() <= keep {
        println!(
            "Only {} experiments found, keeping all (threshold: {}).",
            targets.len(),
            keep
        );
        return Ok(());
    }

    let to_remove = &targets[keep..];

    // Calculate sizes
    let mut total_bytes: u64 = 0;
    println!(
        "\nExperiments to remove (keeping top {} by {}):\n",
        keep, metric
    );
    println!("  {:<30} {:>12} {:>10}", "Name", metric, "Size");
    println!("  {}", "-".repeat(56));

    for exp in to_remove {
        let size = dir_size(&exp.dir);
        total_bytes += size;
        let metric_val = exp
            .best_metrics
            .get(metric)
            .map(|v| format!("{:.4}", v))
            .unwrap_or_else(|| "N/A".into());
        println!(
            "  {:<30} {:>12} {:>10}",
            exp.name,
            metric_val,
            format_size(size)
        );
    }

    println!("\n  Total reclaimable: {}", format_size(total_bytes));
    println!(
        "  Experiments to remove: {}/{}\n",
        to_remove.len(),
        targets.len()
    );

    if !args.force {
        println!("Dry run — no files deleted. Use --force to delete.");
        return Ok(());
    }

    // Actually delete
    for exp in to_remove {
        match std::fs::remove_dir_all(&exp.dir) {
            Ok(()) => println!("  Deleted: {}", exp.name),
            Err(e) => eprintln!("  Failed to delete {}: {}", exp.name, e),
        }
    }

    println!("\nCleanup complete. Freed {}.", format_size(total_bytes));
    Ok(())
}

fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() {
                total += p.metadata().map(|m| m.len()).unwrap_or(0);
            } else if p.is_dir() {
                total += dir_size(&p);
            }
        }
    }
    total
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}
