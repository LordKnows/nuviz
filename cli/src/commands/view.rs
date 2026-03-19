use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use comfy_table::{Cell, Table};

use crate::cli::ViewArgs;
use crate::data::experiment::discover_experiments;
use crate::data::ply::{compute_ply_stats, PlyStats};

pub fn run(args: ViewArgs, base_dir: &Path) -> Result<()> {
    let ply_path = resolve_ply_path(&args.path, base_dir, args.project.as_deref())?;
    let stats = compute_ply_stats(&ply_path)?;

    print_stats(&stats, &ply_path);

    if args.histogram {
        print_histograms(&stats);
    }

    Ok(())
}

/// Resolve the PLY path: either a direct file path or experiment name lookup.
fn resolve_ply_path(path_str: &str, base_dir: &Path, project: Option<&str>) -> Result<PathBuf> {
    let path = Path::new(path_str);

    // Direct file path
    if path_str.contains('.') && path.exists() {
        return Ok(path.to_path_buf());
    }

    // Treat as experiment name — find latest PLY
    let experiments = discover_experiments(base_dir);
    let exp = experiments
        .iter()
        .find(|e| {
            let name_match =
                e.name == path_str || e.dir.file_name().and_then(|n| n.to_str()) == Some(path_str);
            let project_match = project
                .map(|p| e.project.as_deref() == Some(p))
                .unwrap_or(true);
            name_match && project_match
        })
        .ok_or_else(|| anyhow::anyhow!("Experiment '{}' not found", path_str))?;

    let pc_dir = exp.dir.join("pointclouds");
    if !pc_dir.is_dir() {
        bail!("No pointclouds/ directory in experiment '{}'", path_str);
    }

    // Find the latest PLY file
    let mut plys: Vec<_> = std::fs::read_dir(&pc_dir)?
        .flatten()
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "ply")
                .unwrap_or(false)
        })
        .collect();

    plys.sort_by_key(|e| e.file_name());

    plys.last()
        .map(|e| e.path())
        .ok_or_else(|| anyhow::anyhow!("No PLY files found in {}", pc_dir.display()))
}

fn print_stats(stats: &PlyStats, path: &Path) {
    let mut table = Table::new();
    table.set_header(vec!["Property", "Value"]);

    table.add_row(vec![
        Cell::new("File"),
        Cell::new(path.file_name().unwrap_or_default().to_string_lossy()),
    ]);
    table.add_row(vec![
        Cell::new("File Size"),
        Cell::new(format_bytes(stats.file_size_bytes)),
    ]);
    table.add_row(vec![
        Cell::new("Vertices/Gaussians"),
        Cell::new(format_number(stats.num_vertices)),
    ]);

    let (min, max) = &stats.bounding_box;
    table.add_row(vec![
        Cell::new("Bounding Box Min"),
        Cell::new(format!("[{:.3}, {:.3}, {:.3}]", min[0], min[1], min[2])),
    ]);
    table.add_row(vec![
        Cell::new("Bounding Box Max"),
        Cell::new(format!("[{:.3}, {:.3}, {:.3}]", max[0], max[1], max[2])),
    ]);
    let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    table.add_row(vec![
        Cell::new("Extent"),
        Cell::new(format!(
            "[{:.3}, {:.3}, {:.3}]",
            extent[0], extent[1], extent[2]
        )),
    ]);

    table.add_row(vec![
        Cell::new("Has Colors"),
        Cell::new(if stats.has_colors { "yes" } else { "no" }),
    ]);

    if let Some(degree) = stats.sh_degree {
        table.add_row(vec![Cell::new("SH Degree"), Cell::new(degree.to_string())]);
    }

    if let Some(ref op_stats) = stats.opacity_stats {
        table.add_row(vec![
            Cell::new("Opacity (mean ± std)"),
            Cell::new(format!("{:.4} ± {:.4}", op_stats.mean, op_stats.std)),
        ]);
        table.add_row(vec![
            Cell::new("Opacity Range"),
            Cell::new(format!("[{:.4}, {:.4}]", op_stats.min, op_stats.max)),
        ]);
        let pct = op_stats.special_count as f64 / stats.num_vertices as f64 * 100.0;
        table.add_row(vec![
            Cell::new("Near-transparent (<0.01)"),
            Cell::new(format!("{} ({:.1}%)", op_stats.special_count, pct)),
        ]);
    }

    if let Some(ref sc_stats) = stats.scale_stats {
        table.add_row(vec![
            Cell::new("Scale magnitude (mean ± std)"),
            Cell::new(format!("{:.4} ± {:.4}", sc_stats.mean, sc_stats.std)),
        ]);
        table.add_row(vec![
            Cell::new("Scale Range"),
            Cell::new(format!("[{:.4}, {:.4}]", sc_stats.min, sc_stats.max)),
        ]);
        table.add_row(vec![
            Cell::new("Scale outliers (>3σ)"),
            Cell::new(sc_stats.special_count.to_string()),
        ]);
    }

    if stats.custom_property_count > 0 {
        table.add_row(vec![
            Cell::new("Custom Properties"),
            Cell::new(stats.custom_property_count.to_string()),
        ]);
    }

    println!("{table}");
}

fn print_histograms(stats: &PlyStats) {
    if let Some(ref opacities) = stats.opacities {
        println!("\nOpacity Distribution:");
        print_histogram(opacities, 10, 0.0, 1.0);
    }

    if let Some(ref scales) = stats.scales {
        let magnitudes: Vec<f32> = scales
            .iter()
            .map(|s| (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt())
            .collect();
        if let (Some(&min), Some(&max)) = (
            magnitudes.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
            magnitudes.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
        ) {
            println!("\nScale Magnitude Distribution:");
            print_histogram(&magnitudes, 10, min, max);
        }
    }
}

fn print_histogram(values: &[f32], bins: usize, min: f32, max: f32) {
    let range = max - min;
    if range <= 0.0 || bins == 0 {
        println!("  (no variation)");
        return;
    }

    let bin_width = range / bins as f32;
    let mut counts = vec![0usize; bins];

    for &v in values {
        let idx = ((v - min) / bin_width) as usize;
        let idx = idx.min(bins - 1);
        counts[idx] += 1;
    }

    let max_count = *counts.iter().max().unwrap_or(&1);
    let bar_max_width = 40;

    for (i, &count) in counts.iter().enumerate() {
        let lo = min + i as f32 * bin_width;
        let hi = lo + bin_width;
        let bar_len = if max_count > 0 {
            (count * bar_max_width) / max_count
        } else {
            0
        };
        let bar: String = "█".repeat(bar_len);
        let pct = count as f64 / values.len() as f64 * 100.0;
        println!(
            "  [{:>7.3}, {:>7.3}) {:>6} ({:>5.1}%) {bar}",
            lo, hi, count, pct
        );
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1500), "1.5 KB");
        assert_eq!(format_bytes(5_242_880), "5.0 MB");
        assert_eq!(format_bytes(1_073_741_824), "1.00 GB");
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1_000_000), "1,000,000");
    }
}
