use std::path::Path;

use anyhow::{bail, Result};

use crate::cli::DiffArgs;
use crate::data::experiment::discover_experiments;
use crate::data::images::discover_images;
use crate::terminal::capability::detect_capabilities;
use crate::terminal::heatmap::generate_error_heatmap;
use crate::terminal::render;

pub fn run(args: DiffArgs, base_dir: &Path) -> Result<()> {
    let experiments = discover_experiments(base_dir);

    let exp_a = experiments
        .iter()
        .find(|e| {
            e.name == args.experiment_a
                || e.dir.file_name().and_then(|n| n.to_str()) == Some(&args.experiment_a)
        })
        .ok_or_else(|| anyhow::anyhow!("Experiment '{}' not found", args.experiment_a))?;

    let exp_b = experiments
        .iter()
        .find(|e| {
            e.name == args.experiment_b
                || e.dir.file_name().and_then(|n| n.to_str()) == Some(&args.experiment_b)
        })
        .ok_or_else(|| anyhow::anyhow!("Experiment '{}' not found", args.experiment_b))?;

    let caps = detect_capabilities();
    let tag = &args.tag;

    // Find matching image pairs
    let images_a = discover_images(&exp_a.dir);
    let images_b = discover_images(&exp_b.dir);

    let images_a_filtered: Vec<_> = images_a.iter().filter(|e| e.tag == *tag).collect();
    let images_b_filtered: Vec<_> = images_b.iter().filter(|e| e.tag == *tag).collect();

    // Determine which step to compare
    let step = if let Some(s) = args.step {
        s
    } else {
        // Find the latest common step
        let steps_a: std::collections::HashSet<u64> =
            images_a_filtered.iter().map(|e| e.step).collect();
        let steps_b: std::collections::HashSet<u64> =
            images_b_filtered.iter().map(|e| e.step).collect();
        let common: Vec<u64> = steps_a.intersection(&steps_b).copied().collect();

        if common.is_empty() {
            bail!(
                "No common steps found for tag '{}' between '{}' and '{}'",
                tag,
                args.experiment_a,
                args.experiment_b
            );
        }
        *common.iter().max().unwrap()
    };

    let img_entry_a = images_a_filtered
        .iter()
        .find(|e| e.step == step)
        .ok_or_else(|| anyhow::anyhow!("No image at step {} in '{}'", step, args.experiment_a))?;

    let img_entry_b = images_b_filtered
        .iter()
        .find(|e| e.step == step)
        .ok_or_else(|| anyhow::anyhow!("No image at step {} in '{}'", step, args.experiment_b))?;

    println!(
        "Comparing step={step} tag={tag}: {} vs {}",
        args.experiment_a, args.experiment_b
    );

    let (max_w, max_h) = render::get_terminal_pixel_size();

    if args.heatmap {
        // Generate and display error heatmap
        let img_a = image::open(&img_entry_a.path)?;
        let img_b = image::open(&img_entry_b.path)?;
        let result = generate_error_heatmap(&img_a, &img_b);

        render::render_dynamic_image(&result.image, &caps, max_w, max_h.saturating_sub(128))?;

        println!();
        println!("Error Analysis:");
        println!("  MAE:       {:.4}", result.mae);
        println!("  Max Error: {}", result.max_error);
        if result.psnr.is_infinite() {
            println!("  PSNR:      ∞ (identical images)");
        } else {
            println!("  PSNR:      {:.2} dB", result.psnr);
        }
    } else {
        // Side-by-side comparison
        render::render_image_pair(
            &img_entry_a.path,
            &img_entry_b.path,
            &caps,
            max_w,
            max_h.saturating_sub(128),
        )?;

        // Also compute and display metrics
        let img_a = image::open(&img_entry_a.path)?;
        let img_b = image::open(&img_entry_b.path)?;
        let result = generate_error_heatmap(&img_a, &img_b);

        println!();
        println!(
            "  {} (left) vs {} (right)",
            args.experiment_a, args.experiment_b
        );
        if result.psnr.is_infinite() {
            println!("  PSNR: ∞ | MAE: {:.4}", result.mae);
        } else {
            println!("  PSNR: {:.2} dB | MAE: {:.4}", result.psnr, result.mae);
        }
    }

    Ok(())
}
