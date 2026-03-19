use std::io::{self, Write};
use std::path::Path;

use anyhow::{bail, Result};
use crossterm::event::{self, Event, KeyCode, KeyEvent};
use crossterm::terminal;

use crate::cli::ImageArgs;
use crate::data::experiment::discover_experiments;
use crate::data::images::{discover_images, find_latest_image, ImageEntry};
use crate::terminal::capability::detect_capabilities;
use crate::terminal::render;

pub fn run(args: ImageArgs, base_dir: &Path) -> Result<()> {
    let experiments = discover_experiments(base_dir);
    let exp = experiments
        .iter()
        .find(|e| {
            e.name == args.experiment
                || e.dir.file_name().and_then(|n| n.to_str()) == Some(&args.experiment)
        })
        .ok_or_else(|| anyhow::anyhow!("Experiment '{}' not found", args.experiment))?;

    // Filter by project if specified
    if let Some(ref proj) = args.project {
        if exp.project.as_deref() != Some(proj) {
            bail!("Experiment '{}' not in project '{}'", args.experiment, proj);
        }
    }

    let caps = detect_capabilities();

    if args.latest {
        let entry = find_latest_image(&exp.dir, args.tag.as_deref())
            .ok_or_else(|| anyhow::anyhow!("No images found"))?;
        print_image_info(&entry);
        let (max_w, max_h) = render::get_terminal_pixel_size();
        render::render_image(&entry.path, &caps, max_w, max_h.saturating_sub(64))?;
        return Ok(());
    }

    if let Some(ref side_tag) = args.side_by_side {
        return run_side_by_side(
            &exp.dir,
            args.tag.as_deref().unwrap_or("render"),
            side_tag,
            args.step,
            &caps,
        );
    }

    // Collect and filter images
    let mut images = discover_images(&exp.dir);
    if let Some(step_filter) = args.step {
        images.retain(|e| e.step == step_filter);
    }
    if let Some(ref tag_filter) = args.tag {
        images.retain(|e| e.tag == *tag_filter);
    }

    if images.is_empty() {
        bail!("No images found matching filters");
    }

    if images.len() == 1 {
        print_image_info(&images[0]);
        let (max_w, max_h) = render::get_terminal_pixel_size();
        render::render_image(&images[0].path, &caps, max_w, max_h.saturating_sub(64))?;
        return Ok(());
    }

    // Interactive browsing mode
    browse_images(&images, &caps)
}

fn run_side_by_side(
    experiment_dir: &Path,
    tag_a: &str,
    tag_b: &str,
    step_filter: Option<u64>,
    caps: &crate::terminal::capability::TerminalCapabilities,
) -> Result<()> {
    let images = discover_images(experiment_dir);

    let step = if let Some(s) = step_filter {
        s
    } else {
        // Use the latest step that has both tags
        images
            .iter()
            .filter(|e| e.tag == tag_a)
            .filter_map(|e| {
                if images.iter().any(|o| o.step == e.step && o.tag == tag_b) {
                    Some(e.step)
                } else {
                    None
                }
            })
            .max()
            .ok_or_else(|| {
                anyhow::anyhow!("No step found with both tags '{}' and '{}'", tag_a, tag_b)
            })?
    };

    let left = images
        .iter()
        .find(|e| e.step == step && e.tag == tag_a)
        .ok_or_else(|| anyhow::anyhow!("No image found: step={}, tag={}", step, tag_a))?;

    let right = images
        .iter()
        .find(|e| e.step == step && e.tag == tag_b)
        .ok_or_else(|| anyhow::anyhow!("No image found: step={}, tag={}", step, tag_b))?;

    println!("Step {step}: {tag_a} (left) vs {tag_b} (right)");
    let (max_w, max_h) = render::get_terminal_pixel_size();
    render::render_image_pair(
        &left.path,
        &right.path,
        caps,
        max_w,
        max_h.saturating_sub(64),
    )
}

fn browse_images(
    images: &[ImageEntry],
    caps: &crate::terminal::capability::TerminalCapabilities,
) -> Result<()> {
    let mut index: usize = 0;

    terminal::enable_raw_mode()?;
    let result = browse_loop(images, caps, &mut index);
    terminal::disable_raw_mode()?;

    result
}

fn browse_loop(
    images: &[ImageEntry],
    caps: &crate::terminal::capability::TerminalCapabilities,
    index: &mut usize,
) -> Result<()> {
    let (max_w, max_h) = render::get_terminal_pixel_size();

    loop {
        // Clear screen and show current image
        print!("\x1b[2J\x1b[H"); // Clear screen, cursor to top
        io::stdout().flush()?;

        let entry = &images[*index];
        println!(
            "[{}/{}] step={} tag={} ({} bytes)",
            *index + 1,
            images.len(),
            entry.step,
            entry.tag,
            entry.size_bytes
        );
        println!("← prev | → next | q quit");
        println!();

        render::render_image(&entry.path, caps, max_w, max_h.saturating_sub(128))?;

        // Wait for key
        loop {
            if let Event::Key(KeyEvent { code, .. }) = event::read()? {
                match code {
                    KeyCode::Right | KeyCode::Char('l') | KeyCode::Char('n') => {
                        if *index + 1 < images.len() {
                            *index += 1;
                        }
                        break;
                    }
                    KeyCode::Left | KeyCode::Char('h') | KeyCode::Char('p') => {
                        if *index > 0 {
                            *index -= 1;
                        }
                        break;
                    }
                    KeyCode::Char('q') | KeyCode::Esc => {
                        print!("\x1b[2J\x1b[H");
                        io::stdout().flush()?;
                        return Ok(());
                    }
                    KeyCode::Home => {
                        *index = 0;
                        break;
                    }
                    KeyCode::End => {
                        *index = images.len() - 1;
                        break;
                    }
                    _ => {}
                }
            }
        }
    }
}

fn print_image_info(entry: &ImageEntry) {
    println!(
        "Image: step={} tag={} size={} bytes",
        entry.step, entry.tag, entry.size_bytes
    );
    println!("  Path: {}", entry.path.display());
}
