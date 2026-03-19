use std::io::{self, Write};
use std::path::Path;

use anyhow::{Context, Result};
use base64::Engine;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, Rgba};

use super::capability::{ColorDepth, TerminalCapabilities};

/// Render an image to the terminal using the best available protocol.
pub fn render_image(
    path: &Path,
    caps: &TerminalCapabilities,
    max_width: u32,
    max_height: u32,
) -> Result<()> {
    let img =
        image::open(path).with_context(|| format!("Failed to open image: {}", path.display()))?;
    render_dynamic_image(&img, caps, max_width, max_height)
}

/// Render a DynamicImage (already loaded) to the terminal.
pub fn render_dynamic_image(
    img: &DynamicImage,
    caps: &TerminalCapabilities,
    max_width: u32,
    max_height: u32,
) -> Result<()> {
    let img = resize_to_fit(img, max_width, max_height);

    if caps.supports_kitty_graphics {
        render_kitty(&img)?;
    } else if caps.supports_iterm2 {
        render_iterm2(&img)?;
    } else if caps.supports_sixel {
        render_sixel(&img)?;
    } else {
        render_halfblock(&img, caps)?;
    }

    Ok(())
}

/// Render two images side by side.
pub fn render_image_pair(
    left: &Path,
    right: &Path,
    caps: &TerminalCapabilities,
    max_width: u32,
    max_height: u32,
) -> Result<()> {
    let left_img =
        image::open(left).with_context(|| format!("Failed to open: {}", left.display()))?;
    let right_img =
        image::open(right).with_context(|| format!("Failed to open: {}", right.display()))?;

    // Each image gets half the width, minus 2 cols for separator
    let half_width = max_width.saturating_sub(2) / 2;
    let left_resized = resize_to_fit(&left_img, half_width, max_height);
    let right_resized = resize_to_fit(&right_img, half_width, max_height);

    // For graphics protocols, render sequentially with a gap
    if caps.supports_kitty_graphics || caps.supports_iterm2 || caps.supports_sixel {
        // Combine into a single image with separator
        let combined = combine_side_by_side(&left_resized, &right_resized, 4);
        render_dynamic_image(&combined, caps, max_width, max_height)?;
    } else {
        // Half-block: render combined image
        let combined = combine_side_by_side(&left_resized, &right_resized, 4);
        render_halfblock(&combined, caps)?;
    }

    Ok(())
}

/// Resize image to fit within max dimensions while preserving aspect ratio.
fn resize_to_fit(img: &DynamicImage, max_width: u32, max_height: u32) -> DynamicImage {
    let (w, h) = img.dimensions();
    if w <= max_width && h <= max_height {
        return img.clone();
    }
    img.resize(max_width, max_height, FilterType::Lanczos3)
}

/// Combine two images side by side with a separator gap.
fn combine_side_by_side(left: &DynamicImage, right: &DynamicImage, gap: u32) -> DynamicImage {
    let (lw, lh) = left.dimensions();
    let (rw, rh) = right.dimensions();
    let total_width = lw + gap + rw;
    let total_height = lh.max(rh);

    let mut combined = DynamicImage::new_rgba8(total_width, total_height);
    image::imageops::overlay(&mut combined, left, 0, 0);
    image::imageops::overlay(&mut combined, right, (lw + gap) as i64, 0);
    combined
}

/// Kitty Graphics Protocol: transmit PNG as base64 chunks.
fn render_kitty(img: &DynamicImage) -> Result<()> {
    let png_data = encode_png(img)?;
    let encoded = base64::engine::general_purpose::STANDARD.encode(&png_data);

    let stdout = io::stdout();
    let mut out = stdout.lock();

    // Chunked transmission (4096 bytes per chunk)
    let chunk_size = 4096;
    let chunks: Vec<&str> = encoded
        .as_bytes()
        .chunks(chunk_size)
        .map(|c| std::str::from_utf8(c).unwrap_or(""))
        .collect();

    for (i, chunk) in chunks.iter().enumerate() {
        let more = if i < chunks.len() - 1 { 1 } else { 0 };
        if i == 0 {
            // First chunk: include action and format
            write!(out, "\x1b_Ga=T,f=100,m={more};{chunk}\x1b\\")?;
        } else {
            write!(out, "\x1b_Gm={more};{chunk}\x1b\\")?;
        }
    }
    writeln!(out)?;
    out.flush()?;
    Ok(())
}

/// iTerm2 Inline Image Protocol.
fn render_iterm2(img: &DynamicImage) -> Result<()> {
    let png_data = encode_png(img)?;
    let encoded = base64::engine::general_purpose::STANDARD.encode(&png_data);
    let (w, h) = img.dimensions();

    let stdout = io::stdout();
    let mut out = stdout.lock();

    write!(
        out,
        "\x1b]1337;File=inline=1;width={w}px;height={h}px;preserveAspectRatio=1:{encoded}\x07"
    )?;
    writeln!(out)?;
    out.flush()?;
    Ok(())
}

/// Sixel rendering: quantize to 256 colors and emit DCS sequences.
fn render_sixel(img: &DynamicImage) -> Result<()> {
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();

    let stdout = io::stdout();
    let mut out = stdout.lock();

    // Build a simple 256-color palette by uniform quantization
    // DCS q  ... ST
    write!(out, "\x1bPq")?;

    // Register 216 colors (6x6x6 cube) + leave room
    for r in 0..6u8 {
        for g in 0..6u8 {
            for b in 0..6u8 {
                let idx = r as u32 * 36 + g as u32 * 6 + b as u32;
                let ri = (r as u32 * 100) / 5;
                let gi = (g as u32 * 100) / 5;
                let bi = (b as u32 * 100) / 5;
                write!(out, "#{idx};2;{ri};{gi};{bi}")?;
            }
        }
    }

    // Render in 6-pixel-high bands
    let mut y = 0u32;
    while y < height {
        for color_idx in 0..216u32 {
            let mut has_pixels = false;
            let mut sixel_data = Vec::with_capacity(width as usize);

            for x in 0..width {
                let mut sixel_bits: u8 = 0;
                for dy in 0..6u32 {
                    let py = y + dy;
                    if py < height {
                        let pixel = rgba.get_pixel(x, py);
                        if pixel[3] > 127 && nearest_color(pixel) == color_idx {
                            sixel_bits |= 1 << dy;
                            has_pixels = true;
                        }
                    }
                }
                sixel_data.push(sixel_bits + 0x3f);
            }

            if has_pixels {
                write!(out, "#{color_idx}")?;
                for &b in &sixel_data {
                    out.write_all(&[b])?;
                }
                write!(out, "$")?; // CR within sixel band
            }
        }
        write!(out, "-")?; // Next sixel band (line feed)
        y += 6;
    }

    write!(out, "\x1b\\")?; // ST
    writeln!(out)?;
    out.flush()?;
    Ok(())
}

/// Map RGBA pixel to nearest 6x6x6 cube index.
fn nearest_color(pixel: &Rgba<u8>) -> u32 {
    let r = ((pixel[0] as u32 + 25) / 51).min(5);
    let g = ((pixel[1] as u32 + 25) / 51).min(5);
    let b = ((pixel[2] as u32 + 25) / 51).min(5);
    r * 36 + g * 6 + b
}

/// Half-block fallback: use upper/lower half-block chars with 24-bit ANSI colors.
/// Each character cell encodes 2 vertical pixels.
fn render_halfblock(img: &DynamicImage, caps: &TerminalCapabilities) -> Result<()> {
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();

    let stdout = io::stdout();
    let mut out = stdout.lock();

    let mut y = 0u32;
    while y < height {
        for x in 0..width {
            let top = rgba.get_pixel(x, y);
            let bottom = if y + 1 < height {
                *rgba.get_pixel(x, y + 1)
            } else {
                Rgba([0, 0, 0, 0])
            };

            match caps.color_depth {
                ColorDepth::TrueColor => {
                    // Upper half block: foreground = top, background = bottom
                    write!(
                        out,
                        "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m\u{2580}",
                        top[0], top[1], top[2], bottom[0], bottom[1], bottom[2]
                    )?;
                }
                ColorDepth::Colors256 => {
                    let fg = to_256_color(top[0], top[1], top[2]);
                    let bg = to_256_color(bottom[0], bottom[1], bottom[2]);
                    write!(out, "\x1b[38;5;{fg}m\x1b[48;5;{bg}m\u{2580}")?;
                }
                ColorDepth::Colors16 => {
                    // Basic fallback — just use half blocks with default colors
                    write!(out, "\u{2580}")?;
                }
            }
        }
        write!(out, "\x1b[0m")?; // Reset
        writeln!(out)?;
        y += 2;
    }

    out.flush()?;
    Ok(())
}

/// Map RGB to 256-color xterm palette (16-231 color cube).
fn to_256_color(r: u8, g: u8, b: u8) -> u8 {
    let ri = ((r as u16 + 25) / 51).min(5) as u8;
    let gi = ((g as u16 + 25) / 51).min(5) as u8;
    let bi = ((b as u16 + 25) / 51).min(5) as u8;
    16 + 36 * ri + 6 * gi + bi
}

/// Encode a DynamicImage as PNG bytes.
fn encode_png(img: &DynamicImage) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut cursor = io::Cursor::new(&mut buf);
    img.write_to(&mut cursor, image::ImageFormat::Png)?;
    Ok(buf)
}

/// Get terminal dimensions in pixels (if available) or estimate from character cells.
pub fn get_terminal_pixel_size() -> (u32, u32) {
    // Try ioctl TIOCGWINSZ for pixel size
    if let Ok((cols, rows)) = crossterm::terminal::size() {
        // Estimate: typical terminal character is ~8px wide, ~16px tall
        let pixel_width = cols as u32 * 8;
        let pixel_height = rows as u32 * 16;
        return (pixel_width, pixel_height);
    }
    (640, 480) // fallback
}

/// Get terminal size in character cells.
#[allow(dead_code)]
pub fn get_terminal_char_size() -> (u16, u16) {
    crossterm::terminal::size().unwrap_or((80, 24))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nearest_color() {
        assert_eq!(nearest_color(&Rgba([0, 0, 0, 255])), 0);
        assert_eq!(nearest_color(&Rgba([255, 255, 255, 255])), 215);
        assert_eq!(nearest_color(&Rgba([255, 0, 0, 255])), 180);
    }

    #[test]
    fn test_to_256_color() {
        assert_eq!(to_256_color(0, 0, 0), 16);
        assert_eq!(to_256_color(255, 255, 255), 231);
    }

    #[test]
    fn test_resize_to_fit_no_resize_needed() {
        let img = DynamicImage::new_rgba8(100, 100);
        let resized = resize_to_fit(&img, 200, 200);
        assert_eq!(resized.dimensions(), (100, 100));
    }

    #[test]
    fn test_resize_to_fit_downscale() {
        let img = DynamicImage::new_rgba8(400, 200);
        let resized = resize_to_fit(&img, 200, 200);
        assert!(resized.width() <= 200);
        assert!(resized.height() <= 200);
    }

    #[test]
    fn test_combine_side_by_side() {
        let left = DynamicImage::new_rgba8(50, 100);
        let right = DynamicImage::new_rgba8(60, 80);
        let combined = combine_side_by_side(&left, &right, 4);
        assert_eq!(combined.width(), 114); // 50 + 4 + 60
        assert_eq!(combined.height(), 100); // max(100, 80)
    }

    #[test]
    fn test_encode_png() {
        let img = DynamicImage::new_rgba8(10, 10);
        let data = encode_png(&img).unwrap();
        assert!(!data.is_empty());
        // PNG magic bytes
        assert_eq!(&data[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }
}
