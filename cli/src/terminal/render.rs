use std::io::{self, Write};
use std::path::Path;

use anyhow::{Context, Result};
use base64::Engine;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};

use super::capability::{ColorDepth, ImageProtocol, TerminalCapabilities};

/// Open an image file and render it, leaving `reserve_rows` terminal rows
/// free for surrounding text.
pub fn show_file(path: &Path, caps: &TerminalCapabilities, reserve_rows: u16) -> Result<()> {
    let img =
        image::open(path).with_context(|| format!("Failed to open image: {}", path.display()))?;
    show(&img, caps, reserve_rows)
}

/// Render an image to stdout using the terminal's protocol, scaled down to
/// fit the window.
pub fn show(img: &DynamicImage, caps: &TerminalCapabilities, reserve_rows: u16) -> Result<()> {
    let (max_w, max_h) = pixel_budget(caps, reserve_rows);
    let img = resize_to_fit(img, max_w, max_h);
    let stdout = io::stdout();
    let mut out = stdout.lock();
    match caps.protocol {
        ImageProtocol::Kitty => write_kitty(&img, &mut out)?,
        ImageProtocol::Iterm2 => write_iterm2(&img, &mut out)?,
        ImageProtocol::Sixel => write_sixel(&img, &mut out)?,
        ImageProtocol::Halfblock => write_halfblock(&img, caps.color_depth, &mut out)?,
    }
    out.flush()?;
    Ok(())
}

/// Render two image files side by side.
pub fn show_pair(
    left: &Path,
    right: &Path,
    caps: &TerminalCapabilities,
    reserve_rows: u16,
) -> Result<()> {
    let left_img =
        image::open(left).with_context(|| format!("Failed to open: {}", left.display()))?;
    let right_img =
        image::open(right).with_context(|| format!("Failed to open: {}", right.display()))?;

    // Scale each image to half the window so the pair fits; the gap is one
    // character cell wide.
    let (max_w, max_h) = pixel_budget(caps, reserve_rows);
    let gap = match caps.protocol {
        ImageProtocol::Halfblock => 2,
        _ => u32::from(caps.cell_px.0),
    };
    let half = max_w.saturating_sub(gap) / 2;
    let combined = combine_side_by_side(
        &resize_to_fit(&left_img, half.max(1), max_h),
        &resize_to_fit(&right_img, half.max(1), max_h),
        gap,
    );
    show(&combined, caps, reserve_rows)
}

/// How many image pixels fit in the window for the active protocol.
///
/// Graphics protocols draw real pixels. Half blocks draw one image pixel
/// per column and two per row, so the budget is in character cells.
pub fn pixel_budget(caps: &TerminalCapabilities, reserve_rows: u16) -> (u32, u32) {
    let cols = u32::from(caps.cols.max(1));
    let rows = u32::from(caps.rows.saturating_sub(reserve_rows).max(1));
    match caps.protocol {
        ImageProtocol::Halfblock => (cols, rows * 2),
        _ => (
            cols * u32::from(caps.cell_px.0),
            rows * u32::from(caps.cell_px.1),
        ),
    }
}

/// Resize image to fit within max dimensions while preserving aspect ratio.
/// Images that already fit are returned unchanged (never upscaled).
fn resize_to_fit(img: &DynamicImage, max_width: u32, max_height: u32) -> DynamicImage {
    let (w, h) = img.dimensions();
    if w <= max_width && h <= max_height {
        return img.clone();
    }
    img.resize(max_width.max(1), max_height.max(1), FilterType::Lanczos3)
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
fn write_kitty(img: &DynamicImage, out: &mut impl Write) -> Result<()> {
    let encoded = base64::engine::general_purpose::STANDARD.encode(encode_png(img)?);

    // Chunked transmission (4096 bytes per chunk)
    let chunks: Vec<&[u8]> = encoded.as_bytes().chunks(4096).collect();
    for (i, chunk) in chunks.iter().enumerate() {
        let more = u8::from(i + 1 < chunks.len());
        if i == 0 {
            // First chunk: include action and format
            write!(out, "\x1b_Ga=T,f=100,m={more};")?;
        } else {
            write!(out, "\x1b_Gm={more};")?;
        }
        out.write_all(chunk)?;
        write!(out, "\x1b\\")?;
    }
    writeln!(out)?;
    Ok(())
}

/// iTerm2 Inline Image Protocol.
fn write_iterm2(img: &DynamicImage, out: &mut impl Write) -> Result<()> {
    let png = encode_png(img)?;
    let encoded = base64::engine::general_purpose::STANDARD.encode(&png);
    let (w, h) = img.dimensions();
    write!(
        out,
        "\x1b]1337;File=inline=1;size={};width={w}px;height={h}px;preserveAspectRatio=1:{encoded}\x07",
        png.len()
    )?;
    writeln!(out)?;
    Ok(())
}

/// Sixel rendering with a fixed 6x6x6 color cube.
fn write_sixel(img: &DynamicImage, out: &mut impl Write) -> Result<()> {
    let rgba = flatten_alpha(img);
    let (width, height) = rgba.dimensions();
    let (width, height) = (width as usize, height as usize);

    // Quantize once; each band then only visits the colors it contains.
    let indices: Vec<u8> = rgba.pixels().map(nearest_color).collect();

    write!(out, "\x1bPq\"1;1;{width};{height}")?;
    for r in 0..6u32 {
        for g in 0..6u32 {
            for b in 0..6u32 {
                let idx = r * 36 + g * 6 + b;
                write!(out, "#{idx};2;{};{};{}", r * 20, g * 20, b * 20)?;
            }
        }
    }

    let mut row = vec![0u8; width];
    for band_top in (0..height).step_by(6) {
        let band_rows = (height - band_top).min(6);
        let band = &indices[band_top * width..(band_top + band_rows) * width];

        let mut present = [false; 216];
        for &c in band {
            present[c as usize] = true;
        }

        for color in (0..216u8).filter(|c| present[*c as usize]) {
            row.fill(0);
            for (dy, line) in band.chunks(width).enumerate() {
                for (x, &c) in line.iter().enumerate() {
                    if c == color {
                        row[x] |= 1 << dy;
                    }
                }
            }
            write!(out, "#{color}")?;
            write_sixel_row(&row, out)?;
            write!(out, "$")?; // carriage return within the band
        }
        write!(out, "-")?; // next band
    }

    write!(out, "\x1b\\")?;
    writeln!(out)?;
    Ok(())
}

/// Emit one sixel row with run-length encoding (`!<count><char>`).
fn write_sixel_row(row: &[u8], out: &mut impl Write) -> io::Result<()> {
    let mut i = 0;
    while i < row.len() {
        let bits = row[i];
        let run = row[i..].iter().take_while(|&&b| b == bits).count();
        let ch = (bits + 0x3f) as char;
        if run >= 4 {
            write!(out, "!{run}{ch}")?;
        } else {
            for _ in 0..run {
                write!(out, "{ch}")?;
            }
        }
        i += run;
    }
    Ok(())
}

/// Map an (opaque) pixel to the nearest 6x6x6 cube index.
fn nearest_color(pixel: &Rgba<u8>) -> u8 {
    let level = |v: u8| ((u32::from(v) + 25) / 51).min(5);
    (level(pixel[0]) * 36 + level(pixel[1]) * 6 + level(pixel[2])) as u8
}

/// Blend transparency onto black so every protocol sees opaque pixels.
fn flatten_alpha(img: &DynamicImage) -> RgbaImage {
    let mut rgba = img.to_rgba8();
    for p in rgba.pixels_mut() {
        let a = u32::from(p[3]);
        if a < 255 {
            for c in 0..3 {
                p[c] = (u32::from(p[c]) * a / 255) as u8;
            }
            p[3] = 255;
        }
    }
    rgba
}

/// Half-block fallback: each character cell shows two vertically stacked
/// pixels (upper half block: foreground = top pixel, background = bottom).
fn write_halfblock(img: &DynamicImage, depth: ColorDepth, out: &mut impl Write) -> Result<()> {
    let rgba = flatten_alpha(img);
    let (width, height) = rgba.dimensions();

    for y in (0..height).step_by(2) {
        for x in 0..width {
            let top = *rgba.get_pixel(x, y);
            let bottom = if y + 1 < height {
                *rgba.get_pixel(x, y + 1)
            } else {
                Rgba([0, 0, 0, 255])
            };

            match depth {
                ColorDepth::TrueColor => write!(
                    out,
                    "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m\u{2580}",
                    top[0], top[1], top[2], bottom[0], bottom[1], bottom[2]
                )?,
                ColorDepth::Colors256 => write!(
                    out,
                    "\x1b[38;5;{}m\x1b[48;5;{}m\u{2580}",
                    to_256_color(top[0], top[1], top[2]),
                    to_256_color(bottom[0], bottom[1], bottom[2])
                )?,
                ColorDepth::Mono => {
                    // No color: shade by the average brightness of the cell.
                    let luma = |p: Rgba<u8>| {
                        (u32::from(p[0]) * 299 + u32::from(p[1]) * 587 + u32::from(p[2]) * 114)
                            / 1000
                    };
                    let shade = [' ', '░', '▒', '▓', '█'];
                    let level = (luma(top) + luma(bottom)) / 2 * 5 / 256;
                    write!(out, "{}", shade[level as usize])?;
                }
            }
        }
        if depth != ColorDepth::Mono {
            write!(out, "\x1b[0m")?;
        }
        writeln!(out)?;
    }
    Ok(())
}

/// Map RGB to 256-color xterm palette (16-231 color cube).
fn to_256_color(r: u8, g: u8, b: u8) -> u8 {
    let level = |v: u8| ((u16::from(v) + 25) / 51).min(5) as u8;
    16 + 36 * level(r) + 6 * level(g) + level(b)
}

/// Encode a DynamicImage as PNG bytes.
fn encode_png(img: &DynamicImage) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    img.write_to(&mut io::Cursor::new(&mut buf), image::ImageFormat::Png)?;
    Ok(buf)
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

    fn caps(protocol: ImageProtocol, cols: u16, rows: u16) -> TerminalCapabilities {
        TerminalCapabilities {
            protocol,
            color_depth: ColorDepth::TrueColor,
            cols,
            rows,
            cell_px: (10, 20),
        }
    }

    #[test]
    fn test_pixel_budget_depends_on_protocol() {
        // Half blocks: one pixel per column, two per row.
        assert_eq!(
            pixel_budget(&caps(ImageProtocol::Halfblock, 80, 24), 4),
            (80, 40)
        );
        // Graphics protocols: real pixels from the cell size.
        assert_eq!(
            pixel_budget(&caps(ImageProtocol::Kitty, 80, 24), 4),
            (800, 400)
        );
        assert_eq!(
            pixel_budget(&caps(ImageProtocol::Sixel, 80, 3), 10),
            (800, 20)
        );
    }

    #[test]
    fn test_halfblock_output_never_exceeds_terminal_width() {
        // A wide image must be scaled to the column count, not wrapped.
        let img = DynamicImage::new_rgba8(640, 480);
        let c = caps(ImageProtocol::Halfblock, 80, 24);
        let (max_w, max_h) = pixel_budget(&c, 2);
        let fitted = resize_to_fit(&img, max_w, max_h);
        let mut out = Vec::new();
        write_halfblock(&fitted, ColorDepth::Mono, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert!(
            lines.iter().all(|l| l.chars().count() <= 80),
            "{}",
            lines[0].chars().count()
        );
        assert!(lines.len() <= 22);
        assert!(!lines.is_empty());
    }

    #[test]
    fn test_halfblock_color_modes() {
        let mut img = RgbaImage::new(2, 2);
        img.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
        img.put_pixel(0, 1, Rgba([0, 0, 255, 255]));
        let img = DynamicImage::ImageRgba8(img);

        let mut out = Vec::new();
        write_halfblock(&img, ColorDepth::TrueColor, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.starts_with("\x1b[38;2;255;0;0m\x1b[48;2;0;0;255m\u{2580}"),
            "{text:?}"
        );
        assert!(text.ends_with("\x1b[0m\n"));

        let mut out = Vec::new();
        write_halfblock(&img, ColorDepth::Colors256, &mut out).unwrap();
        assert!(String::from_utf8(out)
            .unwrap()
            .starts_with("\x1b[38;5;196m\x1b[48;5;21m"));

        let mut out = Vec::new();
        write_halfblock(
            &DynamicImage::ImageRgba8(RgbaImage::from_pixel(3, 2, Rgba([255, 255, 255, 255]))),
            ColorDepth::Mono,
            &mut out,
        )
        .unwrap();
        assert_eq!(String::from_utf8(out).unwrap(), "███\n");
    }

    #[test]
    fn test_transparent_pixels_blend_to_black() {
        let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(1, 1, Rgba([200, 100, 50, 0])));
        assert_eq!(flatten_alpha(&img).get_pixel(0, 0), &Rgba([0, 0, 0, 255]));
    }

    #[test]
    fn test_sixel_structure_and_run_length_encoding() {
        let img = DynamicImage::ImageRgba8(RgbaImage::from_pixel(10, 7, Rgba([255, 0, 0, 255])));
        let mut out = Vec::new();
        write_sixel(&img, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("\x1bPq\"1;1;10;7"), "{text:?}");
        assert!(text.trim_end().ends_with("\x1b\\"));
        // Pure red is cube index 180. First band: all six rows set (`~`),
        // second band: only the top row (`@`), both run-length encoded.
        assert!(text.contains("#180!10~$-"), "{text:?}");
        assert!(text.contains("#180!10@$-"), "{text:?}");

        let mut row_out = Vec::new();
        write_sixel_row(&[1, 1, 3, 3, 3, 3, 3], &mut row_out).unwrap();
        assert_eq!(String::from_utf8(row_out).unwrap(), "@@!5B");
    }

    #[test]
    fn test_kitty_chunks_and_iterm2_header() {
        let img = DynamicImage::ImageRgba8(RgbaImage::from_fn(64, 64, |x, y| {
            Rgba([(x * 4) as u8, (y * 4) as u8, ((x * y) % 255) as u8, 255])
        }));
        let mut out = Vec::new();
        write_kitty(&img, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("\x1b_Ga=T,f=100,m="));
        let last = text.trim_end().rsplit("\x1b_G").next().unwrap();
        assert!(
            last.starts_with("m=0;") || last.starts_with("a=T,f=100,m=0;"),
            "{last:?}"
        );

        let mut out = Vec::new();
        write_iterm2(&img, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("\x1b]1337;File=inline=1;size="));
        assert!(text.contains(";width=64px;height=64px;"));
    }
}
