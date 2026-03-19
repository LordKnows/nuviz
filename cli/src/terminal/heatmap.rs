use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};

/// Result of computing a per-pixel error heatmap.
pub struct HeatmapResult {
    /// The heatmap image (turbo colormap)
    pub image: DynamicImage,
    /// Mean absolute error across all pixels and channels
    pub mae: f64,
    /// Max absolute error
    pub max_error: u8,
    /// PSNR in dB (infinite if images are identical)
    pub psnr: f64,
}

/// Generate a per-pixel error heatmap between two images.
///
/// If images differ in size, the smaller is resized to match the larger.
/// Uses a turbo-style colormap: blue (low error) → red (high error).
pub fn generate_error_heatmap(img_a: &DynamicImage, img_b: &DynamicImage) -> HeatmapResult {
    let (a, b) = match_dimensions(img_a, img_b);
    let (width, height) = a.dimensions();

    let rgba_a = a.to_rgba8();
    let rgba_b = b.to_rgba8();

    let mut heatmap = RgbaImage::new(width, height);
    let mut total_error: f64 = 0.0;
    let mut max_error: u8 = 0;
    let mut total_sq_error: f64 = 0.0;
    let pixel_count = (width * height) as f64;

    for y in 0..height {
        for x in 0..width {
            let pa = rgba_a.get_pixel(x, y);
            let pb = rgba_b.get_pixel(x, y);

            // Per-pixel error: mean of absolute channel differences (RGB only)
            let dr = (pa[0] as i16 - pb[0] as i16).unsigned_abs() as u8;
            let dg = (pa[1] as i16 - pb[1] as i16).unsigned_abs() as u8;
            let db = (pa[2] as i16 - pb[2] as i16).unsigned_abs() as u8;

            let pixel_error = ((dr as u16 + dg as u16 + db as u16) / 3) as u8;
            max_error = max_error.max(pixel_error);
            total_error += pixel_error as f64;

            // For PSNR: sum of squared errors per channel
            total_sq_error += dr as f64 * dr as f64;
            total_sq_error += dg as f64 * dg as f64;
            total_sq_error += db as f64 * db as f64;

            // Map to turbo colormap
            let color = turbo_colormap(pixel_error);
            heatmap.put_pixel(x, y, color);
        }
    }

    let mae = total_error / pixel_count;
    let mse = total_sq_error / (pixel_count * 3.0); // 3 channels
    let psnr = if mse > 0.0 {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    } else {
        f64::INFINITY
    };

    HeatmapResult {
        image: DynamicImage::ImageRgba8(heatmap),
        mae,
        max_error,
        psnr,
    }
}

/// Ensure both images have the same dimensions by resizing the smaller.
fn match_dimensions(a: &DynamicImage, b: &DynamicImage) -> (DynamicImage, DynamicImage) {
    let (aw, ah) = a.dimensions();
    let (bw, bh) = b.dimensions();

    if aw == bw && ah == bh {
        return (a.clone(), b.clone());
    }

    let target_w = aw.max(bw);
    let target_h = ah.max(bh);

    let a_resized = if aw != target_w || ah != target_h {
        eprintln!(
            "[nuviz] Warning: resizing image A from {}x{} to {}x{}",
            aw, ah, target_w, target_h
        );
        a.resize_exact(target_w, target_h, image::imageops::FilterType::Lanczos3)
    } else {
        a.clone()
    };

    let b_resized = if bw != target_w || bh != target_h {
        eprintln!(
            "[nuviz] Warning: resizing image B from {}x{} to {}x{}",
            bw, bh, target_w, target_h
        );
        b.resize_exact(target_w, target_h, image::imageops::FilterType::Lanczos3)
    } else {
        b.clone()
    };

    (a_resized, b_resized)
}

/// Simplified turbo colormap: maps 0..255 error to blue→cyan→green→yellow→red.
fn turbo_colormap(value: u8) -> Rgba<u8> {
    let t = value as f32 / 255.0;

    let r = (if t < 0.25 {
        0.0
    } else if t < 0.5 {
        (t - 0.25) * 4.0
    } else {
        1.0
    } * 255.0) as u8;

    let g = (if t < 0.25 {
        t * 4.0
    } else if t < 0.75 {
        1.0
    } else {
        (1.0 - t) * 4.0
    } * 255.0) as u8;

    let b = (if t < 0.5 { 1.0 - t * 2.0 } else { 0.0 } * 255.0) as u8;

    Rgba([r, g, b, 255])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_images() {
        let img = DynamicImage::new_rgba8(10, 10);
        let result = generate_error_heatmap(&img, &img);
        assert_eq!(result.mae, 0.0);
        assert_eq!(result.max_error, 0);
        assert!(result.psnr.is_infinite());
    }

    #[test]
    fn test_different_images() {
        let mut a = RgbaImage::new(2, 2);
        let mut b = RgbaImage::new(2, 2);

        for y in 0..2 {
            for x in 0..2 {
                a.put_pixel(x, y, Rgba([255, 0, 0, 255]));
                b.put_pixel(x, y, Rgba([0, 0, 255, 255]));
            }
        }

        let img_a = DynamicImage::ImageRgba8(a);
        let img_b = DynamicImage::ImageRgba8(b);
        let result = generate_error_heatmap(&img_a, &img_b);

        assert!(result.mae > 0.0);
        assert!(result.max_error > 0);
        assert!(result.psnr.is_finite());
    }

    #[test]
    fn test_different_sizes_resized() {
        let a = DynamicImage::new_rgba8(10, 10);
        let b = DynamicImage::new_rgba8(20, 20);
        let result = generate_error_heatmap(&a, &b);
        // Should not panic; heatmap should be 20x20
        assert_eq!(result.image.dimensions(), (20, 20));
    }

    #[test]
    fn test_turbo_colormap_endpoints() {
        let low = turbo_colormap(0);
        assert_eq!(low[2], 255); // blue channel high at 0
        let high = turbo_colormap(255);
        assert_eq!(high[0], 255); // red channel high at 255
    }
}
