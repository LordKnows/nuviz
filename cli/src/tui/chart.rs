/// Braille-character based chart rendering.
///
/// Each terminal cell maps to a 2x4 dot grid using Unicode braille characters
/// (U+2800–U+28FF). The bit pattern for each dot position is:
///
/// ```text
///   Col 0  Col 1
///   0x01   0x08    row 0
///   0x02   0x10    row 1
///   0x04   0x20    row 2
///   0x40   0x80    row 3
/// ```

/// Dot bit positions for braille encoding: [col][row]
const BRAILLE_DOTS: [[u8; 4]; 2] = [
    [0x01, 0x02, 0x04, 0x40], // left column
    [0x08, 0x10, 0x20, 0x80], // right column
];

/// A canvas for drawing with braille characters.
pub struct BrailleCanvas {
    /// Width in terminal cells
    width: usize,
    /// Height in terminal cells
    height: usize,
    /// Pixel buffer: each cell has an 8-bit braille pattern
    cells: Vec<Vec<u8>>,
}

impl BrailleCanvas {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cells: vec![vec![0u8; width]; height],
        }
    }

    /// Pixel dimensions (each cell is 2 wide x 4 tall in dots)
    pub fn pixel_width(&self) -> usize {
        self.width * 2
    }

    pub fn pixel_height(&self) -> usize {
        self.height * 4
    }

    /// Set a single pixel (in dot coordinates).
    pub fn set_pixel(&mut self, x: usize, y: usize) {
        let cell_x = x / 2;
        let cell_y = y / 4;
        let dot_x = x % 2;
        let dot_y = y % 4;

        if cell_x < self.width && cell_y < self.height {
            self.cells[cell_y][cell_x] |= BRAILLE_DOTS[dot_x][dot_y];
        }
    }

    /// Draw a line using Bresenham's algorithm.
    pub fn draw_line(&mut self, x0: usize, y0: usize, x1: usize, y1: usize) {
        let (mut x0, mut y0) = (x0 as isize, y0 as isize);
        let (x1, y1) = (x1 as isize, y1 as isize);

        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx: isize = if x0 < x1 { 1 } else { -1 };
        let sy: isize = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;

        loop {
            if x0 >= 0 && y0 >= 0 {
                self.set_pixel(x0 as usize, y0 as usize);
            }

            if x0 == x1 && y0 == y1 {
                break;
            }

            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
    }

    /// Render the canvas to a vector of strings (one per row).
    pub fn render(&self) -> Vec<String> {
        self.cells
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&bits| char::from_u32(0x2800 + bits as u32).unwrap_or(' '))
                    .collect()
            })
            .collect()
    }
}

/// Plot a data series as a braille line chart.
///
/// Returns rendered lines with Y-axis labels prepended.
pub fn plot_series(
    data: &[f64],
    width: usize,
    height: usize,
    label: &str,
) -> Vec<String> {
    if data.is_empty() || width == 0 || height == 0 {
        return vec![format!("{label}: (no data)")];
    }

    // Reserve space for Y-axis labels
    let label_width = 8;
    let chart_width = width.saturating_sub(label_width);
    if chart_width == 0 {
        return vec![format!("{label}: (too narrow)")];
    }

    let mut canvas = BrailleCanvas::new(chart_width, height);
    let pw = canvas.pixel_width();
    let ph = canvas.pixel_height();

    // Find data range
    let min_val = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = if (max_val - min_val).abs() < f64::EPSILON {
        1.0
    } else {
        max_val - min_val
    };

    // Map data points to pixel coordinates
    let points: Vec<(usize, usize)> = data
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let x = if data.len() > 1 {
                i * (pw - 1) / (data.len() - 1)
            } else {
                pw / 2
            };
            // Y is inverted (0 at top)
            let y = ((max_val - v) / range * (ph - 1) as f64) as usize;
            (x, y.min(ph - 1))
        })
        .collect();

    // Draw lines between consecutive points
    for window in points.windows(2) {
        canvas.draw_line(window[0].0, window[0].1, window[1].0, window[1].1);
    }

    // If single point, just set the pixel
    if points.len() == 1 {
        canvas.set_pixel(points[0].0, points[0].1);
    }

    // Render with Y-axis labels
    let rendered = canvas.render();
    let mut result = Vec::with_capacity(height + 1);

    // Title
    result.push(format!("─ {label} "));

    for (i, line) in rendered.iter().enumerate() {
        let y_val = if i == 0 {
            max_val
        } else if i == height - 1 {
            min_val
        } else {
            max_val - (i as f64 / (height - 1) as f64) * range
        };

        let y_label = format_number(y_val);
        result.push(format!("{y_label:>label_width$}┤{line}"));
    }

    result
}

/// Format a number compactly for axis labels.
fn format_number(v: f64) -> String {
    let abs = v.abs();
    if abs == 0.0 {
        "0".into()
    } else if abs >= 1_000_000.0 {
        format!("{:.1}M", v / 1_000_000.0)
    } else if abs >= 1_000.0 {
        format!("{:.1}k", v / 1_000.0)
    } else if abs >= 1.0 {
        format!("{v:.2}")
    } else if abs >= 0.001 {
        format!("{v:.4}")
    } else {
        format!("{v:.2e}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_braille_canvas_new() {
        let canvas = BrailleCanvas::new(10, 5);
        assert_eq!(canvas.pixel_width(), 20);
        assert_eq!(canvas.pixel_height(), 20);
    }

    #[test]
    fn test_set_pixel_renders_braille() {
        let mut canvas = BrailleCanvas::new(1, 1);
        canvas.set_pixel(0, 0); // top-left dot
        let rendered = canvas.render();
        assert_eq!(rendered.len(), 1);
        // U+2800 + 0x01 = U+2801 = ⠁
        assert_eq!(rendered[0], "\u{2801}");
    }

    #[test]
    fn test_set_multiple_pixels() {
        let mut canvas = BrailleCanvas::new(1, 1);
        canvas.set_pixel(0, 0); // bit 0x01
        canvas.set_pixel(1, 0); // bit 0x08
        let rendered = canvas.render();
        // 0x01 | 0x08 = 0x09 => U+2809 = ⠉
        assert_eq!(rendered[0], "\u{2809}");
    }

    #[test]
    fn test_empty_canvas_renders_blank_braille() {
        let canvas = BrailleCanvas::new(3, 2);
        let rendered = canvas.render();
        assert_eq!(rendered.len(), 2);
        // Empty braille = U+2800 = ⠀
        for line in &rendered {
            assert_eq!(line.chars().count(), 3);
            assert!(line.chars().all(|c| c == '\u{2800}'));
        }
    }

    #[test]
    fn test_draw_line_horizontal() {
        let mut canvas = BrailleCanvas::new(5, 1);
        canvas.draw_line(0, 0, 9, 0);
        let rendered = canvas.render();
        // All cells in the top row should have dots
        for c in rendered[0].chars() {
            assert_ne!(c, '\u{2800}', "expected dots in horizontal line");
        }
    }

    #[test]
    fn test_draw_line_vertical() {
        let mut canvas = BrailleCanvas::new(1, 3);
        canvas.draw_line(0, 0, 0, 11);
        let rendered = canvas.render();
        // All rows should have dots in the left column
        for line in &rendered {
            let c = line.chars().next().unwrap();
            assert_ne!(c, '\u{2800}', "expected dots in vertical line");
        }
    }

    #[test]
    fn test_out_of_bounds_pixel_ignored() {
        let mut canvas = BrailleCanvas::new(2, 2);
        canvas.set_pixel(100, 100); // should not panic
        let rendered = canvas.render();
        assert!(rendered.iter().all(|l| l.chars().all(|c| c == '\u{2800}')));
    }

    #[test]
    fn test_plot_series_empty() {
        let result = plot_series(&[], 40, 10, "test");
        assert_eq!(result.len(), 1);
        assert!(result[0].contains("no data"));
    }

    #[test]
    fn test_plot_series_single_point() {
        let result = plot_series(&[42.0], 40, 5, "value");
        assert!(result.len() > 1);
    }

    #[test]
    fn test_plot_series_monotonic() {
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let result = plot_series(&data, 40, 8, "linear");
        assert!(result.len() > 1);
    }

    #[test]
    fn test_plot_series_constant() {
        let data = vec![5.0; 20];
        let result = plot_series(&data, 40, 5, "constant");
        assert!(result.len() > 1);
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0.0), "0");
        assert_eq!(format_number(1500000.0), "1.5M");
        assert_eq!(format_number(2500.0), "2.5k");
        assert_eq!(format_number(3.14), "3.14");
        assert_eq!(format_number(0.0523), "0.0523");
    }
}
