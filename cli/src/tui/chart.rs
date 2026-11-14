//! Braille line charts.
//!
//! Each terminal cell maps to a 2x4 dot grid using Unicode braille characters
//! (U+2800–U+28FF). The bit pattern for each dot position is:
//!
//! ```text
//!   Col 0  Col 1
//!   0x01   0x08    row 0
//!   0x02   0x10    row 1
//!   0x04   0x20    row 2
//!   0x40   0x80    row 3
//! ```
//!
//! Series are reduced to one min/max/first/last bucket per pixel column
//! before drawing, so the cost of a frame depends on the chart width rather
//! than on how many points were logged.

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};

/// Dot bit positions for braille encoding: [col][row]
const BRAILLE_DOTS: [[u8; 4]; 2] = [
    [0x01, 0x02, 0x04, 0x40], // left column
    [0x08, 0x10, 0x20, 0x80], // right column
];

/// Colors assigned to experiments, in order.
pub const PALETTE: [Color; 8] = [
    Color::Cyan,
    Color::Yellow,
    Color::Green,
    Color::Magenta,
    Color::Red,
    Color::Blue,
    Color::LightCyan,
    Color::LightYellow,
];

pub fn series_color(index: usize) -> Color {
    PALETTE[index % PALETTE.len()]
}

/// A canvas for drawing with braille characters. Each cell remembers which
/// series drew into it last, so series can be colored.
pub struct BrailleCanvas {
    width: usize,
    height: usize,
    cells: Vec<u8>,
    owner: Vec<u8>,
}

impl BrailleCanvas {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cells: vec![0; width * height],
            owner: vec![0; width * height],
        }
    }

    /// Pixel dimensions (each cell is 2 wide x 4 tall in dots)
    pub fn pixel_width(&self) -> usize {
        self.width * 2
    }

    pub fn pixel_height(&self) -> usize {
        self.height * 4
    }

    /// Set a single pixel (in dot coordinates) on behalf of `series`.
    pub fn set_pixel(&mut self, x: usize, y: usize, series: u8) {
        let (cell_x, cell_y) = (x / 2, y / 4);
        if cell_x < self.width && cell_y < self.height {
            let idx = cell_y * self.width + cell_x;
            self.cells[idx] |= BRAILLE_DOTS[x % 2][y % 4];
            self.owner[idx] = series;
        }
    }

    /// Draw a line using Bresenham's algorithm.
    pub fn draw_line(&mut self, x0: usize, y0: usize, x1: usize, y1: usize, series: u8) {
        let (mut x0, mut y0) = (x0 as isize, y0 as isize);
        let (x1, y1) = (x1 as isize, y1 as isize);

        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx: isize = if x0 < x1 { 1 } else { -1 };
        let sy: isize = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;

        loop {
            if x0 >= 0 && y0 >= 0 {
                self.set_pixel(x0 as usize, y0 as usize, series);
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

    /// Character and owning series of the cell at (`col`, `row`).
    fn cell(&self, col: usize, row: usize) -> (char, u8) {
        let idx = row * self.width + col;
        let bits = self.cells[idx];
        if bits == 0 {
            (' ', 0)
        } else {
            (
                char::from_u32(0x2800 + u32::from(bits)).unwrap_or(' '),
                self.owner[idx],
            )
        }
    }

    /// Render the canvas to plain strings (one per row).
    pub fn render(&self) -> Vec<String> {
        (0..self.height)
            .map(|row| (0..self.width).map(|col| self.cell(col, row).0).collect())
            .collect()
    }
}

/// Summary of the points that fall into one pixel column.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bucket {
    pub min: f64,
    pub max: f64,
    pub first: f64,
    pub last: f64,
}

/// Reduce a series to at most `n` buckets across `x_range`.
///
/// * `smoothing` in `[0, 1)` applies a bias-corrected exponential moving
///   average over the whole series (before windowing, so zooming does not
///   change the curve).
/// * With `log_y`, values are mapped to `log10` and non-positive values are
///   dropped.
pub fn bucketize(
    points: impl Iterator<Item = (f64, f64)>,
    x_range: (f64, f64),
    n: usize,
    smoothing: f64,
    log_y: bool,
) -> Vec<Option<Bucket>> {
    let mut buckets: Vec<Option<Bucket>> = vec![None; n];
    if n == 0 {
        return buckets;
    }
    let (x0, x1) = x_range;
    let span = x1 - x0;

    let mut ema = 0.0;
    let mut weight = 1.0; // smoothing^k, for bias correction

    for (x, raw) in points {
        let mut y = raw;
        if smoothing > 0.0 {
            ema = smoothing * ema + (1.0 - smoothing) * raw;
            weight *= smoothing;
            y = ema / (1.0 - weight);
        }
        if log_y {
            if y <= 0.0 {
                continue;
            }
            y = y.log10();
        }
        if !y.is_finite() || x < x0 || x > x1 {
            continue;
        }

        let idx = if span > 0.0 {
            (((x - x0) / span) * (n - 1) as f64).round() as usize
        } else {
            n / 2
        };
        match &mut buckets[idx.min(n - 1)] {
            Some(b) => {
                b.min = b.min.min(y);
                b.max = b.max.max(y);
                b.last = y;
            }
            slot => {
                *slot = Some(Bucket {
                    min: y,
                    max: y,
                    first: y,
                    last: y,
                })
            }
        }
    }
    buckets
}

/// How to label the x axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XFormat {
    Count,
    Duration,
}

pub struct ChartSpec {
    /// Total size in terminal cells, including axes.
    pub width: usize,
    pub height: usize,
    pub x_range: (f64, f64),
    pub x_format: XFormat,
    pub log_y: bool,
    pub smoothing: f64,
}

const Y_LABEL_WIDTH: usize = 9;
const AXIS_STYLE: Style = Style::new().fg(Color::DarkGray);

/// Plot several series on shared axes. Returns exactly `spec.height` lines
/// (or a single message line when there is nothing to draw).
pub fn plot<I>(series: Vec<(Color, I)>, spec: &ChartSpec) -> Vec<Line<'static>>
where
    I: Iterator<Item = (f64, f64)>,
{
    let chart_width = spec.width.saturating_sub(Y_LABEL_WIDTH + 1);
    let chart_height = spec.height.saturating_sub(2); // x axis + labels
    if chart_width < 4 || chart_height < 2 {
        return vec![Line::from("(too small)")];
    }

    let mut canvas = BrailleCanvas::new(chart_width, chart_height);
    let (pw, ph) = (canvas.pixel_width(), canvas.pixel_height());

    let mut colors = Vec::with_capacity(series.len());
    let bucketed: Vec<Vec<Option<Bucket>>> = series
        .into_iter()
        .map(|(color, points)| {
            colors.push(color);
            bucketize(points, spec.x_range, pw, spec.smoothing, spec.log_y)
        })
        .collect();

    let all = || bucketed.iter().flatten().flatten();
    let y_min = all().map(|b| b.min).fold(f64::INFINITY, f64::min);
    let y_max = all().map(|b| b.max).fold(f64::NEG_INFINITY, f64::max);
    if !y_min.is_finite() || !y_max.is_finite() {
        let reason = if spec.log_y {
            "(no positive values to show on a log axis)"
        } else {
            "(no data)"
        };
        return vec![Line::from(reason)];
    }
    // A flat series still needs a visible band around it.
    let (y_min, y_max) = if y_max - y_min < f64::EPSILON * y_max.abs().max(1.0) {
        let pad = if y_max == 0.0 {
            0.5
        } else {
            y_max.abs() * 0.05
        };
        (y_min - pad, y_max + pad)
    } else {
        (y_min, y_max)
    };
    let to_py = |y: f64| -> usize {
        let t = (y_max - y) / (y_max - y_min);
        ((t * (ph - 1) as f64).round() as usize).min(ph - 1)
    };

    for (i, buckets) in bucketed.iter().enumerate() {
        let id = (i + 1).min(255) as u8;
        let mut prev: Option<(usize, f64)> = None;
        for (px, bucket) in buckets.iter().enumerate() {
            let Some(b) = bucket else { continue };
            if let Some((prev_x, prev_y)) = prev {
                canvas.draw_line(prev_x, to_py(prev_y), px, to_py(b.first), id);
            }
            canvas.draw_line(px, to_py(b.max), px, to_py(b.min), id);
            prev = Some((px, b.last));
        }
    }

    let mut lines: Vec<Line<'static>> = Vec::with_capacity(spec.height);

    // Label every row on short charts, every other row on tall ones.
    let label_every = if chart_height > 10 { 2 } else { 1 };
    for row in 0..chart_height {
        let labelled = row == chart_height - 1 || row % label_every == 0;
        let label = if labelled {
            let t = row as f64 / (chart_height - 1) as f64;
            let y = y_max - t * (y_max - y_min);
            if spec.log_y {
                format_number(10f64.powf(y))
            } else {
                format_axis(y, (y_max - y_min) / (chart_height - 1) as f64)
            }
        } else {
            String::new()
        };
        let mut spans = vec![Span::styled(
            format!(
                "{label:>Y_LABEL_WIDTH$}{}",
                if labelled { '┤' } else { '│' }
            ),
            AXIS_STYLE,
        )];

        // Group consecutive cells of the same series into one span.
        let mut run = String::new();
        let mut run_owner = 0u8;
        for col in 0..chart_width {
            let (ch, owner) = canvas.cell(col, row);
            if owner != run_owner && !run.is_empty() {
                spans.push(styled_run(std::mem::take(&mut run), run_owner, &colors));
            }
            run_owner = owner;
            run.push(ch);
        }
        if !run.is_empty() {
            spans.push(styled_run(run, run_owner, &colors));
        }
        lines.push(Line::from(spans));
    }

    let (axis, labels) = x_axis(spec.x_range, spec.x_format, chart_width);
    let pad = " ".repeat(Y_LABEL_WIDTH);
    lines.push(Line::from(Span::styled(
        format!("{pad}└{axis}"),
        AXIS_STYLE,
    )));
    lines.push(Line::from(Span::styled(
        format!("{pad} {labels}"),
        AXIS_STYLE,
    )));
    lines
}

fn styled_run(text: String, owner: u8, colors: &[Color]) -> Span<'static> {
    match owner.checked_sub(1).and_then(|i| colors.get(i as usize)) {
        Some(color) => Span::styled(text, Style::new().fg(*color)),
        None => Span::raw(text),
    }
}

/// Build the x-axis rule with tick marks and the label row beneath it.
fn x_axis(range: (f64, f64), format: XFormat, width: usize) -> (String, String) {
    let ticks = if width >= 60 { 5 } else { 3 };
    let mut axis: Vec<char> = vec!['─'; width];
    let mut labels: Vec<char> = vec![' '; width];
    let mut next_free = 0usize;

    for i in 0..ticks {
        let t = i as f64 / (ticks - 1) as f64;
        let col = (t * (width - 1) as f64).round() as usize;
        axis[col] = '┬';

        let value = range.0 + t * (range.1 - range.0);
        let text: Vec<char> = match format {
            XFormat::Count => format_count(value),
            XFormat::Duration => format_duration(value),
        }
        .chars()
        .collect();

        // Left-align the first label, right-align the last, center the rest.
        let start = if i == 0 {
            0
        } else if i == ticks - 1 {
            width.saturating_sub(text.len())
        } else {
            col.saturating_sub(text.len() / 2)
        };
        if start >= next_free && start + text.len() <= width {
            labels[start..start + text.len()].copy_from_slice(&text);
            next_free = start + text.len() + 1;
        }
    }
    (axis.into_iter().collect(), labels.into_iter().collect())
}

/// Format an axis label with enough decimals that neighbouring labels,
/// `step` apart, read differently (a nearly flat curve would otherwise show
/// the same label on every row).
pub fn format_axis(v: f64, step: f64) -> String {
    let abs = v.abs();
    if abs >= 10_000.0 || step <= 0.0 || !step.is_finite() {
        return format_number(v);
    }
    let decimals = (-step.log10()).ceil().max(0.0) as usize + 1;
    let text = format!("{v:.decimals$}");
    if text.len() <= Y_LABEL_WIDTH {
        text
    } else {
        format!("{v:.2e}")
    }
}

/// Format a number compactly for axis labels.
pub fn format_number(v: f64) -> String {
    let abs = v.abs();
    if abs == 0.0 {
        "0".into()
    } else if abs >= 1_000_000.0 {
        format!("{:.1}M", v / 1_000_000.0)
    } else if abs >= 10_000.0 {
        format!("{:.1}k", v / 1_000.0)
    } else if abs >= 100.0 {
        format!("{v:.1}")
    } else if abs >= 1.0 {
        format!("{v:.2}")
    } else if abs >= 0.001 {
        format!("{v:.4}")
    } else {
        format!("{v:.2e}")
    }
}

/// Format a step count: `0`, `950`, `1.5k`, `20k`, `1.2M`.
pub fn format_count(v: f64) -> String {
    let abs = v.abs();
    let trim = |s: String| s.strip_suffix(".0").map(str::to_string).unwrap_or(s);
    if abs >= 1_000_000.0 {
        trim(format!("{:.1}", v / 1_000_000.0)) + "M"
    } else if abs >= 1_000.0 {
        trim(format!("{:.1}", v / 1_000.0)) + "k"
    } else {
        format!("{}", v.round() as i64)
    }
}

/// Format seconds as a short duration: `45s`, `12m30s`, `1h05m`, `2d03h`.
pub fn format_duration(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "-".into();
    }
    let s = seconds.round() as u64;
    let (d, h, m, sec) = (s / 86_400, (s % 86_400) / 3_600, (s % 3_600) / 60, s % 60);
    if d > 0 {
        format!("{d}d{h:02}h")
    } else if h > 0 {
        format!("{h}h{m:02}m")
    } else if m > 0 {
        format!("{m}m{sec:02}s")
    } else {
        format!("{sec}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(lines: &[Line<'_>]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    fn spec(width: usize, height: usize, x_range: (f64, f64)) -> ChartSpec {
        ChartSpec {
            width,
            height,
            x_range,
            x_format: XFormat::Count,
            log_y: false,
            smoothing: 0.0,
        }
    }

    #[test]
    fn test_set_pixel_renders_braille() {
        let mut canvas = BrailleCanvas::new(1, 1);
        canvas.set_pixel(0, 0, 1); // top-left dot
        canvas.set_pixel(1, 0, 1); // top-right dot
        assert_eq!(canvas.render(), ["\u{2809}"]);
    }

    #[test]
    fn test_empty_cells_are_spaces() {
        let canvas = BrailleCanvas::new(3, 2);
        assert_eq!(canvas.render(), ["   ", "   "]);
    }

    #[test]
    fn test_draw_line_spans_canvas() {
        let mut canvas = BrailleCanvas::new(5, 1);
        canvas.draw_line(0, 0, 9, 0, 1);
        assert!(canvas.render()[0].chars().all(|c| c == '\u{2809}'));

        let mut canvas = BrailleCanvas::new(1, 3);
        canvas.draw_line(0, 0, 0, 11, 1);
        assert!(canvas.render().iter().all(|r| r == "\u{2847}"));
    }

    #[test]
    fn test_out_of_bounds_pixel_ignored() {
        let mut canvas = BrailleCanvas::new(2, 2);
        canvas.set_pixel(100, 100, 1);
        assert_eq!(canvas.render(), ["  ", "  "]);
    }

    #[test]
    fn test_bucketize_keeps_extremes_of_dense_data() {
        // 100k points into 10 buckets: the spike must survive downsampling.
        let points = (0..100_000).map(|i| (i as f64, if i == 54_321 { 99.0 } else { 1.0 }));
        let buckets = bucketize(points, (0.0, 99_999.0), 10, 0.0, false);
        assert_eq!(buckets.len(), 10);
        assert!(buckets.iter().all(Option::is_some));
        let max = buckets.iter().flatten().map(|b| b.max).fold(0.0, f64::max);
        assert_eq!(max, 99.0);
        assert_eq!(buckets[5].unwrap().min, 1.0);
    }

    #[test]
    fn test_bucketize_window_excludes_points_outside_range() {
        let points = (0..100).map(|i| (i as f64, i as f64));
        let buckets = bucketize(points, (90.0, 99.0), 10, 0.0, false);
        assert_eq!(buckets[0].unwrap().first, 90.0);
        assert_eq!(buckets[9].unwrap().last, 99.0);
    }

    #[test]
    fn test_bucketize_log_drops_non_positive() {
        let points = [(0.0, 100.0), (1.0, 0.0), (2.0, -5.0), (3.0, 0.01)].into_iter();
        let buckets = bucketize(points, (0.0, 3.0), 4, 0.0, true);
        assert_eq!(buckets[0].unwrap().first, 2.0);
        assert!(buckets[1].is_none() && buckets[2].is_none());
        assert_eq!(buckets[3].unwrap().first, -2.0);
    }

    #[test]
    fn test_smoothing_is_bias_corrected_and_damps_noise() {
        // A constant series stays constant from the very first point.
        let flat = bucketize(
            (0..50).map(|i| (i as f64, 3.0)),
            (0.0, 49.0),
            50,
            0.9,
            false,
        );
        assert!(flat.iter().flatten().all(|b| (b.first - 3.0).abs() < 1e-9));

        // An alternating series is pulled towards its mean.
        let noisy = (0..200).map(|i| (i as f64, if i % 2 == 0 { 0.0 } else { 2.0 }));
        let smooth = bucketize(noisy, (0.0, 199.0), 200, 0.9, false);
        let tail = smooth[199].unwrap().last;
        assert!((tail - 1.0).abs() < 0.2, "{tail}");
    }

    #[test]
    fn test_degenerate_x_range() {
        let buckets = bucketize([(5.0, 1.0)].into_iter(), (5.0, 5.0), 8, 0.0, false);
        assert_eq!(buckets.iter().filter(|b| b.is_some()).count(), 1);
        assert!(bucketize([(5.0, 1.0)].into_iter(), (0.0, 1.0), 0, 0.0, false).is_empty());
    }

    #[test]
    fn test_plot_layout_axes_and_height() {
        let series = vec![(Color::Cyan, (0..=1000).map(|i| (i as f64, i as f64 / 10.0)))];
        let lines = plot(series, &spec(60, 12, (0.0, 1000.0)));
        let rows = text(&lines);
        assert_eq!(rows.len(), 12);
        assert!(rows[0].trim_start().starts_with("100.0┤"), "{:?}", rows[0]);
        assert!(rows[9].trim_start().starts_with("0.0┤"), "{:?}", rows[9]);
        assert!(rows[10].contains('└') && rows[10].contains('┬'));
        assert!(rows[11].trim_start().starts_with('0') && rows[11].trim_end().ends_with("1k"));
        // An increasing line: bottom-left and top-right corners are drawn.
        assert_ne!(rows[9].chars().nth(Y_LABEL_WIDTH + 1), Some(' '));
        assert_ne!(rows[0].chars().last(), Some(' '));
    }

    #[test]
    fn test_plot_colors_each_series() {
        let series = vec![
            (Color::Cyan, vec![(0.0, 0.0), (10.0, 0.0)].into_iter()),
            (Color::Yellow, vec![(0.0, 10.0), (10.0, 10.0)].into_iter()),
        ];
        let lines = plot(series, &spec(40, 8, (0.0, 10.0)));
        let color_of = |line: &Line<'_>| {
            line.spans
                .iter()
                .find(|s| {
                    s.content
                        .chars()
                        .any(|c| ('\u{2801}'..='\u{28ff}').contains(&c))
                })
                .and_then(|s| s.style.fg)
        };
        assert_eq!(color_of(&lines[0]), Some(Color::Yellow));
        assert_eq!(color_of(&lines[5]), Some(Color::Cyan));
    }

    #[test]
    fn test_plot_flat_series_and_empty_and_tiny() {
        let flat = vec![(Color::Cyan, vec![(0.0, 2.0), (5.0, 2.0)].into_iter())];
        let rows = text(&plot(flat, &spec(40, 8, (0.0, 5.0))));
        assert_eq!(rows.len(), 8);
        assert!(
            rows[0].contains("2.10") && rows[5].contains("1.90"),
            "{rows:?}"
        );

        let empty: Vec<(Color, std::vec::IntoIter<(f64, f64)>)> =
            vec![(Color::Cyan, vec![].into_iter())];
        assert_eq!(text(&plot(empty, &spec(40, 8, (0.0, 1.0)))), ["(no data)"]);

        let tiny = vec![(Color::Cyan, vec![(0.0, 1.0)].into_iter())];
        assert_eq!(text(&plot(tiny, &spec(8, 3, (0.0, 1.0)))), ["(too small)"]);
    }

    #[test]
    fn test_plot_log_axis_labels_are_in_data_units() {
        let series = vec![(Color::Cyan, vec![(0.0, 100.0), (1.0, 0.01)].into_iter())];
        let mut s = spec(40, 8, (0.0, 1.0));
        s.log_y = true;
        let rows = text(&plot(series, &s));
        assert!(rows[0].contains("100.0┤"), "{rows:?}");
        assert!(rows[5].contains("0.0100┤"), "{rows:?}");
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0.0), "0");
        assert_eq!(format_number(1_500_000.0), "1.5M");
        assert_eq!(format_number(25_000.0), "25.0k");
        assert_eq!(format_number(3.25), "3.25");
        assert_eq!(format_number(0.0512), "0.0512");
        assert_eq!(format_number(0.00001), "1.00e-5");
        assert_eq!(format_number(-2.5), "-2.50");
    }

    #[test]
    fn test_format_axis_adapts_to_step() {
        assert_eq!(format_axis(0.001, 0.00002), "0.001000");
        assert_eq!(format_axis(0.5, 0.07), "0.500");
        assert_eq!(format_axis(100.0, 10.0), "100.0");
        assert_eq!(format_axis(25_000.0, 100.0), "25.0k");
        assert_eq!(format_axis(1.0, 1e-12), "1.00e0");
    }

    #[test]
    fn test_format_count_and_duration() {
        assert_eq!(format_count(0.0), "0");
        assert_eq!(format_count(950.0), "950");
        assert_eq!(format_count(1_500.0), "1.5k");
        assert_eq!(format_count(20_000.0), "20k");
        assert_eq!(format_count(1_200_000.0), "1.2M");

        assert_eq!(format_duration(45.0), "45s");
        assert_eq!(format_duration(750.0), "12m30s");
        assert_eq!(format_duration(3_900.0), "1h05m");
        assert_eq!(format_duration(183_600.0), "2d03h");
        assert_eq!(format_duration(f64::NAN), "-");
    }
}
