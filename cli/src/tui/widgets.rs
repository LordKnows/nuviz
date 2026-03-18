use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};

use crate::tui::app::App;
use crate::tui::chart;

/// Render a braille chart as a ratatui widget.
pub struct BrailleChart<'a> {
    pub app: &'a App,
    pub metric: &'a str,
    pub focused: bool,
}

impl<'a> BrailleChart<'a> {
    pub fn render_to_paragraph(&self, area: Rect) -> Paragraph<'a> {
        let border_style = if self.focused {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        };

        let block = Block::default()
            .title(format!(" {} ", self.metric))
            .borders(Borders::ALL)
            .border_style(border_style);

        // Collect series for all experiments
        let inner_width = area.width.saturating_sub(2) as usize;
        let inner_height = area.height.saturating_sub(2) as usize;

        if inner_width < 10 || inner_height < 2 {
            return Paragraph::new("(too small)").block(block);
        }

        // Use first experiment for now
        let lines: Vec<Line<'a>> = if let Some(exp_name) = self.app.experiment_names.first() {
            let data = self.app.metric_series(exp_name, self.metric);
            let rendered = chart::plot_series(&data, inner_width, inner_height, self.metric);
            rendered
                .into_iter()
                .map(|s| Line::from(Span::raw(s)))
                .collect()
        } else {
            vec![Line::from("No experiments")]
        };

        Paragraph::new(lines).block(block)
    }
}

/// Render an info panel showing experiment status.
pub fn info_panel<'a>(app: &App, focused: bool) -> Paragraph<'a> {
    let border_style = if focused {
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let block = Block::default()
        .title(" Info ")
        .borders(Borders::ALL)
        .border_style(border_style);

    let mut lines: Vec<Line> = Vec::new();

    for exp_name in &app.experiment_names {
        lines.push(Line::from(Span::styled(
            format!("Experiment: {exp_name}"),
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        )));

        if let Some(step) = app.current_step(exp_name) {
            lines.push(Line::from(format!("  Step: {step}")));
        }

        // Show best values for detected metrics
        let (m1, m2) = &app.chart_metrics;
        if let Some(best) = app.best_metric(exp_name, m1) {
            lines.push(Line::from(format!("  Best {m1}: {best:.4}")));
        }
        if let Some(best) = app.best_metric(exp_name, m2) {
            lines.push(Line::from(format!("  Best {m2}: {best:.4}")));
        }

        lines.push(Line::from(""));
    }

    // Show alerts
    if !app.alerts.is_empty() {
        lines.push(Line::from(Span::styled(
            "Alerts:",
            Style::default().fg(Color::Yellow),
        )));
        for alert in app.alerts.iter().rev().take(5) {
            lines.push(Line::from(Span::styled(
                format!("  ⚠ {alert}"),
                Style::default().fg(Color::Yellow),
            )));
        }
    }

    // Keybindings help
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "q:quit  Tab:focus  [/]:zoom",
        Style::default().fg(Color::DarkGray),
    )));

    Paragraph::new(lines).block(block)
}
