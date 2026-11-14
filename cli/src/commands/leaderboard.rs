use anyhow::Result;
use comfy_table::{modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL, Cell, Color, Table};
use std::collections::BTreeSet;
use std::path::Path;

use crate::cli::{LeaderboardArgs, TableFormat};
use crate::commands::common::{display_labels, load_scope, print_no_experiments};
use crate::data::aggregation::{format_stat, format_stat_latex, mean_std, Stat};
use crate::data::experiment::{group_by_config, strip_seed_suffix, Experiment};
use crate::data::goal::{cmp_best_first, Goal, Goals};
use crate::data::summary::{RunSummary, ValueMode};
use crate::output::{csv_line, latex_escape, markdown_cell};

pub fn run(args: LeaderboardArgs, base_dir: &Path) -> Result<()> {
    let mut experiments = load_scope(base_dir, args.project.as_deref());
    if let Some(tag) = &args.tag {
        experiments.retain(|e| e.meta.tags.contains(tag));
    }
    if experiments.is_empty() {
        print_no_experiments(base_dir);
        return Ok(());
    }

    let runs: Vec<Run> = experiments
        .iter()
        .map(|e| Run {
            exp: e,
            summary: e.summary(),
        })
        .collect();

    let board = Board::build(&runs, &BoardOptions::from(&args));
    match args.format {
        TableFormat::Table => println!("{}", board.to_table()),
        TableFormat::Markdown => print!("{}", board.to_markdown()),
        TableFormat::Latex => print!("{}", board.to_latex()),
        TableFormat::Csv => print!("{}", board.to_csv()),
    }
    Ok(())
}

pub struct Run<'a> {
    pub exp: &'a Experiment,
    pub summary: RunSummary,
}

pub struct BoardOptions {
    pub sort: Option<String>,
    pub top: Option<usize>,
    pub reverse: bool,
    pub mode: ValueMode,
    pub show_steps: bool,
    pub aggregate: bool,
    pub metrics: Vec<String>,
    pub precision: usize,
}

impl From<&LeaderboardArgs> for BoardOptions {
    fn from(args: &LeaderboardArgs) -> Self {
        Self {
            sort: args.sort.clone(),
            top: args.top,
            reverse: args.asc,
            mode: args.value,
            show_steps: args.steps,
            aggregate: args.aggregate,
            metrics: args.metrics.clone(),
            precision: args.precision,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BoardCell {
    pub stat: Stat,
    /// Step the value was taken at (single runs only).
    pub step: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct Row {
    pub label: String,
    pub cells: Vec<Option<BoardCell>>,
    pub status: String,
    pub runs: usize,
}

/// A ranked table of experiments (or seed groups) by metric.
pub struct Board {
    pub metrics: Vec<String>,
    pub goals: Vec<Goal>,
    pub rows: Vec<Row>,
    pub sort_metric: Option<String>,
    mode: ValueMode,
    show_steps: bool,
    aggregate: bool,
    precision: usize,
    /// Per metric: best and second-best mean among the rows.
    highlights: Vec<(Option<f64>, Option<f64>)>,
}

impl Board {
    pub fn build(runs: &[Run<'_>], opts: &BoardOptions) -> Self {
        let mut goals = Goals::default();
        for run in runs {
            goals.merge(&run.exp.meta.metric_goals);
        }

        let available: BTreeSet<&str> = runs
            .iter()
            .flat_map(|r| r.summary.metrics.keys().map(String::as_str))
            .collect();
        let metrics: Vec<String> = if opts.metrics.is_empty() {
            available.iter().map(|m| m.to_string()).collect()
        } else {
            opts.metrics.clone()
        };
        let metric_goals: Vec<Goal> = metrics.iter().map(|m| goals.get(m)).collect();

        let value =
            |run: &Run<'_>, i: usize| run.summary.value(&metrics[i], opts.mode, metric_goals[i]);

        let mut rows: Vec<Row> = if opts.aggregate {
            let exps: Vec<Experiment> = runs.iter().map(|r| r.exp.clone()).collect();
            group_by_config(&exps)
                .values()
                .map(|members| {
                    let group: Vec<&Run<'_>> = runs
                        .iter()
                        .filter(|r| members.iter().any(|m| m.dir == r.exp.dir))
                        .collect();
                    let cells = (0..metrics.len())
                        .map(|i| {
                            let values: Vec<f64> = group
                                .iter()
                                .filter_map(|r| value(r, i))
                                .map(|(v, _)| v)
                                .collect();
                            mean_std(&values).map(|stat| BoardCell { stat, step: None })
                        })
                        .collect();
                    Row {
                        label: group_label(&group),
                        cells,
                        status: group_status(&group),
                        runs: group.len(),
                    }
                })
                .collect()
        } else {
            let refs: Vec<&Experiment> = runs.iter().map(|r| r.exp).collect();
            runs.iter()
                .zip(display_labels(&refs))
                .map(|(run, label)| Row {
                    label,
                    cells: (0..metrics.len())
                        .map(|i| {
                            value(run, i).and_then(|(v, step)| {
                                mean_std(&[v]).map(|stat| BoardCell {
                                    stat,
                                    step: Some(step),
                                })
                            })
                        })
                        .collect(),
                    status: run.exp.status.to_string(),
                    runs: 1,
                })
                .collect()
        };

        // Rank: best first (or worst first with --asc); rows without the
        // metric always go last. Ties keep a stable, name-based order.
        let sort_metric = opts
            .sort
            .clone()
            .or_else(|| metrics.first().cloned())
            .filter(|m| metrics.contains(m));
        rows.sort_by(|a, b| a.label.cmp(&b.label));
        if let Some(idx) = sort_metric
            .as_ref()
            .and_then(|m| metrics.iter().position(|x| x == m))
        {
            let goal = match (metric_goals[idx], opts.reverse) {
                (g, false) => g,
                (Goal::Min, true) => Goal::Max,
                (Goal::Max, true) => Goal::Min,
            };
            let mean = |r: &Row| r.cells[idx].as_ref().map(|c| c.stat.mean);
            rows.sort_by(|a, b| cmp_best_first(mean(a), mean(b), goal));
        }
        if let Some(top) = opts.top {
            rows.truncate(top);
        }

        let highlights = (0..metrics.len())
            .map(|i| {
                let mut means: Vec<f64> = rows
                    .iter()
                    .filter_map(|r| r.cells[i].as_ref().map(|c| c.stat.mean))
                    .filter(|v| v.is_finite())
                    .collect();
                means.sort_by(|a, b| cmp_best_first(Some(*a), Some(*b), metric_goals[i]));
                means.dedup();
                (means.first().copied(), means.get(1).copied())
            })
            .collect();

        Board {
            metrics,
            goals: metric_goals,
            rows,
            sort_metric,
            mode: opts.mode,
            show_steps: opts.show_steps && !opts.aggregate,
            aggregate: opts.aggregate,
            precision: opts.precision,
            highlights,
        }
    }

    fn is_best(&self, metric: usize, cell: &BoardCell) -> bool {
        self.highlights[metric].0 == Some(cell.stat.mean)
    }

    fn is_second(&self, metric: usize, cell: &BoardCell) -> bool {
        self.highlights[metric].1 == Some(cell.stat.mean)
    }

    fn cell_text(&self, cell: &BoardCell) -> String {
        let mut text = format_stat(&cell.stat, self.precision);
        if self.show_steps {
            if let Some(step) = cell.step {
                text.push_str(&format!(" @{step}"));
            }
        }
        text
    }

    fn mode_label(&self) -> &'static str {
        match self.mode {
            ValueMode::Best => "best",
            ValueMode::Final => "final",
        }
    }

    fn name_header(&self) -> &'static str {
        if self.aggregate {
            "Configuration"
        } else {
            "Experiment"
        }
    }

    pub fn to_table(&self) -> Table {
        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS);

        let mut header = vec!["Rank".to_string(), self.name_header().to_string()];
        for (name, goal) in self.metrics.iter().zip(&self.goals) {
            let marker = if Some(name) == self.sort_metric.as_ref() {
                " *"
            } else {
                ""
            };
            header.push(format!("{name} {}{marker}", goal.arrow()));
        }
        header.push("Status".to_string());
        table.set_header(header);

        for (i, row) in self.rows.iter().enumerate() {
            let mut cells = vec![Cell::new(i + 1), Cell::new(&row.label)];
            for (m, cell) in row.cells.iter().enumerate() {
                cells.push(match cell {
                    Some(c) if self.is_best(m, c) => Cell::new(self.cell_text(c)).fg(Color::Green),
                    Some(c) => Cell::new(self.cell_text(c)),
                    None => Cell::new("-"),
                });
            }
            cells.push(match row.status.as_str() {
                "done" => Cell::new("done").fg(Color::Green),
                "running" => Cell::new("running").fg(Color::Yellow),
                "failed" | "crashed" => Cell::new(&row.status).fg(Color::Red),
                _ => Cell::new(&row.status),
            });
            table.add_row(cells);
        }
        table
    }

    pub fn to_markdown(&self) -> String {
        let mut header = vec!["Rank".to_string(), self.name_header().to_string()];
        for (name, goal) in self.metrics.iter().zip(&self.goals) {
            header.push(format!("{} {}", markdown_cell(name), goal.arrow()));
        }
        header.push("Status".to_string());

        let mut out = format!("| {} |\n", header.join(" | "));
        out.push_str(&format!("| {} |\n", vec!["---"; header.len()].join(" | ")));

        for (i, row) in self.rows.iter().enumerate() {
            let mut cols = vec![format!("{}", i + 1), markdown_cell(&row.label)];
            for (m, cell) in row.cells.iter().enumerate() {
                cols.push(match cell {
                    Some(c) if self.is_best(m, c) => format!("**{}**", self.cell_text(c)),
                    Some(c) => self.cell_text(c),
                    None => "-".into(),
                });
            }
            cols.push(row.status.clone());
            out.push_str(&format!("| {} |\n", cols.join(" | ")));
        }
        out
    }

    pub fn to_latex(&self) -> String {
        let mut out = String::new();
        let cols = "l".to_string() + &"c".repeat(self.metrics.len());
        out.push_str("\\begin{table}[t]\n\\centering\n");
        out.push_str(&format!(
            "\\caption{{Experiment leaderboard ({} values{}).}}\n",
            self.mode_label(),
            if self.aggregate {
                ", mean $\\pm$ std over seeds"
            } else {
                ""
            }
        ));
        out.push_str(&format!("\\begin{{tabular}}{{{cols}}}\n\\toprule\n"));

        let mut header = vec![self.name_header().to_string()];
        for (name, goal) in self.metrics.iter().zip(&self.goals) {
            header.push(format!("{} {}", latex_escape(name), goal.latex_arrow()));
        }
        out.push_str(&format!("{} \\\\\n\\midrule\n", header.join(" & ")));

        for row in &self.rows {
            let mut cols = vec![latex_escape(&row.label)];
            for (m, cell) in row.cells.iter().enumerate() {
                cols.push(match cell {
                    Some(c) => {
                        let text = format_stat_latex(&c.stat, self.precision);
                        if self.is_best(m, c) {
                            format!("\\textbf{{{text}}}")
                        } else if self.is_second(m, c) {
                            format!("\\underline{{{text}}}")
                        } else {
                            text
                        }
                    }
                    None => "-".into(),
                });
            }
            out.push_str(&format!("{} \\\\\n", cols.join(" & ")));
        }

        out.push_str("\\bottomrule\n\\end{tabular}\n\\end{table}\n");
        out
    }

    pub fn to_csv(&self) -> String {
        let mut header = vec!["rank".to_string(), "experiment".to_string()];
        for name in &self.metrics {
            header.push(name.clone());
            if self.aggregate {
                header.push(format!("{name}_std"));
            }
            if self.show_steps {
                header.push(format!("{name}_step"));
            }
        }
        header.push("status".to_string());
        header.push("runs".to_string());

        let mut out = csv_line(&header) + "\n";
        for (i, row) in self.rows.iter().enumerate() {
            let mut cols = vec![format!("{}", i + 1), row.label.clone()];
            for cell in &row.cells {
                let (mean, std, step) = match cell {
                    Some(c) => (
                        format!("{:.6}", c.stat.mean),
                        format!("{:.6}", c.stat.std),
                        c.step.map(|s| s.to_string()).unwrap_or_default(),
                    ),
                    None => Default::default(),
                };
                cols.push(mean);
                if self.aggregate {
                    cols.push(std);
                }
                if self.show_steps {
                    cols.push(step);
                }
            }
            cols.push(row.status.clone());
            cols.push(row.runs.to_string());
            out.push_str(&(csv_line(&cols) + "\n"));
        }
        out
    }
}

/// Name for a group of runs that differ only by seed.
fn group_label(group: &[&Run<'_>]) -> String {
    let mut bases: Vec<String> = group
        .iter()
        .map(|r| strip_seed_suffix(&r.exp.dir_name))
        .collect();
    bases.sort();
    bases.dedup();
    let base = match bases.as_slice() {
        [one] => one.clone(),
        many => {
            let prefix = common_prefix(many);
            let trimmed = prefix.trim_end_matches(['_', '-', '.']);
            if trimmed.len() >= 3 {
                trimmed.to_string()
            } else {
                format!("{}…", many[0])
            }
        }
    };
    if group.len() > 1 {
        format!("{base} (n={})", group.len())
    } else {
        base
    }
}

fn common_prefix(names: &[String]) -> String {
    let first: Vec<char> = names[0].chars().collect();
    let len = names[1..].iter().fold(first.len(), |len, name| {
        name.chars()
            .zip(&first)
            .take(len)
            .take_while(|(a, b)| a == *b)
            .count()
    });
    first[..len].iter().collect()
}

fn group_status(group: &[&Run<'_>]) -> String {
    let mut counts: Vec<(String, usize)> = Vec::new();
    for run in group {
        let s = run.exp.status.to_string();
        match counts.iter_mut().find(|(name, _)| *name == s) {
            Some((_, n)) => *n += 1,
            None => counts.push((s, 1)),
        }
    }
    match counts.as_slice() {
        [(status, _)] => status.clone(),
        many => many
            .iter()
            .map(|(s, n)| format!("{n} {s}"))
            .collect::<Vec<_>>()
            .join(", "),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::experiment::discover_experiments;
    use std::fs;

    /// Write an experiment whose `loss`/`psnr` series are given explicitly.
    fn make(base: &Path, name: &str, meta: &str, loss: &[f64], psnr: &[f64]) {
        let dir = base.join(name);
        fs::create_dir_all(&dir).unwrap();
        let mut jsonl = String::new();
        for i in 0..loss.len().max(psnr.len()) {
            let mut fields = Vec::new();
            if let Some(v) = loss.get(i) {
                fields.push(format!("\"val_loss\":{v}"));
            }
            if let Some(v) = psnr.get(i) {
                fields.push(format!("\"psnr\":{v}"));
            }
            jsonl.push_str(&format!(
                "{{\"step\":{},\"timestamp\":{i}.0,\"metrics\":{{{}}}}}\n",
                i * 10,
                fields.join(",")
            ));
        }
        fs::write(dir.join("metrics.jsonl"), jsonl).unwrap();
        fs::write(dir.join("meta.json"), meta).unwrap();
    }

    fn opts() -> BoardOptions {
        BoardOptions {
            sort: None,
            top: None,
            reverse: false,
            mode: ValueMode::Best,
            show_steps: false,
            aggregate: false,
            metrics: Vec::new(),
            precision: 2,
        }
    }

    fn board(base: &Path, opts: &BoardOptions) -> Board {
        let exps = discover_experiments(base);
        let runs: Vec<Run> = exps
            .iter()
            .map(|e| Run {
                exp: e,
                summary: crate::data::summary::scan_summary_with(&e.dir, None),
            })
            .collect();
        Board::build(&runs, opts)
    }

    fn fixture() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let done = r#"{"status":"done"}"#;
        // a: best loss early, then overfits. b: steadily improves.
        make(dir.path(), "a", done, &[0.5, 0.1, 0.4], &[20.0, 30.0, 25.0]);
        make(dir.path(), "b", done, &[0.6, 0.3, 0.2], &[19.0, 24.0, 28.0]);
        make(dir.path(), "no_loss", done, &[], &[10.0, 11.0]);
        dir
    }

    #[test]
    fn test_best_values_rank_and_missing_goes_last() {
        let dir = fixture();
        let b = board(
            dir.path(),
            &BoardOptions {
                sort: Some("val_loss".into()),
                ..opts()
            },
        );
        assert_eq!(b.metrics, ["psnr", "val_loss"]);
        assert_eq!(b.goals, [Goal::Max, Goal::Min]);
        let labels: Vec<&str> = b.rows.iter().map(|r| r.label.as_str()).collect();
        assert_eq!(labels, ["a", "b", "no_loss"]);
        assert_eq!(b.rows[0].cells[1].as_ref().unwrap().stat.mean, 0.1);
    }

    #[test]
    fn test_final_values_can_reverse_the_ranking() {
        let dir = fixture();
        let b = board(
            dir.path(),
            &BoardOptions {
                sort: Some("val_loss".into()),
                mode: ValueMode::Final,
                show_steps: true,
                ..opts()
            },
        );
        let labels: Vec<&str> = b.rows.iter().map(|r| r.label.as_str()).collect();
        assert_eq!(labels, ["b", "a", "no_loss"]);
        assert!(b.to_markdown().contains("**0.20 @20**"));
    }

    #[test]
    fn test_reverse_keeps_missing_last() {
        let dir = fixture();
        let b = board(
            dir.path(),
            &BoardOptions {
                sort: Some("val_loss".into()),
                reverse: true,
                ..opts()
            },
        );
        let labels: Vec<&str> = b.rows.iter().map(|r| r.label.as_str()).collect();
        assert_eq!(labels, ["b", "a", "no_loss"]);
    }

    #[test]
    fn test_latex_is_escaped_and_marks_best_and_second() {
        let dir = fixture();
        let latex = board(
            dir.path(),
            &BoardOptions {
                sort: Some("val_loss".into()),
                ..opts()
            },
        )
        .to_latex();
        assert!(latex.contains("val\\_loss $\\downarrow$"), "{latex}");
        assert!(latex.contains("no\\_loss &"), "{latex}");
        assert!(
            latex.contains("a & \\textbf{30.00} & \\textbf{0.10} \\\\"),
            "{latex}"
        );
        assert!(
            latex.contains("b & \\underline{28.00} & \\underline{0.20} \\\\"),
            "{latex}"
        );
        assert!(
            !latex.contains("val_loss"),
            "unescaped underscore:\n{latex}"
        );
    }

    #[test]
    fn test_csv_and_top() {
        let dir = fixture();
        let csv = board(
            dir.path(),
            &BoardOptions {
                sort: Some("psnr".into()),
                top: Some(2),
                metrics: vec!["psnr".into()],
                ..opts()
            },
        )
        .to_csv();
        assert_eq!(
            csv,
            "rank,experiment,psnr,status,runs\n1,a,30.000000,done,1\n2,b,28.000000,done,1\n"
        );
    }

    #[test]
    fn test_aggregate_groups_seeds_with_mean_and_std() {
        let dir = tempfile::tempdir().unwrap();
        let meta = |hash: &str| format!("{{\"status\":\"done\",\"config_hash\":\"{hash}\"}}");
        make(dir.path(), "wide_seed0", &meta("w"), &[0.2], &[28.0]);
        make(dir.path(), "wide_seed1", &meta("w"), &[0.4], &[30.0]);
        make(dir.path(), "wide_seed2", &meta("w"), &[0.3], &[29.0]);
        make(dir.path(), "narrow_seed0", &meta("n"), &[0.5], &[25.0]);

        let b = board(
            dir.path(),
            &BoardOptions {
                sort: Some("psnr".into()),
                aggregate: true,
                ..opts()
            },
        );
        assert_eq!(b.rows.len(), 2);
        assert_eq!(b.rows[0].label, "wide (n=3)");
        assert_eq!(b.rows[1].label, "narrow");
        let cell = b.rows[0].cells[0].as_ref().unwrap();
        assert!((cell.stat.mean - 29.0).abs() < 1e-9 && (cell.stat.std - 1.0).abs() < 1e-9);

        let md = b.to_markdown();
        assert!(md.contains("| 1 | wide (n=3) | **29.00±1.00** |"), "{md}");
        assert!(b.to_latex().contains("\\textbf{29.00 $\\pm$ 1.00}"));
        let csv = b.to_csv();
        assert!(
            csv.starts_with("rank,experiment,psnr,psnr_std,val_loss,val_loss_std,status,runs\n")
        );
        assert!(
            csv.contains("1,wide (n=3),29.000000,1.000000,0.300000,0.100000,done,3\n"),
            "{csv}"
        );
    }

    #[test]
    fn test_explicit_goal_from_meta_is_respected() {
        let dir = tempfile::tempdir().unwrap();
        let meta = r#"{"status":"done","metric_goals":{"psnr":"min"}}"#;
        make(dir.path(), "x", meta, &[], &[20.0, 30.0]);
        make(dir.path(), "y", meta, &[], &[25.0, 27.0]);
        let b = board(dir.path(), &opts());
        assert_eq!(b.goals, [Goal::Min]);
        assert_eq!(b.rows[0].label, "x"); // lowest psnr (20) wins under an explicit "min"
    }
}
