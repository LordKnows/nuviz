use clap::{Parser, Subcommand, ValueEnum};

use crate::data::summary::ValueMode;

#[derive(Parser)]
#[command(name = "nuviz")]
#[command(about = "Terminal-native ML training visualization")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Base directory for experiments [default: $NUVIZ_DIR or ~/.nuviz/experiments]
    #[arg(long, global = true)]
    pub dir: Option<String>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Real-time training monitoring (TUI dashboard)
    Watch(WatchArgs),

    /// List all experiments
    Ls(LsArgs),

    /// Experiment metrics leaderboard
    Leaderboard(LeaderboardArgs),

    /// Multi-experiment curve comparison (TUI)
    Compare(CompareArgs),

    /// Ablation experiment matrix view
    Matrix(MatrixArgs),

    /// Per-scene metric breakdown
    Breakdown(BreakdownArgs),

    /// Export experiment data as CSV or JSON
    Export(ExportArgs),

    /// Browse experiment images in terminal
    Image(ImageArgs),

    /// Visual diff between two experiments
    Diff(DiffArgs),

    /// Point cloud (PLY) statistics and inspection
    View(ViewArgs),

    /// Tag experiments for organization
    Tag(TagArgs),

    /// Clean up low-value experiments
    Cleanup(CleanupArgs),

    /// Print reproduction guide for an experiment
    Reproduce(ReproduceArgs),
}

/// Output format for tabular commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TableFormat {
    Table,
    Markdown,
    Latex,
    Csv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ExportFormat {
    Csv,
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum LsSort {
    Name,
    Date,
    Steps,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Align {
    Step,
    #[value(name = "wall_time", alias = "wall-time", alias = "time")]
    WallTime,
}

#[derive(Parser)]
pub struct WatchArgs {
    /// Experiment(s) to watch: name, project/name, run id prefix or path
    pub experiments: Vec<String>,

    /// Watch the latest N experiments
    #[arg(long)]
    pub latest: Option<usize>,

    /// Filter by project name
    #[arg(long)]
    pub project: Option<String>,

    /// Use polling instead of inotify (for NFS/WSL)
    #[arg(long)]
    pub poll: bool,
}

#[derive(Parser)]
pub struct LsArgs {
    /// Filter by project name
    #[arg(long)]
    pub project: Option<String>,

    /// Sort order
    #[arg(long, value_enum, default_value_t = LsSort::Date)]
    pub sort: LsSort,

    /// Metric shown in the "Best" column
    #[arg(long, default_value = "loss")]
    pub metric: String,

    /// Only show experiments with this status (running, done, failed, crashed, unknown)
    #[arg(long)]
    pub status: Option<String>,

    /// Only show experiments carrying this tag
    #[arg(long)]
    pub tag: Option<String>,
}

#[derive(Parser)]
pub struct LeaderboardArgs {
    /// Filter by project name
    #[arg(long)]
    pub project: Option<String>,

    /// Metric to sort by [default: first metric alphabetically]
    #[arg(long)]
    pub sort: Option<String>,

    /// Show only top N rows
    #[arg(long)]
    pub top: Option<usize>,

    /// Reverse the order (worst first)
    #[arg(long, alias = "reverse")]
    pub asc: bool,

    /// Output format
    #[arg(long, value_enum, default_value_t = TableFormat::Table)]
    pub format: TableFormat,

    /// Which value represents a run: its best step or its final step
    #[arg(long, value_enum, default_value_t = ValueMode::Best)]
    pub value: ValueMode,

    /// Show the step each value was taken at
    #[arg(long)]
    pub steps: bool,

    /// Aggregate runs that differ only by seed (mean ± std)
    #[arg(long)]
    pub aggregate: bool,

    /// Only show these metrics (repeatable)
    #[arg(long = "metric")]
    pub metrics: Vec<String>,

    /// Only include experiments carrying this tag
    #[arg(long)]
    pub tag: Option<String>,

    /// Decimal places
    #[arg(long, default_value_t = 4)]
    pub precision: usize,
}

#[derive(Parser)]
pub struct CompareArgs {
    /// Experiments to compare: name, project/name, run id prefix or path
    pub experiments: Vec<String>,

    /// Filter by project name (compare everything in it when no names are given)
    #[arg(long)]
    pub project: Option<String>,

    /// Metric to display
    #[arg(long)]
    pub metric: Option<String>,

    /// X axis: step or wall_time
    #[arg(long, value_enum, default_value_t = Align::Step)]
    pub align: Align,

    /// Use polling for file watching (NFS/WSL)
    #[arg(long)]
    pub poll: bool,
}

#[derive(Parser)]
pub struct MatrixArgs {
    /// Parameter for matrix rows
    #[arg(long)]
    pub rows: String,

    /// Parameter for matrix columns
    #[arg(long)]
    pub cols: String,

    /// Metric to display in cells
    #[arg(long)]
    pub metric: String,

    /// Filter by project name
    #[arg(long)]
    pub project: Option<String>,

    /// Output format
    #[arg(long, value_enum, default_value_t = TableFormat::Table)]
    pub format: TableFormat,

    /// Which value represents a run: its best step or its final step
    #[arg(long, value_enum, default_value_t = ValueMode::Best)]
    pub value: ValueMode,
}

#[derive(Parser)]
pub struct BreakdownArgs {
    /// Experiment: name, project/name, run id prefix or path
    pub experiment: String,

    /// Output as LaTeX table
    #[arg(long)]
    pub latex: bool,

    /// Output as Markdown table
    #[arg(long)]
    pub markdown: bool,

    /// Compare with a second experiment (show deltas)
    #[arg(long)]
    pub diff: Option<String>,

    /// Filter by project name
    #[arg(long)]
    pub project: Option<String>,
}

#[derive(Parser)]
pub struct ExportArgs {
    /// Experiment: name, project/name, run id prefix or path
    pub experiment: String,

    /// Output format
    #[arg(long, value_enum, default_value_t = ExportFormat::Csv)]
    pub format: ExportFormat,

    /// Export only specific metric(s)
    #[arg(long)]
    pub metric: Option<Vec<String>>,

    /// Filter by project name
    #[arg(long)]
    pub project: Option<String>,
}

#[derive(Parser)]
pub struct ImageArgs {
    /// Experiment: name, project/name, run id prefix or path
    pub experiment: String,

    /// Filter by step number
    #[arg(long)]
    pub step: Option<u64>,

    /// Filter by image tag (e.g., "render", "depth")
    #[arg(long)]
    pub tag: Option<String>,

    /// Show only the most recent image
    #[arg(long)]
    pub latest: bool,

    /// Show side-by-side with another tag (e.g., --side-by-side gt)
    #[arg(long)]
    pub side_by_side: Option<String>,

    /// Filter by project name
    #[arg(long)]
    pub project: Option<String>,
}

#[derive(Parser)]
pub struct DiffArgs {
    /// First experiment
    pub experiment_a: String,

    /// Second experiment
    pub experiment_b: String,

    /// Specific step to compare [default: latest common step]
    #[arg(long)]
    pub step: Option<u64>,

    /// Image tag to compare
    #[arg(long, default_value = "render")]
    pub tag: String,

    /// Show per-pixel error heatmap
    #[arg(long)]
    pub heatmap: bool,

    /// Filter by project name
    #[arg(long)]
    pub project: Option<String>,
}

#[derive(Parser)]
pub struct ViewArgs {
    /// Path to PLY file, or experiment to find its latest PLY
    pub path: String,

    /// Show attribute distribution histograms
    #[arg(long)]
    pub histogram: bool,

    /// Filter by project name
    #[arg(long)]
    pub project: Option<String>,
}

#[derive(Parser)]
pub struct TagArgs {
    /// Experiment: name, project/name, run id prefix or path
    pub experiment: String,

    /// Tag to add
    pub tag: Option<String>,

    /// Remove a tag
    #[arg(long)]
    pub remove: Option<String>,

    /// List all tags
    #[arg(long)]
    pub list: bool,

    /// Filter by project name
    #[arg(long)]
    pub project: Option<String>,
}

#[derive(Parser)]
pub struct CleanupArgs {
    /// Filter by project name
    #[arg(long)]
    pub project: Option<String>,

    /// Keep top N experiments by metric
    #[arg(long, default_value_t = 5)]
    pub keep_top: usize,

    /// Metric to rank by
    #[arg(long, default_value = "loss")]
    pub metric: String,

    /// Actually delete (default is dry-run)
    #[arg(long)]
    pub force: bool,

    /// Also consider tagged experiments for removal (kept by default)
    #[arg(long)]
    pub include_tagged: bool,
}

#[derive(Parser)]
pub struct ReproduceArgs {
    /// Experiment: name, project/name, run id prefix or path
    pub experiment: String,

    /// Filter by project name
    #[arg(long)]
    pub project: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn test_cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn test_invalid_enum_values_are_rejected() {
        assert!(Cli::try_parse_from(["nuviz", "leaderboard", "--format", "latex"]).is_ok());
        assert!(Cli::try_parse_from(["nuviz", "leaderboard", "--format", "latxe"]).is_err());
        assert!(Cli::try_parse_from(["nuviz", "ls", "--sort", "size"]).is_err());
        assert!(
            Cli::try_parse_from(["nuviz", "compare", "a", "b", "--align", "wall_time"]).is_ok()
        );
        assert!(
            Cli::try_parse_from(["nuviz", "compare", "a", "b", "--align", "wall-time"]).is_ok()
        );
    }
}
