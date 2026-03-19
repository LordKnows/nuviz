use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "nuviz")]
#[command(about = "Terminal-native ML training visualization")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Base directory for experiments [default: ~/.nuviz/experiments]
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

#[derive(Parser)]
pub struct WatchArgs {
    /// Experiment name(s) to watch
    pub experiments: Vec<String>,

    /// Watch the latest N experiments in a project
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

    /// Sort by field: name, date, steps
    #[arg(long, default_value = "date")]
    pub sort: String,
}

#[derive(Parser)]
pub struct LeaderboardArgs {
    /// Filter by project name
    #[arg(long)]
    pub project: Option<String>,

    /// Metric to sort by
    #[arg(long)]
    pub sort: Option<String>,

    /// Show only top N experiments
    #[arg(long)]
    pub top: Option<usize>,

    /// Sort in ascending order (default: descending)
    #[arg(long)]
    pub asc: bool,

    /// Output format: table, markdown, latex, csv
    #[arg(long, default_value = "table")]
    pub format: String,

    /// Aggregate multi-seed runs (show mean ± std)
    #[arg(long)]
    pub aggregate: bool,
}

#[derive(Parser)]
pub struct CompareArgs {
    /// Experiment names to compare
    pub experiments: Vec<String>,

    /// Filter by project name
    #[arg(long)]
    pub project: Option<String>,

    /// Metric to display
    #[arg(long)]
    pub metric: Option<String>,

    /// Alignment mode: step or wall_time
    #[arg(long, default_value = "step")]
    pub align: String,

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

    /// Output format: table, latex, markdown, csv
    #[arg(long, default_value = "table")]
    pub format: String,
}

#[derive(Parser)]
pub struct BreakdownArgs {
    /// Experiment name
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
    /// Experiment name
    pub experiment: String,

    /// Output format: csv or json
    #[arg(long, default_value = "csv")]
    pub format: String,

    /// Export only specific metric(s)
    #[arg(long)]
    pub metric: Option<Vec<String>>,

    /// Filter by project name
    #[arg(long)]
    pub project: Option<String>,
}

#[derive(Parser)]
pub struct ImageArgs {
    /// Experiment name
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
    /// First experiment name
    pub experiment_a: String,

    /// Second experiment name
    pub experiment_b: String,

    /// Specific step to compare
    #[arg(long)]
    pub step: Option<u64>,

    /// Image tag to compare (default: render)
    #[arg(long, default_value = "render")]
    pub tag: String,

    /// Show per-pixel error heatmap
    #[arg(long)]
    pub heatmap: bool,

    /// Filter by scene name
    #[arg(long)]
    pub scene: Option<String>,

    /// Filter by project name
    #[arg(long)]
    pub project: Option<String>,
}

#[derive(Parser)]
pub struct ViewArgs {
    /// Path to PLY file, or experiment name to find latest PLY
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
    /// Experiment name
    pub experiment: String,

    /// Tag to add
    pub tag: Option<String>,

    /// Remove a tag
    #[arg(long)]
    pub remove: Option<String>,

    /// List all tags
    #[arg(long)]
    pub list: bool,
}

#[derive(Parser)]
pub struct CleanupArgs {
    /// Filter by project name
    #[arg(long)]
    pub project: Option<String>,

    /// Keep top N experiments by metric (default: 5)
    #[arg(long)]
    pub keep_top: Option<usize>,

    /// Metric to rank by (default: loss)
    #[arg(long)]
    pub metric: Option<String>,

    /// Actually delete (default is dry-run)
    #[arg(long)]
    pub force: bool,
}

#[derive(Parser)]
pub struct ReproduceArgs {
    /// Experiment name
    pub experiment: String,
}
