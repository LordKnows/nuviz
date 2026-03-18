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
}
