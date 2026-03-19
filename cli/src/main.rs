mod cli;
mod commands;
mod data;
mod terminal;
mod tui;
mod watcher;

use anyhow::Result;
use clap::Parser;

use cli::{Cli, Commands};

fn main() -> Result<()> {
    let cli = Cli::parse();

    let base_dir = data::experiment::resolve_base_dir(cli.dir.as_deref());

    match cli.command {
        Commands::Watch(args) => commands::watch::run(args, &base_dir),
        Commands::Ls(args) => commands::ls::run(args, &base_dir),
        Commands::Leaderboard(args) => commands::leaderboard::run(args, &base_dir),
        Commands::Compare(args) => commands::compare::run(args, &base_dir),
        Commands::Matrix(args) => commands::matrix::run(args, &base_dir),
        Commands::Breakdown(args) => commands::breakdown::run(args, &base_dir),
        Commands::Export(args) => commands::export::run(args, &base_dir),
    }
}
