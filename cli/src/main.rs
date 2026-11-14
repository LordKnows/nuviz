use anyhow::Result;
use clap::Parser;

use nuviz_cli::cli::{Cli, Commands};
use nuviz_cli::{commands, data};

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
        Commands::Image(args) => commands::image::run(args, &base_dir),
        Commands::Diff(args) => commands::diff::run(args, &base_dir),
        Commands::View(args) => commands::view::run(args, &base_dir),
        Commands::Tag(args) => commands::tag::run(args, &base_dir),
        Commands::Cleanup(args) => commands::cleanup::run(args, &base_dir),
        Commands::Reproduce(args) => commands::reproduce::run(args, &base_dir),
    }
}
