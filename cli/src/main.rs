use anyhow::Result;
use clap::Parser;

use nuviz_cli::cli::{Cli, Commands};
use nuviz_cli::{commands, data};

/// Die quietly when the reader of our output goes away (`nuviz ls | head`),
/// like other command-line tools, instead of reporting a broken pipe.
#[cfg(unix)]
fn restore_default_sigpipe() {
    // SAFETY: called once at startup before any threads exist; SIG_DFL is a
    // valid disposition for SIGPIPE.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

#[cfg(not(unix))]
fn restore_default_sigpipe() {}

fn main() -> Result<()> {
    restore_default_sigpipe();
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
