use anyhow::Result;
use clap::{CommandFactory, Parser};

use rawq::cli::args::*;
use rawq::cli::commands::diff::cmd_diff;
use rawq::cli::commands::chunk::cmd_chunk;
use rawq::cli::commands::embed::cmd_embed;
use rawq::cli::commands::index::{cmd_index, cmd_status, cmd_unindex};
use rawq::cli::commands::map::cmd_map;
use rawq::cli::commands::model::cmd_model;
use rawq::cli::commands::search::cmd_search;
use rawq::cli::commands::watch::cmd_watch;
use rawq::cli::output::{print_json_error, wants_json};

/// Exit codes: 0 = results found, 1 = no results, 2 = error.
fn main() {
    let json_mode = wants_json();
    let code = match run() {
        Ok(code) => code,
        Err(e) => {
            if json_mode {
                print_json_error("error", &format!("{e:#}"), None);
            } else {
                eprintln!("error: {e:#}");
            }
            2
        }
    };
    std::process::exit(code);
}

fn run() -> Result<i32> {
    const KNOWN_SUBCOMMANDS: &[&str] = &[
        "search", "index", "diff", "map", "watch", "model", "daemon", "embed", "chunk", "completions", "help",
    ];

    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            let args: Vec<String> = std::env::args().collect();
            if args.len() > 1
                && !args[1].starts_with('-')
                && !KNOWN_SUBCOMMANDS.contains(&args[1].as_str())
            {
                let mut new_args = vec![args[0].clone(), "search".to_string()];
                new_args.extend_from_slice(&args[1..]);
                Cli::parse_from(new_args)
            } else if args.len() > 2
                && args[1] == "index"
                && !["build", "status", "remove", "help", "--help", "-h"]
                    .contains(&args[2].as_str())
                && !args[2].starts_with('-')
            {
                let mut new_args = vec![args[0].clone(), "index".to_string(), "build".to_string()];
                new_args.extend_from_slice(&args[2..]);
                Cli::parse_from(new_args)
            } else if wants_json() {
                print_json_error("parse_error", &e.to_string(), None);
                std::process::exit(2);
            } else {
                e.exit();
            }
        }
    };

    if std::env::var("NO_COLOR").is_ok() {
        colored::control::set_override(false);
    }

    match cli.command {
        Command::Search(args) => cmd_search(args),
        Command::Index { command } => {
            match command {
                IndexCommand::Build { path, reindex, model, batch_size, json, exclude } => {
                    cmd_index(path, reindex, model, batch_size, json, exclude)?;
                }
                IndexCommand::Status { path, json } => {
                    cmd_status(path, json)?;
                }
                IndexCommand::Remove { path, all } => {
                    cmd_unindex(path, all)?;
                }
            }
            Ok(0)
        }
        Command::Diff(args) => cmd_diff(args),
        Command::Map(args) => {
            cmd_map(args)?;
            Ok(0)
        }
        Command::Watch(args) => {
            cmd_watch(args)?;
            Ok(0)
        }
        Command::Model { command } => {
            cmd_model(command)?;
            Ok(0)
        }
        Command::Daemon { command } => {
            match command {
                DaemonCommand::Start { model, idle_timeout, background } => {
                    rawq::cli::commands::daemon::cmd_daemon_start(model, idle_timeout, background)?;
                }
                DaemonCommand::Stop { model } => {
                    rawq::cli::commands::daemon::cmd_daemon_stop(model)?;
                }
                DaemonCommand::Status { model, json } => {
                    rawq::cli::commands::daemon::cmd_daemon_status(model, json)?;
                }
            }
            Ok(0)
        }
        Command::Embed { text, query, model, tokenizer, no_daemon } => {
            cmd_embed(text, query, model, tokenizer, no_daemon)?;
            Ok(0)
        }
        Command::Chunk { path } => {
            cmd_chunk(path)?;
            Ok(0)
        }
        Command::Completions { shell } => {
            clap_complete::generate(shell, &mut Cli::command(), "rawq", &mut std::io::stdout());
            Ok(0)
        }
    }
}
