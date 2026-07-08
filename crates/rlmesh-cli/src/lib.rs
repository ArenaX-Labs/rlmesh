//! The `rlmesh` command-line binary.
//!
//! [`run_cli`] parses argv with clap and dispatches the available subcommands:
//! `version` reports this build's version, its workflow edition (the
//! commit-stamped build identity, so two builds sharing a package version stay
//! distinguishable), and the distribution it shipped in (read from the
//! `RLMESH_CLI_DISTRIBUTION` environment variable, defaulting to `standalone`)
//! so a wheel- or container-bundled CLI can identify how it was packaged;
//! `login`/`logout` sign in to and out of a managed platform via the OAuth
//! device flow; `whoami` reports the active profile's platform and sign-in
//! state, verifying a stored credential against the platform when one exists;
//! `profile` manages the AWS-CLI-style
//! named profiles those commands act on (each remembering its own platform and
//! credential); and `registry login` authenticates docker with the platform's
//! image registry.

mod cli;
mod platform;
mod viewtest;

use std::ffi::{OsStr, OsString};
use std::io::{self, Write};

use anyhow::Result;
use clap::Parser;
use clap::error::ErrorKind;
use cli::{Cli, Command};

pub async fn run_cli() -> Result<i32> {
    run_cli_with_args(std::env::args_os().skip(1).collect::<Vec<_>>()).await
}

pub async fn run_cli_with_args(argv: Vec<OsString>) -> Result<i32> {
    let mut stdout = io::stdout();
    let mut stderr = io::stderr();
    run_cli_with_writers(argv, &mut stdout, &mut stderr).await
}

async fn run_cli_with_writers(
    argv: Vec<OsString>,
    stdout: &mut impl Write,
    stderr: &mut impl Write,
) -> Result<i32> {
    let cli = match Cli::try_parse_from(std::iter::once(OsString::from("rlmesh")).chain(argv)) {
        Ok(cli) => cli,
        Err(err) => {
            let exit_code = err.exit_code();
            match err.kind() {
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => write!(stdout, "{err}")?,
                _ => write!(stderr, "{err}")?,
            }
            return Ok(exit_code);
        }
    };

    match cli.command {
        Command::Version => version(stdout),
        Command::Login(args) => platform::login(&args, stdout).await,
        Command::Logout(args) => platform::logout(&args, stdout),
        Command::Whoami(args) => platform::whoami(&args, stdout).await,
        Command::Registry(args) => match args.command {
            cli::RegistryCommand::Login(args) => platform::registry_login(&args, stdout).await,
        },
        Command::Profile(args) => match args.command {
            cli::ProfileCommand::List => platform::profile_list(stdout),
            cli::ProfileCommand::Use { name } => platform::profile_use(&name, stdout),
            cli::ProfileCommand::Remove { name } => platform::profile_remove(&name, stdout),
        },
        Command::Viewtest(args) => viewtest::run(&args, stderr),
    }
}

fn version(stdout: &mut impl Write) -> Result<i32> {
    writeln!(stdout, "rlmesh-cli {}", env!("CARGO_PKG_VERSION"))?;
    writeln!(
        stdout,
        "edition: {}",
        rlmesh_proto::CURRENT_WORKFLOW_EDITION
    )?;
    writeln!(stdout, "distribution: {}", cli_distribution())?;
    Ok(0)
}

fn cli_distribution() -> String {
    cli_distribution_from(std::env::var_os("RLMESH_CLI_DISTRIBUTION"))
}

fn cli_distribution_from(value: Option<OsString>) -> String {
    value
        .as_deref()
        .and_then(OsStr::to_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("standalone")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    async fn run_for_test(args: &[&str]) -> (i32, String, String) {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let code = run_cli_with_writers(
            args.iter().map(OsString::from).collect(),
            &mut stdout,
            &mut stderr,
        )
        .await
        .unwrap();
        (
            code,
            String::from_utf8(stdout).unwrap(),
            String::from_utf8(stderr).unwrap(),
        )
    }

    #[tokio::test]
    async fn help_lists_real_commands_and_omits_viewer() {
        let (code, stdout, stderr) = run_for_test(&["--help"]).await;

        assert_eq!(code, 0);
        assert!(stderr.is_empty());
        assert!(stdout.contains("version"));
        assert!(stdout.contains("login"));
        assert!(stdout.contains("logout"));
        assert!(stdout.contains("registry"));
        assert!(stdout.contains("profile"));
        assert!(stdout.contains("whoami"));
        assert!(!stdout.contains("viewer"));

        let mut command = cli::Cli::command();
        let subcommands: Vec<&str> = command.get_subcommands().map(|c| c.get_name()).collect();
        for command_name in [
            "auth", "init", "doctor", "probe", "build", "catalog", "eval",
        ] {
            assert!(!subcommands.contains(&command_name), "{command_name}");
        }

        let help = command.render_help().to_string();
        assert!(!help.contains("viewer"));
    }

    #[tokio::test]
    async fn version_reports_cli_version_and_distribution() {
        let (code, stdout, stderr) = run_for_test(&["version"]).await;

        assert_eq!(code, 0);
        assert!(stderr.is_empty());
        assert!(stdout.contains(concat!("rlmesh-cli ", env!("CARGO_PKG_VERSION"))));
        assert!(stdout.contains(&format!(
            "edition: {}",
            rlmesh_proto::CURRENT_WORKFLOW_EDITION
        )));
        assert!(stdout.contains("distribution: "));
    }

    #[test]
    fn distribution_defaults_to_standalone() {
        assert_eq!(cli_distribution_from(None), "standalone");
        assert_eq!(
            cli_distribution_from(Some(OsString::from("  "))),
            "standalone"
        );
        assert_eq!(
            cli_distribution_from(Some(OsString::from("python-wheel"))),
            "python-wheel"
        );
    }
}
