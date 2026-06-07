mod cli;
mod viewer;

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
        Command::Viewer(args) => viewer::run(&args),
    }
}

fn version(stdout: &mut impl Write) -> Result<i32> {
    writeln!(stdout, "rlmesh-cli {}", env!("CARGO_PKG_VERSION"))?;
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
    async fn help_lists_real_commands_and_hides_viewer() {
        let (code, stdout, stderr) = run_for_test(&["--help"]).await;

        assert_eq!(code, 0);
        assert!(stderr.is_empty());
        assert!(stdout.contains("version"));
        for command in [
            "auth", "init", "doctor", "probe", "build", "catalog", "eval",
        ] {
            assert!(!stdout.contains(command), "{command}");
        }
        assert!(!stdout.contains("viewer"));

        let mut command = cli::Cli::command();
        let help = command.render_help().to_string();
        assert!(!help.contains("viewer"));
    }

    #[tokio::test]
    async fn version_reports_cli_version_and_distribution() {
        let (code, stdout, stderr) = run_for_test(&["version"]).await;

        assert_eq!(code, 0);
        assert!(stderr.is_empty());
        assert!(stdout.contains(concat!("rlmesh-cli ", env!("CARGO_PKG_VERSION"))));
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
