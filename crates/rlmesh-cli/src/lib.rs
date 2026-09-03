mod auth;
mod cli;
mod config;
mod helpers;
mod platform;
mod profile;
mod registry;
mod render;
mod viewtest;

use std::ffi::{OsStr, OsString};
use std::io::Write;

use anyhow::Result;
use clap::error::ErrorKind;
use clap::{ColorChoice, CommandFactory, FromArgMatches};
use cli::{Cli, Command};
use config::ProfileStore;
use render::{Style, write_error, write_heading, write_key_value};

pub async fn run_cli(
    argv: Vec<OsString>,
    stdout: &mut impl Write,
    stderr: &mut impl Write,
    stdout_is_terminal: bool,
    stderr_is_terminal: bool,
) -> Result<i32> {
    let stdout_style = Style::for_terminal(stdout_is_terminal);
    let stderr_style = Style::for_terminal(stderr_is_terminal);
    let command = Cli::command().color(ColorChoice::Never);
    let cli = match command
        .try_get_matches_from(std::iter::once(OsString::from("rlmesh")).chain(argv))
        .and_then(|matches| Cli::from_arg_matches(&matches))
    {
        Ok(cli) => cli,
        Err(error) => {
            let exit_code = error.exit_code();
            match error.kind() {
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => write!(stdout, "{error}")?,
                _ => write!(stderr, "{error}")?,
            }
            return Ok(exit_code);
        }
    };

    let mut profiles = match &cli.command {
        Command::Version | Command::Viewtest(_) => None,
        _ => match ProfileStore::load() {
            Ok(profiles) => Some(profiles),
            Err(error) => {
                write_error(stderr, stderr_style, &error)?;
                return Ok(1);
            }
        },
    };

    let result = match cli.command {
        Command::Version => version(stdout, stdout_style).map(|()| 0),
        Command::Login(args) => {
            auth::login(profile_store(&mut profiles), &args, stdout, stdout_style)
                .await
                .map(|()| 0)
        }
        Command::Logout(args) => {
            auth::logout(profile_store(&mut profiles), &args, stdout, stdout_style).map(|()| 0)
        }
        Command::Whoami(args) => {
            auth::whoami(profile_store(&mut profiles), &args, stdout, stdout_style).await
        }
        Command::Registry(args) => match args.command {
            cli::RegistryCommand::Login(args) => {
                registry::registry_login(profile_store(&mut profiles), &args, stdout, stdout_style)
                    .await
                    .map(|()| 0)
            }
            cli::RegistryCommand::CredentialHelper(args) => {
                registry::credential_helper(profile_store(&mut profiles), &args, stdout).await
            }
        },
        Command::Profile(args) => match args.command {
            cli::ProfileCommand::List => {
                profile::profile_list(profile_store(&mut profiles), stdout, stdout_style)
                    .map(|()| 0)
            }
            cli::ProfileCommand::Use { name } => {
                profile::profile_use(profile_store(&mut profiles), &name, stdout, stdout_style)
                    .map(|()| 0)
            }
            cli::ProfileCommand::Remove { name } => {
                profile::profile_remove(profile_store(&mut profiles), &name, stdout, stdout_style)
                    .map(|()| 0)
            }
        },
        Command::Org(args) => match args.command {
            cli::OrgCommand::List(args) => {
                auth::org_list(profile_store(&mut profiles), &args, stdout, stdout_style)
                    .await
                    .map(|()| 0)
            }
            cli::OrgCommand::Switch { id, profile } => auth::org_switch(
                profile_store(&mut profiles),
                &id,
                &profile,
                stdout,
                stdout_style,
            )
            .await
            .map(|()| 0),
        },
        Command::Token(args) => platform::token(profile_store(&mut profiles), &args, stdout)
            .await
            .map(|()| 0),
        Command::Eval(args) => match args.command {
            cli::EvalCommand::Submit(args) => {
                platform::submit(profile_store(&mut profiles), &args, stdout, stdout_style).await
            }
            cli::EvalCommand::List(args) => {
                platform::list(profile_store(&mut profiles), &args, stdout, stdout_style)
                    .await
                    .map(|()| 0)
            }
            cli::EvalCommand::Get(args) => {
                platform::get(profile_store(&mut profiles), &args, stdout)
                    .await
                    .map(|()| 0)
            }
            cli::EvalCommand::Wait(args) => {
                platform::wait(profile_store(&mut profiles), &args, stdout, stdout_style).await
            }
            cli::EvalCommand::Cancel(args) => {
                platform::cancel(profile_store(&mut profiles), &args, stdout, stdout_style)
                    .await
                    .map(|()| 0)
            }
        },
        Command::Viewtest(args) => viewtest::run(&args, stderr).map(|_| 0),
    };

    match result {
        Ok(code) => Ok(code),
        Err(error) => {
            write_error(stderr, stderr_style, &error)?;
            Ok(1)
        }
    }
}

/// Terminal entrypoint shared by the `rlmesh` and `docker-credential-rlmesh`
/// binaries; `prefix` routes the latter to its subcommand.
pub async fn run_terminal(prefix: &[&str]) -> i32 {
    let mut stdout = std::io::stdout();
    let mut stderr = std::io::stderr();
    let stdout_is_terminal = std::io::IsTerminal::is_terminal(&stdout);
    let stderr_is_terminal = std::io::IsTerminal::is_terminal(&stderr);

    let argv = prefix
        .iter()
        .map(OsString::from)
        .chain(std::env::args_os().skip(1))
        .collect();
    match run_cli(
        argv,
        &mut stdout,
        &mut stderr,
        stdout_is_terminal,
        stderr_is_terminal,
    )
    .await
    {
        Ok(code) => code,
        Err(error) => {
            #[allow(clippy::print_stderr)]
            {
                eprintln!("Error: {error:#}");
            }
            1
        }
    }
}

fn profile_store(profiles: &mut Option<ProfileStore>) -> &mut ProfileStore {
    profiles
        .as_mut()
        .expect("profile command has profile state")
}

fn version(stdout: &mut impl Write, style: Style) -> Result<()> {
    write_heading(stdout, style, "RLMesh CLI")?;
    write_key_value(stdout, style, "Version", env!("CARGO_PKG_VERSION"))?;
    write_key_value(
        stdout,
        style,
        "Edition",
        rlmesh_proto::CURRENT_WORKFLOW_EDITION,
    )?;
    write_key_value(stdout, style, "Distribution", &cli_distribution())?;
    Ok(())
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

    async fn run_for_test(args: &[&str]) -> (i32, String, String) {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let code = run_cli(
            args.iter().map(OsString::from).collect(),
            &mut stdout,
            &mut stderr,
            false,
            false,
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
        assert!(stdout.contains("token"));
        assert!(stdout.contains("eval"));
        assert!(!stdout.contains("viewer"));

        let mut command = cli::Cli::command();
        let subcommands: Vec<&str> = command.get_subcommands().map(|c| c.get_name()).collect();
        for command_name in ["auth", "init", "doctor", "probe", "build", "catalog"] {
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
        assert!(stdout.contains(env!("CARGO_PKG_VERSION")));
        assert!(stdout.contains(rlmesh_proto::CURRENT_WORKFLOW_EDITION));
        assert!(stdout.contains("Distribution"));
        assert!(!stdout.contains('\x1b'));
    }

    #[tokio::test]
    async fn invalid_command_is_plain_and_goes_to_stderr() {
        let (code, stdout, stderr) = run_for_test(&["not-a-command"]).await;

        assert_eq!(code, 2);
        assert!(stdout.is_empty());
        assert!(stderr.contains("unrecognized subcommand"));
        assert!(!stderr.contains('\x1b'));
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
