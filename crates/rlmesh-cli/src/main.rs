use std::io::{self, IsTerminal};

#[tokio::main]
async fn main() {
    let mut stdout = io::stdout();
    let mut stderr = io::stderr();
    let stdout_is_terminal = stdout.is_terminal();
    let stderr_is_terminal = stderr.is_terminal();

    let exit_code = match rlmesh_cli::run_cli(
        std::env::args_os().skip(1).collect(),
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
    };
    std::process::exit(exit_code);
}
