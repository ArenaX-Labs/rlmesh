#[tokio::main]
async fn main() {
    let exit_code = match rlmesh_cli::run_cli().await {
        Ok(code) => code,
        Err(err) => {
            // Top-level fatal-error reporter for the binary; stderr is the right sink here.
            #[allow(clippy::print_stderr)]
            {
                eprintln!("Error: {err:#}");
            }
            1
        }
    };

    std::process::exit(exit_code);
}
