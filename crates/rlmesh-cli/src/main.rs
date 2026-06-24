#[tokio::main]
async fn main() {
    let exit_code = match rlmesh_cli::run_cli().await {
        Ok(code) => code,
        Err(err) => {
            #[allow(clippy::print_stderr)]
            {
                eprintln!("Error: {err:#}");
            }
            1
        }
    };
    std::process::exit(exit_code);
}
