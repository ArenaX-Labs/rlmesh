//! rlmeshd: `rlmesh host` under its own name, for a service.

#[tokio::main]
async fn main() {
    std::process::exit(rlmesh_cli::run_terminal(&["host"]).await);
}
