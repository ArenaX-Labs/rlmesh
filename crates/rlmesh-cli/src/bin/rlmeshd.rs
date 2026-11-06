//! rlmeshd: `rlmesh host` under its own name, for a service (design §3).

#[tokio::main]
async fn main() {
    std::process::exit(rlmesh_cli::run_terminal(&["host"]).await);
}
