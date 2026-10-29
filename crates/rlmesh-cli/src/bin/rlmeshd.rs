//! rlmeshd: `rlmesh daemon` under its own name (design §3).

#[tokio::main]
async fn main() {
    std::process::exit(rlmesh_cli::run_terminal(&["daemon"]).await);
}
