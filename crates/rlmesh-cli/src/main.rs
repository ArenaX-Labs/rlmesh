#[tokio::main]
async fn main() {
    std::process::exit(rlmesh_cli::run_terminal(&[]).await);
}
