//! Docker credential helper: docker invokes this binary (by the name in
//! credHelpers) with the operation as argv[1] and the payload on stdin.

#[tokio::main]
async fn main() {
    std::process::exit(rlmesh_cli::run_terminal(&["registry", "credential-helper"]).await);
}
