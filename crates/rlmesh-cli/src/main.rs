#[tokio::main]
async fn main() {
    let prefix = std::env::args_os()
        .next()
        .map_or(&[][..], |argv0| rlmesh_cli::invoked_as(&argv0));
    std::process::exit(rlmesh_cli::run_terminal(prefix).await);
}
