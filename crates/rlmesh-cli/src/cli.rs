use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "rlmesh",
    about = "RLMesh - Gymnasium-compatible infrastructure for model-environment evaluation",
    version,
    arg_required_else_help = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Print RLMesh CLI version and distribution details
    Version,
    /// Open the minimal render viewer driven over stdin
    #[command(name = "viewer", hide = true)]
    Viewer(ViewerArgs),
}

#[derive(Args, Debug, Clone)]
pub struct ViewerArgs {
    /// Window title
    #[arg(long, default_value = "RLMesh Render")]
    pub title: String,
}
