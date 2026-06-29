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
    /// Smoke-test the terminal/HTTP renderer with synthetic frames (diagnostic).
    #[command(hide = true)]
    Viewtest(ViewtestArgs),
}

/// Flags for the hidden `viewtest` diagnostic.
#[derive(Args, Debug)]
pub struct ViewtestArgs {
    /// Serve a full-res browser view on this port instead of the terminal.
    #[arg(long, value_name = "PORT")]
    pub http: Option<u16>,
    /// Drive the terminal AND the browser at once (use with --http).
    #[arg(long)]
    pub both: bool,
    /// Target frames per second.
    #[arg(long, default_value_t = 30)]
    pub fps: u32,
    /// Stop after this many frames.
    #[arg(long, default_value_t = 900)]
    pub frames: u32,
    /// Feed only the HUD, never an image (mimics an env with no camera frames).
    #[arg(long)]
    pub no_frames: bool,
}
