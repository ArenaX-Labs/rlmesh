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
    /// Sign in to an RLMesh managed platform (OAuth device flow).
    Login(LoginArgs),
    /// Delete the stored credential for a profile.
    Logout(ProfileArgs),
    /// Show the active profile, its platform, and sign-in state (exits
    /// nonzero unless signed in with a verified session).
    Whoami(ProfileArgs),
    /// Authenticate container tooling with the platform's image registry.
    Registry(RegistryArgs),
    /// Manage named platform profiles.
    Profile(ProfileCommandArgs),
    /// List the organizations you belong to, or switch the active one.
    Org(OrgArgs),
    /// Print a fresh access token for scripts: `Authorization: Bearer $(rlmesh token)`.
    Token(TokenArgs),
    /// Submit, list, watch, and cancel evaluations on the signed-in platform.
    Eval(EvalArgs),
    /// Smoke-test the terminal/HTTP renderer with synthetic frames (diagnostic).
    #[command(hide = true)]
    Viewtest(ViewtestArgs),
}

/// Shared flag selecting which named profile to act on.
#[derive(Args, Debug)]
pub struct ProfileArgs {
    /// Profile name (defaults to the configured default profile).
    #[arg(long, value_name = "NAME", env = "RLMESH_PROFILE")]
    pub profile: Option<String>,
}

/// Flags for `rlmesh login`.
#[derive(Args, Debug)]
pub struct LoginArgs {
    #[command(flatten)]
    pub profile: ProfileArgs,
    /// Platform base URL, e.g. https://platform.example.com (defaults to the hosted platform, https://api.rlmesh.dev; remembered per profile afterwards).
    #[arg(long, value_name = "URL", env = "RLMESH_PLATFORM_URL")]
    pub platform: Option<String>,
}

/// Container-registry authentication subcommands.
#[derive(Args, Debug)]
pub struct RegistryArgs {
    #[command(subcommand)]
    pub command: RegistryCommand,
}

#[derive(Subcommand, Debug)]
pub enum RegistryCommand {
    /// Log Docker in to the platform's image registry using the current session.
    Login(ProfileArgs),
    /// Docker credential-helper protocol endpoint (invoked by docker as
    /// docker-credential-rlmesh, not by hand).
    #[command(hide = true)]
    CredentialHelper(CredentialHelperArgs),
}

/// The docker credential-helper operation, per its get/store/erase protocol.
#[derive(Args, Debug)]
pub struct CredentialHelperArgs {
    pub operation: String,
}

/// Organization subcommands.
#[derive(Args, Debug)]
pub struct OrgArgs {
    #[command(subcommand)]
    pub command: OrgCommand,
}

#[derive(Subcommand, Debug)]
pub enum OrgCommand {
    /// List the organizations the signed-in user belongs to.
    List(ProfileArgs),
    /// Make an organization the profile's active one (WorkOS org id, org_...).
    Switch {
        id: String,
        #[command(flatten)]
        profile: ProfileArgs,
    },
}

/// Named-profile management subcommands.
#[derive(Args, Debug)]
pub struct ProfileCommandArgs {
    #[command(subcommand)]
    pub command: ProfileCommand,
}

#[derive(Subcommand, Debug)]
pub enum ProfileCommand {
    /// List profiles, marking the default and each profile's sign-in state.
    List,
    /// Set the default profile used when --profile/RLMESH_PROFILE is absent.
    Use { name: String },
    /// Delete a profile: its stored credential and its config entry.
    Remove { name: String },
}

/// Flags for `rlmesh token`.
#[derive(Args, Debug)]
pub struct TokenArgs {
    #[command(flatten)]
    pub profile: ProfileArgs,
    /// Print `{"platform", "token"}` instead of the bare token.
    #[arg(long)]
    pub json: bool,
}

/// Evaluation subcommands.
#[derive(Args, Debug)]
pub struct EvalArgs {
    #[command(subcommand)]
    pub command: EvalCommand,
}

#[derive(Subcommand, Debug)]
pub enum EvalCommand {
    /// Submit an evaluation request (JSON file, or `-` for stdin).
    Submit(EvalSubmitArgs),
    /// List evaluations, newest first.
    List(EvalListArgs),
    /// Print one evaluation as JSON.
    Get(EvalIdArgs),
    /// Poll an evaluation until it finishes; exits nonzero unless it completed.
    Wait(EvalIdArgs),
    /// Cancel a running evaluation.
    Cancel(EvalIdArgs),
}

#[derive(Args, Debug)]
pub struct EvalSubmitArgs {
    /// Path to the request JSON (`-` reads stdin).
    pub request: String,
    /// Validate and size the request without launching it.
    #[arg(long)]
    pub preview: bool,
    /// Block until the evaluation finishes.
    #[arg(long)]
    pub wait: bool,
    #[command(flatten)]
    pub profile: ProfileArgs,
    /// Print the raw API response.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct EvalListArgs {
    /// Filter by status (queued, running, completed, failed, cancelled, ...).
    #[arg(long)]
    pub status: Option<String>,
    /// Filter by tag as key:value; repeatable, every pair must match.
    #[arg(long = "tag", value_name = "KEY:VALUE")]
    pub tags: Vec<String>,
    /// Case-insensitive substring match on the evaluation name.
    #[arg(long)]
    pub q: Option<String>,
    #[arg(long, default_value_t = 25)]
    pub limit: u32,
    #[command(flatten)]
    pub profile: ProfileArgs,
    /// Print the raw API response.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct EvalIdArgs {
    /// Evaluation id (eval_...).
    pub id: String,
    #[command(flatten)]
    pub profile: ProfileArgs,
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
