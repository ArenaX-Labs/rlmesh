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
    Whoami(WhoamiArgs),
    /// Authenticate container tooling with the platform's image registry, and
    /// publish multi-variant image indexes.
    Registry(RegistryArgs),
    /// Manage named platform profiles.
    Profile(ProfileCommandArgs),
    /// List the organizations you belong to, or switch the active one.
    Org(OrgArgs),
    /// Print a fresh access token for scripts: `Authorization: Bearer $(rlmesh token)`.
    Token(TokenArgs),
    /// Submit, list, watch, and cancel evaluations on the signed-in platform.
    Eval(EvalArgs),
    /// Check an env or model class for packaging and contract mistakes before
    /// building its image (runs the Python describe; exits 1 on a failure).
    Check(CheckArgs),
    /// Check a built image's config the way the platform's admission will,
    /// before pushing (docker image inspect; exits 1 on a failure).
    CheckImage(CheckImageArgs),
    /// Print a class's describe envelope, or its `dev.rlmesh.describe` label
    /// for a custom-entrypoint image build.
    Describe(DescribeArgs),
    /// Host v2 sessions on this machine until Ctrl-C: the default session,
    /// and any other a participant names. Prints the address it listens on.
    Host(HostArgs),
    /// Run rlmeshd in the foreground (also installed as `rlmeshd`): v2
    /// sessions on the LAN under this machine's host identity, admitting the
    /// devices of its device table.
    Daemon(DaemonArgs),
    /// List and change the devices and robot pins of this machine's
    /// rlmeshd, and provision boards over USB; it applies a change at once.
    #[command(visible_alias = "device")]
    Devices(DevicesArgs),
    /// Pair this machine with a robot's rlmeshd (shows a code to type on
    /// the robot), or, on the robot, approve a machine by its code.
    Pair(PairArgs),
    /// Run an env or model image and check it the way the platform's probe
    /// does: handshake, describe, conformance, reproducibility, and an env's
    /// preview clip (exits 1 on a failed check).
    Doctor(DoctorArgs),
    /// Run a model image and sweep how many concurrent sessions it serves,
    /// recommending `max_batch_size` (exits 1 when no size runs).
    Bench(BenchArgs),
}

/// Shared flag selecting which named profile to act on.
#[derive(Args, Debug)]
pub struct ProfileArgs {
    /// Profile name (defaults to the configured default profile).
    #[arg(long, value_name = "NAME", env = "RLMESH_PROFILE")]
    pub profile: Option<String>,
}

/// Flags for `rlmesh whoami`.
#[derive(Args, Debug)]
pub struct WhoamiArgs {
    #[command(flatten)]
    pub profile: ProfileArgs,
    /// Print `{"profile","platform","status","verified","identity","error"}` instead of text.
    #[arg(long)]
    pub json: bool,
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
    /// Assemble per-variant images into one OCI image index (a version whose
    /// children are compute variants) and push it under TARGET: the pushed
    /// SOURCE images, or each variant of `rlmesh.toml`, built and pushed first.
    Publish(PublishArgs),
    /// Docker credential-helper protocol endpoint (invoked by docker as
    /// docker-credential-rlmesh, not by hand).
    #[command(hide = true)]
    CredentialHelper(CredentialHelperArgs),
}

/// Flags for `rlmesh registry publish`.
#[derive(Args, Debug)]
pub struct PublishArgs {
    /// The version to publish, as `REPOSITORY:TAG` (e.g. `registry.rlmesh.dev/ns/pi0:v3`).
    pub target: String,
    /// The per-variant images, already pushed (e.g. `ns/pi0:v3-cuda12`). Each must
    /// be one linux image (attestations are carried along) whose
    /// `dev.rlmesh.package` label declares a unique `variant.key`. Without
    /// any, each variant of `rlmesh.toml` is built and pushed first.
    #[arg(value_name = "SOURCE")]
    pub sources: Vec<String>,
    /// The manifest to build from when no SOURCE is given (`-` reads stdin);
    /// defaults to `rlmesh.toml` in the current directory.
    #[arg(long, value_name = "FILE", conflicts_with_all = ["sources", "index_package"])]
    pub config: Option<std::path::PathBuf>,
    /// Another tag for the index in TARGET's repository; repeatable.
    #[arg(long = "tag", value_name = "TAG")]
    pub tags: Vec<String>,
    /// A moving channel tag (e.g. `latest`) to point at this version too. It
    /// moves without `--force`, unless it is also TARGET or a `--tag`, which
    /// keeps it protected like them.
    #[arg(long, value_name = "TAG")]
    pub channel: Option<String>,
    /// A JSON file of version-level package data (checkpoints, compatibility),
    /// set as the index's `dev.rlmesh.package` annotation.
    #[arg(long, value_name = "FILE")]
    pub index_package: Option<std::path::PathBuf>,
    /// Print the per-variant summary and the index JSON without pushing (with
    /// `rlmesh.toml`, without building); fails on a tag the real run would
    /// refuse.
    #[arg(long)]
    pub dry_run: bool,
    /// With `--dry-run` and `rlmesh.toml`, print each variant's build and
    /// rendered `dev.rlmesh.package` label, and the index annotation, as JSON.
    /// Only renders: no tag is read or checked.
    #[arg(long, requires = "dry_run", conflicts_with = "sources")]
    pub json: bool,
    /// Push TARGET or a `--tag` that already points at a different index, or
    /// that cannot be read (the channel tag needs no `--force`, unless it is
    /// also TARGET or a `--tag`). With `rlmesh.toml`, also build over a
    /// per-variant tag holding another image, and rebuild even where TARGET
    /// already holds the index of the builds at those tags.
    #[arg(long)]
    pub force: bool,
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
    List(OrgListArgs),
    /// Make an organization the profile's active one (identity-provider
    /// organization id, org_...).
    Switch {
        /// Organization id (org_...).
        id: String,
        #[command(flatten)]
        profile: ProfileArgs,
    },
}

/// Flags for `rlmesh org list`.
#[derive(Args, Debug)]
pub struct OrgListArgs {
    #[command(flatten)]
    pub profile: ProfileArgs,
    /// Print the platform's organizations array as JSON.
    #[arg(long)]
    pub json: bool,
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
    List(ProfileListArgs),
    /// Set the default profile used when --profile/RLMESH_PROFILE is absent.
    Use {
        /// Profile name to make the default.
        name: String,
    },
    /// Delete a profile: its stored credential and its config entry.
    Remove {
        /// Profile name to delete.
        name: String,
    },
}

/// Flags for `rlmesh profile list`.
#[derive(Args, Debug)]
pub struct ProfileListArgs {
    /// Print `[{"name","platform","status","default"}]` instead of a table.
    #[arg(long)]
    pub json: bool,
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
    /// Print per-task, per-suite and overall success of saved result bundles.
    Report(EvalReportArgs),
    /// Run an evaluation request on this machine, in Docker.
    Run(EvalRunArgs),
}

#[derive(Args, Debug)]
pub struct EvalRunArgs {
    /// Path to the request JSON (`-` reads stdin), the file `eval submit` takes.
    pub request: String,
    /// Resolve catalog names (`prebuilt/<model>/<env>`, `prebuilt/<env>`): a
    /// local build `rlmesh-<model>-<env>:TAG` when there is one (TAG is
    /// `latest` unless given), else the registry's image.
    #[arg(long, value_name = "TAG", num_args = 0..=1, default_missing_value = "latest")]
    pub catalog: Option<String>,
    /// `docker run --gpus` for the images whose package declares a GPU, e.g. `all`.
    #[arg(long)]
    pub gpus: Option<String>,
    /// A `docker run -v` mount for every container, e.g. a checkpoint store
    /// at the path its package label names; repeatable.
    #[arg(long = "volume", value_name = "HOST:CONTAINER")]
    pub volumes: Vec<String>,
    /// The most env containers alive at once: a model's workloads run in
    /// waves of this many.
    #[arg(long, value_name = "N", default_value_t = 16, value_parser = clap::value_parser!(u32).range(1..))]
    pub max_envs: u32,
    /// The folder for the run index and one result bundle per workload.
    #[arg(long, default_value = "rlmesh-eval")]
    pub out: std::path::PathBuf,
    /// Continue the run in `--out`: skip the workloads whose bundles already
    /// hold their episodes. Refused when the plan or an image changed.
    #[arg(long)]
    pub resume: bool,
    /// Print the report as JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct EvalReportArgs {
    /// Result bundles (folders or zips), or folders of bundles.
    #[arg(required = true)]
    pub paths: Vec<std::path::PathBuf>,
    /// Print the report as JSON.
    #[arg(long)]
    pub json: bool,
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
    /// Maximum evaluations to return.
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

/// Flags for `rlmesh check`.
#[derive(Args, Debug)]
pub struct CheckArgs {
    /// The class to check, as `module:Class` (an EnvFactory or Model subclass;
    /// the kind is detected).
    pub target: String,
    /// Print the report as JSON: `{"failed","warnings","not_checked","passed"}`.
    #[arg(long)]
    pub json: bool,
}

/// Flags for `rlmesh check-image`.
#[derive(Args, Debug)]
pub struct CheckImageArgs {
    /// A local image reference (`name:tag` or id) that `docker image inspect` resolves.
    pub image: String,
    /// The workflow editions the target platform can drive, comma-separated;
    /// the image's rlmesh must share one. Defaults to this CLI's own rlmesh.
    #[arg(
        long,
        value_name = "LIST",
        value_delimiter = ',',
        conflicts_with = "platform_version"
    )]
    pub platform_editions: Vec<String>,
    /// The rlmesh version the target platform runs. Only this CLI's own version
    /// is known (each release retains its own edition list); any other version
    /// reports the edition check as not checked, so pass --platform-editions
    /// instead. `0.1.0rc13` and `0.1.0-rc.13` name the same build.
    #[arg(long, value_name = "VERSION")]
    pub platform_version: Option<String>,
    /// Print the report as JSON: `{"failed","warnings","not_checked","passed"}`.
    #[arg(long)]
    pub json: bool,
}

/// Flags `doctor` and `bench` share: the image and how it runs.
#[derive(Args, Debug)]
pub struct ProbeImageArgs {
    /// A local image reference (`name:tag` or id).
    pub image: String,
    /// For a model that declares no ModelSpec: an env image whose spaces it is
    /// checked against (run beside it, never stepped).
    #[arg(long, value_name = "IMAGE")]
    pub env: Option<String>,
    /// GPUs for the container(s), as `docker run --gpus` takes them (`all`, `device=0`).
    #[arg(long, value_name = "SPEC")]
    pub gpus: Option<String>,
    /// Seconds the image gets to start serving and to answer each call.
    #[arg(long, value_name = "SECONDS", default_value_t = 120)]
    pub timeout: u64,
    /// Print the report as JSON, in the platform's `probe-report.json` shape.
    #[arg(long)]
    pub json: bool,
}

/// Flags for `rlmesh doctor`.
#[derive(Args, Debug)]
pub struct DoctorArgs {
    #[command(flatten)]
    pub image: ProbeImageArgs,
    /// Probe the image as this kind instead of the one its labels and command declare.
    #[arg(long, value_enum)]
    pub kind: Option<ProbeKind>,
    /// Write `probe-report.json` and an env's preview clip here.
    #[arg(long, value_name = "DIR", default_value = "rlmesh-doctor")]
    pub output: std::path::PathBuf,
}

/// What `doctor` probes an image as.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeKind {
    Env,
    Model,
}

/// Flags for `rlmesh bench`.
#[derive(Args, Debug)]
pub struct BenchArgs {
    #[command(flatten)]
    pub image: ProbeImageArgs,
    /// Concurrent sessions per sweep point, comma-separated and ascending.
    #[arg(
        long,
        value_name = "LIST",
        value_delimiter = ',',
        default_value = "1,2,4,8,16,32"
    )]
    pub sizes: Vec<u32>,
    /// Predicts each session times per point, after 3 untimed ones.
    #[arg(long, value_name = "N", default_value_t = 20)]
    pub iterations: u32,
}

/// Flags for `rlmesh host`.
#[derive(Args, Debug)]
pub struct HostArgs {
    /// Listen here (this port and the next three) instead of 127.0.0.1 on
    /// the well-known port (`RLMESH_PORT`, else 7447). Port 0: OS-assigned.
    #[arg(long, value_name = "ADDR")]
    pub listen: Option<std::net::SocketAddr>,
    /// Record every session here, always on: the newest 30 minutes or
    /// 20 GB, in one-minute MCAP segments under `rolling/{session}/`.
    /// Default: `rlmesh/recordings` in the user data directory.
    #[arg(long, value_name = "DIR")]
    pub record_dir: Option<std::path::PathBuf>,
    /// Do not record.
    #[arg(long, conflicts_with = "record_dir")]
    pub no_record: bool,
    /// Listen beyond loopback anyway (`--listen 0.0.0.0:…`): anyone who can
    /// reach it can try this host. Sessions for other machines are
    /// rlmeshd's.
    #[arg(long)]
    pub insecure: bool,
}

/// Flags for `rlmesh daemon`.
#[derive(Args, Debug)]
pub struct DaemonArgs {
    /// Keep the host's keys and device table here (default: `rlmesh/daemon`
    /// in the user state directory).
    #[arg(long, value_name = "DIR", env = "RLMESH_STATE_DIR")]
    pub state_dir: Option<std::path::PathBuf>,
    /// Listen here (this port and the next three) instead of every
    /// interface on the well-known port (`RLMESH_PORT`, else 7447).
    #[arg(long, value_name = "ADDR")]
    pub listen: Option<std::net::SocketAddr>,
}

/// Flags for `rlmesh devices`.
#[derive(Args, Debug)]
pub struct DevicesArgs {
    /// rlmeshd's state directory (default: `rlmesh/daemon` in the user
    /// state directory).
    #[arg(long, value_name = "DIR", env = "RLMESH_STATE_DIR", global = true)]
    pub state_dir: Option<std::path::PathBuf>,
    #[command(subcommand)]
    pub command: DevicesCommand,
}

#[derive(Subcommand, Debug)]
pub enum DevicesCommand {
    /// List the devices with their entries, and the robot pins.
    Ls,
    /// Change a paired device's entry; its participants re-attach under it.
    Allow {
        /// The device.
        device: String,
        /// The highest class it may attach and lease at: observer, scripted,
        /// model or operator.
        #[arg(long, value_name = "CLASS")]
        max: Option<String>,
        /// The parts it may lease and read: globs over `{robot}/{path}`,
        /// comma-separated.
        #[arg(long, value_name = "GLOBS", value_delimiter = ',')]
        parts: Option<Vec<String>>,
        /// The verbs it may send besides estop: arm, disarm, comma-separated.
        #[arg(long, value_name = "VERBS", value_delimiter = ',')]
        verbs: Option<Vec<String>>,
    },
    /// Remove a device: it is detached at once, and its robot pins stay.
    Revoke {
        /// The device.
        device: String,
    },
    /// Pin a robot name to another device (or board); its embodiment is
    /// detached, and an active session goes SAFE.
    Move {
        /// The robot name.
        robot: String,
        /// The device that embodies it from now on (`local`: this machine).
        #[arg(long, value_name = "DEVICE")]
        to: String,
        /// The tethered board behind that device, 16 hex digits.
        #[arg(long, value_name = "ID", conflicts_with = "no_board_id")]
        board_id: Option<String>,
        /// The robot has no board behind the device any more.
        #[arg(long)]
        no_board_id: bool,
    },
    /// Remove a robot name's pin; its embodiment is detached.
    Unpin {
        /// The robot name.
        robot: String,
    },
    /// Provision a networked board (an ESP32) plugged in over USB: give it
    /// this rlmeshd's key, id and addresses, enroll its key as a board and
    /// pin its robot name to it. Run it on the robot's machine, with
    /// rlmeshd running; the board keeps its key when provisioned again.
    Provision {
        /// The board's serial port (`/dev/ttyUSB0`, `/dev/ttyACM0`).
        #[arg(long, value_name = "TTY")]
        port: std::path::PathBuf,
        /// The robot name compiled into the board's sketch.
        #[arg(long, value_name = "ROBOT")]
        name: String,
        /// The highest class it may attach and lease at (default: model,
        /// or what it already has).
        #[arg(long, value_name = "CLASS")]
        max: Option<String>,
    },
}

/// Flags for `rlmesh pair`.
#[derive(Args, Debug)]
#[command(args_conflicts_with_subcommands = true, subcommand_negates_reqs = true)]
pub struct PairArgs {
    /// The robot's host (`robot.local`, `10.0.0.5:7447`).
    #[arg(value_name = "HOST", required = true)]
    pub host: Option<String>,
    /// The name this machine pairs as (default: its host name).
    #[arg(long, value_name = "NAME")]
    pub name: Option<String>,
    #[command(subcommand)]
    pub command: Option<PairCommand>,
}

#[derive(Subcommand, Debug)]
pub enum PairCommand {
    /// On the robot: list the machines waiting, and approve the one
    /// showing CODE (typed, or asked for).
    Approve {
        /// The 6-digit code the machine shows.
        code: Option<String>,
        /// The highest class it may attach and lease at: observer,
        /// scripted, model or operator (default operator).
        #[arg(long, value_name = "CLASS")]
        max: Option<String>,
        /// rlmeshd's state directory (default: `rlmesh/daemon` in the user
        /// state directory).
        #[arg(long, value_name = "DIR", env = "RLMESH_STATE_DIR")]
        state_dir: Option<std::path::PathBuf>,
    },
}

/// Flags for `rlmesh describe`.
#[derive(Args, Debug)]
pub struct DescribeArgs {
    /// The class to describe, as `module:Class` (the kind is detected).
    pub target: String,
    /// Print `dev.rlmesh.describe=<envelope>`, ready for `docker build --label`.
    /// Run it inside the built image (`docker run --rm --entrypoint rlmesh IMAGE
    /// describe ... --label`): the envelope records the machine it ran on, and
    /// the platform fails a label generated off linux.
    #[arg(long)]
    pub label: bool,
}
