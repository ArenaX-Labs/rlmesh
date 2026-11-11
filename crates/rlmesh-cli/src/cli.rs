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

/// `rlmesh cockpit` in a build without the window.
#[cfg(not(feature = "cockpit"))]
#[derive(Args, Debug)]
pub struct NoCockpitArgs {
    #[arg(trailing_var_arg = true, allow_hyphen_values = true, hide = true)]
    pub args: Vec<String>,
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
    /// Check a built image's config for packaging and contract mistakes,
    /// before pushing (docker image inspect; exits 1 on a failure).
    CheckImage(CheckImageArgs),
    /// Print a class's describe envelope, or its `dev.rlmesh.describe` label
    /// for a custom-entrypoint image build.
    Describe(DescribeArgs),
    /// Host real-time sessions on this machine until Ctrl-C (also installed
    /// as `rlmeshd`, to run as a service): on the LAN, under a host identity
    /// kept in its state directory, admitting the devices of its device
    /// table.
    Host(HostArgs),
    /// Open the cockpit on a live session (`rlmesh cockpit ADDRESS`) or a
    /// recording (`--replay FILE`): live channels, rates, why each robot is
    /// or isn't moving, the timeline, and arm and e-stop.
    #[cfg(feature = "cockpit")]
    Cockpit(Box<rlmesh_cockpit::CockpitArgs>),
    /// Open the cockpit (not in this build: install `rlmesh` on a desktop
    /// machine).
    #[cfg(not(feature = "cockpit"))]
    Cockpit(NoCockpitArgs),
    /// List and change the devices and robot pins of this machine's
    /// rlmeshd, and provision boards over USB; it applies a change at once.
    #[command(visible_alias = "device")]
    Devices(DevicesArgs),
    /// Check, run, stop and arm scenes on their robots' home rlmeshd.
    #[cfg(feature = "fleet")]
    Scene(SceneArgs),
    /// List the robots waiting in the home rlmeshd's lobby.
    #[cfg(feature = "fleet")]
    Lobby(HomeArgs),
    /// Pair this machine with a robot's rlmeshd (shows a code to type on
    /// the robot), or, on the robot, approve a machine by its code.
    Pair(PairArgs),
    /// Run an env or model image and check its handshake, describe,
    /// conformance, reproducibility, and preview clip (exits 1 on a failed check).
    Doctor(DoctorArgs),
    /// Run a model image and sweep how many concurrent sessions it serves,
    /// recommending `max_batch_size` (exits 1 when no size runs).
    Bench(BenchArgs),
    /// Make the identity a session host another process runs is started
    /// under (`RLMESH_HOST_IDENTITY`, a secret), or print the address key of
    /// one: what a launcher passes a container, and the key it dials it by.
    HostIdentity(HostIdentityArgs),
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
    /// Print the report as JSON, in the `probe-report.json` shape.
    #[arg(long)]
    pub json: bool,
}

/// `rlmesh host-identity` subcommands.
#[derive(Args, Debug)]
pub struct HostIdentityArgs {
    #[command(subcommand)]
    pub command: HostIdentityCommand,
}

#[derive(Subcommand, Debug)]
pub enum HostIdentityCommand {
    /// Print a fresh host identity and its address key, as `identity=` and
    /// `key=` lines. The identity is a secret: pass it to the host in
    /// `RLMESH_HOST_IDENTITY` through the environment, never argv, and keep
    /// it out of logs; `--out` writes it to a 0600 file instead of stdout.
    New(HostIdentityNewArgs),
    /// Print the address key (`key=`) of the host identity in
    /// `RLMESH_HOST_IDENTITY`, or in `--identity-file`. The identity is a
    /// secret, so it is never taken on the command line.
    Key(HostIdentityKeyArgs),
}

/// The keyed address `rlmesh host-identity` prints beside the key.
#[derive(Args, Debug)]
pub struct LaunchAddressArgs {
    /// Also print the keyed address (`address=`) of the session the host
    /// serves at HOST:PORT (its base port; the default session) or
    /// HOST:PORT/SESSION, as participants dial it from where they run.
    #[arg(long, value_name = "HOST:PORT[/SESSION]")]
    pub address: Option<String>,
    /// With `--address`: print participant NAME's address in that session.
    #[arg(long, value_name = "NAME", requires = "address")]
    pub participant: Option<String>,
}

/// Flags for `rlmesh host-identity new`.
#[derive(Args, Debug)]
pub struct HostIdentityNewArgs {
    /// Write the identity (a secret) to FILE, mode 0600, instead of
    /// printing it.
    #[arg(long, value_name = "FILE")]
    pub out: Option<std::path::PathBuf>,
    #[command(flatten)]
    pub address: LaunchAddressArgs,
    /// Print `{"identity","identityFile","key","address"}` instead of lines.
    #[arg(long)]
    pub json: bool,
}

/// Flags for `rlmesh host-identity key`.
#[derive(Args, Debug)]
pub struct HostIdentityKeyArgs {
    /// Read the identity (a secret) from FILE, `-` for stdin, instead of
    /// `RLMESH_HOST_IDENTITY`.
    #[arg(long, value_name = "FILE")]
    pub identity_file: Option<std::path::PathBuf>,
    #[command(flatten)]
    pub address: LaunchAddressArgs,
    /// Print `{"key","address"}` instead of lines.
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
    /// Keep the host's keys and device table here (default: `rlmesh/daemon`
    /// in the user state directory). They are made on first start and
    /// kept, so boards provisioned for this host stay valid across restarts.
    #[arg(long, value_name = "DIR", env = "RLMESH_STATE_DIR")]
    pub state_dir: Option<std::path::PathBuf>,
    /// Listen here (this port and the next three) instead of every
    /// interface on the well-known port (`RLMESH_PORT`, else 7447). Only the
    /// devices in `devices.toml` (and this user, `local`) can attach.
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
    /// How much of each session to keep recorded (`30m`, `2h`; default
    /// 30m): older segments go.
    #[arg(long, value_name = "SPAN", value_parser = span, conflicts_with = "no_record")]
    pub record_keep: Option<std::time::Duration>,
    /// How much disk the rolling recordings may use in all (`20GB`,
    /// `500MB`; default 20GB): the oldest segments go first.
    #[arg(long, value_name = "SIZE", value_parser = size, conflicts_with = "no_record")]
    pub record_max_size: Option<u64>,
    /// Pause recording while the disk has less free than this (default
    /// 2GB).
    #[arg(long, value_name = "SIZE", value_parser = size, conflicts_with = "no_record")]
    pub record_min_free: Option<u64>,
    /// How long kept incidents stay (`14d`, `48h`; default 14d).
    #[arg(long, value_name = "SPAN", value_parser = span, conflicts_with = "no_record")]
    pub incidents_keep: Option<std::time::Duration>,
    /// How much disk kept incidents may use in all (default 10GB): the
    /// oldest go first.
    #[arg(long, value_name = "SIZE", value_parser = size, conflicts_with = "no_record")]
    pub incidents_max_size: Option<u64>,
    /// How long a participant's heartbeats may stop before it is lost
    /// (`3s`, `1500ms`; default 1 s). Raise it for robots on Wi-Fi.
    #[arg(long, value_name = "DURATION", value_parser = duration)]
    pub presence_grace: Option<std::time::Duration>,
    /// How long a lease holder's heartbeats may stop before its leases lapse
    /// (default 250 ms).
    #[arg(long, value_name = "DURATION", value_parser = duration)]
    pub lease_grace: Option<std::time::Duration>,
    /// A robot that went safe, restarted or came back from a loss is driven
    /// again at once, instead of waiting to be re-armed by name. Other robots
    /// never stop for one either way.
    #[arg(long)]
    pub rearm_robots: bool,
    /// Only authenticate parts' samples and chunks (a MAC every reader of a
    /// part can make) instead of encrypting them. Beyond loopback a host
    /// seals (encrypts) every part by default; on loopback it MACs.
    #[arg(long)]
    pub no_seal: bool,
    /// How boards carry their realtime parts (action chunks down, states
    /// and `_applied` up): `tcp` over the session's links, or `udp` over the
    /// host's datagram path on UDP port `RLMESH_PORT + 1` (open it in the
    /// firewall), where a lost packet holds nothing up. A board that can't
    /// take the datagram path, a virtual-clock session and every
    /// non-board participant keep TCP. Boards re-attach to switch.
    #[arg(long, value_enum, env = "RLMESH_BOARD_DATA", default_value_t = BoardData::Tcp)]
    pub board_data: BoardData,
    /// Bound the bytes each accepted TCP link may hold unsent
    /// (`TCP_NOTSENT_LOWAT`, Linux only): what a slow link can't take waits
    /// in the host's queues, where stale samples are dropped, instead of in
    /// the socket. Off by default; for measuring lossy links.
    #[arg(long, value_name = "BYTES", value_parser = clap::value_parser!(u32).range(1..))]
    pub tcp_notsent_lowat: Option<u32>,
    /// Host at most this many sessions: the default one and those its
    /// participants name. A session no participant attaches to within 10 s
    /// stops.
    #[arg(long, value_name = "N", default_value_t = 8, value_parser = clap::value_parser!(u32).range(1..))]
    pub max_sessions: u32,
    /// Also host the lobby and run scenes (unreleased).
    #[cfg(feature = "fleet")]
    #[arg(long)]
    pub lobby: bool,
}

/// How boards carry their realtime parts (`rlmesh host --board-data`).
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BoardData {
    /// Over the session's TCP links.
    #[default]
    Tcp,
    /// Over the host's datagram path (UDP).
    Udp,
}

/// A duration flag: a number with `ms` or `s` (`250ms`, `1.5s`).
fn duration(text: &str) -> Result<std::time::Duration, String> {
    let (number, scale) = if let Some(ms) = text.strip_suffix("ms") {
        (ms, 1e-3)
    } else if let Some(s) = text.strip_suffix('s') {
        (s, 1.0)
    } else {
        return Err(format!("{text:?}: a duration with its unit, 250ms or 1.5s"));
    };
    let value: f64 = number
        .trim()
        .parse()
        .map_err(|_| format!("{text:?}: a duration with its unit, 250ms or 1.5s"))?;
    if !value.is_finite() || value <= 0.0 || value * scale > 3600.0 {
        return Err(format!("{text:?}: between 1ms and 1 hour"));
    }
    Ok(std::time::Duration::from_secs_f64(value * scale))
}

/// A span flag: a number with `s`, `m`, `h` or `d` (`30m`, `14d`).
fn span(text: &str) -> Result<std::time::Duration, String> {
    let wrong = || format!("{text:?}: a span with its unit, 90s, 30m, 2h or 14d");
    let unit = text.chars().last().ok_or_else(wrong)?;
    let scale = match unit {
        's' => 1.0,
        'm' => 60.0,
        'h' => 3600.0,
        'd' => 86_400.0,
        _ => return Err(wrong()),
    };
    let value: f64 = text[..text.len() - 1].trim().parse().map_err(|_| wrong())?;
    if !value.is_finite() || value <= 0.0 || value * scale > 366.0 * 86_400.0 {
        return Err(format!("{text:?}: between 1s and a year"));
    }
    Ok(std::time::Duration::from_secs_f64(value * scale))
}

/// A size flag: a number with `B`, `KB`, `MB`, `GB` or `TB` (powers of
/// 1000; `KiB` to `TiB` for powers of 1024).
fn size(text: &str) -> Result<u64, String> {
    let wrong = || format!("{text:?}: a size with its unit, 500MB or 20GB");
    let upper = text.trim().to_ascii_uppercase();
    let digits = upper
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .ok_or_else(wrong)?;
    let (number, unit) = upper.split_at(digits);
    let scale: f64 = match unit.trim() {
        "B" => 1.0,
        "KB" => 1e3,
        "MB" => 1e6,
        "GB" => 1e9,
        "TB" => 1e12,
        "KIB" => 1024.0,
        "MIB" => 1024.0 * 1024.0,
        "GIB" => 1024.0 * 1024.0 * 1024.0,
        "TIB" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        _ => return Err(wrong()),
    };
    let value: f64 = number.parse().map_err(|_| wrong())?;
    let bytes = value * scale;
    if !bytes.is_finite() || bytes < 1.0 || bytes > 1e18 {
        return Err(format!("{text:?}: between 1B and 1000PB"));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(bytes as u64)
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
        /// The day (YYYY-MM-DD, UTC) from which it may no longer attach, or
        /// `never`.
        #[arg(long, value_name = "DATE")]
        expires: Option<String>,
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
    /// a robot name and this rlmeshd's key, id and addresses, enroll its key as a board and
    /// pin its robot name to it. Run it on the robot's machine, with
    /// rlmeshd running; the board keeps its key when provisioned again.
    Provision {
        /// The board's serial port (`/dev/ttyUSB0`, `/dev/ttyACM0`).
        #[arg(long, value_name = "TTY")]
        port: std::path::PathBuf,
        /// Assign this robot name (1–128 bytes of [A-Za-z0-9._-]); may rename a board.
        #[arg(long, value_name = "ROBOT")]
        name: String,
        /// The highest class it may attach and lease at (default: model,
        /// or what it already has).
        #[arg(long, value_name = "CLASS")]
        max: Option<String>,
    },
}

#[cfg(feature = "fleet")]
/// Flags for `rlmesh scene`.
#[derive(Args, Debug)]
pub struct SceneArgs {
    /// rlmeshd's state directory (default: `rlmesh/daemon` in the user
    /// state directory).
    #[arg(long, value_name = "DIR", env = "RLMESH_STATE_DIR", global = true)]
    pub state_dir: Option<std::path::PathBuf>,
    #[command(subcommand)]
    pub command: SceneCommand,
}

#[cfg(feature = "fleet")]
#[derive(Subcommand, Debug)]
pub enum SceneCommand {
    /// Check a scene file offline, as a run's first stage will: its
    /// format, members, places and security, and its robots and
    /// controllers against the device table and pins. Prints every problem
    /// with its line and fix, and exits 1 if there is one.
    Check {
        /// The scene file.
        #[arg(value_name = "SCENE.TOML")]
        file: std::path::PathBuf,
    },
    /// Run a scene on its robots' home rlmeshd and follow its stages until
    /// it ends; Ctrl-C stops it.
    Run(RunArgs),
    /// Stop a run.
    Stop(RunIdArgs),
    /// Arm a run that waits for an explicit arm (`--arm manual`, or after a
    /// lost member rejoined), or re-arm one robot of a running run.
    Arm(ArmArgs),
}

#[cfg(feature = "fleet")]
/// Where the home rlmeshd is.
#[derive(Args, Debug)]
pub struct HomeArgs {
    /// The home daemon (`rig-pi.local`, `10.0.0.5:7447`, or the host id
    /// this machine paired with, `rlm-3f9a2c`, found again over mDNS when
    /// its address changed; default: this machine's).
    #[arg(long, value_name = "HOST")]
    pub home: Option<String>,
}

#[cfg(feature = "fleet")]
/// Flags for `rlmesh scene run`.
#[derive(Args, Debug)]
pub struct RunArgs {
    /// The scene file.
    #[arg(value_name = "SCENE.TOML")]
    pub file: std::path::PathBuf,
    /// `manual`: wait for `rlmesh scene arm` even when the scene arms itself.
    #[arg(long, value_name = "auto|manual", default_value = "auto")]
    pub arm: String,
    #[command(flatten)]
    pub home: HomeArgs,
}

#[cfg(feature = "fleet")]
/// Flags for `rlmesh scene stop`.
#[derive(Args, Debug)]
pub struct RunIdArgs {
    /// The run (`r-7k2m`), as `rlmesh scene run` printed it.
    pub run: String,
    #[command(flatten)]
    pub home: HomeArgs,
}

#[cfg(feature = "fleet")]
/// Flags for `rlmesh scene arm`.
#[derive(Args, Debug)]
pub struct ArmArgs {
    /// The run (`r-7k2m`), as `rlmesh scene run` printed it.
    pub run: String,
    /// Re-arm this robot alone: it stopped on its own (a fault, a restart)
    /// and the run went on without it.
    #[arg(value_name = "ROBOT")]
    pub robot: Option<String>,
    #[command(flatten)]
    pub home: HomeArgs,
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
    /// Pair a browser token instead: a device of its own (default cap
    /// model), which this machine's rlmesh-gateway attaches its browser
    /// with. Prints the token once.
    #[arg(long)]
    pub browser: bool,
    #[command(subcommand)]
    pub command: Option<PairCommand>,
}

#[derive(Subcommand, Debug)]
pub enum PairCommand {
    /// On the robot: list the machines and browsers waiting, and approve
    /// the one showing CODE (typed, or asked for).
    Approve {
        /// The 6-digit code the machine shows.
        code: Option<String>,
        /// The highest class it may attach and lease at: observer,
        /// scripted, model or operator (default operator for a machine,
        /// model for a browser).
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
    /// labels generated off linux fail validation.
    #[arg(long)]
    pub label: bool,
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{size, span};

    #[test]
    fn spans_and_sizes_read_with_their_units() {
        assert_eq!(span("30m"), Ok(Duration::from_secs(1800)));
        assert_eq!(span("14d"), Ok(Duration::from_secs(14 * 86_400)));
        assert_eq!(span("1.5h"), Ok(Duration::from_secs(5400)));
        for bad in ["", "30", "m", "0s", "-1h", "400d", "1w"] {
            assert!(span(bad).is_err(), "{bad}");
        }
        assert_eq!(size("20GB"), Ok(20_000_000_000));
        assert_eq!(size("500 mb"), Ok(500_000_000));
        assert_eq!(size("1.5GiB"), Ok(1_610_612_736));
        assert_eq!(size("512B"), Ok(512));
        for bad in ["", "20", "GB", "0GB", "1XB", "1e30TB"] {
            assert!(size(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_host_carries_boards_over_tcp_unless_told_udp() {
        use clap::Parser;

        use super::{BoardData, Cli, Command};
        let host = |args: &[&str]| match Cli::try_parse_from(args).map(|c| c.command) {
            Ok(Command::Host(host)) => Ok((host.board_data, host.tcp_notsent_lowat)),
            Ok(_) => Err("not host".to_string()),
            Err(err) => Err(err.to_string()),
        };
        assert_eq!(host(&["rlmesh", "host"]), Ok((BoardData::Tcp, None)));
        assert_eq!(
            host(&[
                "rlmesh",
                "host",
                "--board-data",
                "udp",
                "--tcp-notsent-lowat",
                "4096"
            ]),
            Ok((BoardData::Udp, Some(4096)))
        );
        assert!(host(&["rlmesh", "host", "--board-data", "quic"]).is_err());
        assert!(host(&["rlmesh", "host", "--tcp-notsent-lowat", "0"]).is_err());
    }
}
