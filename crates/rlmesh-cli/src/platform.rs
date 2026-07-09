//! Sign-in, sign-out, and registry authentication against an RLMesh managed platform.
//!
//! The CLI is platform-agnostic: it learns everything it needs at runtime from
//! the platform URL a named profile points at (AWS-CLI-style profiles, each
//! remembering its own platform and holding its own stored credential).
//! The closed control-plane repo implements the endpoints below; keep the wire
//! shapes in [`tests`] in lockstep with it.
//!
//! # CLI ↔ control-plane contract
//!
//! - `GET <platform>/v1/auth/cli-config` (public, unauthenticated) returns
//!   `{"authkitDomain": "https://auth.example.com", "clientId": "client_..."}`
//!   (camelCase), or HTTP 404 when CLI auth is not configured on the platform.
//!   The CLI runs the WorkOS AuthKit OAuth device-authorization flow against
//!   `authkitDomain` with `clientId`.
//! - `POST <platform>/v1/api-keys` with `Authorization: Bearer <device-flow
//!   access token>` and JSON body exactly `{"name": "<key name>"}` (the server
//!   binds strictly; no extra fields) mints the CLI's durable credential,
//!   returning HTTP 201 with `{"id": "...", "name": "...", "value": "<secret>",
//!   "createdAt": "..."}` — `value` is only present on create. HTTP 403 means
//!   the access token carries no organization claim. `id` becomes the docker
//!   username, `value` the stored API key.
//! - `GET <platform>/v1/registry/info` with `Authorization: Bearer <api_key>`
//!   returns `{"host": "registry.example.com", "namespaces": ["acme", ...]}`
//!   describing the image registry to point container tooling at; the
//!   namespaces are the push-capable ones only.
//! - `GET <platform>/v1/me` with `Authorization: Bearer <api_key>` echoes the
//!   caller's identity (camelCase: `organizationId`, `subjectType`, and
//!   optionally `role`, `userId`, among other fields the CLI does not model).

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::cli::{LoginArgs, ProfileArgs};
use crate::style::Style;

/// The hosted RLMesh platform, used by `login` when neither `--platform`,
/// `RLMESH_PLATFORM_URL`, nor a remembered profile supplies one. Already in the
/// normalized form [`normalize_base_url`] produces (explicit scheme, no trailing
/// slash), so it needs no further massaging.
const DEFAULT_PLATFORM_URL: &str = "https://api.rlmesh.dev";

/// Normalize an operator-supplied base URL: trim whitespace, drop a trailing
/// slash, and prefix a scheme unless one is already present — `http://` for
/// loopback hosts (a local dev server almost never terminates TLS, and trying
/// anyway dies in an opaque rustls handshake error), `https://` for everything
/// else. An explicit scheme always wins.
fn normalize_base_url(raw: &str) -> String {
    let trimmed = raw.trim().trim_end_matches('/');
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        return trimmed.to_string();
    }
    let scheme = if is_loopback_host(trimmed) {
        "http"
    } else {
        "https"
    };
    format!("{scheme}://{trimmed}")
}

/// Whether a scheme-less base URL points at the local machine: `localhost`,
/// any `*.localhost` name, a `127.0.0.0/8` address, or IPv6 `::1` (bare or
/// bracketed), each with or without a port.
fn is_loopback_host(rest: &str) -> bool {
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    if authority == "::1" || authority.starts_with("[::1]") {
        return true;
    }
    let host = authority
        .rsplit_once(':')
        .map_or(authority, |(host, _port)| host);
    if let Ok(ip) = host.parse::<std::net::Ipv4Addr>() {
        return ip.is_loopback();
    }
    host.eq_ignore_ascii_case("localhost") || host.to_ascii_lowercase().ends_with(".localhost")
}

/// Resolve which named profile to act on.
///
/// Precedence: the `--profile` flag (into which clap already folds the
/// `RLMESH_PROFILE` environment variable), then `default_profile` from the
/// config file, then the literal `"default"`.
fn resolve_profile_name(flag: Option<&str>, config: &Config) -> String {
    if let Some(name) = flag {
        return name.to_string();
    }
    if let Some(name) = &config.default_profile {
        return name.clone();
    }
    "default".to_string()
}

/// On-disk CLI configuration, stored as TOML at `~/.config/rlmesh/config.toml`.
#[derive(Serialize, Deserialize, Default)]
struct Config {
    #[serde(default)]
    default_profile: Option<String>,
    #[serde(default)]
    profiles: BTreeMap<String, Profile>,
}

/// A single named profile: the platform it signs in to. Its credential lives in
/// the keychain or file store keyed by the profile name, not here.
#[derive(Serialize, Deserialize, Default, Clone)]
struct Profile {
    #[serde(default)]
    platform_url: Option<String>,
}

/// Directory holding the CLI's config and file-mode credential store.
fn config_dir() -> Result<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME")
        && !xdg.is_empty()
    {
        return Ok(PathBuf::from(xdg).join("rlmesh"));
    }
    if let Some(home) = std::env::home_dir() {
        return Ok(home.join(".config/rlmesh"));
    }
    bail!("cannot locate a config directory; set XDG_CONFIG_HOME or HOME")
}

fn config_path() -> Result<PathBuf> {
    Ok(config_dir()?.join("config.toml"))
}

fn load_config() -> Result<Config> {
    load_config_from(&config_path()?)
}

/// Read the config file, returning defaults if it does not exist yet.
fn load_config_from(path: &Path) -> Result<Config> {
    match fs::read_to_string(path) {
        Ok(text) => {
            toml::from_str(&text).with_context(|| format!("parsing config file {}", path.display()))
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Config::default()),
        Err(err) => Err(err).with_context(|| format!("reading config file {}", path.display())),
    }
}

/// Persist the full config back to `config.toml`.
fn save_config(config: &Config) -> Result<()> {
    let dir = config_dir()?;
    ensure_config_dir(&dir)?;
    let path = dir.join("config.toml");
    let text = toml::to_string_pretty(config).context("serializing config")?;
    let tmp = path.with_extension("toml.tmp");
    fs::write(&tmp, text).with_context(|| format!("writing config file {}", tmp.display()))?;
    fs::rename(&tmp, &path).with_context(|| format!("writing config file {}", path.display()))
}

/// A platform API key and its identifier, as minted by `POST /v1/api-keys`.
#[derive(Serialize, Deserialize, Clone)]
struct Credentials {
    api_key_id: String,
    api_key: String,
}

/// Persist a credential under `profile_name`, preferring the OS keychain and
/// falling back to a mode-0600 file if no keychain is usable.
///
/// The two stores are kept mutually exclusive: whichever one is written, the
/// other is cleared. [`load_credentials`] always prefers the keychain, so a
/// stale keychain entry left behind after a file-store fallback would otherwise
/// shadow the freshly written credential.
///
/// Returns a human-readable description of where it landed, for display.
fn save_credentials(profile_name: &str, creds: &Credentials) -> Result<String> {
    let payload = serde_json::to_string(creds).context("serializing credentials")?;
    match keyring::Entry::new("rlmesh", profile_name).and_then(|entry| entry.set_password(&payload))
    {
        Ok(()) => {
            if let Ok(dir) = config_dir() {
                let _ = delete_file_credential(&dir, profile_name);
            }
            Ok("the OS keychain".to_string())
        }
        Err(_) => {
            let dir = config_dir()?;
            if let Ok(entry) = keyring::Entry::new("rlmesh", profile_name) {
                let _ = entry.delete_credential();
            }
            save_file_credentials(&dir, profile_name, creds)?;
            let path = dir.join("credentials.json");
            Ok(format!(
                "{} (no usable OS keychain; file mode 0600)",
                path.display()
            ))
        }
    }
}

/// Look up the stored credential for `profile_name`, checking the keychain first
/// and falling back to the file store on any keychain failure.
fn load_credentials(profile_name: &str) -> Result<Option<Credentials>> {
    if let Ok(entry) = keyring::Entry::new("rlmesh", profile_name)
        && let Ok(payload) = entry.get_password()
    {
        let creds =
            serde_json::from_str(&payload).context("parsing stored credentials from keychain")?;
        return Ok(Some(creds));
    }
    let map = read_file_credentials(&config_dir()?)?;
    Ok(map.get(profile_name).cloned())
}

/// Remove any stored credential for `profile_name` from both the keychain and
/// the file store. Returns whether anything was actually removed.
///
/// A keychain delete that fails for any reason other than "no such entry" (a
/// locked or permission-denied keychain that still holds the secret) is
/// surfaced as an error rather than silently swallowed, so `logout` cannot
/// report success while the credential remains behind. The file store is always
/// cleaned regardless.
fn delete_credentials(profile_name: &str) -> Result<bool> {
    let mut removed = false;
    let keychain_err = match keyring::Entry::new("rlmesh", profile_name) {
        Ok(entry) => match entry.delete_credential() {
            Ok(()) => {
                removed = true;
                None
            }
            Err(keyring::Error::NoEntry) => None,
            Err(err) => Some(err),
        },
        Err(_) => None,
    };
    if delete_file_credential(&config_dir()?, profile_name)? {
        removed = true;
    }
    if let Some(err) = keychain_err {
        return Err(anyhow!(err)).with_context(|| {
            format!("removing the keychain credential for profile \"{profile_name}\"")
        });
    }
    Ok(removed)
}

fn credentials_path(dir: &Path) -> PathBuf {
    dir.join("credentials.json")
}

/// Read the file-mode credential map, returning an empty map if absent.
fn read_file_credentials(dir: &Path) -> Result<BTreeMap<String, Credentials>> {
    let path = credentials_path(dir);
    match fs::read(&path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing credential file {}", path.display())),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(err) => Err(err).with_context(|| format!("reading credential file {}", path.display())),
    }
}

/// Write a credential into the file-mode store, keyed by profile name.
fn save_file_credentials(dir: &Path, profile_name: &str, creds: &Credentials) -> Result<()> {
    let mut map = read_file_credentials(dir)?;
    map.insert(profile_name.to_string(), creds.clone());
    write_file_credentials(dir, &map)
}

/// Drop a credential from the file store, rewriting the file if it changed.
fn delete_file_credential(dir: &Path, profile_name: &str) -> Result<bool> {
    let mut map = read_file_credentials(dir)?;
    if map.remove(profile_name).is_some() {
        write_file_credentials(dir, &map)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Serialize the credential map to `credentials.json` with owner-only permissions.
fn write_file_credentials(dir: &Path, map: &BTreeMap<String, Credentials>) -> Result<()> {
    ensure_config_dir(dir)?;
    let path = credentials_path(dir);
    let bytes = serde_json::to_vec_pretty(map).context("serializing credentials")?;
    write_private_file(&path, &bytes)
        .with_context(|| format!("writing credential file {}", path.display()))
}

/// Create the config directory, restricting it to the owner on unix.
fn ensure_config_dir(dir: &Path) -> Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(dir)
        .with_context(|| format!("creating config directory {}", dir.display()))
}

/// Write bytes to a file, forcing mode 0600 on unix.
fn write_private_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    Ok(())
}

/// Build an HTTP client with a bounded timeout. The separate connect timeout
/// keeps commands that merely *verify* state (whoami) from hanging half a
/// minute on an unreachable host.
fn http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .connect_timeout(Duration::from_secs(10))
        .build()
        .context("building HTTP client")
}

/// The transport-level failures (no HTTP response at all) that have a
/// specific, actionable remedy worth naming.
#[derive(Debug, PartialEq)]
enum TransportProblem {
    /// The server answered our TLS handshake with plain HTTP — an https URL
    /// pointed at an http server.
    PlaintextServer,
    /// TLS connected but the certificate did not verify.
    BadCertificate,
    /// Nothing is listening on that host/port.
    ConnectionRefused,
    /// The hostname did not resolve.
    UnknownHost,
}

/// Classify the rendered error chain of a failed request. reqwest exposes no
/// structured cause for these, so this matches the stable substrings that
/// rustls, hyper, and the OS put in the chain.
fn classify_transport_error(chain: &str) -> Option<TransportProblem> {
    let chain = chain.to_ascii_lowercase();
    if chain.contains("corrupt message") || chain.contains("invalidcontenttype") {
        Some(TransportProblem::PlaintextServer)
    } else if chain.contains("certificate") {
        Some(TransportProblem::BadCertificate)
    } else if chain.contains("connection refused") {
        Some(TransportProblem::ConnectionRefused)
    } else if chain.contains("dns error") || chain.contains("failed to lookup") {
        Some(TransportProblem::UnknownHost)
    } else {
        None
    }
}

/// Turn a transport-level failure into a diagnostic that names the failing URL
/// and, for the classifiable cases, the likely remedy — reqwest's own
/// rendering ("received corrupt message of type InvalidContentType") diagnoses
/// nothing for the operator.
fn describe_transport_error(err: reqwest::Error, url: &str, what: &str) -> anyhow::Error {
    if err.is_timeout() {
        return anyhow!("{what}: {url} did not respond in time");
    }
    let chain = format!("{:#}", anyhow::Error::from(err.without_url()));
    match classify_transport_error(&chain) {
        Some(TransportProblem::PlaintextServer) => anyhow!(
            "{what}: {url} speaks plain HTTP, not TLS. If it is a local dev server, use an explicit http:// URL, e.g. {}",
            url.replacen("https://", "http://", 1)
        ),
        Some(TransportProblem::BadCertificate) => {
            anyhow!("{what}: could not verify the TLS certificate of {url} ({chain})")
        }
        Some(TransportProblem::ConnectionRefused) => {
            anyhow!("{what}: nothing is listening at {url} (connection refused)")
        }
        Some(TransportProblem::UnknownHost) => {
            anyhow!("{what}: cannot resolve the host in {url}")
        }
        None => anyhow!("{what}: could not reach {url}: {chain}"),
    }
}

/// Issue a GET expecting a JSON body, optionally with a bearer token.
async fn get_json<T: DeserializeOwned>(
    client: &reqwest::Client,
    url: &str,
    bearer: Option<&str>,
    what: &str,
) -> Result<T> {
    let mut request = client.get(url);
    if let Some(token) = bearer {
        request = request.bearer_auth(token);
    }
    let resp = request
        .send()
        .await
        .map_err(|err| describe_transport_error(err, url, what))?;
    expect_json(resp, what).await
}

/// Turn a response into a decoded JSON value, mapping non-success statuses
/// through [`http_failure_message`].
async fn expect_json<T: DeserializeOwned>(resp: reqwest::Response, what: &str) -> Result<T> {
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        bail!(http_failure_message(status, &body, what));
    }
    resp.json::<T>()
        .await
        .with_context(|| format!("parsing {what} response"))
}

/// Render a non-success HTTP response as a one-line diagnostic: the server's
/// own `error.message` when the body carries the platform error shape,
/// otherwise the truncated raw body.
fn http_failure_message(status: reqwest::StatusCode, body: &str, what: &str) -> String {
    match parse_error_detail(body) {
        Some(message) => format!("{what} failed: {message} (HTTP {status})"),
        None => {
            let truncated: String = body.chars().take(300).collect();
            format!("{what} failed with HTTP {status}: {truncated}")
        }
    }
}

/// Public `cli-config` response describing the AuthKit tenant to use.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CliAuthConfig {
    authkit_domain: String,
    client_id: String,
}

/// Device-authorization grant returned by AuthKit's device flow.
#[derive(Deserialize)]
struct DeviceAuthorization {
    device_code: String,
    user_code: String,
    verification_uri: String,
    #[serde(default)]
    verification_uri_complete: Option<String>,
    expires_in: u64,
    #[serde(default)]
    interval: Option<u64>,
}

/// Successful token-endpoint response carrying the AuthKit access token.
#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
}

/// Token-endpoint denial, whose `error` code drives the polling loop.
#[derive(Deserialize, Default)]
struct TokenDenial {
    #[serde(default)]
    error: String,
}

/// Request body for `POST /v1/api-keys`; the server binds strictly, so this
/// must carry exactly the `name` field and nothing else.
#[derive(Serialize)]
struct CreateApiKeyRequest<'a> {
    name: &'a str,
}

/// API key minted by `POST /v1/api-keys`. The full response also carries
/// `name`, `createdAt`, and other fields the CLI does not need; `value` (the
/// secret) is only present on create.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreatedApiKey {
    id: String,
    #[serde(default)]
    value: Option<String>,
}

/// Identity echo from `GET /v1/me`; only the fields the CLI prints are modeled
/// (the server also returns permissions, platform role, provider, and more).
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MeResponse {
    organization_id: String,
    subject_type: String,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    user_id: Option<String>,
}

/// Registry coordinates returned by `registry/info`; `namespaces` lists only
/// the push-capable namespaces.
#[derive(Deserialize)]
struct RegistryInfo {
    host: String,
    #[serde(default)]
    namespaces: Vec<String>,
}

/// Sign in to a platform via the OAuth device flow and store the issued API key
/// under the selected profile.
///
/// The platform URL is resolved by precedence: `--platform` (into which clap
/// folds `RLMESH_PLATFORM_URL`), then the profile's remembered platform, then
/// [`DEFAULT_PLATFORM_URL`] — so `rlmesh login` with no flags signs in to the
/// hosted platform, while `--platform` still points a profile at any other one.
pub async fn login(args: &LoginArgs, stdout: &mut impl Write) -> Result<i32> {
    let mut config = load_config()?;
    let name = resolve_profile_name(args.profile.profile.as_deref(), &config);
    let platform = match &args.platform {
        Some(raw) => normalize_base_url(raw),
        None => match config
            .profiles
            .get(&name)
            .and_then(|p| p.platform_url.clone())
        {
            Some(url) => normalize_base_url(&url),
            None => DEFAULT_PLATFORM_URL.to_string(),
        },
    };
    let style = Style::stdout();
    let client = http_client()?;

    writeln!(
        stdout,
        "Signing in to {} (profile \"{name}\")",
        style.bold(&platform)
    )?;
    stdout.flush()?;

    let config_url = format!("{platform}/v1/auth/cli-config");
    let config_resp = client.get(&config_url).send().await.map_err(|err| {
        describe_transport_error(err, &config_url, "fetching sign-in configuration")
    })?;
    if config_resp.status() == reqwest::StatusCode::NOT_FOUND {
        bail!(
            "{platform} does not offer CLI sign-in (GET /v1/auth/cli-config returned 404); check the platform URL"
        );
    }
    let auth_config: CliAuthConfig =
        expect_json(config_resp, "fetching sign-in configuration").await?;
    let authkit = normalize_base_url(&auth_config.authkit_domain);

    let device_auth_url = format!("{authkit}/oauth2/device_authorization");
    let grant: DeviceAuthorization = expect_json(
        client
            .post(&device_auth_url)
            .form(&[("client_id", auth_config.client_id.as_str())])
            .send()
            .await
            .map_err(|err| {
                describe_transport_error(err, &device_auth_url, "requesting a device authorization")
            })?,
        "requesting a device authorization",
    )
    .await?;

    let open_url = grant
        .verification_uri_complete
        .as_deref()
        .unwrap_or(&grant.verification_uri);
    let opened = style.interactive() && try_open_browser(open_url);
    writeln!(stdout)?;
    if opened {
        writeln!(
            stdout,
            "Opening your browser to approve this sign-in; if nothing appears, visit:"
        )?;
    } else {
        writeln!(stdout, "To approve this sign-in, visit:")?;
    }
    writeln!(stdout)?;
    writeln!(stdout, "    {}", style.cyan(open_url))?;
    writeln!(stdout)?;
    if grant.verification_uri_complete.is_some() {
        writeln!(
            stdout,
            "and confirm that the page shows code {}.",
            style.bold(&grant.user_code)
        )?;
    } else {
        writeln!(
            stdout,
            "and enter the code {}.",
            style.bold(&grant.user_code)
        )?;
    }
    writeln!(stdout)?;
    let minutes = grant.expires_in.div_ceil(60);
    write!(
        stdout,
        "Waiting for approval (the code is valid for {minutes} minutes)"
    )?;
    stdout.flush()?;

    let poll_result = poll_for_token(
        &client,
        &authkit,
        &auth_config.client_id,
        &grant,
        stdout,
        style,
    )
    .await;
    writeln!(stdout)?;
    let access_token = poll_result?;

    let created = create_api_key(&client, &platform, &access_token).await?;
    let Some(value) = created.value else {
        bail!("the platform did not return the API key secret");
    };

    config
        .profiles
        .entry(name.clone())
        .or_default()
        .platform_url = Some(platform.clone());
    if config.default_profile.is_none() {
        config.default_profile = Some(name.clone());
    }
    save_config(&config)?;

    let creds = Credentials {
        api_key_id: created.id,
        api_key: value,
    };
    let location = save_credentials(&name, &creds)?;

    writeln!(
        stdout,
        "{} Signed in to {} (profile \"{name}\").",
        style.green("✓"),
        style.bold(&platform)
    )?;
    writeln!(stdout, "  API key stored in {location}.")?;
    writeln!(
        stdout,
        "  Next: run {} to authenticate docker with the platform registry.",
        style.bold("rlmesh registry login")
    )?;
    Ok(0)
}

/// Poll AuthKit's token endpoint until the device grant is approved, denied, or
/// expires, returning the access token on approval.
async fn poll_for_token(
    client: &reqwest::Client,
    authkit: &str,
    client_id: &str,
    grant: &DeviceAuthorization,
    stdout: &mut impl Write,
    style: Style,
) -> Result<String> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(grant.expires_in);
    let mut interval = Duration::from_secs(grant.interval.unwrap_or(5).max(1));

    loop {
        if tokio::time::Instant::now() + interval >= deadline {
            bail!("the sign-in code expired before approval; run `rlmesh login` again");
        }
        tokio::time::sleep(interval).await;
        if style.interactive() {
            let _ = write!(stdout, ".");
            let _ = stdout.flush();
        }

        let resp = client
            .post(format!("{authkit}/oauth2/token"))
            .form(&[
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ("device_code", grant.device_code.as_str()),
                ("client_id", client_id),
            ])
            .send()
            .await
            .context("polling the token endpoint")?;

        if resp.status().is_success() {
            let token: TokenResponse = resp
                .json()
                .await
                .context("parsing the token endpoint response")?;
            return Ok(token.access_token);
        }

        let denial: TokenDenial = resp.json().await.unwrap_or_default();
        match denial.error.as_str() {
            "slow_down" => interval += Duration::from_secs(5),
            "access_denied" => bail!("sign-in was declined"),
            "expired_token" => {
                bail!("the sign-in code expired before approval; run `rlmesh login` again")
            }
            "authorization_pending" | "" => {}
            other => bail!("the token endpoint rejected the sign-in: {other}"),
        }
    }
}

/// Best-effort launch of the platform-default browser at `url`, returning
/// whether the opener command was spawned successfully. Never blocks on the
/// browser and never fails the sign-in: a headless or locked-down host just
/// falls back to the printed URL. The URL comes from the platform's own
/// AuthKit response, not user input, so it is not shell-injected — and it is
/// passed as an argv element regardless.
fn try_open_browser(url: &str) -> bool {
    #[cfg(target_os = "macos")]
    let (program, args): (&str, &[&str]) = ("open", &[]);
    #[cfg(target_os = "windows")]
    let (program, args): (&str, &[&str]) = ("cmd", &["/C", "start", ""]);
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let (program, args): (&str, &[&str]) = ("xdg-open", &[]);

    Command::new(program)
        .args(args)
        .arg(url)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .stdin(Stdio::null())
        .spawn()
        .is_ok()
}

/// Name for the minted API key: `rlmesh-cli <hostname>`, or plain `rlmesh-cli`
/// when the hostname cannot be determined.
fn api_key_name() -> String {
    let hostname = Command::new("hostname")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|raw| raw.trim().to_string())
        .filter(|name| !name.is_empty());
    match hostname {
        Some(host) => format!("rlmesh-cli {host}"),
        None => "rlmesh-cli".to_string(),
    }
}

/// Mint the CLI's durable credential via `POST /v1/api-keys`, authorized with
/// the device-flow access token. On HTTP 403 the platform's own reason is
/// surfaced verbatim (the causes — organization not selected, organization not
/// linked to the platform, or a non-user session — carry distinct messages and
/// distinct remedies), followed by the two next steps that cover them.
async fn create_api_key(
    client: &reqwest::Client,
    platform: &str,
    access_token: &str,
) -> Result<CreatedApiKey> {
    let key_name = api_key_name();
    let resp = client
        .post(format!("{platform}/v1/api-keys"))
        .bearer_auth(access_token)
        .json(&CreateApiKeyRequest { name: &key_name })
        .send()
        .await
        .context("creating a platform API key")?;
    if resp.status() == reqwest::StatusCode::FORBIDDEN {
        let reason = error_detail(resp)
            .await
            .unwrap_or_else(|| "no organization is available for API key creation".to_string());
        bail!(
            "the platform rejected API key creation: {reason}. Run `rlmesh login` again to sign in to a different organization, or ask an administrator to link your organization to the platform."
        );
    }
    expect_json(resp, "creating a platform API key").await
}

/// Best-effort extraction of the human-readable reason from the platform's
/// `{"error": {"code": "...", "message": "..."}}` error body, so a failed call
/// can surface the server's own message instead of a generic guess. Returns
/// `None` when the body is missing, unparsable, or carries an empty message.
async fn error_detail(resp: reqwest::Response) -> Option<String> {
    parse_error_detail(&resp.text().await.ok()?)
}

/// Parse the `message` out of the platform's `{"error": {..., "message"}}` body.
fn parse_error_detail(text: &str) -> Option<String> {
    #[derive(Deserialize)]
    struct Body {
        error: Detail,
    }
    #[derive(Deserialize)]
    struct Detail {
        message: String,
    }
    serde_json::from_str::<Body>(text)
        .ok()
        .map(|body| body.error.message)
        .filter(|message| !message.is_empty())
}

/// Delete the stored credential for a profile, leaving its config entry in place.
pub fn logout(args: &ProfileArgs, stdout: &mut impl Write) -> Result<i32> {
    let config = load_config()?;
    let name = resolve_profile_name(args.profile.as_deref(), &config);
    if delete_credentials(&name)? {
        writeln!(stdout, "Signed out of profile \"{name}\".")?;
    } else {
        writeln!(stdout, "No stored credential for profile \"{name}\".")?;
    }
    Ok(0)
}

/// Render the `whoami` report: aligned key-value lines describing the resolved
/// profile, its platform, and sign-in state. Never sees the API key secret —
/// only the key id (the public docker username).
fn render_whoami(
    name: &str,
    is_default: bool,
    platform_url: Option<&str>,
    api_key_id: Option<&str>,
    login_cmd: &str,
) -> String {
    let profile = if is_default {
        format!("{name} (default)")
    } else {
        name.to_string()
    };
    let platform = match platform_url {
        Some(url) => url.to_string(),
        None => format!("(none; run `{login_cmd}`)"),
    };
    let api_key = match api_key_id {
        Some(id) => format!("{id} (signed in)"),
        None => "(signed out; run `rlmesh login`)".to_string(),
    };
    format!(
        "{:<9}  {profile}\n{:<9}  {platform}\n{:<9}  {api_key}\n",
        "profile:", "platform:", "api key:"
    )
}

/// Render the `/v1/me` verification lines appended to the `whoami` report: the
/// organization (annotated with the user id or subject type) and role on
/// success, or a single `(unverified: ...)` line on failure. Never sees the
/// API key.
fn render_identity(me: Option<&MeResponse>, err: Option<&str>) -> String {
    if let Some(me) = me {
        let subject = match &me.user_id {
            Some(user) => format!("user {user}"),
            None => me.subject_type.clone(),
        };
        let mut out = format!("{:<9}  {} ({subject})\n", "org:", me.organization_id);
        if let Some(role) = &me.role {
            out.push_str(&format!("{:<9}  {role}\n", "role:"));
        }
        return out;
    }
    match err {
        Some(reason) => format!("{:<9}  (unverified: {reason})\n", "org:"),
        None => String::new(),
    }
}

/// Fetch the caller's identity from `GET /v1/me`, mapping every failure into a
/// short display reason (never the key itself).
async fn fetch_identity(platform: &str, api_key: &str) -> Result<MeResponse, String> {
    let client = http_client().map_err(|err| err.to_string())?;
    let resp = client
        .get(format!("{platform}/v1/me"))
        .bearer_auth(api_key)
        .send()
        .await
        .map_err(|err| err.without_url().to_string())?;
    let status = resp.status();
    if !status.is_success() {
        return Err(format!("HTTP {status}"));
    }
    resp.json::<MeResponse>()
        .await
        .map_err(|_| "unparsable identity response".to_string())
}

/// Show the active profile, its platform, and sign-in state. The three profile
/// lines are local-only; when a credential exists its identity is additionally
/// verified against the platform's `/v1/me`, with any failure reported inline.
/// Always exits 0: whoami reports state.
pub async fn whoami(args: &ProfileArgs, stdout: &mut impl Write) -> Result<i32> {
    let config = load_config()?;
    let name = resolve_profile_name(args.profile.as_deref(), &config);
    let is_default = config.default_profile.as_deref() == Some(name.as_str());
    let platform_url = config
        .profiles
        .get(&name)
        .and_then(|profile| profile.platform_url.as_deref());
    let creds = load_credentials(&name)?;
    let hint = login_hint(&name, is_effective_default(&config, &name));
    let report = render_whoami(
        &name,
        is_default,
        platform_url,
        creds.as_ref().map(|creds| creds.api_key_id.as_str()),
        &hint,
    );
    write!(stdout, "{report}")?;

    if let Some(creds) = &creds {
        let identity = match platform_url {
            Some(url) => fetch_identity(&normalize_base_url(url), &creds.api_key).await,
            None => Err("no platform configured for this profile".to_string()),
        };
        let lines = match &identity {
            Ok(me) => render_identity(Some(me), None),
            Err(reason) => render_identity(None, Some(reason)),
        };
        write!(stdout, "{lines}")?;
    }
    Ok(0)
}

/// Whether `name` is the profile a flagless `rlmesh login` resolves to: the
/// configured default, or the literal `"default"` when none is configured yet.
/// This is what makes `rlmesh login` (no flags) sign the profile in, so it also
/// governs whether the suggested hint can drop `--profile`/`--platform`.
fn is_effective_default(config: &Config, name: &str) -> bool {
    config.default_profile.as_deref().unwrap_or("default") == name
}

/// Format the `rlmesh login` command to suggest for `name`. The effective
/// default profile (see [`is_effective_default`]) signs in to the hosted
/// platform ([`DEFAULT_PLATFORM_URL`]) with no flags; any other named profile is
/// assumed to target its own platform, so its hint carries `--profile` and a
/// `--platform <url>` placeholder to point it there.
fn login_hint(name: &str, is_default: bool) -> String {
    if is_default {
        "rlmesh login".to_string()
    } else {
        format!("rlmesh login --profile {name} --platform <url>")
    }
}

/// Authenticate the local docker client with the platform's image registry,
/// using the selected profile's platform and stored credential.
pub async fn registry_login(args: &ProfileArgs, stdout: &mut impl Write) -> Result<i32> {
    let config = load_config()?;
    let name = resolve_profile_name(args.profile.as_deref(), &config);
    let is_default = is_effective_default(&config, &name);
    let platform = match config
        .profiles
        .get(&name)
        .and_then(|p| p.platform_url.clone())
    {
        Some(url) => normalize_base_url(&url),
        None => {
            bail!(
                "no profile \"{name}\" yet; run `{}` to sign in",
                login_hint(&name, is_default)
            )
        }
    };
    let creds = load_credentials(&name)?.ok_or_else(|| {
        anyhow!(
            "profile \"{name}\" is not signed in; run `{}`",
            login_hint(&name, is_default)
        )
    })?;

    let client = http_client()?;
    let info: RegistryInfo = get_json(
        &client,
        &format!("{platform}/v1/registry/info"),
        Some(&creds.api_key),
        "fetching registry info",
    )
    .await?;

    docker_login(&info.host, &creds)?;

    writeln!(stdout, "docker is now signed in to {}.", info.host)?;
    if !info.namespaces.is_empty() {
        let targets = info
            .namespaces
            .iter()
            .map(|ns| format!("{}/{ns}", info.host))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(stdout, "You can push to: {targets}")?;
    }
    Ok(0)
}

/// Render the profile table: one aligned line per profile (BTreeMap order),
/// marking the default with `*` and reporting each profile's sign-in state.
///
/// Assumes at least one profile; the empty case is handled by [`profile_list`].
fn render_profiles(config: &Config, is_signed_in: impl Fn(&str) -> bool) -> String {
    let name_width = config.profiles.keys().map(String::len).max().unwrap_or(0);
    let url_width = config
        .profiles
        .values()
        .map(|profile| profile.platform_url.as_deref().unwrap_or("").len())
        .max()
        .unwrap_or(0);

    let mut out = String::new();
    for (name, profile) in &config.profiles {
        let marker = if config.default_profile.as_deref() == Some(name.as_str()) {
            "* "
        } else {
            "  "
        };
        let url = profile.platform_url.as_deref().unwrap_or("");
        let state = if is_signed_in(name) {
            "signed in"
        } else {
            "signed out"
        };
        out.push_str(&format!(
            "{marker}{name:<name_width$}  {url:<url_width$}  {state}\n"
        ));
    }
    out
}

/// List configured profiles, marking the default and each profile's sign-in
/// state. A locked keychain is treated as signed out rather than failing.
pub fn profile_list(stdout: &mut impl Write) -> Result<i32> {
    let config = load_config()?;
    if config.profiles.is_empty() {
        writeln!(
            stdout,
            "No profiles yet; run `rlmesh login` to sign in to the hosted platform."
        )?;
        return Ok(0);
    }
    let file_creds = read_file_credentials(&config_dir()?).unwrap_or_default();
    let table = render_profiles(&config, |name| {
        keyring::Entry::new("rlmesh", name)
            .and_then(|entry| entry.get_password())
            .is_ok()
            || file_creds.contains_key(name)
    });
    write!(stdout, "{table}")?;
    Ok(0)
}

/// Set the default profile, requiring that it already exists.
pub fn profile_use(name: &str, stdout: &mut impl Write) -> Result<i32> {
    let mut config = load_config()?;
    if !config.profiles.contains_key(name) {
        bail!(
            "no profile named \"{name}\"; run `rlmesh login --profile {name} --platform <url>` to create it"
        );
    }
    config.default_profile = Some(name.to_string());
    save_config(&config)?;
    writeln!(stdout, "Default profile is now \"{name}\".")?;
    Ok(0)
}

/// Mutate `config` for profile removal, returning (existed, default_cleared).
///
/// Removes the named entry from `config.profiles`; if it existed and
/// `default_profile` pointed at it, clears `default_profile`.
fn drop_profile(config: &mut Config, name: &str) -> (bool, bool) {
    let existed = config.profiles.remove(name).is_some();
    let default_cleared = existed && config.default_profile.as_deref() == Some(name);
    if default_cleared {
        config.default_profile = None;
    }
    (existed, default_cleared)
}

/// Delete a profile: its stored credential and its config entry.
pub fn profile_remove(name: &str, stdout: &mut impl Write) -> Result<i32> {
    let mut config = load_config()?;
    let (existed, default_cleared) = drop_profile(&mut config, name);
    let creds_removed = delete_credentials(name)?;
    if !existed && !creds_removed {
        bail!("no profile named \"{name}\"");
    }
    if existed {
        save_config(&config)?;
    }
    writeln!(stdout, "Removed profile \"{name}\".")?;
    if default_cleared && !config.profiles.is_empty() {
        writeln!(
            stdout,
            "Default profile cleared; run `rlmesh profile use <name>` to pick a new one."
        )?;
    }
    Ok(0)
}

/// Run `docker login` for `host`, feeding the API key over stdin so it never
/// appears on a command line or in any error message.
fn docker_login(host: &str, creds: &Credentials) -> Result<()> {
    let mut child = Command::new("docker")
        .arg("login")
        .arg(host)
        .arg("--username")
        .arg(&creds.api_key_id)
        .arg("--password-stdin")
        .stdin(Stdio::piped())
        .spawn()
        .context("launching docker (is it installed and on PATH?)")?;

    {
        let mut stdin = child.stdin.take().context("capturing docker's stdin")?;
        stdin
            .write_all(creds.api_key.as_bytes())
            .context("passing the API key to docker")?;
    }

    let status = child.wait().context("waiting for docker login")?;
    if !status.success() {
        bail!("docker login failed with {status}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_platform_urls() {
        assert_eq!(
            normalize_base_url("platform.example.com"),
            "https://platform.example.com"
        );
        assert_eq!(
            normalize_base_url("https://platform.example.com/"),
            "https://platform.example.com"
        );
        assert_eq!(
            normalize_base_url("http://localhost:8080"),
            "http://localhost:8080"
        );
        assert_eq!(
            normalize_base_url("  platform.example.com/  "),
            "https://platform.example.com"
        );
    }

    #[test]
    fn normalize_defaults_loopback_to_http() {
        // A scheme-less loopback host gets http:// (local dev servers rarely
        // terminate TLS); everything else still gets https://.
        assert_eq!(
            normalize_base_url("localhost:3300"),
            "http://localhost:3300"
        );
        assert_eq!(
            normalize_base_url("127.0.0.1:8080"),
            "http://127.0.0.1:8080"
        );
        assert_eq!(normalize_base_url("[::1]:3300"), "http://[::1]:3300");
        assert_eq!(
            normalize_base_url("api.rlmesh.dev"),
            "https://api.rlmesh.dev"
        );
        // An explicit scheme is always honored, even for loopback.
        assert_eq!(
            normalize_base_url("https://localhost:3300"),
            "https://localhost:3300"
        );
    }

    #[test]
    fn loopback_host_detection() {
        for host in [
            "localhost",
            "localhost:3300",
            "LOCALHOST:3300",
            "foo.localhost",
            "127.0.0.1",
            "127.0.0.1:8080",
            "127.5.6.7",
            "::1",
            "[::1]:3300",
        ] {
            assert!(is_loopback_host(host), "{host} should be loopback");
        }
        for host in [
            "api.rlmesh.dev",
            "example.com:443",
            "10.0.0.1",
            "localhosting.example.com",
        ] {
            assert!(!is_loopback_host(host), "{host} should not be loopback");
        }
    }

    #[test]
    fn transport_errors_are_classified() {
        assert_eq!(
            classify_transport_error(
                "error sending request: received corrupt message of type InvalidContentType"
            ),
            Some(TransportProblem::PlaintextServer)
        );
        assert_eq!(
            classify_transport_error("invalid peer certificate: UnknownIssuer"),
            Some(TransportProblem::BadCertificate)
        );
        assert_eq!(
            classify_transport_error("tcp connect error: Connection refused (os error 111)"),
            Some(TransportProblem::ConnectionRefused)
        );
        assert_eq!(
            classify_transport_error("dns error: failed to lookup address information"),
            Some(TransportProblem::UnknownHost)
        );
        assert_eq!(classify_transport_error("some other failure"), None);
    }

    #[test]
    fn http_failure_prefers_server_message() {
        let with_detail = http_failure_message(
            reqwest::StatusCode::UNAUTHORIZED,
            r#"{"error": {"code": "unauthorized", "message": "missing bearer token"}}"#,
            "fetching sign-in configuration",
        );
        assert_eq!(
            with_detail,
            "fetching sign-in configuration failed: missing bearer token (HTTP 401 Unauthorized)"
        );

        let plain = http_failure_message(
            reqwest::StatusCode::BAD_GATEWAY,
            "upstream is down",
            "fetching registry info",
        );
        assert_eq!(
            plain,
            "fetching registry info failed with HTTP 502 Bad Gateway: upstream is down"
        );
    }

    #[test]
    fn resolve_profile_name_precedence() {
        let mut config = Config::default();
        assert_eq!(resolve_profile_name(None, &config), "default");

        config.default_profile = Some("staging".to_string());
        assert_eq!(resolve_profile_name(None, &config), "staging");
        assert_eq!(resolve_profile_name(Some("prod"), &config), "prod");
    }

    #[test]
    fn load_config_from_missing_file_is_default() {
        let dir = tempfile::tempdir().unwrap();
        let config = load_config_from(&dir.path().join("does-not-exist.toml")).unwrap();
        assert!(config.default_profile.is_none());
        assert!(config.profiles.is_empty());
    }

    #[test]
    fn config_toml_round_trip() {
        let config: Config = toml::from_str(
            "default_profile = \"staging\"\n\n[profiles.staging]\nplatform_url = \"https://staging.example.com\"\n",
        )
        .unwrap();
        assert_eq!(config.default_profile.as_deref(), Some("staging"));
        assert_eq!(
            config.profiles["staging"].platform_url.as_deref(),
            Some("https://staging.example.com")
        );

        let empty: Config = toml::from_str("").unwrap();
        assert!(empty.default_profile.is_none());
        assert!(empty.profiles.is_empty());

        let text = toml::to_string_pretty(&config).unwrap();
        let reparsed: Config = toml::from_str(&text).unwrap();
        assert_eq!(reparsed.default_profile.as_deref(), Some("staging"));
        assert_eq!(
            reparsed.profiles["staging"].platform_url.as_deref(),
            Some("https://staging.example.com")
        );
    }

    #[test]
    fn login_hint_omits_flags_for_default_profile() {
        assert_eq!(login_hint("default", true), "rlmesh login");
        assert_eq!(
            login_hint("staging", false),
            "rlmesh login --profile staging --platform <url>"
        );
    }

    #[test]
    fn effective_default_covers_the_implicit_default() {
        // No configured default: only the literal "default" is effective.
        let empty = Config::default();
        assert!(is_effective_default(&empty, "default"));
        assert!(!is_effective_default(&empty, "staging"));

        // A configured default wins, and "default" is no longer special.
        let config = Config {
            default_profile: Some("staging".to_string()),
            profiles: BTreeMap::new(),
        };
        assert!(is_effective_default(&config, "staging"));
        assert!(!is_effective_default(&config, "default"));
    }

    #[test]
    fn default_platform_url_is_normalized() {
        assert_eq!(
            normalize_base_url(DEFAULT_PLATFORM_URL),
            DEFAULT_PLATFORM_URL
        );
    }

    #[test]
    fn render_profiles_marks_default_and_state() {
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "staging".to_string(),
            Profile {
                platform_url: Some("https://staging.example.com".to_string()),
            },
        );
        profiles.insert(
            "prod".to_string(),
            Profile {
                platform_url: Some("https://prod.example.com".to_string()),
            },
        );
        let config = Config {
            default_profile: Some("staging".to_string()),
            profiles,
        };

        let rendered = render_profiles(&config, |name| name == "staging");
        let lines: Vec<&str> = rendered.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("  prod"));
        assert!(lines[0].contains("signed out"));
        assert!(lines[1].starts_with("* staging"));
        assert!(lines[1].contains("signed in"));
    }

    fn config_with_profile(name: &str, default: Option<&str>) -> Config {
        let mut profiles = BTreeMap::new();
        profiles.insert(name.to_string(), Profile::default());
        Config {
            default_profile: default.map(str::to_string),
            profiles,
        }
    }

    #[test]
    fn drop_profile_removes_and_clears_default() {
        let mut config = config_with_profile("staging", Some("staging"));
        assert_eq!(drop_profile(&mut config, "staging"), (true, true));
        assert!(config.profiles.is_empty());
        assert!(config.default_profile.is_none());

        let mut config = config_with_profile("staging", Some("prod"));
        assert_eq!(drop_profile(&mut config, "staging"), (true, false));
        assert_eq!(config.default_profile.as_deref(), Some("prod"));

        let mut config = config_with_profile("staging", Some("staging"));
        assert_eq!(drop_profile(&mut config, "missing"), (false, false));
        assert!(config.profiles.contains_key("staging"));
        assert_eq!(config.default_profile.as_deref(), Some("staging"));
    }

    #[test]
    fn render_whoami_reports_state_without_secrets() {
        let signed_in = render_whoami(
            "staging",
            true,
            Some("https://staging.example.com"),
            Some("key_123"),
            "rlmesh login",
        );
        assert_eq!(
            signed_in,
            "profile:   staging (default)\nplatform:  https://staging.example.com\napi key:   key_123 (signed in)\n"
        );

        let signed_out = render_whoami(
            "scratch",
            false,
            None,
            None,
            "rlmesh login --profile scratch --platform <url>",
        );
        assert_eq!(
            signed_out,
            "profile:   scratch\nplatform:  (none; run `rlmesh login --profile scratch --platform <url>`)\napi key:   (signed out; run `rlmesh login`)\n"
        );

        // The literal "default" profile with no configured default still gets
        // the flagless hint, since `rlmesh login` alone would sign it in.
        let fresh_default = render_whoami("default", false, None, None, "rlmesh login");
        assert!(fresh_default.contains("(none; run `rlmesh login`)"));

        let creds = Credentials {
            api_key_id: "key_123".to_string(),
            api_key: "sentinel-secret-value".to_string(),
        };
        let rendered = render_whoami(
            "staging",
            true,
            Some("https://staging.example.com"),
            Some(creds.api_key_id.as_str()),
            "rlmesh login",
        );
        assert!(!rendered.contains("sentinel-secret-value"));
    }

    #[test]
    fn file_credentials_round_trip_and_delete() {
        let dir = tempfile::tempdir().unwrap();
        let profile = "staging";
        let creds = Credentials {
            api_key_id: "key_123".to_string(),
            api_key: "secret-value".to_string(),
        };

        save_file_credentials(dir.path(), profile, &creds).unwrap();

        let path = credentials_path(dir.path());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }

        let map = read_file_credentials(dir.path()).unwrap();
        let stored = map.get(profile).unwrap();
        assert_eq!(stored.api_key_id, "key_123");
        assert_eq!(stored.api_key, "secret-value");

        assert!(delete_file_credential(dir.path(), profile).unwrap());
        assert!(!delete_file_credential(dir.path(), profile).unwrap());
        let map = read_file_credentials(dir.path()).unwrap();
        assert!(!map.contains_key(profile));
    }

    #[test]
    fn wire_shapes_match_the_documented_contract() {
        let config: CliAuthConfig = serde_json::from_str(
            r#"{"authkitDomain": "https://auth.example.com", "clientId": "client_abc"}"#,
        )
        .unwrap();
        assert_eq!(config.authkit_domain, "https://auth.example.com");
        assert_eq!(config.client_id, "client_abc");

        let grant: DeviceAuthorization = serde_json::from_str(
            r#"{"device_code": "dc", "user_code": "WXYZ-1234", "verification_uri": "https://auth.example.com/device", "expires_in": 300, "interval": 5}"#,
        )
        .unwrap();
        assert_eq!(grant.user_code, "WXYZ-1234");
        assert_eq!(grant.expires_in, 300);
        assert_eq!(grant.interval, Some(5));
        assert!(grant.verification_uri_complete.is_none());

        let created: CreatedApiKey = serde_json::from_str(
            r#"{"id": "key_01", "name": "rlmesh-cli host", "value": "sk", "createdAt": "2026-07-07T00:00:00Z"}"#,
        )
        .unwrap();
        assert_eq!(created.id, "key_01");
        assert_eq!(created.value.as_deref(), Some("sk"));

        let listed: CreatedApiKey = serde_json::from_str(
            r#"{"id": "key_02", "name": "rlmesh-cli host", "createdAt": "2026-07-07T00:00:00Z"}"#,
        )
        .unwrap();
        assert_eq!(listed.id, "key_02");
        assert!(listed.value.is_none());

        assert_eq!(
            serde_json::to_string(&CreateApiKeyRequest {
                name: "rlmesh-cli host"
            })
            .unwrap(),
            r#"{"name":"rlmesh-cli host"}"#
        );

        let me: MeResponse = serde_json::from_str(
            r#"{"userId": "user_01", "organizationId": "org_01", "subjectType": "user", "permissions": ["a"], "role": "admin", "provider": "workos", "demoOrganization": false, "demoPublisher": false}"#,
        )
        .unwrap();
        assert_eq!(me.organization_id, "org_01");
        assert_eq!(me.subject_type, "user");
        assert_eq!(me.role.as_deref(), Some("admin"));
        assert_eq!(me.user_id.as_deref(), Some("user_01"));

        let info: RegistryInfo = serde_json::from_str(
            r#"{"host": "registry.example.com", "namespaces": ["acme", "beta"]}"#,
        )
        .unwrap();
        assert_eq!(info.host, "registry.example.com");
        assert_eq!(info.namespaces, vec!["acme", "beta"]);
    }

    #[test]
    fn parse_error_detail_pulls_server_message() {
        assert_eq!(
            parse_error_detail(
                r#"{"error": {"code": "forbidden", "message": "organization is not linked"}}"#
            )
            .as_deref(),
            Some("organization is not linked")
        );
        assert_eq!(
            parse_error_detail(
                r#"{"error": {"code": "forbidden", "message": "API keys require an organization"}}"#
            )
            .as_deref(),
            Some("API keys require an organization")
        );
        assert_eq!(parse_error_detail(r#"{"error": {"message": ""}}"#), None);
        assert_eq!(parse_error_detail("not json"), None);
        assert_eq!(parse_error_detail(r#"{"detail": "other shape"}"#), None);
    }

    #[test]
    fn render_identity_reports_org_role_and_failures() {
        let me = MeResponse {
            organization_id: "org_01ABC".to_string(),
            subject_type: "user".to_string(),
            role: Some("admin".to_string()),
            user_id: Some("user_01".to_string()),
        };
        assert_eq!(
            render_identity(Some(&me), None),
            "org:       org_01ABC (user user_01)\nrole:      admin\n"
        );

        let key_subject = MeResponse {
            organization_id: "org_01ABC".to_string(),
            subject_type: "api_key".to_string(),
            role: None,
            user_id: None,
        };
        assert_eq!(
            render_identity(Some(&key_subject), None),
            "org:       org_01ABC (api_key)\n"
        );

        assert_eq!(
            render_identity(None, Some("HTTP 401 Unauthorized")),
            "org:       (unverified: HTTP 401 Unauthorized)\n"
        );
        assert_eq!(render_identity(None, None), "");
    }
}
