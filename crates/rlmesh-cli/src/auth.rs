use crate::cli::{LoginArgs, OrgListArgs, ProfileArgs, WhoamiArgs};
use crate::config::{
    CredentialStatus, CredentialStorage, Credentials, Identity, ProfileStore, ResolvedProfile,
};
use crate::helpers::{
    expect_json, get_json, http_client, require_pinned_host, require_trusted_endpoint,
};
use crate::platform::Platform;
use crate::render::{Style, write_heading, write_key_value};
use crate::session::{Refresh, ensure_fresh_session, switch_organization};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

#[derive(Deserialize)]
struct PlatformInfo {
    auth: AuthConfig,
}

/// The platform /v1/info auth block, RFC 8414 vocabulary. Only the fields the
/// CLI drives the device flow with; everything else in the document is
/// ignored. The CLI never falls back to a built-in provider: a platform that
/// does not advertise its endpoints cannot be signed in to.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AuthConfig {
    cli: AppAuth,
    #[serde(default)]
    device_authorization_endpoint: Option<String>,
    #[serde(default)]
    token_endpoint: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AppAuth {
    client_id: String,
}

/// The advertised, validated endpoints the device and refresh grants go to.
#[derive(Debug)]
pub(crate) struct SignInEndpoints {
    pub client_id: String,
    pub device_authorization_endpoint: String,
    pub token_endpoint: String,
}

async fn fetch_auth_config(
    client: &reqwest::Client,
    platform_url: &str,
) -> Result<SignInEndpoints> {
    let info: PlatformInfo = get_json(
        client,
        &format!("{platform_url}/v1/info"),
        None,
        "fetching sign-in configuration",
    )
    .await?;
    sign_in_endpoints(info, platform_url)
}

fn sign_in_endpoints(info: PlatformInfo, platform_url: &str) -> Result<SignInEndpoints> {
    let (Some(device_authorization_endpoint), Some(token_endpoint)) = (
        info.auth.device_authorization_endpoint,
        info.auth.token_endpoint,
    ) else {
        bail!(
            "platform {platform_url} does not advertise sign-in endpoints \
             (auth.deviceAuthorizationEndpoint and auth.tokenEndpoint in /v1/info); \
             upgrade the platform or check the URL"
        );
    };
    require_trusted_endpoint(
        &device_authorization_endpoint,
        "device authorization endpoint",
    )?;
    require_trusted_endpoint(&token_endpoint, "token endpoint")?;
    Ok(SignInEndpoints {
        client_id: info.auth.cli.client_id,
        device_authorization_endpoint,
        token_endpoint,
    })
}

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

/// The RFC 6749 token response. Only the two standard fields are read;
/// identity comes from the platform's /v1/me, never from the provider.
#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    // A missing refresh token degrades the login to Incomplete instead of
    // failing after the user already approved in the browser.
    #[serde(default)]
    refresh_token: String,
}

impl TokenResponse {
    fn into_credentials(self) -> Credentials {
        Credentials {
            access_token: self.access_token,
            refresh_token: self.refresh_token,
        }
    }
}

#[derive(Serialize)]
struct RefreshTokenRequest<'a> {
    client_id: &'a str,
    grant_type: &'static str,
    refresh_token: &'a str,
    /// The identity provider switches the session's active organization
    /// when set: the platform's documented extension to the refresh grant.
    #[serde(skip_serializing_if = "Option::is_none")]
    organization_id: Option<&'a str>,
}

impl<'a> RefreshTokenRequest<'a> {
    fn new(client_id: &'a str, refresh_token: &'a str, organization_id: Option<&'a str>) -> Self {
        Self {
            client_id,
            grant_type: "refresh_token",
            refresh_token,
            organization_id,
        }
    }
}

#[derive(Deserialize)]
struct TokenDenial {
    #[serde(default)]
    error: String,
    #[serde(default)]
    error_description: String,
}

impl TokenDenial {
    fn detail(&self) -> String {
        let detail = if self.error_description.trim().is_empty() {
            self.error.clone()
        } else {
            format!("{}: {}", self.error, self.error_description)
        };
        detail.chars().take(300).collect()
    }
}

#[derive(Deserialize)]
struct MeOrganizationsResponse {
    organizations: Vec<MeMembership>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MeMembership {
    #[serde(default)]
    name: String,
    #[serde(default)]
    provider_id: String,
    #[serde(default)]
    registry_namespace: String,
    active: bool,
}

pub async fn org_list(
    profiles: &mut ProfileStore,
    args: &OrgListArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<()> {
    // Always the profile session, never the API key: organizations are an
    // account-level concept and a key is bound to one of them.
    let profile = profiles.resolve(args.profile.profile.as_deref());
    let mut platform = Platform::connect_profile(profiles, profile).await?;
    let page = platform.get("/v1/me/organizations", &[]).await?;
    if args.json {
        writeln!(
            stdout,
            "{}",
            serde_json::to_string_pretty(&page["organizations"])?
        )?;
        return Ok(());
    }
    let response: MeOrganizationsResponse =
        serde_json::from_value(page).context("parsing organizations")?;

    write_heading(stdout, style, "Organizations")?;
    for org in response.organizations {
        let marker = if org.active {
            style.green("●")
        } else {
            " ".to_owned()
        };
        let namespace = if org.registry_namespace.is_empty() {
            style.muted("not provisioned")
        } else {
            style.muted(&format!("registry {}", org.registry_namespace))
        };
        writeln!(
            stdout,
            "  {marker} {} {}  {namespace}",
            style.bold(&org.name),
            org.provider_id
        )?;
    }
    writeln!(stdout)?;
    writeln!(
        stdout,
        "  Switch with {}.",
        style.bold("rlmesh org switch <org_id>")
    )?;
    Ok(())
}

pub async fn org_switch(
    profiles: &mut ProfileStore,
    id: &str,
    args: &ProfileArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<()> {
    let profile = profiles.resolve(args.profile.as_deref());
    let client = http_client()?;
    let session = switch_organization(&client, profiles, &profile, id).await?;
    let platform = profile.platform_url.as_deref().unwrap_or_default();
    // The platform, not the token, says which organization the session now
    // belongs to: /v1/me reports it by provider id and supplies the name.
    let identity = fetch_identity(&client, platform, &session.credentials.access_token).await?;
    if identity.organization_id != id {
        bail!(
            "the platform kept {:?} active; is {id:?} an organization you belong to?",
            identity.organization_id
        );
    }
    profiles.update_identity(&profile.name, identity.clone())?;

    let name = if identity.organization_name.is_empty() {
        id.to_owned()
    } else {
        format!("{} ({id})", identity.organization_name)
    };
    writeln!(
        stdout,
        "{}",
        style.success(&format!("Profile {:?} now uses {name}", profile.name))
    )?;
    Ok(())
}

#[derive(Deserialize)]
struct MeResponse {
    subject: MeSubject,
    #[serde(default)]
    organization: Option<MeOrganization>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MeSubject {
    id: String,
    #[serde(default)]
    email: String,
    #[serde(default)]
    display_name: String,
}

// The platform also reports its own public id for a linked organization;
// the CLI keys everything by the provider id, which is what `org switch`
// sends on the refresh grant.
#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct MeOrganization {
    #[serde(default)]
    name: String,
    #[serde(default)]
    provider_id: String,
}

pub async fn login(
    profiles: &mut ProfileStore,
    args: &LoginArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<()> {
    let mut profile =
        profiles.resolve_login(args.profile.profile.as_deref(), args.platform.as_deref())?;
    let platform_url = profile
        .platform_url
        .clone()
        .context("resolved login has no platform")?;
    let platform_url = platform_url.as_str();
    require_trusted_endpoint(platform_url, "platform URL")?;
    let client = http_client()?;

    write_heading(stdout, style, "Sign in to RLMesh")?;
    write_key_value(stdout, style, "Profile", &profile.name)?;
    write_key_value(stdout, style, "Platform", platform_url)?;
    writeln!(stdout)?;
    stdout.flush()?;

    let auth_config = fetch_auth_config(&client, platform_url).await?;
    // Pin the token endpoint's host: the stored refresh token is only ever
    // sent where this sign-in sent it, whatever a later /v1/info says.
    profile.token_endpoint = Some(auth_config.token_endpoint.clone());

    let authorization: DeviceAuthorization = expect_json(
        client
            .post(&auth_config.device_authorization_endpoint)
            .form(&[("client_id", auth_config.client_id.as_str())])
            .send()
            .await
            .context("requesting device authorization")?,
        "requesting device authorization",
    )
    .await?;

    let browser_url = authorization
        .verification_uri_complete
        .as_deref()
        .unwrap_or(&authorization.verification_uri);
    let browser_opened = style.interactive() && try_open_browser(browser_url);

    if browser_opened {
        writeln!(stdout, "{}", style.success("Browser opened"))?;
        writeln!(stdout, "  If it did not open, visit:")?;
    } else {
        writeln!(stdout, "Open this URL in your browser:")?;
    }
    writeln!(stdout, "  {}", style.cyan(browser_url))?;
    writeln!(stdout)?;
    write_key_value(
        stdout,
        style,
        "Confirm code",
        &style.bold(&authorization.user_code),
    )?;
    writeln!(stdout)?;

    let minutes = authorization.expires_in.div_ceil(60);
    write!(
        stdout,
        "{} Waiting for approval (expires in {minutes} min)",
        style.muted("◌")
    )?;
    stdout.flush()?;

    let credentials = poll_for_token(&client, &auth_config, &authorization, stdout, style).await;
    writeln!(stdout)?;
    let credentials = credentials?;

    // The provider vouched for the sign-in, so the credentials are stored
    // either way; a platform that cannot confirm the identity is reported,
    // not hidden behind "Signed in".
    let identity = match fetch_identity(&client, platform_url, &credentials.access_token).await {
        Ok(identity) => Some(identity),
        Err(error) => {
            writeln!(
                stdout,
                "{}",
                style.yellow(&format!(
                    "Signed in, but the platform did not confirm your identity: {error:#}"
                ))
            )?;
            None
        }
    };
    let storage = profiles.record_login(&profile, identity, &credentials)?;

    writeln!(stdout)?;
    writeln!(
        stdout,
        "{}",
        style.success(&format!("Signed in as profile {:?}", profile.name))
    )?;
    if let CredentialStorage::File(path) = storage {
        writeln!(
            stdout,
            "  {}",
            style.muted(&format!(
                "No usable OS keychain; credentials stored in {} (mode 0600)",
                path.display()
            ))
        )?;
    }
    writeln!(stdout, "  {}", style.muted("Next: rlmesh registry login"))?;
    Ok(())
}

/// Discovers the platform's sign-in endpoints for a refresh and enforces
/// the pin the profile took at sign-in: the stored refresh token only ever
/// goes to the host that issued it. A profile without a pin predates
/// pinning and has to sign in once more.
pub(crate) async fn discover_sign_in_endpoints(
    client: &reqwest::Client,
    profile: &ResolvedProfile,
) -> Result<SignInEndpoints> {
    let platform = profile
        .platform_url
        .as_deref()
        .with_context(|| format!("profile {:?} has no configured platform", profile.name))?;
    let endpoints = fetch_auth_config(client, platform).await?;
    let Some(pinned) = profile.token_endpoint.as_deref() else {
        bail!(
            "profile {:?} was signed in before the CLI pinned sign-in endpoints; run `{}` to sign in again",
            profile.name,
            profile.login_hint()
        );
    };
    require_pinned_host(pinned, &endpoints.token_endpoint, "token endpoint")
        .with_context(|| format!("run `{}` to sign in against it", profile.login_hint()))?;
    Ok(endpoints)
}

/// The RFC 6749 `refresh_token` grant against the platform's advertised
/// token endpoint. `organization_id` is the platform's documented extension
/// that re-issues the session against another organization.
pub(crate) async fn exchange_refresh_token(
    client: &reqwest::Client,
    auth_config: &SignInEndpoints,
    refresh_token: &str,
    organization_id: Option<&str>,
    login_hint: &str,
) -> Result<Credentials> {
    let response = client
        .post(&auth_config.token_endpoint)
        .form(&RefreshTokenRequest::new(
            &auth_config.client_id,
            refresh_token,
            organization_id,
        ))
        .send()
        .await
        .context("refreshing session")?;
    let status = response.status();
    let body = response
        .text()
        .await
        .context("reading session refresh response")?;

    if !status.is_success() {
        let denial: TokenDenial = serde_json::from_str(&body).with_context(|| {
            format!("session refresh returned HTTP {status} with an invalid response")
        })?;
        bail!(
            "session refresh was rejected: {}; run `{login_hint}`",
            denial.detail()
        );
    }

    let refreshed: TokenResponse =
        serde_json::from_str(&body).context("parsing session refresh response")?;
    Ok(refreshed.into_credentials())
}

async fn poll_for_token(
    client: &reqwest::Client,
    auth_config: &SignInEndpoints,
    authorization: &DeviceAuthorization,
    progress: &mut impl Write,
    style: Style,
) -> Result<Credentials> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(authorization.expires_in);
    let mut interval = Duration::from_secs(authorization.interval.unwrap_or(5).max(1));

    loop {
        let now = tokio::time::Instant::now();
        if now >= deadline {
            bail!("the sign-in code expired; run `rlmesh login` again");
        }
        tokio::time::sleep(interval.min(deadline - now)).await;
        if tokio::time::Instant::now() >= deadline {
            bail!("the sign-in code expired; run `rlmesh login` again");
        }

        let response = client
            .post(&auth_config.token_endpoint)
            .form(&[
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ("device_code", authorization.device_code.as_str()),
                ("client_id", auth_config.client_id.as_str()),
            ])
            .send()
            .await
            .context("polling for sign-in approval")?;
        let status = response.status();
        let body = response
            .text()
            .await
            .context("reading token endpoint response")?;

        if status.is_success() {
            let token: TokenResponse =
                serde_json::from_str(&body).context("parsing token response")?;
            return Ok(token.into_credentials());
        }

        // A gateway's HTML 502/429 page mid-poll must not abort a sign-in
        // the user is busy approving; the expiry deadline bounds the loop.
        let Ok(denial) = serde_json::from_str::<TokenDenial>(&body) else {
            continue;
        };
        match denial.error.as_str() {
            "" | "authorization_pending" => {
                if style.interactive() {
                    write!(progress, ".")?;
                    progress.flush()?;
                }
            }
            "slow_down" => interval += Duration::from_secs(5),
            "access_denied" => bail!("sign-in was declined"),
            "expired_token" => bail!("the sign-in code expired; run `rlmesh login` again"),
            _ => bail!(
                "the token endpoint rejected the sign-in: {}",
                denial.detail()
            ),
        }
    }
}

pub async fn logout(
    profiles: &mut ProfileStore,
    args: &ProfileArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<()> {
    let profile = profiles.resolve(args.profile.as_deref());
    // Best effort all the way down: nothing about the revocation, its
    // client, or reporting it may keep the local credential on disk.
    let _ = revoke_platform_session(profiles, &profile, stdout, style).await;
    if profiles.logout(&profile.name)? {
        writeln!(
            stdout,
            "{}",
            style.success(&format!("Signed out of profile {:?}", profile.name))
        )?;
    } else {
        writeln!(
            stdout,
            "{}",
            style.muted(&format!(
                "Profile {:?} is already signed out.",
                profile.name
            ))
        )?;
    }
    Ok(())
}

/// Asks the platform to end the session before the local credential is
/// deleted, so the refresh token stops working server-side too. Best
/// effort: a platform without the route, or one that cannot be reached,
/// never blocks a local sign-out.
async fn revoke_platform_session(
    profiles: &mut ProfileStore,
    profile: &ResolvedProfile,
    stdout: &mut impl Write,
    style: Style,
) -> Result<()> {
    let Some(platform) = profile.platform_url.as_deref() else {
        return Ok(());
    };
    let Some(stored) = profiles.credentials(&profile.name)? else {
        return Ok(());
    };
    let client = http_client()?;
    let access_token =
        match ensure_fresh_session(&client, profiles, profile, Refresh::IfStale).await {
            Ok(session) => session.credentials.access_token,
            // A session the provider already rejected may still be known to the
            // platform; try with what is on hand rather than skipping.
            Err(_) if !stored.access_token.trim().is_empty() => stored.access_token,
            Err(_) => return Ok(()),
        };

    let outcome = client
        .delete(format!("{platform}/v1/me/session"))
        .bearer_auth(&access_token)
        .send()
        .await;
    let note = match outcome {
        Ok(response) if response.status().is_success() => {
            "Session revoked on the platform".to_owned()
        }
        Ok(response) if matches!(response.status().as_u16(), 404 | 405) => {
            "Session not revoked on the platform (endpoint not available)".to_owned()
        }
        Ok(response) => format!(
            "Session not revoked on the platform (HTTP {})",
            response.status()
        ),
        Err(error) => format!(
            "Session not revoked on the platform ({})",
            error.without_url()
        ),
    };
    writeln!(stdout, "  {}", style.muted(&note))?;
    Ok(())
}

/// Exits 0 only for a signed-in, verified session, so scripts can gate on it.
pub async fn whoami(
    profiles: &mut ProfileStore,
    args: &WhoamiArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<i32> {
    if let Some((api_key, platform_url)) = profiles.api_key() {
        return whoami_api_key(api_key, platform_url, args.json, stdout, style).await;
    }
    let profile = profiles.resolve(args.profile.profile.as_deref());
    let status = profiles.credential_status(&profile.name)?;
    let mut identity = profile.identity.clone();
    let mut verification = None;

    if status == CredentialStatus::SignedIn {
        verification = Some(verify_session(profiles, &profile, &mut identity).await);
    }
    let healthy = matches!(verification, Some(Ok(())));

    if args.json {
        let report = whoami_report(
            Some(&profile.name),
            profile.platform_url.as_deref(),
            status,
            healthy,
            identity
                .as_ref()
                .filter(|_| status == CredentialStatus::SignedIn),
            verification
                .as_ref()
                .and_then(|outcome| outcome.as_ref().err())
                .map(|error| format!("{error:#}")),
        );
        writeln!(stdout, "{}", serde_json::to_string_pretty(&report)?)?;
        return Ok(i32::from(!healthy));
    }

    write_heading(stdout, style, "Authentication")?;
    let profile_name = if profile.is_default {
        format!("{} {}", profile.name, style.muted("(default)"))
    } else {
        profile.name.clone()
    };
    write_key_value(stdout, style, "Profile", &profile_name)?;
    write_key_value(
        stdout,
        style,
        "Platform",
        profile.platform_url.as_deref().unwrap_or("not configured"),
    )?;
    write_key_value(stdout, style, "Status", &style.status(status))?;

    if status == CredentialStatus::SignedIn
        && let Some(identity) = identity.as_ref()
    {
        if !identity.email.is_empty() {
            write_key_value(stdout, style, "Account", &identity.email)?;
        }
        if !identity.display_name.is_empty() {
            write_key_value(stdout, style, "Name", &identity.display_name)?;
        }
        if !identity.user_id.is_empty() {
            write_key_value(stdout, style, "User", &identity.user_id)?;
        }
        let organization = if identity.organization_name.is_empty() {
            identity.organization_id.clone()
        } else {
            format!(
                "{} ({})",
                identity.organization_name, identity.organization_id
            )
        };
        write_key_value(stdout, style, "Organization", &organization)?;
    }

    if let Some(verification) = verification {
        let value = match verification {
            Ok(()) => style.green("✓ verified"),
            Err(error) => style.yellow(&format!("not verified — {error:#}")),
        };
        write_key_value(stdout, style, "Session", &value)?;
    } else if status != CredentialStatus::SignedIn {
        writeln!(stdout)?;
        writeln!(
            stdout,
            "  Run {} to sign in.",
            style.bold(&profile.login_hint())
        )?;
    }

    Ok(if healthy { 0 } else { 1 })
}

/// The `whoami --json` document. `identity` is present only for a verified
/// or at least signed-in session; `error` carries the verification failure.
fn whoami_report(
    profile: Option<&str>,
    platform: Option<&str>,
    status: CredentialStatus,
    verified: bool,
    identity: Option<&Identity>,
    error: Option<String>,
) -> Value {
    json!({
        "profile": profile,
        "platform": platform,
        "status": status.label(),
        "verified": verified,
        "identity": identity.map(|identity| json!({
            "userId": identity.user_id,
            "email": identity.email,
            "displayName": identity.display_name,
            "organizationId": identity.organization_id,
            "organizationName": identity.organization_name,
        })),
        "error": error,
    })
}

/// `whoami` for an API key from the environment: no profile, no cached
/// identity, just whether the platform accepts the key and as whom.
async fn whoami_api_key(
    api_key: &str,
    platform_url: &str,
    json_output: bool,
    stdout: &mut impl Write,
    style: Style,
) -> Result<i32> {
    if json_output {
        let client = http_client()?;
        let (identity, error) = match fetch_identity(&client, platform_url, api_key).await {
            Ok(identity) => (Some(identity), None),
            Err(error) => (None, Some(format!("{error:#}"))),
        };
        let verified = identity.is_some();
        let report = whoami_report(
            None,
            Some(platform_url),
            CredentialStatus::ApiKey,
            verified,
            identity.as_ref(),
            error,
        );
        writeln!(stdout, "{}", serde_json::to_string_pretty(&report)?)?;
        return Ok(i32::from(!verified));
    }

    write_heading(stdout, style, "Authentication")?;
    write_key_value(
        stdout,
        style,
        "Profile",
        &style.muted("none (RLMESH_API_KEY is set)"),
    )?;
    write_key_value(stdout, style, "Platform", platform_url)?;
    write_key_value(
        stdout,
        style,
        "Status",
        &style.status(CredentialStatus::ApiKey),
    )?;

    let client = http_client()?;
    match fetch_identity(&client, platform_url, api_key).await {
        Ok(identity) => {
            if !identity.email.is_empty() {
                write_key_value(stdout, style, "Account", &identity.email)?;
            }
            write_key_value(stdout, style, "Subject", &identity.user_id)?;
            if !identity.organization_id.is_empty() {
                write_key_value(stdout, style, "Organization", &identity.organization_id)?;
            }
            write_key_value(stdout, style, "Session", &style.green("✓ verified"))?;
            Ok(0)
        }
        Err(error) => {
            write_key_value(
                stdout,
                style,
                "Session",
                &style.yellow(&format!("not verified — {error:#}")),
            )?;
            Ok(1)
        }
    }
}

/// Verifies the stored session against /v1/me, updating the cached identity.
///
/// Goes through `Platform` so the stored access token is reused while it is
/// fresh and rotated only if the platform actually rejects it: refresh
/// tokens are single-use, and a read-only status command must not burn one.
async fn verify_session(
    profiles: &mut ProfileStore,
    profile: &ResolvedProfile,
    identity: &mut Option<Identity>,
) -> Result<()> {
    let current = {
        let mut platform = Platform::connect_profile(profiles, profile.clone()).await?;
        let response: MeResponse = serde_json::from_value(platform.get("/v1/me", &[]).await?)
            .context("parsing identity")?;
        identity_from_me(response)
    };
    record_identity(profiles, profile, identity, current)
}

fn record_identity(
    profiles: &mut ProfileStore,
    profile: &ResolvedProfile,
    identity: &mut Option<Identity>,
    current: Identity,
) -> Result<()> {
    if identity.as_ref() != Some(&current) {
        profiles.update_identity(&profile.name, current.clone())?;
    }
    *identity = Some(current);
    Ok(())
}

async fn fetch_identity(
    client: &reqwest::Client,
    platform: &str,
    access_token: &str,
) -> Result<Identity> {
    let response: MeResponse = get_json(
        client,
        &format!("{platform}/v1/me"),
        Some(access_token),
        "fetching identity",
    )
    .await?;
    Ok(identity_from_me(response))
}

fn identity_from_me(response: MeResponse) -> Identity {
    let organization = response.organization.unwrap_or_default();
    Identity {
        user_id: response.subject.id,
        email: response.subject.email,
        display_name: response.subject.display_name,
        organization_id: organization.provider_id,
        organization_name: organization.name,
    }
}

fn try_open_browser(url: &str) -> bool {
    #[cfg(target_os = "macos")]
    let (program, args): (&str, &[&str]) = ("open", &[]);
    #[cfg(target_os = "windows")]
    let (program, args): (&str, &[&str]) = ("rundll32.exe", &["url.dll,FileProtocolHandler"]);
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let (program, args): (&str, &[&str]) = ("xdg-open", &[]);

    Command::new(program)
        .args(args)
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_info_requires_advertised_endpoints() {
        let discovered: PlatformInfo = serde_json::from_str(
            r#"{"auth":{
                "cli":{"clientId":"client_cli"},
                "issuer":"https://auth.example.com",
                "deviceAuthorizationEndpoint":"https://id.example.com/device",
                "tokenEndpoint":"https://id.example.com/token"
            }}"#,
        )
        .unwrap();
        let endpoints = sign_in_endpoints(discovered, "https://platform.example.com").unwrap();
        assert_eq!(endpoints.client_id, "client_cli");
        assert_eq!(
            endpoints.device_authorization_endpoint,
            "https://id.example.com/device"
        );
        assert_eq!(endpoints.token_endpoint, "https://id.example.com/token");

        // No built-in provider: a platform without the fields is refused.
        let legacy: PlatformInfo =
            serde_json::from_str(r#"{"auth":{"cli":{"clientId":"client_cli"}}}"#).unwrap();
        let error = sign_in_endpoints(legacy, "https://platform.example.com").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not advertise sign-in endpoints"),
            "{error}"
        );

        // Plaintext endpoints are refused even when advertised.
        let plain: PlatformInfo = serde_json::from_str(
            r#"{"auth":{
                "cli":{"clientId":"client_cli"},
                "deviceAuthorizationEndpoint":"http://id.example.com/device",
                "tokenEndpoint":"http://id.example.com/token"
            }}"#,
        )
        .unwrap();
        assert!(sign_in_endpoints(plain, "https://platform.example.com").is_err());
    }

    #[test]
    fn refresh_request_uses_the_refresh_token_grant() {
        let request =
            serde_json::to_value(RefreshTokenRequest::new("client_123", "refresh_123", None))
                .unwrap();

        assert_eq!(request["client_id"], "client_123");
        assert_eq!(request["grant_type"], "refresh_token");
        assert_eq!(request["refresh_token"], "refresh_123");
    }

    #[test]
    fn token_response_reads_only_the_standard_fields() {
        let token: TokenResponse = serde_json::from_str(
            r#"{
                "access_token":"access_new",
                "refresh_token":"refresh_new",
                "organization_id":"org_123",
                "user":{"id":"user_123","email":"dev@example.com"}
            }"#,
        )
        .unwrap();
        let credentials = token.into_credentials();

        assert_eq!(credentials.access_token, "access_new");
        assert_eq!(credentials.refresh_token, "refresh_new");
    }

    #[test]
    fn identity_comes_from_me_subject_and_provider_id() {
        let response: MeResponse = serde_json::from_str(
            r#"{
                "subject":{
                    "type":"userSession",
                    "id":"user_123",
                    "email":"dev@example.com",
                    "displayName":"Dev User"
                },
                "organization":{
                    "id":"org_9f3c2a",
                    "name":"Dev Org",
                    "providerId":"org_01H",
                    "linked":true
                },
                "access":{"roles":[],"permissions":[]}
            }"#,
        )
        .unwrap();
        let identity = identity_from_me(response);

        assert_eq!(identity.user_id, "user_123");
        assert_eq!(identity.email, "dev@example.com");
        assert_eq!(identity.display_name, "Dev User");
        // The platform's public id is never what the CLI keys on.
        assert_eq!(identity.organization_id, "org_01H");
        assert_eq!(identity.organization_name, "Dev Org");

        // An older platform (no email) or an org-less session still parses.
        let bare: MeResponse = serde_json::from_str(r#"{"subject":{"id":"user_123"}}"#).unwrap();
        let identity = identity_from_me(bare);
        assert!(identity.email.is_empty());
        assert!(identity.organization_id.is_empty());
    }

    #[test]
    fn oauth_errors_keep_the_human_readable_detail() {
        let denial: TokenDenial = serde_json::from_str(
            r#"{"error":"invalid_client","error_description":"Unknown client"}"#,
        )
        .unwrap();

        assert_eq!(denial.error, "invalid_client");
        assert_eq!(denial.error_description, "Unknown client");
    }
}
