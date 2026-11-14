use crate::cli::{LoginArgs, OrgListArgs, ProfileArgs, WhoamiArgs};
use crate::render::{Style, write_heading, write_key_value};

use anyhow::{Context, Result, bail};
use rlmesh::binding::managed::{
    self as store, CredentialStatus, CredentialStorage, Credentials, Identity, ProfileStore,
    Refresh, ResolvedProfile, SignInEndpoints, TokenDenial, TokenResponse, ensure_fresh_session,
    expect_json, fetch_auth_config, get_json, http_client, require_trusted_endpoint,
    switch_organization,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

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
    let mut platform = store::connect_profile(profiles.clone(), profile).await?;
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
/// Goes through `Client` so the stored access token is reused while it is
/// fresh and rotated only if the platform actually rejects it: refresh
/// tokens are single-use, and a read-only status command must not burn one.
async fn verify_session(
    profiles: &mut ProfileStore,
    profile: &ResolvedProfile,
    identity: &mut Option<Identity>,
) -> Result<()> {
    let current = {
        let mut platform = store::connect_profile(profiles.clone(), profile.clone()).await?;
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
}
