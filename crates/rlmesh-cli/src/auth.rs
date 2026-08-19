use crate::cli::{LoginArgs, ProfileArgs};
use crate::config::{CredentialStatus, Credentials, Identity, ProfileStore, ResolvedProfile};
use crate::helpers::{expect_json, get_json, http_client};
use crate::render::{Style, write_heading, write_key_value};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

// Fallback OAuth endpoints for control planes whose /v1/info predates the
// discovery fields. Newer platforms advertise the endpoints themselves, so a
// provider change never requires a CLI upgrade.
const FALLBACK_DEVICE_AUTHORIZATION_ENDPOINT: &str =
    "https://api.workos.com/user_management/authorize/device";
const FALLBACK_TOKEN_ENDPOINT: &str = "https://api.workos.com/user_management/authenticate";

#[derive(Deserialize)]
struct PlatformInfo {
    auth: AuthConfig,
}

/// The platform /v1/info auth block, RFC 8414 vocabulary. Only the fields the
/// CLI drives the device flow with; everything else in the document is
/// ignored.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AuthConfig {
    cli: AppAuth,
    #[serde(default = "fallback_device_authorization_endpoint")]
    device_authorization_endpoint: String,
    #[serde(default = "fallback_token_endpoint")]
    token_endpoint: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AppAuth {
    client_id: String,
}

fn fallback_device_authorization_endpoint() -> String {
    FALLBACK_DEVICE_AUTHORIZATION_ENDPOINT.to_string()
}

fn fallback_token_endpoint() -> String {
    FALLBACK_TOKEN_ENDPOINT.to_string()
}

async fn fetch_auth_config(client: &reqwest::Client, platform_url: &str) -> Result<AuthConfig> {
    let info: PlatformInfo = get_json(
        client,
        &format!("{platform_url}/v1/info"),
        None,
        "fetching sign-in configuration",
    )
    .await?;
    Ok(info.auth)
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

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: String,
    #[serde(default)]
    organization_id: Option<String>,
    #[serde(default)]
    user: Option<AuthUser>,
}

impl TokenResponse {
    fn into_login_result(self) -> LoginResult {
        let identity = match (self.user, self.organization_id) {
            (Some(user), Some(organization_id)) => Some(Identity {
                user_id: user.id,
                email: user.email,
                first_name: user.first_name,
                last_name: user.last_name,
                organization_id,
                // The token endpoint knows nothing platform-side; the org's
                // display name is merged in from /v1/me by fetch_identity.
                organization_name: String::new(),
            }),
            _ => None,
        };

        LoginResult {
            credentials: Credentials {
                access_token: self.access_token,
                refresh_token: self.refresh_token,
            },
            identity,
        }
    }
}

#[derive(Serialize)]
struct RefreshTokenRequest<'a> {
    client_id: &'a str,
    grant_type: &'static str,
    refresh_token: &'a str,
}

impl<'a> RefreshTokenRequest<'a> {
    fn new(client_id: &'a str, refresh_token: &'a str) -> Self {
        Self {
            client_id,
            grant_type: "refresh_token",
            refresh_token,
        }
    }
}

#[derive(Deserialize)]
struct AuthUser {
    id: String,
    #[serde(default)]
    email: String,
    #[serde(default)]
    first_name: String,
    #[serde(default)]
    last_name: String,
}

#[derive(Deserialize)]
struct TokenDenial {
    error: String,
    #[serde(default)]
    error_description: String,
}

impl TokenDenial {
    fn detail(&self) -> String {
        if self.error_description.trim().is_empty() {
            self.error.clone()
        } else {
            format!("{}: {}", self.error, self.error_description)
        }
    }
}

struct LoginResult {
    credentials: Credentials,
    identity: Option<Identity>,
}

pub(crate) struct RefreshedSession {
    pub credentials: Credentials,
    pub identity: Option<Identity>,
}

#[derive(Deserialize)]
struct MeResponse {
    subject: MeSubject,
    #[serde(default)]
    organization: Option<MeOrganization>,
}

#[derive(Deserialize)]
struct MeSubject {
    id: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MeOrganization {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: String,
    provider_id: String,
}

pub async fn login(
    profiles: &mut ProfileStore,
    args: &LoginArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<()> {
    let profile =
        profiles.resolve_login(args.profile.profile.as_deref(), args.platform.as_deref())?;
    let platform_url = profile
        .platform_url
        .as_deref()
        .context("resolved login has no platform")?;
    let client = http_client()?;

    write_heading(stdout, style, "Sign in to RLMesh")?;
    write_key_value(stdout, style, "Profile", &profile.name)?;
    write_key_value(stdout, style, "Platform", platform_url)?;
    writeln!(stdout)?;
    stdout.flush()?;

    let auth_config = fetch_auth_config(&client, platform_url).await?;

    let authorization: DeviceAuthorization = expect_json(
        client
            .post(&auth_config.device_authorization_endpoint)
            .form(&[("client_id", auth_config.cli.client_id.as_str())])
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

    let login = poll_for_token(&client, &auth_config, &authorization, stdout, style).await;
    writeln!(stdout)?;
    let login = login?;

    // The token response and /v1/me each know half the identity: profile
    // fields come from the provider, org id/name from the platform. Merge,
    // keeping whichever half succeeded.
    let identity = match fetch_identity(
        &client,
        platform_url,
        &login.credentials.access_token,
        login.identity.as_ref(),
    )
    .await
    {
        Ok(identity) => Some(identity),
        Err(_) => login.identity,
    };
    profiles.record_login(&profile, identity, &login.credentials)?;

    writeln!(stdout)?;
    writeln!(
        stdout,
        "{}",
        style.success(&format!("Signed in as profile {:?}", profile.name))
    )?;
    writeln!(stdout, "  {}", style.muted("Next: rlmesh registry login"))?;
    Ok(())
}

pub(crate) async fn refresh_session(
    client: &reqwest::Client,
    profiles: &mut ProfileStore,
    profile: &ResolvedProfile,
) -> Result<RefreshedSession> {
    let platform = profile
        .platform_url
        .as_deref()
        .with_context(|| format!("profile {:?} has no configured platform", profile.name))?;
    let credentials = profiles.credentials(&profile.name)?.with_context(|| {
        format!(
            "profile {:?} is not signed in; run `{}`",
            profile.name,
            profile.login_hint()
        )
    })?;
    if credentials.refresh_token.trim().is_empty() {
        bail!(
            "profile {:?} has no refresh token; run `{}`",
            profile.name,
            profile.login_hint()
        );
    }

    let auth_config = fetch_auth_config(client, platform).await?;
    let response = client
        .post(&auth_config.token_endpoint)
        .form(&RefreshTokenRequest::new(
            &auth_config.cli.client_id,
            &credentials.refresh_token,
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
            "session refresh was rejected: {}; run `{}`",
            denial.detail(),
            profile.login_hint()
        );
    }

    let refreshed: TokenResponse =
        serde_json::from_str(&body).context("parsing session refresh response")?;
    let mut refreshed = refreshed.into_login_result();

    // The refresh response never carries the org name (that comes from
    // /v1/me at login); keep the stored one instead of clobbering it.
    if let Some(identity) = refreshed.identity.as_mut()
        && identity.organization_name.is_empty()
        && let Some(existing) = profile
            .identity
            .as_ref()
            .filter(|existing| existing.organization_id == identity.organization_id)
    {
        identity.organization_name = existing.organization_name.clone();
    }

    profiles
        .replace_credentials(&profile.name, &refreshed.credentials)
        .context("saving refreshed credentials")?;
    if let Some(identity) = refreshed.identity.as_ref()
        && profile.identity.as_ref() != Some(identity)
    {
        profiles.update_identity(&profile.name, identity.clone())?;
    }

    Ok(RefreshedSession {
        credentials: refreshed.credentials,
        identity: refreshed.identity.or_else(|| profile.identity.clone()),
    })
}

async fn poll_for_token(
    client: &reqwest::Client,
    auth_config: &AuthConfig,
    authorization: &DeviceAuthorization,
    progress: &mut impl Write,
    style: Style,
) -> Result<LoginResult> {
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
                ("client_id", auth_config.cli.client_id.as_str()),
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
            return Ok(token.into_login_result());
        }

        let denial: TokenDenial = serde_json::from_str(&body).with_context(|| {
            format!("token endpoint returned HTTP {status} with an invalid response")
        })?;
        match denial.error.as_str() {
            "authorization_pending" => {
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

pub fn logout(
    profiles: &mut ProfileStore,
    args: &ProfileArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<()> {
    let profile = profiles.resolve(args.profile.as_deref());
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

pub async fn whoami(
    profiles: &mut ProfileStore,
    args: &ProfileArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<()> {
    let profile = profiles.resolve(args.profile.as_deref());
    let status = profiles.credential_status(&profile.name)?;
    let mut identity = profile.identity.clone();
    let mut verification = None;

    if status == CredentialStatus::SignedIn {
        let client = http_client()?;
        match refresh_session(&client, profiles, &profile).await {
            Ok(session) => {
                identity = session.identity;
                let platform = profile
                    .platform_url
                    .as_deref()
                    .expect("session refresh requires a platform");
                match fetch_identity(
                    &client,
                    platform,
                    &session.credentials.access_token,
                    identity.as_ref(),
                )
                .await
                {
                    Ok(current_identity) => {
                        if identity.as_ref() != Some(&current_identity) {
                            profiles.update_identity(&profile.name, current_identity.clone())?;
                        }
                        identity = Some(current_identity);
                        verification = Some(Ok(()));
                    }
                    Err(error) => verification = Some(Err(error)),
                }
            }
            Err(error) => verification = Some(Err(error)),
        }
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
        if !identity.user_id.is_empty() {
            write_key_value(stdout, style, "User", &identity.user_id)?;
        }
        write_key_value(stdout, style, "Organization", &identity.organization_id)?;
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

    Ok(())
}

async fn fetch_identity(
    client: &reqwest::Client,
    platform: &str,
    access_token: &str,
    cached: Option<&Identity>,
) -> Result<Identity> {
    let response: MeResponse = get_json(
        client,
        &format!("{platform}/v1/me"),
        Some(access_token),
        "fetching identity",
    )
    .await?;

    let organization = response.organization;
    Ok(Identity {
        user_id: response.subject.id,
        email: cached
            .map(|identity| identity.email.clone())
            .unwrap_or_default(),
        first_name: cached
            .map(|identity| identity.first_name.clone())
            .unwrap_or_default(),
        last_name: cached
            .map(|identity| identity.last_name.clone())
            .unwrap_or_default(),
        organization_id: organization
            .as_ref()
            .and_then(|org| org.id.clone())
            .or_else(|| organization.as_ref().map(|org| org.provider_id.clone()))
            .unwrap_or_default(),
        organization_name: organization.map(|org| org.name).unwrap_or_default(),
    })
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
    fn platform_info_supplies_endpoints_with_fallbacks() {
        let discovered: PlatformInfo = serde_json::from_str(
            r#"{"auth":{
                "cli":{"clientId":"client_cli"},
                "deviceAuthorizationEndpoint":"https://id.example.com/device",
                "tokenEndpoint":"https://id.example.com/token"
            }}"#,
        )
        .unwrap();
        assert_eq!(discovered.auth.cli.client_id, "client_cli");
        assert_eq!(
            discovered.auth.device_authorization_endpoint,
            "https://id.example.com/device"
        );
        assert_eq!(
            discovered.auth.token_endpoint,
            "https://id.example.com/token"
        );

        // A platform that predates the discovery fields falls back to WorkOS.
        let legacy: PlatformInfo =
            serde_json::from_str(r#"{"auth":{"cli":{"clientId":"client_cli"}}}"#).unwrap();
        assert_eq!(
            legacy.auth.device_authorization_endpoint,
            FALLBACK_DEVICE_AUTHORIZATION_ENDPOINT
        );
        assert_eq!(legacy.auth.token_endpoint, FALLBACK_TOKEN_ENDPOINT);
    }

    #[test]
    fn refresh_request_uses_the_workos_contract() {
        let request =
            serde_json::to_value(RefreshTokenRequest::new("client_123", "refresh_123")).unwrap();

        assert_eq!(request["client_id"], "client_123");
        assert_eq!(request["grant_type"], "refresh_token");
        assert_eq!(request["refresh_token"], "refresh_123");
    }

    #[test]
    fn token_response_keeps_rotated_credentials_and_identity() {
        let token: TokenResponse = serde_json::from_str(
            r#"{
                "access_token":"access_new",
                "refresh_token":"refresh_new",
                "organization_id":"org_123",
                "user":{
                    "id":"user_123",
                    "email":"dev@example.com",
                    "first_name":"Dev",
                    "last_name":"User"
                }
            }"#,
        )
        .unwrap();
        let login = token.into_login_result();

        assert_eq!(login.credentials.access_token, "access_new");
        assert_eq!(login.credentials.refresh_token, "refresh_new");
        let identity = login.identity.unwrap();
        assert_eq!(identity.user_id, "user_123");
        assert_eq!(identity.email, "dev@example.com");
        assert_eq!(identity.organization_id, "org_123");
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
