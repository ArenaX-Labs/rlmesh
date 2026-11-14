//! Sign-in discovery and the refresh grant: the platform's advertised OAuth
//! endpoints and the token exchange a stored session is refreshed with. The
//! interactive device flow is the CLI's (`rlmesh login`).

use super::http::{get_json, require_pinned_host, require_trusted_endpoint};
use super::store::{Credentials, ResolvedProfile};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

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
pub struct SignInEndpoints {
    /// The CLI's OAuth client id.
    pub client_id: String,
    /// Where the device grant starts.
    pub device_authorization_endpoint: String,
    /// Where device codes and refresh tokens are exchanged.
    pub token_endpoint: String,
}

/// The platform's advertised sign-in endpoints (its `/v1/info` auth block).
pub async fn fetch_auth_config(
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

/// The RFC 6749 token response. Only the two standard fields are read;
/// identity comes from the platform's /v1/me, never from the provider.
#[derive(Deserialize)]
pub struct TokenResponse {
    access_token: String,
    // A missing refresh token degrades the login to Incomplete instead of
    // failing after the user already approved in the browser.
    #[serde(default)]
    refresh_token: String,
}

impl TokenResponse {
    /// The tokens as stored credentials.
    pub fn into_credentials(self) -> Credentials {
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

/// An RFC 6749 error response from the token endpoint.
#[derive(Deserialize)]
pub struct TokenDenial {
    /// The error code (`authorization_pending`, `slow_down`, ...).
    #[serde(default)]
    pub error: String,
    /// The provider's human-readable detail.
    #[serde(default)]
    pub error_description: String,
}

impl TokenDenial {
    /// The code and detail, capped at 300 characters.
    pub fn detail(&self) -> String {
        let detail = if self.error_description.trim().is_empty() {
            self.error.clone()
        } else {
            format!("{}: {}", self.error, self.error_description)
        };
        detail.chars().take(300).collect()
    }
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
    fn oauth_errors_keep_the_human_readable_detail() {
        let denial: TokenDenial = serde_json::from_str(
            r#"{"error":"invalid_client","error_description":"Unknown client"}"#,
        )
        .unwrap();

        assert_eq!(denial.error, "invalid_client");
        assert_eq!(denial.error_description, "Unknown client");
    }
}
