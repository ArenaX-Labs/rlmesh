use anyhow::{Context, Result, bail};
use serde::de::DeserializeOwned;
use std::net::{IpAddr, Ipv6Addr};
use std::time::Duration;

pub(crate) fn normalize_base_url(raw: &str) -> String {
    let value = raw.trim().trim_end_matches('/');

    if value.is_empty() || has_http_scheme(value) {
        return value.to_owned();
    }

    let scheme = if is_loopback_host(value) {
        "http"
    } else {
        "https"
    };

    format!("{scheme}://{value}")
}

/// Rejects a platform-advertised OAuth endpoint unless it is https (or http
/// to a loopback host, for dev servers): these URLs come from an unsigned
/// /v1/info document and receive device codes and refresh tokens.
pub(crate) fn require_trusted_endpoint(url: &str, what: &str) -> Result<()> {
    let value = url.trim();
    let is_https = value
        .get(..8)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("https://"));
    let is_local_http = value
        .get(..7)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("http://"))
        && is_loopback_host(&value[7..]);

    if is_https || is_local_http {
        return Ok(());
    }
    bail!(
        "the platform advertised a non-https {what} ({value}); refusing to send credentials to it"
    )
}

fn has_http_scheme(value: &str) -> bool {
    ["http://", "https://"].iter().any(|scheme| {
        value
            .get(..scheme.len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(scheme))
    })
}

fn is_loopback_host(value: &str) -> bool {
    let authority = value.split(['/', '?', '#']).next().unwrap_or_default();

    let authority = authority
        .rsplit_once('@')
        .map_or(authority, |(_, authority)| authority);

    let host = extract_host(authority);

    host.eq_ignore_ascii_case("localhost")
        || host.to_ascii_lowercase().ends_with(".localhost")
        || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

fn extract_host(authority: &str) -> &str {
    if let Some(bracketed) = authority.strip_prefix('[') {
        return bracketed
            .split_once(']')
            .map_or(bracketed, |(host, _)| host);
    }

    if authority.parse::<Ipv6Addr>().is_ok() {
        return authority;
    }

    authority
        .rsplit_once(':')
        .map_or(authority, |(host, _)| host)
}

pub(crate) fn http_client() -> Result<reqwest::Client> {
    // reqwest is built with `rustls-no-provider` to keep aws-lc-rs (and its
    // cmake/C toolchain requirement) out of clean builds; ring must therefore
    // be installed as the process-wide provider before building a client.
    let _ = rustls::crypto::ring::default_provider().install_default();

    reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .connect_timeout(Duration::from_secs(10))
        .build()
        .context("building HTTP client")
}

pub(crate) async fn get_json<T: DeserializeOwned>(
    client: &reqwest::Client,
    url: &str,
    bearer: Option<&str>,
    what: &str,
) -> Result<T> {
    let mut request = client.get(url);
    if let Some(token) = bearer {
        request = request.bearer_auth(token);
    }

    let response = request
        .send()
        .await
        .with_context(|| format!("{what}: could not reach {url}"))?;
    expect_json(response, what).await
}

pub(crate) async fn expect_json<T: DeserializeOwned>(
    response: reqwest::Response,
    what: &str,
) -> Result<T> {
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        bail!("{what} failed with HTTP {status}: {}", error_message(&body));
    }

    response
        .json()
        .await
        .with_context(|| format!("parsing {what} response"))
}

fn error_message(body: &str) -> String {
    #[derive(serde::Deserialize)]
    struct Body {
        error: Error,
    }
    #[derive(serde::Deserialize)]
    struct Error {
        message: String,
    }

    serde_json::from_str::<Body>(body)
        .ok()
        .map(|body| body.error.message)
        .filter(|message| !message.is_empty())
        .unwrap_or_else(|| body.to_owned())
        .chars()
        .take(300)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_remote_and_local_urls() {
        assert_eq!(
            normalize_base_url("api.rlmesh.dev/"),
            "https://api.rlmesh.dev"
        );
        assert_eq!(
            normalize_base_url("localhost:3000"),
            "http://localhost:3000"
        );
        assert_eq!(
            normalize_base_url("https://localhost:3000/"),
            "https://localhost:3000"
        );
    }

    #[test]
    fn uses_structured_error_message_when_available() {
        assert_eq!(
            error_message(r#"{"error":{"message":"not allowed"}}"#),
            "not allowed"
        );
    }

    #[test]
    fn truncates_huge_server_messages() {
        let huge = "x".repeat(10_000);
        assert_eq!(error_message(&huge).chars().count(), 300);
        assert_eq!(
            error_message(&format!(r#"{{"error":{{"message":"{huge}"}}}}"#))
                .chars()
                .count(),
            300
        );
    }

    #[test]
    fn trusts_only_https_or_loopback_endpoints() {
        assert!(require_trusted_endpoint("https://id.example.com/token", "token endpoint").is_ok());
        assert!(require_trusted_endpoint("http://localhost:3000/token", "token endpoint").is_ok());
        assert!(require_trusted_endpoint("http://127.0.0.1/token", "token endpoint").is_ok());
        assert!(require_trusted_endpoint("http://id.example.com/token", "token endpoint").is_err());
        assert!(require_trusted_endpoint("ftp://id.example.com/token", "token endpoint").is_err());
    }
}
