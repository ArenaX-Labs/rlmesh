use crate::auth::refresh_session;
use crate::cli::{CredentialHelperArgs, ProfileArgs};
use crate::config::{Identity, ProfileStore};
use crate::helpers::{get_json, http_client};
use crate::render::{Style, write_heading, write_key_value};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const HELPER_BINARY: &str = "docker-credential-rlmesh";
const HELPER_SUFFIX: &str = "rlmesh";
// docker's sentinel for "no stored credential"; it must go to stdout verbatim.
const CREDENTIALS_NOT_FOUND: &str = "credentials not found in native keychain";

#[derive(Deserialize)]
struct RegistryInfo {
    host: String,
    #[serde(default)]
    namespaces: Vec<String>,
}

pub async fn registry_login(
    profiles: &mut ProfileStore,
    args: &ProfileArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<()> {
    let profile = profiles.resolve(args.profile.as_deref());
    let platform = profile.platform_url.as_deref().ok_or_else(|| {
        anyhow!(
            "profile {:?} has no configured platform; run `{}`",
            profile.name,
            profile.login_hint()
        )
    })?;
    write_heading(stdout, style, "Container registry")?;
    write_key_value(stdout, style, "Profile", &profile.name)?;
    write_key_value(stdout, style, "Platform", platform)?;
    writeln!(stdout)?;
    writeln!(stdout, "{} Refreshing session…", style.muted("◌"))?;
    stdout.flush()?;

    let client = http_client()?;
    let session = refresh_session(&client, profiles, &profile).await?;

    writeln!(stdout, "{} Looking up registry…", style.muted("◌"))?;
    stdout.flush()?;
    let info: RegistryInfo = get_json(
        &client,
        &format!("{platform}/v1/registry/info"),
        Some(&session.credentials.access_token),
        "fetching registry info",
    )
    .await?;

    let host = registry_host_key(&info.host);
    profiles.set_registry_host(&profile.name, &host)?;
    let docker_config = docker_config_path()?;
    register_credential_helper(&docker_config, &host)?;

    writeln!(
        stdout,
        "{}",
        style.success(&format!(
            "Docker will authenticate to {host} with the rlmesh credential helper"
        ))
    )?;
    writeln!(
        stdout,
        "  {}",
        style.muted(&format!("Registered in {}", docker_config.display()))
    )?;
    if !helper_on_path() {
        writeln!(
            stdout,
            "  {}",
            style.yellow(&format!(
                "Warning: {HELPER_BINARY} is not on PATH; docker cannot invoke it. \
                 Install it alongside the rlmesh binary."
            ))
        )?;
    }
    if !info.namespaces.is_empty() {
        writeln!(stdout)?;
        writeln!(stdout, "{}", style.bold("Push targets"))?;
        for namespace in info.namespaces {
            writeln!(stdout, "  • {host}/{namespace}")?;
        }
    }
    Ok(())
}

/// Implements docker's credential-helper protocol: the operation arrives as
/// argv, its payload on stdin, and the response goes to stdout. `get` mints a
/// fresh short-lived access token from the profile's refresh token, so docker
/// never holds a credential that outlives a session.
pub async fn credential_helper(
    profiles: &mut ProfileStore,
    args: &CredentialHelperArgs,
    stdout: &mut impl Write,
) -> Result<i32> {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .context("reading credential-helper input")?;

    match args.operation.as_str() {
        "get" => credential_get(profiles, input.trim(), stdout).await,
        // Credentials are managed by `rlmesh login`, not by docker: store and
        // erase are acknowledged and ignored.
        "store" | "erase" => Ok(0),
        "list" => credential_list(profiles, stdout),
        other => bail!("unknown credential-helper operation {other:?}"),
    }
}

#[derive(Serialize)]
struct HelperCredentials {
    #[serde(rename = "ServerURL")]
    server_url: String,
    #[serde(rename = "Username")]
    username: String,
    #[serde(rename = "Secret")]
    secret: String,
}

async fn credential_get(
    profiles: &mut ProfileStore,
    server_url: &str,
    stdout: &mut impl Write,
) -> Result<i32> {
    let Some(profile) = profiles.resolve_registry(&registry_host_key(server_url)) else {
        writeln!(stdout, "{CREDENTIALS_NOT_FOUND}")?;
        return Ok(1);
    };

    let client = http_client()?;
    // docker surfaces the helper's stdout as the failure reason, so errors
    // are reported there instead of propagating to stderr.
    let session = match refresh_session(&client, profiles, &profile).await {
        Ok(session) => session,
        Err(error) => {
            writeln!(stdout, "{error:#}")?;
            return Ok(1);
        }
    };

    let response = HelperCredentials {
        server_url: server_url.to_owned(),
        username: registry_username(session.identity.as_ref()),
        secret: session.credentials.access_token,
    };
    writeln!(
        stdout,
        "{}",
        serde_json::to_string(&response).context("serializing credential response")?
    )?;
    Ok(0)
}

fn credential_list(profiles: &ProfileStore, stdout: &mut impl Write) -> Result<i32> {
    let hosts: BTreeMap<&str, String> = profiles
        .registry_profiles()
        .filter_map(|profile| {
            profile
                .registry_host
                .as_deref()
                .map(|host| (host, registry_username(profile.identity.as_ref())))
        })
        .collect();
    writeln!(
        stdout,
        "{}",
        serde_json::to_string(&hosts).context("serializing registry list")?
    )?;
    Ok(0)
}

fn registry_username(identity: Option<&Identity>) -> String {
    identity
        .map(|identity| identity.user_id.as_str())
        .filter(|user_id| !user_id.is_empty())
        .unwrap_or(HELPER_SUFFIX)
        .to_owned()
}

/// Normalizes a registry reference (bare host, host:port, or full URL) to the
/// key docker uses in credHelpers and passes back to the helper.
fn registry_host_key(value: &str) -> String {
    let value = value.trim();
    let value = ["https://", "http://"]
        .iter()
        .find_map(|scheme| {
            value
                .get(..scheme.len())
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case(scheme))
                .then(|| &value[scheme.len()..])
        })
        .unwrap_or(value);
    value
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase()
}

fn docker_config_path() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("DOCKER_CONFIG") {
        return Ok(PathBuf::from(dir).join("config.json"));
    }
    let home = dirs::home_dir().context("cannot locate the home directory")?;
    Ok(home.join(".docker").join("config.json"))
}

fn register_credential_helper(path: &Path, host: &str) -> Result<()> {
    let mut config: serde_json::Value = match fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing docker config {}", path.display()))?,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => serde_json::json!({}),
        Err(err) => {
            return Err(err).with_context(|| format!("reading docker config {}", path.display()));
        }
    };

    config
        .as_object_mut()
        .with_context(|| format!("docker config {} is not a JSON object", path.display()))?
        .entry("credHelpers")
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .with_context(|| {
            format!(
                "credHelpers in docker config {} is not a JSON object",
                path.display()
            )
        })?
        .insert(
            host.to_owned(),
            serde_json::Value::String(HELPER_SUFFIX.to_owned()),
        );

    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)
            .with_context(|| format!("creating docker config directory {}", dir.display()))?;
    }
    let mut bytes = serde_json::to_vec_pretty(&config).context("serializing docker config")?;
    bytes.push(b'\n');
    fs::write(path, bytes).with_context(|| format!("writing docker config {}", path.display()))
}

fn helper_on_path() -> bool {
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&paths).any(|dir| {
        let candidate = dir.join(HELPER_BINARY);
        candidate.is_file() || candidate.with_extension("exe").is_file()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_registry_references_to_host_keys() {
        assert_eq!(
            registry_host_key("registry.rlmesh.dev"),
            "registry.rlmesh.dev"
        );
        assert_eq!(
            registry_host_key("https://Registry.RLMesh.dev/v2/"),
            "registry.rlmesh.dev"
        );
        assert_eq!(registry_host_key("http://localhost:5000"), "localhost:5000");
        assert_eq!(
            registry_host_key("  registry.rlmesh.dev/ns  "),
            "registry.rlmesh.dev"
        );
    }

    #[test]
    fn registers_helper_preserving_existing_docker_config() {
        let dir = std::env::temp_dir().join(format!("rlmesh-docker-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("config.json");

        register_credential_helper(&path, "registry.rlmesh.dev").unwrap();
        let config: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(config["credHelpers"]["registry.rlmesh.dev"], "rlmesh");

        fs::write(
            &path,
            r#"{"auths":{"ghcr.io":{"auth":"abc"}},"credHelpers":{"gcr.io":"gcloud"}}"#,
        )
        .unwrap();
        register_credential_helper(&path, "registry.rlmesh.dev").unwrap();
        let config: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(config["auths"]["ghcr.io"]["auth"], "abc");
        assert_eq!(config["credHelpers"]["gcr.io"], "gcloud");
        assert_eq!(config["credHelpers"]["registry.rlmesh.dev"], "rlmesh");

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn credential_response_uses_docker_field_names() {
        let response = serde_json::to_value(HelperCredentials {
            server_url: "registry.rlmesh.dev".to_owned(),
            username: "user_123".to_owned(),
            secret: "token".to_owned(),
        })
        .unwrap();
        assert_eq!(response["ServerURL"], "registry.rlmesh.dev");
        assert_eq!(response["Username"], "user_123");
        assert_eq!(response["Secret"], "token");
    }
}
