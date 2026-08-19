use crate::auth::refresh_session;
use crate::cli::ProfileArgs;
use crate::config::ProfileStore;
use crate::helpers::{get_json, http_client};
use crate::render::{Style, write_heading, write_key_value};

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use std::io::Write;
use std::process::{Command, Stdio};

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
    let username = session
        .identity
        .as_ref()
        .map(|identity| identity.user_id.as_str())
        .filter(|user_id| !user_id.is_empty())
        .unwrap_or("rlmesh");
    docker_login(&info.host, username, &session.credentials.access_token)?;

    writeln!(
        stdout,
        "{}",
        style.success(&format!("Docker is signed in to {}", info.host))
    )?;
    if !info.namespaces.is_empty() {
        writeln!(stdout)?;
        writeln!(stdout, "{}", style.bold("Push targets"))?;
        for namespace in info.namespaces {
            writeln!(stdout, "  • {}/{namespace}", info.host)?;
        }
    }
    Ok(())
}

fn docker_login(host: &str, username: &str, access_token: &str) -> Result<()> {
    let mut child = Command::new("docker")
        .arg("login")
        .arg(host)
        .arg("--username")
        .arg(username)
        .arg("--password-stdin")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("launching docker (is it installed and on PATH?)")?;

    child
        .stdin
        .take()
        .context("capturing docker stdin")?
        .write_all(access_token.as_bytes())
        .context("passing credentials to docker")?;

    let output = child
        .wait_with_output()
        .context("waiting for docker login")?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr);
        let detail = detail.trim();
        if detail.is_empty() {
            bail!("docker login failed with {}", output.status);
        }
        bail!("docker login failed: {detail}");
    }
    Ok(())
}
