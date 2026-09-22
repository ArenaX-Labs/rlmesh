use crate::cli::ProfileListArgs;
use crate::config::ProfileStore;
use crate::render::{Style, render_profiles};

use anyhow::{Result, bail};
use serde_json::{Value, json};
use std::io::Write;

pub fn profile_list(
    profiles: &mut ProfileStore,
    args: &ProfileListArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<()> {
    let config = profiles.config().clone();
    if args.json {
        let default_profile = config.default_profile.as_deref().unwrap_or("default");
        let mut rows = Vec::with_capacity(config.profiles.len());
        for (name, profile) in &config.profiles {
            rows.push(json!({
                "name": name,
                "platform": profile.platform_url,
                "status": profiles.credential_status(name)?.label(),
                "default": name == default_profile,
            }));
        }
        writeln!(
            stdout,
            "{}",
            serde_json::to_string_pretty(&Value::Array(rows))?
        )?;
        return Ok(());
    }
    let output = render_profiles(&config, |name| profiles.credential_status(name), style)?;
    write!(stdout, "{output}")?;
    Ok(())
}

pub fn profile_use(
    profiles: &mut ProfileStore,
    name: &str,
    stdout: &mut impl Write,
    style: Style,
) -> Result<()> {
    profiles.set_default(name)?;
    writeln!(
        stdout,
        "{}",
        style.success(&format!("Using profile {name:?} by default"))
    )?;
    Ok(())
}

pub fn profile_remove(
    profiles: &mut ProfileStore,
    name: &str,
    stdout: &mut impl Write,
    style: Style,
) -> Result<()> {
    let (removed, default_cleared) = profiles.remove(name)?;
    if !removed {
        bail!("no profile named {name:?}");
    }

    writeln!(
        stdout,
        "{}",
        style.success(&format!("Removed profile {name:?}"))
    )?;
    if default_cleared && !profiles.config().profiles.is_empty() {
        writeln!(
            stdout,
            "  {}",
            style.muted("Choose a new default with `rlmesh profile use <name>`.")
        )?;
    }
    Ok(())
}
