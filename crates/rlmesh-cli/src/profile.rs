use crate::config::ProfileStore;
use crate::render::{Style, render_profiles};

use anyhow::{Result, bail};
use std::io::Write;

pub fn profile_list(
    profiles: &mut ProfileStore,
    stdout: &mut impl Write,
    style: Style,
) -> Result<()> {
    let config = profiles.config().clone();
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
