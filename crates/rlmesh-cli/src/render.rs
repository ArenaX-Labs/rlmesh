use crate::config::{Config, CredentialStatus};

use anyhow::Result;
use std::io::Write;

#[derive(Clone, Copy)]
pub(crate) struct Style {
    color: bool,
    interactive: bool,
}

impl Style {
    pub(crate) fn for_terminal(is_terminal: bool) -> Self {
        Self {
            color: is_terminal && color_enabled(),
            interactive: is_terminal,
        }
    }

    pub(crate) fn interactive(self) -> bool {
        self.interactive
    }

    pub(crate) fn bold(self, value: &str) -> String {
        self.paint("1", value)
    }

    pub(crate) fn muted(self, value: &str) -> String {
        self.paint("2", value)
    }

    pub(crate) fn cyan(self, value: &str) -> String {
        self.paint("36", value)
    }

    pub(crate) fn green(self, value: &str) -> String {
        self.paint("32", value)
    }

    pub(crate) fn yellow(self, value: &str) -> String {
        self.paint("33", value)
    }

    pub(crate) fn red_bold(self, value: &str) -> String {
        self.paint("1;31", value)
    }

    pub(crate) fn success(self, message: &str) -> String {
        format!("{} {message}", self.green("✓"))
    }

    pub(crate) fn status(self, status: CredentialStatus) -> String {
        match status {
            CredentialStatus::SignedIn => self.green("● signed in"),
            CredentialStatus::Incomplete => self.yellow("◐ incomplete"),
            CredentialStatus::SignedOut => self.muted("○ signed out"),
        }
    }

    fn paint(self, code: &str, value: &str) -> String {
        if self.color {
            format!("\x1b[{code}m{value}\x1b[0m")
        } else {
            value.to_owned()
        }
    }
}

pub(crate) fn write_error(
    stderr: &mut impl Write,
    style: Style,
    error: &anyhow::Error,
) -> Result<()> {
    writeln!(stderr, "{} {error:#}", style.red_bold("Error:"))?;
    Ok(())
}

pub(crate) fn write_heading(output: &mut impl Write, style: Style, heading: &str) -> Result<()> {
    writeln!(output, "{}", style.bold(heading))?;
    Ok(())
}

pub(crate) fn write_key_value(
    output: &mut impl Write,
    style: Style,
    label: &str,
    value: &str,
) -> Result<()> {
    writeln!(
        output,
        "  {}  {value}",
        style.muted(&format!("{label:<12}"))
    )?;
    Ok(())
}

pub(crate) fn render_profiles(
    config: &Config,
    mut credential_status: impl FnMut(&str) -> Result<CredentialStatus>,
    style: Style,
) -> Result<String> {
    if config.profiles.is_empty() {
        return Ok(format!(
            "{}\n\n  Run {} to create one.\n",
            style.muted("No profiles configured."),
            style.bold("rlmesh login")
        ));
    }

    let mut rows = Vec::with_capacity(config.profiles.len());
    for (name, profile) in &config.profiles {
        rows.push((
            name.as_str(),
            profile.platform_url.as_deref().unwrap_or("—"),
            credential_status(name)?,
        ));
    }

    let name_width = column_width("PROFILE", rows.iter().map(|(name, _, _)| *name));
    let platform_width = column_width("PLATFORM", rows.iter().map(|(_, platform, _)| *platform));
    let status_width = "● signed in"
        .chars()
        .count()
        .max("◐ incomplete".chars().count())
        .max("○ signed out".chars().count())
        .max("STATUS".chars().count());
    let default_profile = config.default_profile.as_deref().unwrap_or("default");
    let table_width = 3 + name_width + 2 + platform_width + 2 + status_width;

    let mut output = String::new();
    output.push_str(&format!("{}\n", style.bold("Profiles")));
    output.push_str(&format!(
        "   {}  {}  {}\n",
        style.muted(&format!("{:<name_width$}", "PROFILE")),
        style.muted(&format!("{:<platform_width$}", "PLATFORM")),
        style.muted(&format!("{:<status_width$}", "STATUS")),
    ));
    output.push_str(&format!("{}\n", style.muted(&"─".repeat(table_width))));

    for (name, platform, status) in rows {
        let is_default = name == default_profile;
        let marker = if is_default {
            style.yellow("◆")
        } else {
            " ".to_owned()
        };
        let name = format!("{name:<name_width$}");
        let name = if is_default {
            style.yellow(&name)
        } else {
            style.cyan(&name)
        };
        let platform = format!("{platform:<platform_width$}");
        let status = pad_styled(&style.status(status), status_width);
        output.push_str(&format!("{marker}  {name}  {platform}  {status}\n"));
    }

    output.push_str(&format!("\n{}\n", style.muted("◆ default profile")));
    Ok(output)
}

fn pad_styled(value: &str, visible_width: usize) -> String {
    let plain_width = strip_ansi(value).chars().count();
    if plain_width > visible_width {
        return value.to_owned();
    }
    format!("{value}{}", " ".repeat(visible_width - plain_width))
}

fn strip_ansi(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    while let Some(character) = chars.next() {
        if character == '\x1b' && chars.peek() == Some(&'[') {
            chars.next();
            for code in chars.by_ref() {
                if code == 'm' {
                    break;
                }
            }
        } else {
            output.push(character);
        }
    }
    output
}

fn column_width<'a>(heading: &str, values: impl Iterator<Item = &'a str>) -> usize {
    values
        .map(|value| value.chars().count())
        .max()
        .unwrap_or_default()
        .max(heading.chars().count())
}

fn color_enabled() -> bool {
    let no_color = std::env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty());
    let dumb_terminal = std::env::var_os("TERM").is_some_and(|value| value == "dumb");
    !no_color && !dumb_terminal
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Profile;
    use std::collections::BTreeMap;

    const PLAIN: Style = Style {
        color: false,
        interactive: false,
    };

    #[test]
    fn profile_columns_remain_aligned_for_different_name_lengths() {
        let config = Config {
            default_profile: Some("dev".to_owned()),
            profiles: BTreeMap::from([
                (
                    "dev".to_owned(),
                    Profile {
                        platform_url: Some("http://localhost:3000".to_owned()),
                        identity: None,
                    },
                ),
                (
                    "production".to_owned(),
                    Profile {
                        platform_url: Some("https://api.rlmesh.dev".to_owned()),
                        identity: None,
                    },
                ),
            ]),
        };

        let output = render_profiles(&config, |_| Ok(CredentialStatus::SignedOut), PLAIN).unwrap();
        let rows: Vec<&str> = output
            .lines()
            .filter(|line| line.contains("localhost") || line.contains("api.rlmesh"))
            .collect();
        let platform_columns: Vec<usize> = rows
            .iter()
            .map(|row| {
                let byte_offset = row.find("http").expect("row has a platform");
                row[..byte_offset].chars().count()
            })
            .collect();

        assert_eq!(platform_columns, vec![15, 15]);
    }

    #[test]
    fn plain_style_emits_no_escape_codes() {
        assert_eq!(PLAIN.success("Done"), "✓ Done");
        assert!(!PLAIN.status(CredentialStatus::SignedIn).contains('\x1b'));
    }

    #[test]
    fn leaves_invalid_ansi_sequences_unchanged() {
        assert_eq!(strip_ansi("\x1bx"), "\x1bx");
    }
}
