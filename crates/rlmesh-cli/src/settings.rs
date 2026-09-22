//! Process-level settings the CLI reads once at startup: where its files
//! live, whether the OS keychain is used, and an API key that bypasses
//! profiles altogether.

use crate::helpers::normalize_base_url;

use std::fmt;
use std::path::PathBuf;

#[derive(Clone)]
pub struct Settings {
    /// Overrides the config directory (`RLMESH_CONFIG_DIR`).
    pub config_dir: Option<PathBuf>,
    /// Overrides the data directory holding credentials (`RLMESH_DATA_DIR`).
    pub data_dir: Option<PathBuf>,
    /// Whether to store credentials in the OS keychain when one is
    /// available (`RLMESH_KEYCHAIN=off` disables it; the 0600 file is used).
    pub keychain: bool,
    /// A platform API key (`RLMESH_API_KEY`); `token`, `eval`, and `whoami`
    /// use it instead of a signed-in profile.
    pub api_key: Option<String>,
    /// The platform the API key belongs to (`RLMESH_PLATFORM_URL`).
    pub platform_url: Option<String>,
}

impl fmt::Debug for Settings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Settings")
            .field("config_dir", &self.config_dir)
            .field("data_dir", &self.data_dir)
            .field("keychain", &self.keychain)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("platform_url", &self.platform_url)
            .finish()
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            config_dir: None,
            data_dir: None,
            keychain: true,
            api_key: None,
            platform_url: None,
        }
    }
}

impl Settings {
    pub fn from_env() -> Self {
        let var = |name: &str| {
            std::env::var(name)
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        };
        Self {
            config_dir: var("RLMESH_CONFIG_DIR").map(PathBuf::from),
            data_dir: var("RLMESH_DATA_DIR").map(PathBuf::from),
            keychain: !var("RLMESH_KEYCHAIN").is_some_and(|value| is_off(&value)),
            api_key: var("RLMESH_API_KEY"),
            platform_url: var("RLMESH_PLATFORM_URL")
                .map(|value| normalize_base_url(&value))
                .filter(|value| !value.is_empty()),
        }
    }

    /// Settings confined to the given directories with the keychain off:
    /// what a test harness or a sandboxed CI job wants.
    pub fn isolated(config_dir: impl Into<PathBuf>, data_dir: impl Into<PathBuf>) -> Self {
        Self {
            config_dir: Some(config_dir.into()),
            data_dir: Some(data_dir.into()),
            keychain: false,
            ..Self::default()
        }
    }

    pub fn with_api_key(mut self, api_key: &str, platform_url: Option<&str>) -> Self {
        self.api_key = Some(api_key.to_owned());
        self.platform_url = platform_url.map(normalize_base_url);
        self
    }
}

fn is_off(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "0" | "off" | "false" | "no"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keychain_switch_accepts_the_usual_spellings() {
        for value in ["0", "off", "OFF", "false", "No"] {
            assert!(is_off(value), "{value}");
        }
        for value in ["1", "on", "true", "yes", ""] {
            assert!(!is_off(value), "{value}");
        }
    }

    #[test]
    fn isolated_settings_keep_the_keychain_out() {
        let settings =
            Settings::isolated("/tmp/c", "/tmp/d").with_api_key(" key ", Some("api.example.com/"));
        assert!(!settings.keychain);
        assert_eq!(
            settings.config_dir.as_deref(),
            Some(std::path::Path::new("/tmp/c"))
        );
        assert_eq!(settings.api_key.as_deref(), Some(" key "));
        assert_eq!(
            settings.platform_url.as_deref(),
            Some("https://api.example.com")
        );
    }
}
