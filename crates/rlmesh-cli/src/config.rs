use crate::helpers::normalize_base_url;

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fmt, fs,
    io::Write,
    path::{Path, PathBuf},
};

const APP_NAME: &str = "rlmesh";
const DEFAULT_PLATFORM_URL: &str = "https://api.rlmesh.dev";

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct Profile {
    #[serde(default)]
    pub platform_url: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity: Option<Identity>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct Config {
    #[serde(default)]
    pub default_profile: Option<String>,

    #[serde(default)]
    pub profiles: BTreeMap<String, Profile>,
}

impl Config {
    pub fn load() -> Result<Self> {
        let path = config_path()?;
        match fs::read_to_string(&path) {
            Ok(text) => {
                toml::from_str(&text).with_context(|| format!("parsing config {}", path.display()))
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(err) => Err(err).with_context(|| format!("reading config {}", path.display())),
        }
    }

    pub fn save(&self) -> Result<()> {
        let path = config_path()?;
        let text = toml::to_string_pretty(self).context("serializing config")?;
        write_private(&path, text.as_bytes())
            .with_context(|| format!("writing config {}", path.display()))
    }

    pub fn profile_name(&self, flag: Option<&str>) -> String {
        flag.map(str::to_owned)
            .or_else(|| self.default_profile.clone())
            .unwrap_or_else(|| "default".to_owned())
    }

    fn is_effective_default(&self, name: &str) -> bool {
        self.default_profile.as_deref().unwrap_or("default") == name
    }
}

// Lenient on missing fields so a config written by an older CLI still loads.
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq, Eq)]
#[serde(default)]
pub struct Identity {
    pub user_id: String,
    pub email: String,
    pub first_name: String,
    pub last_name: String,
    pub organization_id: String,
    pub organization_name: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Credentials {
    pub access_token: String,
    pub refresh_token: String,
}

impl fmt::Debug for Credentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Credentials")
            .field("access_token", &"[REDACTED]")
            .field("refresh_token", &"[REDACTED]")
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CredentialStorage {
    Keychain,
    File(PathBuf),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialStatus {
    SignedOut,
    Incomplete,
    SignedIn,
}

impl CredentialStatus {
    fn from_credentials(credentials: Option<&Credentials>) -> Self {
        let Some(credentials) = credentials else {
            return Self::SignedOut;
        };

        match (
            credentials.access_token.trim().is_empty(),
            credentials.refresh_token.trim().is_empty(),
        ) {
            (false, false) => Self::SignedIn,
            _ => Self::Incomplete,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedProfile {
    pub name: String,
    pub platform_url: Option<String>,
    pub identity: Option<Identity>,
    pub is_default: bool,
}

impl ResolvedProfile {
    pub fn login_hint(&self) -> String {
        if self.is_default {
            "rlmesh login".to_owned()
        } else if self
            .platform_url
            .as_deref()
            .is_some_and(|platform| !platform.is_empty())
        {
            format!("rlmesh login --profile {}", self.name)
        } else {
            format!("rlmesh login --profile {} --platform <url>", self.name)
        }
    }
}

pub struct ProfileStore {
    config: Config,
    credentials: CredentialsStore,
}

impl ProfileStore {
    pub fn load() -> Result<Self> {
        Ok(Self {
            config: Config::load()?,
            credentials: CredentialsStore::load()?,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn resolve(&self, profile_override: Option<&str>) -> ResolvedProfile {
        let name = self.config.profile_name(profile_override);
        let profile = self.config.profiles.get(&name);

        ResolvedProfile {
            platform_url: profile
                .and_then(|profile| profile.platform_url.as_deref())
                .map(normalize_base_url),
            identity: profile.and_then(|profile| profile.identity.clone()),
            is_default: self.config.is_effective_default(&name),
            name,
        }
    }

    pub fn resolve_login(
        &self,
        profile_override: Option<&str>,
        platform_override: Option<&str>,
    ) -> Result<ResolvedProfile> {
        let mut profile = self.resolve(profile_override);
        profile.platform_url = match platform_override {
            Some(platform) => Some(normalize_base_url(platform)),
            None => profile
                .platform_url
                .take()
                .or_else(|| profile.is_default.then(|| DEFAULT_PLATFORM_URL.to_owned())),
        };

        if profile.platform_url.as_deref().is_none_or(str::is_empty) {
            bail!(
                "profile {:?} has no platform; run `{}`",
                profile.name,
                profile.login_hint()
            );
        }

        Ok(profile)
    }

    pub fn credential_status(&mut self, profile: &str) -> Result<CredentialStatus> {
        let credentials = self.credentials.get(profile)?;
        Ok(CredentialStatus::from_credentials(credentials.as_ref()))
    }

    pub fn credentials(&mut self, profile: &str) -> Result<Option<Credentials>> {
        self.credentials.get(profile)
    }

    pub fn record_login(
        &mut self,
        profile: &ResolvedProfile,
        identity: Option<Identity>,
        credentials: &Credentials,
    ) -> Result<CredentialStorage> {
        let platform_url = profile
            .platform_url
            .clone()
            .context("cannot record a login without a platform")?;
        let configured_profile = self
            .config
            .profiles
            .entry(profile.name.clone())
            .or_default();
        configured_profile.platform_url = Some(platform_url);
        configured_profile.identity = identity;

        if self.config.default_profile.is_none() {
            self.config.default_profile = Some(profile.name.clone());
        }

        self.config.save()?;
        self.credentials.save(&profile.name, credentials)
    }

    pub fn replace_credentials(&mut self, profile: &str, credentials: &Credentials) -> Result<()> {
        self.credentials.save(profile, credentials).map(|_| ())
    }

    pub fn update_identity(&mut self, profile: &str, identity: Identity) -> Result<()> {
        let configured_profile = self
            .config
            .profiles
            .get_mut(profile)
            .with_context(|| format!("no profile named {profile:?}"))?;
        configured_profile.identity = Some(identity);
        self.config.save()
    }

    pub fn logout(&mut self, profile: &str) -> Result<bool> {
        self.credentials.delete(profile)
    }

    pub fn set_default(&mut self, name: &str) -> Result<()> {
        if !self.config.profiles.contains_key(name) {
            bail!("no profile named {name:?}; run `rlmesh login --profile <name>` to create it");
        }

        self.config.default_profile = Some(name.to_owned());
        self.config.save()
    }

    pub fn remove(&mut self, name: &str) -> Result<(bool, bool)> {
        let credentials_removed = self.credentials.delete(name)?;
        let profile_removed = self.config.profiles.remove(name).is_some();
        let default_cleared =
            profile_removed && self.config.default_profile.as_deref() == Some(name);

        if default_cleared {
            self.config.default_profile = None;
        }
        if profile_removed {
            self.config.save()?;
        }

        Ok((profile_removed || credentials_removed, default_cleared))
    }
}

pub struct CredentialsStore {
    file: PathBuf,
    cached_credentials: BTreeMap<String, Credentials>,
}

impl CredentialsStore {
    fn load() -> Result<Self> {
        Ok(Self {
            file: credentials_path()?,
            cached_credentials: BTreeMap::new(),
        })
    }

    fn save(&mut self, profile: &str, credentials: &Credentials) -> Result<CredentialStorage> {
        if let Ok(entry) = keyring::Entry::new(APP_NAME, profile) {
            let payload = serde_json::to_string(credentials).context("serializing credentials")?;
            if entry.set_password(&payload).is_ok() {
                self.remove_file_credentials(profile)?;
                self.cached_credentials
                    .insert(profile.to_owned(), credentials.clone());
                return Ok(CredentialStorage::Keychain);
            }

            let _ = entry.delete_credential();
        }

        self.save_file_credentials(profile, credentials)?;
        self.cached_credentials
            .insert(profile.to_owned(), credentials.clone());
        Ok(CredentialStorage::File(self.file.clone()))
    }

    fn get(&mut self, profile: &str) -> Result<Option<Credentials>> {
        if let Some(credentials) = self.cached_credentials.get(profile) {
            return Ok(Some(credentials.clone()));
        }

        // The keychain wins over the file fallback: `save` writes the
        // keychain and only then removes the file entry, so a file entry that
        // outlives a failed removal is stale. A payload that doesn't parse
        // (an older CLI's schema) reads as signed out rather than failing
        // every profile-touching command; other keychain failures (locked
        // keychain, denied prompt) fall through to the file.
        if let Ok(entry) = keyring::Entry::new(APP_NAME, profile)
            && let Ok(payload) = entry.get_password()
        {
            let credentials = serde_json::from_str::<Credentials>(&payload).ok();
            if let Some(credentials) = &credentials {
                self.cached_credentials
                    .insert(profile.to_owned(), credentials.clone());
            }
            return Ok(credentials);
        }

        let credentials_by_profile = self.read_credentials_file()?;
        if let Some(credentials) = credentials_by_profile.get(profile) {
            self.cached_credentials
                .insert(profile.to_owned(), credentials.clone());
            return Ok(Some(credentials.clone()));
        }

        Ok(None)
    }

    fn delete(&mut self, profile: &str) -> Result<bool> {
        self.cached_credentials.remove(profile);

        let (keychain_removed, keychain_error) = match keyring::Entry::new(APP_NAME, profile) {
            Ok(entry) => match entry.delete_credential() {
                Ok(()) => (true, None),
                Err(keyring::Error::NoEntry) => (false, None),
                Err(err) => (false, Some(anyhow!(err))),
            },
            Err(_) => (false, None),
        };

        let file_removed = self.remove_file_credentials(profile)?;

        if let Some(err) = keychain_error {
            return Err(err)
                .with_context(|| format!("deleting keychain credentials for profile {profile:?}"));
        }

        Ok(keychain_removed || file_removed)
    }

    fn save_file_credentials(&self, profile: &str, credentials: &Credentials) -> Result<()> {
        let mut credentials_by_profile = self.read_credentials_file()?;
        credentials_by_profile.insert(profile.to_owned(), credentials.clone());
        self.write_credentials_file(credentials_by_profile)
    }

    fn remove_file_credentials(&self, profile: &str) -> Result<bool> {
        let mut credentials_by_profile = self.read_credentials_file()?;
        let removed = credentials_by_profile.remove(profile).is_some();
        if removed {
            self.write_credentials_file(credentials_by_profile)?;
        }
        Ok(removed)
    }

    fn write_credentials_file(
        &self,
        credentials_by_profile: BTreeMap<String, Credentials>,
    ) -> Result<()> {
        if credentials_by_profile.is_empty() {
            return self.remove_credentials_file();
        }

        let bytes = serde_json::to_vec_pretty(&credentials_by_profile)
            .context("serializing credentials")?;
        write_private(&self.file, &bytes)
            .with_context(|| format!("writing credential file {}", self.file.display()))
    }

    fn remove_credentials_file(&self) -> Result<()> {
        match fs::remove_file(&self.file) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err)
                .with_context(|| format!("deleting credential file {}", self.file.display())),
        }
    }

    fn read_credentials_file(&self) -> Result<BTreeMap<String, Credentials>> {
        match fs::read(&self.file) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .with_context(|| format!("parsing credential file {}", self.file.display())),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
            Err(err) => {
                Err(err).with_context(|| format!("reading credential file {}", self.file.display()))
            }
        }
    }
}

fn config_path() -> Result<PathBuf> {
    let dir = dirs::config_dir()
        .map(|path| path.join(APP_NAME))
        .context("cannot locate the user config directory")?;

    Ok(dir.join("config.toml"))
}

fn credentials_path() -> Result<PathBuf> {
    let dir = dirs::data_local_dir()
        .map(|path| path.join(APP_NAME))
        .context("cannot locate the local data directory")?;

    Ok(dir.join("credentials.json"))
}

fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().context("file has no parent directory")?;

    fs::create_dir_all(dir)
        .with_context(|| format!("creating config directory {}", dir.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("securing config directory {}", dir.display()))?;
    }

    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));

    let mut options = fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }

    let mut file = options
        .open(&temporary)
        .with_context(|| format!("opening temporary file {}", temporary.display()))?;

    file.write_all(bytes)
        .with_context(|| format!("writing temporary file {}", temporary.display()))?;

    file.sync_all()
        .with_context(|| format!("syncing temporary file {}", temporary.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        file.set_permissions(fs::Permissions::from_mode(0o600))
            .with_context(|| format!("securing temporary file {}", temporary.display()))?;
    }

    drop(file);

    fs::rename(&temporary, path).with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with(config: Config) -> ProfileStore {
        ProfileStore {
            config,
            credentials: CredentialsStore {
                file: PathBuf::new(),
                cached_credentials: BTreeMap::new(),
            },
        }
    }

    #[test]
    fn profile_resolution_applies_overrides_and_defaults() {
        let config = Config {
            default_profile: Some("staging".to_owned()),
            profiles: BTreeMap::from([(
                "staging".to_owned(),
                Profile {
                    platform_url: Some("staging.example.com/".to_owned()),
                    identity: None,
                },
            )]),
        };
        let store = store_with(config);

        let resolved = store.resolve_login(None, None).unwrap();
        assert_eq!(resolved.name, "staging");
        assert_eq!(
            resolved.platform_url.as_deref(),
            Some("https://staging.example.com")
        );

        let overridden = store
            .resolve_login(Some("development"), Some("localhost:3000"))
            .unwrap();
        assert_eq!(overridden.name, "development");
        assert_eq!(
            overridden.platform_url.as_deref(),
            Some("http://localhost:3000")
        );
    }

    #[test]
    fn named_profile_does_not_inherit_the_hosted_platform() {
        let store = store_with(Config::default());

        assert!(store.resolve_login(Some("staging"), None).is_err());
        assert_eq!(
            store
                .resolve_login(None, None)
                .unwrap()
                .platform_url
                .as_deref(),
            Some(DEFAULT_PLATFORM_URL)
        );
    }

    #[test]
    fn file_credentials_replace_rotated_tokens() {
        let directory =
            std::env::temp_dir().join(format!("rlmesh-cli-credential-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        let store = CredentialsStore {
            file: directory.join("credentials.json"),
            cached_credentials: BTreeMap::new(),
        };

        store
            .save_file_credentials(
                "default",
                &Credentials {
                    access_token: "access_old".to_owned(),
                    refresh_token: "refresh_old".to_owned(),
                },
            )
            .unwrap();
        store
            .save_file_credentials(
                "default",
                &Credentials {
                    access_token: "access_new".to_owned(),
                    refresh_token: "refresh_new".to_owned(),
                },
            )
            .unwrap();

        let credentials = store
            .read_credentials_file()
            .unwrap()
            .remove("default")
            .unwrap();
        assert_eq!(credentials.access_token, "access_new");
        assert_eq!(credentials.refresh_token, "refresh_new");

        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn profile_toml_contains_identity_but_not_credentials() {
        let config = Config {
            default_profile: Some("default".to_owned()),
            profiles: BTreeMap::from([(
                "default".to_owned(),
                Profile {
                    platform_url: Some(DEFAULT_PLATFORM_URL.to_owned()),
                    identity: Some(Identity {
                        user_id: "user_123".to_owned(),
                        email: "dev@example.com".to_owned(),
                        first_name: "Dev".to_owned(),
                        last_name: "User".to_owned(),
                        organization_id: "org_123".to_owned(),
                        organization_name: "Dev Org".to_owned(),
                    }),
                },
            )]),
        };

        let text = toml::to_string(&config).unwrap();
        assert!(text.contains("user_123"));
        assert!(!text.contains("access_token"));
        assert!(!text.contains("refresh_token"));
    }
}
