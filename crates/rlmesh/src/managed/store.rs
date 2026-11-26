//! The CLI's profile store: `config.toml` (profiles, the default, each
//! profile's platform and cached identity) and the credentials, kept in the
//! OS keychain or, without one, in an owner-only `credentials.json`.

#![allow(missing_docs)]

use super::http::normalize_base_url;
use super::settings::Settings;

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fmt, fs,
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};

const APP_NAME: &str = "rlmesh";
pub(crate) const DEFAULT_PLATFORM_URL: &str = "https://api.rlmesh.dev";

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct Profile {
    #[serde(default)]
    pub platform_url: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registry_host: Option<String>,

    /// The token endpoint the profile signed in against; refreshes refuse
    /// to send the stored session to another host.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_endpoint: Option<String>,

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
    pub fn load(settings: &Settings) -> Result<Self> {
        let path = config_path(settings)?;
        match fs::read_to_string(&path) {
            Ok(text) => {
                toml::from_str(&text).with_context(|| format!("parsing config {}", path.display()))
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(err) => Err(err).with_context(|| format!("reading config {}", path.display())),
        }
    }

    fn set_registry_host(&mut self, profile: &str, host: &str) -> Result<()> {
        if !self.profiles.contains_key(profile) {
            bail!("no profile named {profile:?}");
        }
        for (name, configured_profile) in &mut self.profiles {
            if name == profile {
                configured_profile.registry_host = Some(host.to_owned());
            } else if configured_profile.registry_host.as_deref() == Some(host) {
                configured_profile.registry_host = None;
            }
        }
        Ok(())
    }

    fn save_to(&self, path: &Path) -> Result<()> {
        let text = toml::to_string_pretty(self).context("serializing config")?;
        write_private(path, text.as_bytes())
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

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq, Eq)]
#[serde(default)]
pub struct Identity {
    pub user_id: String,
    pub email: String,
    pub display_name: String,
    /// The identity provider's organization id (the `providerId` the
    /// platform reports and the id the refresh grant switches on).
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
    /// `RLMESH_API_KEY` is set; no profile session is involved.
    ApiKey,
}

impl CredentialStatus {
    /// The stable machine-readable name used by `--json` outputs.
    pub fn label(self) -> &'static str {
        match self {
            Self::SignedOut => "signed_out",
            Self::Incomplete => "incomplete",
            Self::SignedIn => "signed_in",
            Self::ApiKey => "api_key",
        }
    }

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
    pub token_endpoint: Option<String>,
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

/// The profiles and their credentials, loaded once per process.
#[derive(Clone)]
pub struct ProfileStore {
    config: Config,
    config_path: PathBuf,
    credentials: CredentialsStore,
    api_key: Option<(String, String)>,
}

impl ProfileStore {
    pub fn load(settings: &Settings) -> Result<Self> {
        Ok(Self {
            config: Config::load(settings)?,
            config_path: config_path(settings)?,
            credentials: CredentialsStore::load(settings)?,
            api_key: settings.api_key.as_ref().map(|key| {
                (
                    key.clone(),
                    settings
                        .platform_url
                        .clone()
                        .unwrap_or_else(|| DEFAULT_PLATFORM_URL.to_owned()),
                )
            }),
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The API key and platform URL from the environment, when set. Commands
    /// that talk to the platform use these instead of a profile session.
    pub fn api_key(&self) -> Option<(&str, &str)> {
        self.api_key
            .as_ref()
            .map(|(key, platform)| (key.as_str(), platform.as_str()))
    }

    fn save_config(&self) -> Result<()> {
        self.config.save_to(&self.config_path)
    }

    pub fn resolve(&self, profile_override: Option<&str>) -> ResolvedProfile {
        let name = self.config.profile_name(profile_override);
        let profile = self.config.profiles.get(&name);

        ResolvedProfile {
            platform_url: profile
                .and_then(|profile| profile.platform_url.as_deref())
                .map(normalize_base_url),
            token_endpoint: profile.and_then(|profile| profile.token_endpoint.clone()),
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
                .or_else(|| Some(DEFAULT_PLATFORM_URL.to_owned())),
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

    /// Re-reads the credential store, bypassing this process's cache, so a
    /// rotation done by another process is seen.
    pub fn reload_credentials(&mut self, profile: &str) -> Result<Option<Credentials>> {
        self.credentials.reload(profile)
    }

    /// The per-profile lock file that serializes session refreshes across
    /// processes; lives next to the credential file.
    pub fn lock_path(&self, profile: &str) -> PathBuf {
        let name: String = profile
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        self.credentials.file.with_file_name(format!("{name}.lock"))
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
        configured_profile.token_endpoint = profile.token_endpoint.clone();
        if let Some(identity) = identity {
            configured_profile.identity = Some(identity);
        }

        if self.config.default_profile.is_none() {
            self.config.default_profile = Some(profile.name.clone());
        }

        self.save_config()?;
        self.credentials.save(&profile.name, credentials)
    }

    pub fn replace_credentials(&mut self, profile: &str, credentials: &Credentials) -> Result<()> {
        self.credentials.save(profile, credentials).map(|_| ())
    }

    pub fn set_registry_host(&mut self, profile: &str, host: &str) -> Result<()> {
        self.config.set_registry_host(profile, host)?;
        self.save_config()
    }

    pub fn resolve_registry(&self, host: &str) -> Option<ResolvedProfile> {
        self.config
            .profiles
            .iter()
            .find(|(_, profile)| profile.registry_host.as_deref() == Some(host))
            .map(|(name, _)| self.resolve(Some(name)))
    }

    pub fn registry_profiles(&self) -> impl Iterator<Item = &Profile> {
        self.config
            .profiles
            .values()
            .filter(|profile| profile.registry_host.is_some())
    }

    pub fn update_identity(&mut self, profile: &str, identity: Identity) -> Result<()> {
        let configured_profile = self
            .config
            .profiles
            .get_mut(profile)
            .with_context(|| format!("no profile named {profile:?}"))?;
        configured_profile.identity = Some(identity);
        self.save_config()
    }

    pub fn logout(&mut self, profile: &str) -> Result<bool> {
        self.credentials.delete(profile)
    }

    pub fn set_default(&mut self, name: &str) -> Result<()> {
        if !self.config.profiles.contains_key(name) {
            bail!("no profile named {name:?}; run `rlmesh login --profile <name>` to create it");
        }

        self.config.default_profile = Some(name.to_owned());
        self.save_config()
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
            self.save_config()?;
        }

        Ok((profile_removed || credentials_removed, default_cleared))
    }
}

/// One generic-password item per profile under the service `rlmesh`. Each
/// method is a single keychain call: on macOS every call that reads the
/// secret can raise an access prompt.
trait Keychain: Send + Sync {
    /// The stored payload; `None` when there is none or it cannot be read.
    fn read(&self, profile: &str) -> Option<String>;
    /// Creates or replaces the item; `false` when the keychain refused.
    fn write(&self, profile: &str, payload: &str) -> bool;
    /// Whether an item was removed.
    fn delete(&self, profile: &str) -> Result<bool>;
}

struct SystemKeychain;

#[cfg(target_os = "macos")]
impl Keychain for SystemKeychain {
    fn read(&self, profile: &str) -> Option<String> {
        use security_framework::passwords::{PasswordOptions, generic_password};
        let bytes =
            generic_password(PasswordOptions::new_generic_password(APP_NAME, profile)).ok()?;
        String::from_utf8(bytes).ok()
    }

    // SecItemAdd then SecItemUpdate never decrypts the existing item, unlike
    // SecKeychainFindGenericPassword followed by a modify.
    fn write(&self, profile: &str, payload: &str) -> bool {
        security_framework::passwords::set_generic_password(APP_NAME, profile, payload.as_bytes())
            .is_ok()
    }

    fn delete(&self, profile: &str) -> Result<bool> {
        const ERR_SEC_ITEM_NOT_FOUND: i32 = -25300;
        match security_framework::passwords::delete_generic_password(APP_NAME, profile) {
            Ok(()) => Ok(true),
            Err(err) if err.code() == ERR_SEC_ITEM_NOT_FOUND => Ok(false),
            Err(err) => Err(anyhow!(err)),
        }
    }
}

#[cfg(not(target_os = "macos"))]
impl Keychain for SystemKeychain {
    fn read(&self, profile: &str) -> Option<String> {
        keyring::Entry::new(APP_NAME, profile)
            .ok()?
            .get_password()
            .ok()
    }

    fn write(&self, profile: &str, payload: &str) -> bool {
        keyring::Entry::new(APP_NAME, profile)
            .is_ok_and(|entry| entry.set_password(payload).is_ok())
    }

    fn delete(&self, profile: &str) -> Result<bool> {
        let Ok(entry) = keyring::Entry::new(APP_NAME, profile) else {
            return Ok(false);
        };
        match entry.delete_credential() {
            Ok(()) => Ok(true),
            Err(keyring::Error::NoEntry) => Ok(false),
            Err(err) => Err(anyhow!(err)),
        }
    }
}

#[derive(Clone)]
struct CredentialsStore {
    file: PathBuf,
    keychain: Option<Arc<dyn Keychain>>,
    cached_credentials: BTreeMap<String, Credentials>,
}

impl CredentialsStore {
    fn load(settings: &Settings) -> Result<Self> {
        Ok(Self {
            file: credentials_path(settings)?,
            keychain: settings
                .keychain
                .then(|| Arc::new(SystemKeychain) as Arc<dyn Keychain>),
            cached_credentials: BTreeMap::new(),
        })
    }

    fn save(&mut self, profile: &str, credentials: &Credentials) -> Result<CredentialStorage> {
        if let Some(keychain) = &self.keychain {
            let payload = serde_json::to_string(credentials).context("serializing credentials")?;
            if keychain.write(profile, &payload) {
                self.remove_file_credentials(profile)?;
                self.cached_credentials
                    .insert(profile.to_owned(), credentials.clone());
                return Ok(CredentialStorage::Keychain);
            }

            // A stale item left behind would win over the file on the next read.
            let _ = keychain.delete(profile);
        }

        self.save_file_credentials(profile, credentials)?;
        self.cached_credentials
            .insert(profile.to_owned(), credentials.clone());
        Ok(CredentialStorage::File(self.file.clone()))
    }

    fn reload(&mut self, profile: &str) -> Result<Option<Credentials>> {
        self.cached_credentials.remove(profile);
        self.get(profile)
    }

    fn get(&mut self, profile: &str) -> Result<Option<Credentials>> {
        if let Some(credentials) = self.cached_credentials.get(profile) {
            return Ok(Some(credentials.clone()));
        }

        // The keychain wins over the file fallback; an unparsable payload reads as signed out.
        if let Some(payload) = self
            .keychain
            .as_ref()
            .and_then(|keychain| keychain.read(profile))
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

        let (keychain_removed, keychain_error) = match &self.keychain {
            Some(keychain) => match keychain.delete(profile) {
                Ok(removed) => (removed, None),
                Err(err) => (false, Some(err)),
            },
            None => (false, None),
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

fn config_path(settings: &Settings) -> Result<PathBuf> {
    let dir = match &settings.config_dir {
        Some(dir) => dir.clone(),
        None => dirs::config_dir()
            .map(|path| path.join(APP_NAME))
            .context("cannot locate the user config directory")?,
    };
    Ok(dir.join("config.toml"))
}

fn credentials_path(settings: &Settings) -> Result<PathBuf> {
    let dir = match &settings.data_dir {
        Some(dir) => dir.clone(),
        None => dirs::data_local_dir()
            .map(|path| path.join(APP_NAME))
            .context("cannot locate the local data directory")?,
    };
    Ok(dir.join("credentials.json"))
}

pub(crate) fn ensure_private_dir(dir: &Path) -> Result<()> {
    fs::create_dir_all(dir)
        .with_context(|| format!("creating config directory {}", dir.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("securing config directory {}", dir.display()))?;
    }
    Ok(())
}

fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().context("file has no parent directory")?;
    ensure_private_dir(dir)?;

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
            config_path: PathBuf::new(),
            credentials: CredentialsStore {
                file: PathBuf::new(),
                keychain: None,
                cached_credentials: BTreeMap::new(),
            },
            api_key: None,
        }
    }

    #[cfg(unix)]
    #[test]
    fn write_private_keeps_credentials_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join(format!("rlmesh-cli-private-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("nested").join("credentials.json");
        let mode = |path: &Path| fs::metadata(path).unwrap().permissions().mode() & 0o777;

        write_private(&path, b"{}").unwrap();
        assert_eq!(mode(&path), 0o600);
        assert_eq!(mode(path.parent().unwrap()), 0o700);
        assert_eq!(fs::read(&path).unwrap(), b"{}");

        write_private(&path, b"{\"rotated\":true}").unwrap();
        assert_eq!(mode(&path), 0o600);
        assert_eq!(fs::read(&path).unwrap(), b"{\"rotated\":true}");

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn profile_resolution_applies_overrides_and_defaults() {
        let config = Config {
            default_profile: Some("staging".to_owned()),
            profiles: BTreeMap::from([(
                "staging".to_owned(),
                Profile {
                    platform_url: Some("staging.example.com/".to_owned()),
                    ..Profile::default()
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
    fn every_profile_defaults_to_the_hosted_platform() {
        let store = store_with(Config::default());

        assert_eq!(
            store
                .resolve_login(Some("bench"), None)
                .unwrap()
                .platform_url
                .as_deref(),
            Some(DEFAULT_PLATFORM_URL)
        );
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
            keychain: None,
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

    #[derive(Default)]
    struct RecordingKeychain {
        items: std::sync::Mutex<BTreeMap<String, String>>,
        calls: std::sync::Mutex<Vec<&'static str>>,
        refuse_writes: bool,
    }

    impl RecordingKeychain {
        fn take_calls(&self) -> Vec<&'static str> {
            std::mem::take(&mut *self.calls.lock().unwrap())
        }
    }

    impl Keychain for RecordingKeychain {
        fn read(&self, profile: &str) -> Option<String> {
            self.calls.lock().unwrap().push("read");
            self.items.lock().unwrap().get(profile).cloned()
        }

        fn write(&self, profile: &str, payload: &str) -> bool {
            self.calls.lock().unwrap().push("write");
            if !self.refuse_writes {
                self.items
                    .lock()
                    .unwrap()
                    .insert(profile.to_owned(), payload.to_owned());
            }
            !self.refuse_writes
        }

        fn delete(&self, profile: &str) -> Result<bool> {
            self.calls.lock().unwrap().push("delete");
            Ok(self.items.lock().unwrap().remove(profile).is_some())
        }
    }

    fn keychain_store(directory: &Path, keychain: &Arc<RecordingKeychain>) -> CredentialsStore {
        CredentialsStore {
            file: directory.join("credentials.json"),
            keychain: Some(keychain.clone()),
            cached_credentials: BTreeMap::new(),
        }
    }

    fn tokens(access: &str, refresh: &str) -> Credentials {
        Credentials {
            access_token: access.to_owned(),
            refresh_token: refresh.to_owned(),
        }
    }

    #[test]
    fn a_session_refresh_is_two_keychain_reads_and_one_write() {
        let directory = tempfile::tempdir().unwrap();
        let keychain = Arc::new(RecordingKeychain::default());
        let mut store = keychain_store(directory.path(), &keychain);
        store
            .save("default", &tokens("access_old", "refresh_old"))
            .unwrap();
        assert_eq!(keychain.take_calls(), ["write"]);

        let mut store = keychain_store(directory.path(), &keychain);
        assert_eq!(
            store.get("default").unwrap().unwrap().refresh_token,
            "refresh_old"
        );
        store.get("default").unwrap();
        assert_eq!(
            store.reload("default").unwrap().unwrap().refresh_token,
            "refresh_old"
        );
        assert_eq!(
            store
                .save("default", &tokens("access_new", "refresh_new"))
                .unwrap(),
            CredentialStorage::Keychain
        );
        assert_eq!(
            store.get("default").unwrap().unwrap().access_token,
            "access_new"
        );
        assert_eq!(keychain.take_calls(), ["read", "read", "write"]);

        assert!(store.delete("default").unwrap());
        assert!(!store.delete("default").unwrap());
        assert_eq!(keychain.take_calls(), ["delete", "delete"]);
        assert!(!store.file.exists());
    }

    #[test]
    fn a_refused_keychain_write_falls_back_to_the_file() {
        let directory = tempfile::tempdir().unwrap();
        let keychain = Arc::new(RecordingKeychain {
            refuse_writes: true,
            ..RecordingKeychain::default()
        });
        keychain
            .items
            .lock()
            .unwrap()
            .insert("default".to_owned(), "stale".to_owned());
        let mut store = keychain_store(directory.path(), &keychain);

        assert_eq!(
            store.save("default", &tokens("access", "refresh")).unwrap(),
            CredentialStorage::File(store.file.clone())
        );
        assert_eq!(keychain.take_calls(), ["write", "delete"]);
        let mut fresh = keychain_store(directory.path(), &keychain);
        assert_eq!(
            fresh.get("default").unwrap().unwrap().refresh_token,
            "refresh"
        );
    }

    #[test]
    fn profile_toml_contains_identity_but_not_credentials() {
        let config = Config {
            default_profile: Some("default".to_owned()),
            profiles: BTreeMap::from([(
                "default".to_owned(),
                Profile {
                    platform_url: Some(DEFAULT_PLATFORM_URL.to_owned()),
                    registry_host: None,
                    token_endpoint: Some("https://id.example.com/token".to_owned()),
                    identity: Some(Identity {
                        user_id: "user_123".to_owned(),
                        email: "dev@example.com".to_owned(),
                        display_name: "Dev User".to_owned(),
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

    #[test]
    fn registry_host_moves_to_the_latest_profile() {
        let mut store = store_with(Config {
            default_profile: Some("default".to_owned()),
            profiles: BTreeMap::from([
                ("default".to_owned(), Profile::default()),
                ("bench".to_owned(), Profile::default()),
            ]),
        });
        store
            .config
            .set_registry_host("default", "registry.rlmesh.dev")
            .unwrap();
        store
            .config
            .set_registry_host("bench", "registry.rlmesh.dev")
            .unwrap();

        assert_eq!(
            store.resolve_registry("registry.rlmesh.dev").unwrap().name,
            "bench"
        );
        assert!(store.config.profiles["default"].registry_host.is_none());
        assert!(
            store
                .config
                .set_registry_host("missing", "registry.rlmesh.dev")
                .is_err()
        );
    }
}
