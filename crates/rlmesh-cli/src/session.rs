//! Access-token lifecycle for a signed-in profile.
//!
//! A stored access token is used until it is about to expire; only then is
//! the single-use refresh token exchanged, under a per-profile lock so two
//! concurrent commands (a parallel `docker push`, a script looping over
//! `rlmesh token`) rotate the session once instead of racing each other.

use crate::auth::{discover_sign_in_endpoints, exchange_refresh_token};
use crate::config::{Credentials, ProfileStore, ResolvedProfile, ensure_private_dir};

use anyhow::{Context, Result, bail};
use base64::Engine;
use serde::Deserialize;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// A token this close to expiry is refreshed ahead of time so a request
/// started now does not land on the server after it lapses.
pub(crate) const REFRESH_LEEWAY: Duration = Duration::from_secs(60);
// The lock guards exactly one request (the token exchange), which the HTTP
// client bounds at 10s connect + 30s total. The stale window must exceed
// that worst case, and a waiter must outlast the window so it can reclaim
// an abandoned lock instead of giving up first.
const LOCK_STALE_AFTER: Duration = Duration::from_secs(60);
const LOCK_WAIT: Duration = Duration::from_secs(90);
const LOCK_POLL: Duration = Duration::from_millis(100);

/// The one claim the CLI reads from an access token: the standard `exp`.
/// The payload is decoded without verifying the signature; the platform
/// validates tokens, the CLI only schedules refreshes. Everything else
/// about a session (the active organization included) comes from /v1/me,
/// so no provider-specific claim name ever leaks into the CLI.
#[derive(Deserialize, Default)]
pub(crate) struct Claims {
    #[serde(default)]
    pub exp: Option<u64>,
}

pub(crate) fn decode_claims(jwt: &str) -> Option<Claims> {
    let payload = jwt.split('.').nth(1)?.trim_end_matches('=');
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload)
        .ok()?;
    serde_json::from_slice(&bytes).ok()
}

pub(crate) fn access_token_expires_at(jwt: &str) -> Option<SystemTime> {
    let exp = decode_claims(jwt)?.exp?;
    // An absurd exp from a hostile or buggy endpoint is "unknown", not a panic.
    UNIX_EPOCH.checked_add(Duration::from_secs(exp))
}

/// False when the expiry is unknown (an opaque token): those are refreshed
/// on every use, which is what the CLI did before it read `exp` at all.
pub(crate) fn is_fresh_at(jwt: &str, now: SystemTime) -> bool {
    access_token_expires_at(jwt).is_some_and(|expires_at| expires_at > now + REFRESH_LEEWAY)
}

/// RFC 3339 in UTC with second precision, e.g. `2026-09-22T14:03:00Z`.
pub(crate) fn rfc3339_utc(time: SystemTime) -> String {
    let secs = time
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (hour, minute, second) = (rem / 3_600, rem % 3_600 / 60, rem % 60);

    // Civil date from days since the epoch (proleptic Gregorian).
    let z = i64::try_from(days).unwrap_or(i64::MAX / 2) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// A usable access token for the profile plus when it stops being one.
pub(crate) struct Session {
    pub credentials: Credentials,
    pub expires_at: Option<SystemTime>,
}

impl Session {
    fn from_credentials(credentials: Credentials) -> Self {
        Self {
            expires_at: access_token_expires_at(&credentials.access_token),
            credentials,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Refresh {
    /// Reuse the stored access token while it is fresh.
    IfStale,
    /// Exchange the refresh token regardless (the stored token was rejected).
    Force,
}

pub(crate) async fn ensure_fresh_session(
    client: &reqwest::Client,
    profiles: &mut ProfileStore,
    profile: &ResolvedProfile,
    refresh: Refresh,
) -> Result<Session> {
    refresh_session(client, profiles, profile, refresh, None).await
}

/// Re-issues the session against another organization the account belongs
/// to, using the platform's `organization_id` refresh-grant extension.
pub(crate) async fn switch_organization(
    client: &reqwest::Client,
    profiles: &mut ProfileStore,
    profile: &ResolvedProfile,
    organization_id: &str,
) -> Result<Session> {
    refresh_session(
        client,
        profiles,
        profile,
        Refresh::Force,
        Some(organization_id),
    )
    .await
}

async fn refresh_session(
    client: &reqwest::Client,
    profiles: &mut ProfileStore,
    profile: &ResolvedProfile,
    refresh: Refresh,
    organization_id: Option<&str>,
) -> Result<Session> {
    let credentials = stored_credentials(profiles, profile, false)?;
    if refresh == Refresh::IfStale && is_fresh_at(&credentials.access_token, SystemTime::now()) {
        return Ok(Session::from_credentials(credentials));
    }

    // Discovery (and the pin check) run before the lock so the critical
    // section is exactly one request: the token exchange itself.
    let endpoints = discover_sign_in_endpoints(client, profile).await?;
    let _lock = acquire_session_lock(&profiles.lock_path(&profile.name)).await?;
    // Another process may have rotated the session while this one waited
    // for the lock; its refresh token is the only one that still works.
    let credentials = stored_credentials(profiles, profile, true)?;
    if refresh == Refresh::IfStale && is_fresh_at(&credentials.access_token, SystemTime::now()) {
        return Ok(Session::from_credentials(credentials));
    }
    if credentials.refresh_token.trim().is_empty() {
        bail!(
            "profile {:?} has no refresh token; run `{}`",
            profile.name,
            profile.login_hint()
        );
    }

    let refreshed = exchange_refresh_token(
        client,
        &endpoints,
        &credentials.refresh_token,
        organization_id,
        &profile.login_hint(),
    )
    .await?;
    profiles
        .replace_credentials(&profile.name, &refreshed)
        .context("saving refreshed credentials")?;
    Ok(Session::from_credentials(refreshed))
}

fn stored_credentials(
    profiles: &mut ProfileStore,
    profile: &ResolvedProfile,
    reload: bool,
) -> Result<Credentials> {
    let credentials = if reload {
        profiles.reload_credentials(&profile.name)?
    } else {
        profiles.credentials(&profile.name)?
    };
    credentials.with_context(|| {
        format!(
            "profile {:?} is not signed in; run `{}`",
            profile.name,
            profile.login_hint()
        )
    })
}

/// A `create_new` lock file: atomic on every platform the CLI ships for,
/// with no extra dependency. A crashed holder is detected by age, since the
/// critical section is one bounded HTTP request.
struct SessionLock {
    path: PathBuf,
}

impl Drop for SessionLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

async fn acquire_session_lock(path: &Path) -> Result<SessionLock> {
    let dir = path.parent().context("lock file has no parent directory")?;
    ensure_private_dir(dir)?;
    let deadline = tokio::time::Instant::now() + LOCK_WAIT;

    loop {
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(mut file) => {
                let _ = writeln!(file, "{}", std::process::id());
                // The handle closes here, before the guard is handed out, so
                // the file can be removed on Windows too.
                drop(file);
                return Ok(SessionLock {
                    path: path.to_owned(),
                });
            }
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                if lock_is_stale(path) {
                    let _ = fs::remove_file(path);
                    continue;
                }
                if tokio::time::Instant::now() >= deadline {
                    bail!(
                        "another rlmesh process is refreshing this session; retry in a moment \
                         (lock file: {})",
                        path.display()
                    );
                }
                tokio::time::sleep(LOCK_POLL).await;
            }
            Err(err) => {
                return Err(err).with_context(|| format!("creating lock file {}", path.display()));
            }
        }
    }
}

fn lock_is_stale(path: &Path) -> bool {
    fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.elapsed().ok())
        .is_some_and(|age| age > LOCK_STALE_AFTER)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jwt(payload: &str) -> String {
        let encode = |part: &str| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(part);
        format!("{}.{}.sig", encode(r#"{"alg":"none"}"#), encode(payload))
    }

    #[test]
    fn decodes_standard_claims_from_an_unverified_payload() {
        let token = jwt(r#"{"sub":"user_1","exp":1700000000,"org_id":"org_01H"}"#);
        let claims = decode_claims(&token).unwrap();
        assert_eq!(claims.exp, Some(1_700_000_000));

        // Padded payloads and payloads without exp still decode.
        let padded = format!(
            "h.{}=.s",
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode("{}")
        );
        assert!(decode_claims(&padded).unwrap().exp.is_none());
        assert!(decode_claims("opaque-token").is_none());
        assert!(decode_claims("a.%%%.c").is_none());
    }

    #[test]
    fn freshness_applies_the_leeway_and_distrusts_unknown_expiry() {
        let now = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let fresh = jwt(r#"{"exp":1700000061}"#);
        let stale = jwt(r#"{"exp":1700000060}"#);
        assert!(is_fresh_at(&fresh, now));
        assert!(!is_fresh_at(&stale, now));
        assert!(!is_fresh_at(&jwt("{}"), now));
        assert!(!is_fresh_at("opaque-token", now));
        // An exp beyond what SystemTime can hold is unknown, never a panic.
        assert!(access_token_expires_at(&jwt(r#"{"exp":18446744073709551615}"#)).is_none());
        assert!(!is_fresh_at(&jwt(r#"{"exp":18446744073709551615}"#), now));
    }

    #[test]
    fn formats_rfc3339_utc() {
        assert_eq!(rfc3339_utc(UNIX_EPOCH), "1970-01-01T00:00:00Z");
        assert_eq!(
            rfc3339_utc(UNIX_EPOCH + Duration::from_secs(1_700_000_000)),
            "2023-11-14T22:13:20Z"
        );
        assert_eq!(
            rfc3339_utc(UNIX_EPOCH + Duration::from_secs(951_782_400)),
            "2000-02-29T00:00:00Z"
        );
    }

    #[tokio::test]
    async fn lock_is_exclusive_and_reclaimed_when_stale() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("default.lock");

        let held = acquire_session_lock(&path).await.unwrap();
        assert!(path.exists());
        let contended =
            tokio::time::timeout(Duration::from_millis(400), acquire_session_lock(&path)).await;
        assert!(contended.is_err(), "a live lock must block a second holder");
        drop(held);
        assert!(!path.exists());

        // A lock left behind by a crashed process is old, so it is reclaimed.
        fs::write(&path, "1\n").unwrap();
        let file = fs::File::options().write(true).open(&path).unwrap();
        file.set_modified(SystemTime::now() - LOCK_STALE_AFTER - Duration::from_secs(5))
            .unwrap();
        drop(file);
        let reclaimed = acquire_session_lock(&path).await.unwrap();
        assert!(path.exists());
        drop(reclaimed);
    }
}
