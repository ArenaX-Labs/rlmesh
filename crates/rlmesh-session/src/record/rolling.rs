//! Always-on capture: rolling segments, their purge, and kept incidents.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::Participant;
#[cfg(doc)]
use crate::record::Recorder;
use crate::record::capture::{Metadata, TopicKey};
use crate::record::writer::{Message, Segment};
use crate::record::{RecordError, RecordStats};
use crate::wire::{Health, SessionId};
use tracing::{debug, info, warn};

/// Budgets for always-on recording ([`Recorder::start_rolling`]).
#[derive(Clone, Debug)]
pub struct Rolling {
    /// The recordings root; segments go under `rolling/{session}/`.
    pub dir: PathBuf,
    /// How much session time one segment covers.
    pub segment: Duration,
    /// Segments older than this go.
    pub max_age: Duration,
    /// The oldest segments go while all of them together exceed this.
    pub max_bytes: u64,
    /// Below this much free space on the disk, capture pauses.
    pub min_free_bytes: u64,
    /// How far back an incident reaches.
    pub incident_before: Duration,
    /// How long after an automatic incident segments are still kept.
    pub incident_after: Duration,
    /// Incidents older than this go.
    pub incident_max_age: Duration,
    /// The oldest incidents go while all of them together exceed this.
    pub incident_max_bytes: u64,
}

impl Rolling {
    /// The defaults (newest 30 min or 20 GB, one-minute segments, pause
    /// below 2 GB free; incidents of the 2 min before and 30 s after, kept
    /// 14 days or 10 GB) under `dir`.
    // `rlmesh host` overrides the budgets with its `--record-*` and
    // `--incidents-*` flags.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            segment: Duration::from_secs(60),
            max_age: Duration::from_secs(30 * 60),
            max_bytes: 20_000_000_000,
            min_free_bytes: 2_000_000_000,
            incident_before: Duration::from_secs(2 * 60),
            incident_after: Duration::from_secs(30),
            incident_max_age: Duration::from_secs(14 * 24 * 60 * 60),
            incident_max_bytes: 10_000_000_000,
        }
    }

    /// The user's default recordings root (`rlmesh/recordings` in the data
    /// directory), shared by `rlmesh host` and the cockpit.
    pub fn default_dir() -> Option<PathBuf> {
        dirs::data_dir().map(|d| d.join("rlmesh").join("recordings"))
    }

    /// Where `session`'s rolling segments are.
    pub fn session_dir(&self, session: SessionId) -> PathBuf {
        self.dir.join("rolling").join(session.to_string())
    }
}

/// Rolling capture's writer side.
pub(super) struct RollingSink {
    pub(super) rolling: Rolling,
    /// This session's segments.
    pub(super) dir: PathBuf,
    pub(super) session: SessionId,
    /// Reports degraded health while paused.
    pub(super) participant: Arc<Participant>,
    pub(super) current: Option<Segment>,
    /// While paused, the log time to try opening a segment again.
    pub(super) retry_at: i64,
    /// The newest authority snapshot, written first into every segment.
    pub(super) authority: Option<Message>,
    pub(super) paused: bool,
    /// Open incidents: until when (log time) new segments go into each.
    pub(super) keeping: Vec<(i64, PathBuf)>,
}

impl RollingSink {
    pub(super) fn write(&mut self, m: Message, stats: &mut RecordStats) -> Result<(), RecordError> {
        let segment = i64::try_from(self.rolling.segment.as_nanos()).unwrap_or(i64::MAX);
        let due = match &self.current {
            Some(current) => m.log_time >= current.started.saturating_add(segment),
            None => m.log_time >= self.retry_at,
        };
        if due {
            self.rotate(m.log_time, stats)?;
            if self.current.is_none() {
                self.retry_at = m.log_time.saturating_add(segment);
            }
        }
        let written = match &mut self.current {
            Some(current) => current.write(&m, stats),
            None => {
                stats.paused_drops += 1;
                Ok(())
            }
        };
        if let Err(err) = written {
            // The segment stays readable up to the failed write (a full
            // disk, most likely); capture retries next period.
            warn!(%err, "recording paused: a segment write failed");
            self.current = None;
            self.pause(true);
            self.retry_at = m.log_time.saturating_add(segment);
            stats.paused_drops += 1;
        }
        if m.topic.key == TopicKey::Authority {
            self.authority = Some(m);
        }
        Ok(())
    }

    /// Closes the current segment, applies the budgets and opens the next
    /// one, unless the disk is below the floor.
    fn rotate(&mut self, log_time: i64, stats: &mut RecordStats) -> Result<(), RecordError> {
        if let Some(current) = self.current.take() {
            current.finish(&Metadata::new(), stats)?;
        }
        purge(&self.rolling);
        // The root, which outlives the session directory the purge may
        // just have removed.
        if free_bytes(&self.rolling.dir).is_some_and(|free| free < self.rolling.min_free_bytes) {
            if !self.paused {
                warn!(dir = %self.dir.display(), "recording paused: less than {} bytes free", self.rolling.min_free_bytes);
            }
            self.pause(true);
            return Ok(());
        }
        if self.paused {
            info!(dir = %self.dir.display(), "recording resumed");
            self.pause(false);
        }
        let wall_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        let path = self.dir.join(format!("{wall_ms}.mcap"));
        // The purge removes a session directory it empties, this one too.
        let created = std::fs::create_dir_all(&self.dir)
            .map_err(|source| RecordError::Io {
                path: self.dir.clone(),
                source,
            })
            .and_then(|()| Segment::create(&path, log_time));
        let mut segment = match created {
            Ok(segment) => segment,
            Err(err) => {
                warn!(%err, "recording paused: cannot open a segment");
                self.pause(true);
                return Ok(());
            }
        };
        if let Some(authority) = &self.authority {
            segment.write(authority, stats)?;
        }
        self.keeping.retain(|(until, _)| log_time < *until);
        for (_, incident) in &self.keeping {
            if let Err(err) = link_into(&path, incident) {
                warn!(%err, incident = %incident.display(), "segment not kept");
            }
        }
        self.current = Some(segment);
        Ok(())
    }

    fn pause(&mut self, paused: bool) {
        self.paused = paused;
        self.participant
            .set_health(if paused { Health::Degraded } else { Health::Ok });
    }

    /// Keeps the recent segments as an incident and the ones opened until
    /// [`Rolling::incident_after`] from now, unless an incident is still
    /// open (it holds this one's start already).
    pub(super) fn incident(&mut self, label: &'static str, log_time: i64) {
        if self.keeping.iter().any(|(until, _)| log_time < *until) {
            debug!(session = %self.session, label, "incident within an open one");
            return;
        }
        let until = log_time.saturating_add(
            i64::try_from(self.rolling.incident_after.as_nanos()).unwrap_or(i64::MAX),
        );
        match keep(&self.rolling, self.session, label) {
            Ok(dir) => {
                info!(session = %self.session, label, dir = %dir.display(), "incident kept");
                self.keeping.push((until, dir));
            }
            Err(err) => warn!(session = %self.session, label, %err, "incident not kept"),
        }
    }
}

/// Keeps `session`'s rolling segments of the last
/// [`Rolling::incident_before`] as an incident named `label`, in
/// `incidents/{session}/{unix_ms}-{label}/`, out of the rolling purge's
/// reach; returns that directory. The segments are hard links, so the one
/// being written keeps growing in the incident until it rotates.
// A recording server answers the `Keep` host verb with this, so a cockpit
// on any machine can keep.
pub fn keep(rolling: &Rolling, session: SessionId, label: &str) -> Result<PathBuf, RecordError> {
    let now = std::time::SystemTime::now();
    let since = now.checked_sub(rolling.incident_before).unwrap_or(now);
    let dir = rolling
        .dir
        .join("incidents")
        .join(session.to_string())
        .join(format!("{}-{}", unix_ms(now), file_name(label)));
    link_since(rolling, session, since, &dir)?;
    Ok(dir)
}

/// A collected episode's outcome, written beside its segments as
/// `episode.json` (`task`, `start_ns`, `end_ns`, `success`).
#[derive(Clone, Debug, PartialEq)]
pub struct Episode {
    /// What the operator was doing.
    pub task: String,
    /// When it started, session time (as recorded log times count it).
    pub start_ns: i64,
    /// When it ended.
    pub end_ns: i64,
    /// The operator's verdict, if given.
    pub success: Option<bool>,
}

/// Collects `session`'s rolling segments written since `started` (wall
/// clock) as an episode of `dataset`, in `collected/{dataset}/{unix_ms}/`
/// with its `episode.json`. Nothing purges collected episodes; returns
/// the directory.
// TODO: an episode longer than the rolling budget (30 min or 20 GB) loses
// its purged start; link segments as they open while an episode runs when
// takes get that long.
pub fn collect(
    rolling: &Rolling,
    session: SessionId,
    dataset: &str,
    episode: &Episode,
    started: std::time::SystemTime,
) -> Result<PathBuf, RecordError> {
    let dir = rolling
        .dir
        .join("collected")
        .join(file_name(dataset))
        .join(unix_ms(started).to_string());
    link_since(rolling, session, started, &dir)?;
    let json = serde_json::json!({
        "session": session.to_string(),
        "task": episode.task,
        "start_ns": episode.start_ns,
        "end_ns": episode.end_ns,
        "success": episode.success,
    });
    let path = dir.join("episode.json");
    std::fs::write(&path, format!("{json:#}\n"))
        .map_err(|source| RecordError::Io { path, source })?;
    Ok(dir)
}

fn unix_ms(t: std::time::SystemTime) -> u128 {
    t.duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

/// `name` with anything but ASCII letters, digits and `-` replaced by `_`.
fn file_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Hard-links `session`'s rolling segments that hold anything since
/// `since` into a new `dir`: each segment is named for when it opened, so
/// it covers until the next one opened (the newest, until now).
fn link_since(
    rolling: &Rolling,
    session: SessionId,
    since: std::time::SystemTime,
    dir: &Path,
) -> Result<(), RecordError> {
    let source = rolling.session_dir(session);
    let io = |path: &Path| {
        let path = path.to_path_buf();
        move |source| RecordError::Io { path, source }
    };
    let mut segments: Vec<(u128, PathBuf)> = std::fs::read_dir(&source)
        .map_err(io(&source))?
        .flatten()
        .filter_map(|f| {
            let path = f.path();
            let opened = path.file_stem()?.to_str()?.parse().ok()?;
            path.extension()
                .is_some_and(|e| e == "mcap")
                .then_some((opened, path))
        })
        .collect();
    segments.sort();
    let since = unix_ms(since);
    let covering: Vec<&PathBuf> = segments
        .iter()
        .enumerate()
        .filter(|(i, _)| segments.get(i + 1).is_none_or(|(next, _)| *next > since))
        .map(|(_, (_, path))| path)
        .collect();
    std::fs::create_dir_all(dir).map_err(io(dir))?;
    let mut linked = 0;
    for segment in covering {
        match link_into(segment, dir) {
            Ok(()) => linked += 1,
            Err(err) => warn!(%err, segment = %segment.display(), "segment not kept"),
        }
    }
    if linked == 0 {
        let _ = std::fs::remove_dir(dir);
        return Err(RecordError::Io {
            path: source,
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "no recent segments"),
        });
    }
    Ok(())
}

/// Hard-links `segment` into `dir`, or copies it where links are refused;
/// a link already there is kept (never copied over: it is the same file).
fn link_into(segment: &Path, dir: &Path) -> std::io::Result<()> {
    let Some(name) = segment.file_name() else {
        return Ok(());
    };
    let target = dir.join(name);
    match std::fs::hard_link(segment, &target) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(_) if !target.exists() => std::fs::copy(segment, &target).map(drop),
        Err(err) => Err(err),
    }
}

/// Removes the oldest rolling segments, of every session under the root,
/// past the age and size budgets. Racing purgers (one per session) are
/// fine: a segment already gone is skipped.
fn purge(rolling: &Rolling) {
    let mut segments: Vec<(std::time::SystemTime, u64, PathBuf)> = Vec::new();
    let sessions = std::fs::read_dir(rolling.dir.join("rolling"))
        .into_iter()
        .flatten();
    for session in sessions.flatten() {
        for file in std::fs::read_dir(session.path())
            .into_iter()
            .flatten()
            .flatten()
        {
            let path = file.path();
            if path.extension().is_some_and(|e| e == "mcap")
                && let Ok(meta) = file.metadata()
                && let Ok(modified) = meta.modified()
            {
                segments.push((modified, meta.len(), path));
            }
        }
    }
    segments.sort();
    let mut total: u64 = segments.iter().map(|(_, len, _)| len).sum();
    let now = std::time::SystemTime::now();
    // A segment written within the last period may be another session's
    // open one: never removed.
    let open = now.checked_sub(rolling.segment).unwrap_or(now);
    for (modified, len, path) in segments {
        if modified >= open {
            break;
        }
        let old = now
            .duration_since(modified)
            .is_ok_and(|age| age > rolling.max_age);
        if !old && total <= rolling.max_bytes {
            break;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => total = total.saturating_sub(len),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => warn!(%err, path = %path.display(), "could not remove an old segment"),
        }
        if let Some(dir) = path.parent() {
            // Only succeeds once the session's last segment is gone.
            let _ = std::fs::remove_dir(dir);
        }
    }
    // Incidents, by their directories' age; a segment linked into two
    // counts twice.
    let mut incidents: Vec<(std::time::SystemTime, u64, PathBuf)> = Vec::new();
    let sessions = std::fs::read_dir(rolling.dir.join("incidents"))
        .into_iter()
        .flatten()
        .flatten();
    for session in sessions {
        for incident in std::fs::read_dir(session.path())
            .into_iter()
            .flatten()
            .flatten()
        {
            let Ok(modified) = incident.metadata().and_then(|m| m.modified()) else {
                continue;
            };
            let len = std::fs::read_dir(incident.path())
                .into_iter()
                .flatten()
                .flatten()
                .filter_map(|f| f.metadata().ok())
                .map(|m| m.len())
                .sum();
            incidents.push((modified, len, incident.path()));
        }
    }
    incidents.sort();
    let mut total: u64 = incidents.iter().map(|(_, len, _)| len).sum();
    for (modified, len, path) in incidents {
        let old = now
            .duration_since(modified)
            .is_ok_and(|age| age > rolling.incident_max_age);
        if !old && total <= rolling.incident_max_bytes {
            break;
        }
        match std::fs::remove_dir_all(&path) {
            Ok(()) => total = total.saturating_sub(len),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => warn!(%err, path = %path.display(), "could not remove an old incident"),
        }
        if let Some(dir) = path.parent() {
            let _ = std::fs::remove_dir(dir);
        }
    }
}

/// Free bytes on the disk holding `dir`, for an unprivileged user.
#[cfg(unix)]
#[allow(unsafe_code, clippy::useless_conversion)]
fn free_bytes(dir: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
    // SAFETY: an all-zero statvfs is a valid value to be overwritten.
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `path` is NUL-terminated and `stat` is valid and writable.
    if unsafe { libc::statvfs(path.as_ptr(), &raw mut stat) } != 0 {
        return None;
    }
    Some(u64::from(stat.f_bavail).saturating_mul(u64::from(stat.f_frsize)))
}

// TODO: no free-space floor off unix; add GetDiskFreeSpaceExW when the
// recorder runs on Windows hosts.
#[cfg(not(unix))]
fn free_bytes(_: &Path) -> Option<u64> {
    None
}
