//! `devices.toml`, rlmeshd's device table (design §4): the devices besides
//! `local` it admits, what each may do, and the device each robot name is
//! pinned to; and the `rlmesh devices` commands that edit it. rlmeshd reads
//! it again within a second of every change, so an edit takes effect live.
//!
//! ```toml
//! [devices.laptop-josh]
//! public_key = "…"            # its static X25519 key, base32
//! max_priority = "operator"   # observer | scripted | model | operator
//! parts = ["ur10e/**"]        # default ["**"]
//! verbs = ["arm", "disarm"]   # default: both from operator up
//!
//! [robots.eye-1]
//! device = "laptop-josh"      # "local": this machine
//! board_id = "0011223344556677"
//! ```
//!
//! Pairing (`rlmesh pair`, [`crate::pair`]) adds machines and provisioning
//! (`rlmesh devices provision`, [`crate::provision`]) boards, with their
//! `kind` and `name`.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rlmesh_session::address::write_private;
use rlmesh_session::auth::address::unbase32;
use rlmesh_session::auth::{Device, LOCAL_DEVICE, Pin as HostPin};
use rlmesh_session::authority::{DeviceAccess, Verb};
use rlmesh_session::wire::{ParticipantId, PriorityClass, Role};
use serde::{Deserialize, Serialize};

use crate::cli::{DevicesArgs, DevicesCommand};

/// The table's file in the state directory.
pub const DEVICES: &str = "devices.toml";
/// Held while the table is read, changed and written back.
const LOCK: &str = "devices.lock";

/// The whole table.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Table {
    /// Every device id revoked: never given to another device (design §4),
    /// so a later pairing under the same name inherits none of its pins.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub revoked: BTreeSet<String>,
    /// Device id → its entry.
    #[serde(default)]
    pub devices: BTreeMap<String, Entry>,
    /// Robot name → its pin.
    #[serde(default)]
    pub robots: BTreeMap<String, Pin>,
}

/// One device's entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    /// Its static X25519 public key, base32.
    pub public_key: String,
    /// What it is (`machine`, `board`, `browser`), as pairing records it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// The robot name a board was provisioned under, which it may pin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// `observer`, `scripted`, `model` or `operator`.
    pub max_priority: String,
    /// Globs over `{robot}/{path}`; `["**"]` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parts: Option<Vec<String>>,
    /// `arm`, `disarm`; both from `operator` up when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verbs: Option<Vec<String>>,
    /// `embodiment`, `controller` (an observer always may attach); by its
    /// kind when absent: a board embodies, a browser controls, a machine
    /// does both.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roles: Option<Vec<String>>,
    /// The day (`YYYY-MM-DD`, UTC) from which it may no longer attach; it
    /// never expires when absent. Pairing sets one 180 days out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires: Option<String>,
}

/// A robot name's pin (design §4, tether spec §7).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pin {
    /// The device that embodies it.
    pub device: String,
    /// A tethered board's id behind that device, 16 hex digits: the host
    /// refuses a `DeclareParts` from another board.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub board_id: Option<String>,
}

/// The class a `max_priority` names (`None`: observer).
pub fn class(name: &str) -> Result<Option<PriorityClass>> {
    Ok(match name {
        "observer" => None,
        "scripted" => Some(PriorityClass::Scripted),
        "model" => Some(PriorityClass::Model),
        "operator" => Some(PriorityClass::Operator),
        "safety" => bail!("safety is reserved for the robot's own machine (local)"),
        other => bail!("max_priority {other:?}: observer, scripted, model or operator"),
    })
}

/// A board id from its 16 hex digits.
fn board_id(hex: &str) -> Option<[u8; 8]> {
    if hex.len() != 16 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let mut id = [0; 8];
    for (i, byte) in id.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(id)
}

fn role(name: &str) -> Result<Role> {
    match name {
        "embodiment" => Ok(Role::Embodiment),
        "controller" => Ok(Role::Controller),
        "observer" => Ok(Role::Observer),
        other => bail!("role {other:?}: embodiment, controller or observer"),
    }
}

fn verb(name: &str) -> Result<Verb> {
    match name {
        "arm" => Ok(Verb::Arm),
        "disarm" => Ok(Verb::Disarm),
        other => bail!("verb {other:?}: arm or disarm (every device may estop)"),
    }
}

/// Days since 1970-01-01 of the civil (proleptic Gregorian) date.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The civil date of `days` since 1970-01-01.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// Today, in days since 1970-01-01 (UTC).
pub fn today() -> i64 {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    i64::try_from(secs / 86_400).unwrap_or(i64::MAX)
}

/// The `YYYY-MM-DD` date `days` from today.
pub fn date_in(days: i64) -> String {
    let (y, m, d) = civil_from_days(today() + days);
    format!("{y:04}-{m:02}-{d:02}")
}

/// A `YYYY-MM-DD` date, in days since 1970-01-01.
pub fn parse_date(text: &str) -> Result<i64> {
    let bad = || format!("{text:?} is not a YYYY-MM-DD date");
    let mut parts = text.splitn(3, '-').map(str::parse::<i64>);
    let (Some(Ok(y)), Some(Ok(m)), Some(Ok(d))) = (parts.next(), parts.next(), parts.next()) else {
        bail!(bad());
    };
    if !(1..=12).contains(&m)
        || !(1..=31).contains(&d)
        || civil_from_days(days_from_civil(y, m, d)) != (y, m, d)
    {
        bail!(bad());
    }
    Ok(days_from_civil(y, m, d))
}

impl Entry {
    /// Whether it has expired by `today` (days since 1970-01-01).
    ///
    /// # Errors
    /// An `expires` that is not a date.
    pub fn expired(&self, today: i64) -> Result<bool> {
        match &self.expires {
            Some(date) => Ok(parse_date(date).context("expires")? <= today),
            None => Ok(false),
        }
    }

    /// What it may do.
    pub fn access(&self) -> Result<DeviceAccess> {
        let mut access = DeviceAccess::new(class(&self.max_priority)?);
        if let Some(parts) = &self.parts {
            access.parts.clone_from(parts);
        }
        if let Some(verbs) = &self.verbs {
            access.verbs = verbs.iter().map(|v| verb(v)).collect::<Result<_>>()?;
        }
        access.roles = match &self.roles {
            Some(roles) => roles.iter().map(|r| role(r)).collect::<Result<_>>()?,
            None => match self.kind.as_deref() {
                Some("board") => vec![Role::Embodiment],
                Some("browser") => vec![Role::Controller],
                _ => vec![Role::Embodiment, Role::Controller],
            },
        };
        access.roles.push(Role::Observer);
        Ok(access)
    }
}

impl Table {
    /// The table in `dir` (none yet: empty).
    pub fn read(dir: &Path) -> Result<Self> {
        match std::fs::read_to_string(dir.join(DEVICES)) {
            Ok(text) => Self::parse(&text),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(err) => {
                Err(err).with_context(|| format!("reading {}", dir.join(DEVICES).display()))
            }
        }
    }

    /// The table `text` holds, checked.
    pub fn parse(text: &str) -> Result<Self> {
        let table: Self = toml::from_str(text).context(DEVICES)?;
        table.host()?;
        Ok(table)
    }

    /// What a host takes from it: the devices with their entries, and the
    /// pins.
    pub fn host(&self) -> Result<(Vec<Device>, BTreeMap<ParticipantId, HostPin>)> {
        let today = today();
        let mut devices: Vec<Device> = Vec::new();
        for (id, entry) in &self.devices {
            let device = || format!("{DEVICES}: device {id}");
            if id == LOCAL_DEVICE {
                bail!("{}: local is this machine, with no entry", device());
            }
            // An expired device is as good as unknown: it may not attach.
            if entry.expired(today).with_context(device)? {
                continue;
            }
            let public = unbase32(&entry.public_key)
                .with_context(|| format!("{}: public_key is not 52 base32 characters", device()))?;
            if let Some(other) = devices.iter().find(|d| d.public == public) {
                bail!("{}: the same key as {}", device(), other.id);
            }
            let access = entry.access().with_context(device)?;
            devices.push(Device {
                id: id.clone(),
                public,
                access,
            });
        }
        let mut pins = BTreeMap::new();
        for (robot, pin) in &self.robots {
            let name = ParticipantId::new(robot).with_context(|| format!("{DEVICES}: {robot}"))?;
            let board_id = pin
                .board_id
                .as_ref()
                .map(|id| {
                    board_id(id).with_context(|| {
                        format!("{DEVICES}: robot {robot}: board_id {id:?} is not 16 hex digits")
                    })
                })
                .transpose()?;
            pins.insert(
                name,
                HostPin {
                    device: pin.device.clone(),
                    board_id,
                },
            );
        }
        Ok((devices, pins))
    }

    /// Whether `device`'s first embodiment of the unpinned `robot` pins it
    /// (design §4): `local`, a board under its own provisioned name, or a
    /// device with the operator verb.
    pub fn may_pin(&self, robot: &str, device: &str) -> bool {
        if device == LOCAL_DEVICE {
            return true;
        }
        self.devices.get(device).is_some_and(|entry| {
            let board =
                entry.kind.as_deref() == Some("board") && entry.name.as_deref() == Some(robot);
            board || entry.access().is_ok_and(|a| a.verbs.contains(&Verb::Arm))
        })
    }

    /// Whether `device` may be a satellite (scene design §3.2, §4.4): a
    /// paired machine (not a board or a browser) with the `arm` verb.
    #[cfg(feature = "fleet")]
    pub fn may_host(&self, device: &str) -> bool {
        self.devices.get(device).is_some_and(|entry| {
            matches!(entry.kind.as_deref(), None | Some("machine"))
                && entry.access().is_ok_and(|a| a.verbs.contains(&Verb::Arm))
        })
    }

    /// A device id for a new device called `name`: its characters outside
    /// `[A-Za-z0-9._-]` replaced by `-` (`fallback` for none), and `-2`,
    /// `-3`… after it while that is taken, or was revoked, or a pin names
    /// it.
    pub fn free_id(&self, name: &str, fallback: &str) -> String {
        let base: String = name
            .chars()
            .take(32)
            .map(|c| {
                if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                    c
                } else {
                    '-'
                }
            })
            .collect();
        let base = match base.trim_matches('-') {
            "" | LOCAL_DEVICE => fallback.to_string(),
            base => base.to_string(),
        };
        let mut id = base.clone();
        let mut n = 1;
        let taken = |id: &String| {
            self.devices.contains_key(id)
                || self.revoked.contains(id)
                || self.robots.values().any(|pin| &pin.device == id)
        };
        while taken(&id) {
            n += 1;
            id = format!("{base}-{n}");
        }
        id
    }

    /// Revokes device `id`: its entry goes, its id is never given out
    /// again, and its pins stay, so its robots' names can't be taken.
    ///
    /// # Errors
    /// A device not in the table.
    pub fn revoke(&mut self, id: &str) -> Result<()> {
        self.devices
            .remove(id)
            .with_context(|| format!("{id} is not in {DEVICES}"))?;
        self.revoked.insert(id.to_string());
        Ok(())
    }

    /// The table as `devices.toml` holds it.
    fn text(&self) -> Result<String> {
        Ok(toml::to_string(self)?)
    }
}

/// Changes the table in `dir` with `change`, under its lock, and writes it
/// back whole (0600) unless `change` fails.
pub fn edit<R>(dir: &Path, change: impl FnOnce(&mut Table) -> Result<R>) -> Result<R> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder
        .create(dir)
        .with_context(|| format!("{}", dir.display()))?;
    let mut options = std::fs::File::options();
    options.create(true).truncate(false).write(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let lock = options.open(dir.join(LOCK))?;
    lock.lock()?;
    let mut table = Table::read(dir)?;
    let out = change(&mut table)?;
    table.host()?;
    write_private(&dir.join(DEVICES), table.text()?.as_bytes())?;
    Ok(out)
}

/// `rlmesh devices`.
pub fn devices(args: &DevicesArgs, stdout: &mut impl Write) -> Result<()> {
    let dir = crate::host::state_dir(args.state_dir.as_deref())?;
    match &args.command {
        DevicesCommand::Ls => {
            let table = Table::read(&dir)?;
            for (id, entry) in &table.devices {
                let access = entry.access()?;
                let verbs: Vec<&str> = access
                    .verbs
                    .iter()
                    .map(|v| match v {
                        Verb::Arm => "arm",
                        Verb::Disarm => "disarm",
                    })
                    .collect();
                let expires = match &entry.expires {
                    Some(date) if entry.expired(today())? => format!(" expired {date}"),
                    Some(date) => format!(" expires {date}"),
                    None => String::new(),
                };
                writeln!(
                    stdout,
                    "device {id}: {} parts {:?} verbs {verbs:?}{expires}",
                    entry.max_priority, access.parts
                )?;
            }
            for (robot, pin) in &table.robots {
                match &pin.board_id {
                    Some(board) => writeln!(stdout, "robot {robot}: {} board {board}", pin.device)?,
                    None => writeln!(stdout, "robot {robot}: {}", pin.device)?,
                }
            }
        }
        DevicesCommand::Allow {
            device,
            max,
            parts,
            verbs,
            expires,
        } => edit(&dir, |table| {
            let entry = table.devices.get_mut(device).with_context(|| {
                format!("{device} is not in {DEVICES}: a device joins it by pairing")
            })?;
            if let Some(max) = max {
                entry.max_priority.clone_from(max);
            }
            if let Some(parts) = parts {
                entry.parts = Some(parts.clone());
            }
            if let Some(verbs) = verbs {
                entry.verbs = Some(verbs.clone());
            }
            match expires.as_deref() {
                None => {}
                Some("never") => entry.expires = None,
                Some(date) => {
                    parse_date(date)?;
                    entry.expires = Some(date.to_string());
                }
            }
            Ok(())
        })?,
        DevicesCommand::Revoke { device } => edit(&dir, |table| table.revoke(device))?,
        DevicesCommand::Move {
            robot,
            to,
            board_id,
            no_board_id,
        } => edit(&dir, |table| {
            if to != LOCAL_DEVICE && !table.devices.contains_key(to) {
                bail!("{to} is not in {DEVICES}");
            }
            let pin = table
                .robots
                .get_mut(robot)
                .with_context(|| format!("{robot} is not pinned"))?;
            pin.device.clone_from(to);
            if board_id.is_some() || *no_board_id {
                pin.board_id.clone_from(board_id);
            }
            Ok(())
        })?,
        DevicesCommand::Unpin { robot } => edit(&dir, |table| {
            table
                .robots
                .remove(robot)
                .with_context(|| format!("{robot} is not pinned"))?;
            Ok(())
        })?,
        DevicesCommand::Provision { port, name, max } => {
            crate::provision::provision(&dir, port, name, max.as_deref(), stdout)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_round_trip_and_an_expired_device_may_not_attach() {
        for date in ["1970-01-01", "2000-02-29", "2026-11-07", "2027-05-06"] {
            let days = parse_date(date).expect(date);
            let (y, m, d) = civil_from_days(days);
            assert_eq!(format!("{y:04}-{m:02}-{d:02}"), date);
        }
        assert!(parse_date("2026-02-30").is_err());
        assert!(parse_date("tomorrow").is_err());
        let key = |n: u8| rlmesh_session::auth::address::base32(&[n; 32]);
        let mut table = Table::parse(&format!(
            "[devices.old]\npublic_key = \"{}\"\nmax_priority = \"model\"\nexpires = \"2001-01-01\"\n\n\
             [devices.new]\npublic_key = \"{}\"\nmax_priority = \"model\"\nexpires = \"{}\"\n",
            key(1),
            key(2),
            date_in(30)
        ))
        .expect("table");
        let (devices, _) = table.host().expect("host");
        let ids: Vec<&str> = devices.iter().map(|d| d.id.as_str()).collect();
        assert_eq!(ids, ["new"], "the expired device is dropped");
        table.devices.get_mut("new").expect("new").expires = Some("someday".into());
        assert!(table.host().is_err(), "a bad date is refused");
    }

    #[test]
    fn a_revoked_device_id_is_never_given_out_again_so_its_pins_go_to_no_one() {
        let dir = tempfile::tempdir().expect("dir");
        let key = |n: u8| rlmesh_session::auth::address::base32(&[n; 32]);
        std::fs::write(
            dir.path().join(DEVICES),
            format!(
                "[devices.laptop]\npublic_key = \"{}\"\nmax_priority = \"operator\"\n\n\
                 [robots.eye-1]\ndevice = \"laptop\"\n",
                key(1)
            ),
        )
        .expect("devices.toml");
        edit(dir.path(), |table| table.revoke("laptop")).expect("revoked");
        // Read back as a restarted daemon would: another machine asking
        // for the name gets another id, and the pin names no device.
        let table = Table::read(dir.path()).expect("read");
        let id = table.free_id("laptop", "machine");
        assert_ne!(id, "laptop");
        let (devices, pins) = table.host().expect("host");
        assert!(devices.iter().all(|d| d.id != "laptop"));
        assert_eq!(pins.len(), 1);
        let pin = table.robots.get("eye-1").map(|p| p.device.as_str());
        assert_eq!(pin, Some("laptop"));
        // Nor does an id a pin names but no entry has (an older table).
        let mut older = table.clone();
        older.revoked.clear();
        assert_ne!(older.free_id("laptop", "machine"), "laptop");
    }
}
