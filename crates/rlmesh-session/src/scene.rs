//! Scene files (scene design §2): a TOML file naming a run's members, their
//! caps, places, security and where its host runs.
//!
//! ```toml
//! [scene]
//! format = 1
//! name = "eyeball-rig"
//!
//! [members.eye-l]
//! role = "robot"
//! place = { at = "eyes/left", pose = { xyz = [0.035, 0, 0.30], quat_xyzw = [0, 0, 0, 1] } }
//!
//! [members.tracker]
//! role = "controller"
//! device = "laptop-josh"
//! ```
//!
//! [`Scene::parse`] reads and checks a file's text on its own: format,
//! keys (`x_*` aside), names, roles, places, security and durations.
//! Checking it against a daemon's devices and pins is the daemon's
//! (`rlmesh scene check`), with the same [`Problem`]s. Nothing here reads a
//! file, a clock or the network: the daemon and every SDK share it.
//!
//! Every refusal names its line, its key and the fix, in the style of the
//! binder's (spec §6.7):
//!
//! ```text
//! scene refused: eyeball-rig (1 problem)
//!
//!   [INVALID_REQUEST] line 12, members.skull.max_priority: a robot leases nothing
//!     fix: remove max_priority from [members.skull]
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use serde_json::json;
use toml::{Table, Value};

use crate::auth::{LOCAL_DEVICE, Pin};
use crate::authority::{Member as HostMember, SceneConfig};
use crate::sdk::status_name;
use crate::spec::{canonical, view};
use crate::wire::{
    CtlStatus, Fingerprint, PartPath, ParticipantId, Place, Pose, PriorityClass, SceneInfo,
};

/// The scene file format this build reads.
pub const FORMAT: i64 = 1;
/// How long each stage waits by default.
pub const DEFAULT_WAIT: Duration = Duration::from_secs(30);
/// How long a run waits for a lost member by default.
pub const DEFAULT_REJOIN: Duration = Duration::from_secs(60);

/// One problem with a scene, as `rlmesh scene check` and `RunScene` refuse
/// it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Problem {
    /// The refusal class (`INVALID_REQUEST`, `NOT_AUTHORIZED`, ...).
    pub status: CtlStatus,
    /// The 1-based line it is on, when the key is in the file.
    pub line: Option<usize>,
    /// The key (`members.arm.device`).
    pub key: String,
    /// What is wrong.
    pub what: String,
    /// How to fix it.
    pub fix: String,
}

/// A refused scene: every problem found, in file order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// The scene's name (the file's, when it has none).
    pub scene: String,
    /// The problems.
    pub problems: Vec<Problem>,
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.problems.len();
        writeln!(
            f,
            "scene refused: {} ({n} problem{})",
            self.scene,
            if n == 1 { "" } else { "s" }
        )?;
        for p in &self.problems {
            writeln!(f)?;
            let at = match p.line {
                Some(line) => format!("line {line}, {}", p.key),
                None => p.key.clone(),
            };
            writeln!(f, "  [{}] {at}: {}", status_name(p.status), p.what)?;
            if !p.fix.is_empty() {
                writeln!(f, "    fix: {}", p.fix)?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for Refusal {}

/// The session-wide default for data and chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Security {
    /// Integrity only.
    Sign,
    /// Encrypted.
    Seal,
}

/// How a run arms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    /// Once everyone declared and bound.
    Auto,
    /// On an operator's arm.
    Manual,
}

/// What a robot member seals beyond the scene's default.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Seal {
    /// The scene's default.
    Default,
    /// Every part.
    All,
    /// These parts.
    Parts(Vec<PartPath>),
}

/// Where a controller member's processes come from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// Any process of this device (`local`: the home machine).
    Device(String),
    /// A catalog model a daemon serves `on` a machine (`home` or a paired
    /// device).
    Serve {
        /// The catalog model (`prebuilt/smolvla/so100`).
        model: String,
        /// The machine.
        on: String,
    },
}

/// A member's role and settings.
#[derive(Clone, Debug, PartialEq)]
pub enum Role {
    /// A robot: its pinned name.
    Robot {
        /// The device asserted to embody it (checked against its pin).
        device: Option<String>,
        /// Where its parts appear in the view (`None`: outside it).
        place: Option<Place>,
        /// What it seals beyond the scene's default.
        seal: Seal,
    },
    /// A controller: a script or a served model.
    Controller {
        /// Where its processes come from.
        source: Source,
        /// The highest class it attaches and leases at.
        max_priority: PriorityClass,
        /// The one robot it binds against rather than the view.
        robot: Option<ParticipantId>,
    },
}

/// One member.
#[derive(Clone, Debug, PartialEq)]
pub struct Member {
    /// Its name: a participant id.
    pub name: ParticipantId,
    /// Its role and settings.
    pub role: Role,
}

impl Member {
    /// The device a controller member's processes attach from: its own, or
    /// that of the machine serving it (`home` is `local`).
    pub fn device(&self) -> Option<&str> {
        match &self.role {
            Role::Controller {
                source: Source::Device(device),
                ..
            } => Some(device),
            Role::Controller {
                source: Source::Serve { on, .. },
                ..
            } => Some(if on == "home" { LOCAL_DEVICE } else { on }),
            Role::Robot { device, .. } => device.as_deref(),
        }
    }
}

/// A checked scene file, defaults filled.
#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    /// Its name.
    pub name: String,
    /// The daemon whose lobby its robots wait in (`None`: the one paired
    /// host).
    pub home: Option<String>,
    /// Where its session host runs: `home` or a paired machine (`None`: the
    /// default placement).
    pub host: Option<String>,
    /// The session-wide default for data and chunks.
    pub security: Security,
    /// How long each stage waits for robots, declares and bindings.
    pub wait: Duration,
    /// How it arms.
    pub arm: Arm,
    /// How long a run waits for a lost member.
    pub rejoin: Duration,
    /// Whether the host daemon's recorder joins.
    pub record: bool,
    /// Its members, by name.
    pub members: Vec<Member>,
    text: String,
}

/// The 1-based line of `key` in table `[table]` of `text` (the header's
/// line when `key` is empty or absent), found by reading the lines.
fn line_of(text: &str, table: &str, key: &str) -> Option<usize> {
    let mut lines = text.lines().enumerate();
    let header = |line: &str| -> Option<String> {
        let inner = line.trim().strip_prefix('[')?.split(']').next()?;
        Some(
            inner
                .chars()
                .filter(|c| !matches!(c, '"' | '\'' | ' '))
                .collect(),
        )
    };
    let start = lines
        .by_ref()
        .find(|(_, l)| header(l).as_deref() == Some(table))?
        .0;
    if key.is_empty() {
        return Some(start + 1);
    }
    for (i, line) in lines {
        let line = line.trim();
        if line.starts_with('[') {
            break;
        }
        if let Some(rest) = line.strip_prefix(key)
            && rest.trim_start().starts_with('=')
        {
            return Some(i + 1);
        }
    }
    Some(start + 1)
}

/// The 1-based line of byte `offset`.
fn line_at(text: &str, offset: usize) -> usize {
    text[..offset.min(text.len())].lines().count().max(1)
}

/// A duration: a number and `ms`, `s`, `min` or `h` (`0` alone is none).
fn duration(text: &str) -> Option<Duration> {
    if text == "0" {
        return Some(Duration::ZERO);
    }
    let split = text.find(|c: char| !(c.is_ascii_digit() || c == '.'))?;
    let (number, unit) = text.split_at(split);
    let number: f64 = number.parse().ok()?;
    let seconds = match unit {
        "ms" => number / 1e3,
        "s" => number,
        "min" => number * 60.0,
        "h" => number * 3600.0,
        _ => return None,
    };
    Duration::try_from_secs_f64(seconds).ok()
}

fn priority(name: &str) -> Option<PriorityClass> {
    match name {
        "scripted" => Some(PriorityClass::Scripted),
        "model" => Some(PriorityClass::Model),
        "operator" => Some(PriorityClass::Operator),
        "safety" => Some(PriorityClass::Safety),
        _ => None,
    }
}

/// The parser's state: the text and the problems found so far.
struct Reader<'a> {
    text: &'a str,
    problems: Vec<Problem>,
}

impl Reader<'_> {
    /// Records a problem with `key` of `[table]`.
    fn refuse(&mut self, status: CtlStatus, table: &str, key: &str, what: String, fix: String) {
        let full = if key.is_empty() {
            table.to_string()
        } else {
            format!("{table}.{key}")
        };
        self.problems.push(Problem {
            status,
            line: line_of(self.text, table, key.split('.').next().unwrap_or(key)),
            key: full,
            what,
            fix,
        });
    }

    fn invalid(&mut self, table: &str, key: &str, what: impl Into<String>, fix: impl Into<String>) {
        self.refuse(
            CtlStatus::InvalidRequest,
            table,
            key,
            what.into(),
            fix.into(),
        );
    }

    /// Refuses every key of `t` outside `known` and `x_*`.
    fn keys(&mut self, table: &str, t: &Table, known: &[&str]) {
        for key in t.keys() {
            if known.contains(&key.as_str()) || key.starts_with("x_") {
                continue;
            }
            let fix = if table == "scene" && key == "clock" {
                "remove it: scenes run in wall mode".to_string()
            } else {
                format!("remove it (known: {}; x_* for your own)", known.join(", "))
            };
            self.invalid(table, key, format!("unknown key {key:?}"), fix);
        }
    }

    fn string(&mut self, table: &str, t: &Table, key: &str) -> Option<String> {
        match t.get(key)? {
            Value::String(s) => Some(s.clone()),
            _ => {
                self.invalid(table, key, "is not a string", format!("{key} = \"...\""));
                None
            }
        }
    }

    fn choice<T: Copy>(
        &mut self,
        table: &str,
        t: &Table,
        key: &str,
        choices: &[(&str, T)],
        default: T,
    ) -> T {
        let Some(value) = self.string(table, t, key) else {
            return default;
        };
        if let Some((_, v)) = choices.iter().find(|(name, _)| *name == value) {
            return *v;
        }
        let names: Vec<&str> = choices.iter().map(|(n, _)| *n).collect();
        self.invalid(
            table,
            key,
            format!("{value:?} is not one of {}", names.join(", ")),
            format!("{key} = {:?}", names[0]),
        );
        default
    }

    fn duration(
        &mut self,
        table: &str,
        t: &Table,
        key: &str,
        range: (Duration, Duration),
        default: Duration,
    ) -> Duration {
        let Some(text) = self.string(table, t, key) else {
            return default;
        };
        match duration(&text) {
            Some(d) if d >= range.0 && d <= range.1 => d,
            _ => {
                let bounds = |d: Duration| match d.as_secs() {
                    0 => "0".to_string(),
                    s if s % 3600 == 0 => format!("{}h", s / 3600),
                    s if s % 60 == 0 => format!("{}min", s / 60),
                    s => format!("{s}s"),
                };
                self.invalid(
                    table,
                    key,
                    format!(
                        "{text:?} is not a duration from {} to {}",
                        bounds(range.0),
                        bounds(range.1)
                    ),
                    format!("{key} = \"{}\" (ms, s, min or h)", bounds(default)),
                );
                default
            }
        }
    }

    fn floats<const N: usize>(value: Option<&Value>) -> Option<[f64; N]> {
        let items = value?.as_array().filter(|a| a.len() == N)?;
        let mut out = [0.0; N];
        for (slot, item) in out.iter_mut().zip(items) {
            *slot = match item {
                Value::Float(f) => *f,
                Value::Integer(i) => *i as f64,
                _ => return None,
            };
        }
        Some(out)
    }

    fn place(&mut self, table: &str, robot: &ParticipantId, value: &Value) -> Option<Place> {
        let fix =
            "place = { at = \"eyes/left\", pose = { xyz = [0, 0, 0], quat_xyzw = [0, 0, 0, 1] } }";
        let Value::Table(t) = value else {
            self.invalid(table, "place", "is not a table", fix);
            return None;
        };
        let before = self.problems.len();
        for key in t.keys() {
            if !["at", "from", "frame", "pose"].contains(&key.as_str()) && !key.starts_with("x_") {
                self.invalid(
                    table,
                    &format!("place.{key}"),
                    format!("unknown key {key:?}"),
                    "remove it (known: at, from, frame, pose)",
                );
            }
        }
        let text = |key: &str| t.get(key).and_then(Value::as_str).map(str::to_string);
        let pose = t.get("pose").and_then(Value::as_table);
        let xyz = Self::floats::<3>(pose.and_then(|p| p.get("xyz")));
        let quat = Self::floats::<4>(pose.and_then(|p| p.get("quat_xyzw")));
        let (Some(at), Some(xyz), Some(quat_xyzw)) = (text("at"), xyz, quat) else {
            self.invalid(
                table,
                "place",
                "needs at and a pose of 3 xyz metres and a quat_xyzw",
                fix,
            );
            return None;
        };
        let place = Place {
            robot: robot.clone(),
            at,
            from: text("from").unwrap_or_default(),
            frame: text("frame").unwrap_or_default(),
            pose: Pose { xyz, quat_xyzw },
        };
        if let Err(why) = view::check_place(&place) {
            self.invalid(
                table,
                "place",
                why,
                "at is a path (spec §2.1) with no _ segment; frame one of the robot's own frames",
            );
        }
        (self.problems.len() == before).then_some(place)
    }

    fn seal(&mut self, table: &str, value: Option<&Value>) -> Seal {
        let fix = "seal = true, or a list of part paths: seal = [\"cam\"]";
        match value {
            None | Some(Value::Boolean(false)) => Seal::Default,
            Some(Value::Boolean(true)) => Seal::All,
            Some(Value::Array(items)) => {
                let mut parts = Vec::new();
                for item in items {
                    match item.as_str().map(PartPath::new) {
                        Some(Ok(path)) => parts.push(path),
                        _ => self.invalid(table, "seal", format!("{item} is not a part path"), fix),
                    }
                }
                Seal::Parts(parts)
            }
            Some(other) => {
                self.invalid(
                    table,
                    "seal",
                    format!("{other} is neither true nor a list"),
                    fix,
                );
                Seal::Default
            }
        }
    }

    fn member(&mut self, name: &str, value: &Value) -> Option<Member> {
        let table = format!("members.{name}");
        let valid = !name.is_empty()
            && name.len() <= 128
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
        let reserved = name.starts_with('_') || name.starts_with("rlmeshd.");
        let id = match ParticipantId::new(name) {
            Ok(id) if valid && !reserved => id,
            _ => {
                self.invalid(
                    &table,
                    "",
                    format!(
                        "{name:?} is not a member name: [A-Za-z0-9._-], 1-128 characters, not \
                         starting with _ or rlmeshd."
                    ),
                    "rename the member",
                );
                return None;
            }
        };
        let Value::Table(t) = value else {
            self.invalid(
                &table,
                "",
                "is not a table",
                format!("[{table}] role = \"robot\""),
            );
            return None;
        };
        let role = match t.get("role").and_then(Value::as_str) {
            Some("robot") => {
                if t.contains_key("max_priority") {
                    self.invalid(
                        &table,
                        "max_priority",
                        "a robot leases nothing, so it takes no max_priority",
                        format!("remove max_priority from [{table}]"),
                    );
                }
                let known = ["role", "device", "place", "seal", "max_priority"];
                self.keys(&table, t, &known);
                Role::Robot {
                    device: self.string(&table, t, "device"),
                    place: t.get("place").and_then(|p| self.place(&table, &id, p)),
                    seal: self.seal(&table, t.get("seal")),
                }
            }
            Some("controller") => {
                self.keys(
                    &table,
                    t,
                    &["role", "device", "serve", "on", "max_priority", "robot"],
                );
                let device = self.string(&table, t, "device");
                let serve = self.string(&table, t, "serve");
                let on = self.string(&table, t, "on");
                let source = match (device, serve, on) {
                    (Some(device), None, None) => Source::Device(device),
                    (None, Some(model), Some(on)) => Source::Serve { model, on },
                    _ => {
                        self.invalid(
                            &table,
                            "",
                            "a controller has a device, or serve and on",
                            "device = \"laptop-josh\", or serve = \"prebuilt/...\" with on = \"home\"",
                        );
                        return None;
                    }
                };
                let max_priority = match self.string(&table, t, "max_priority") {
                    None => PriorityClass::Model,
                    Some(name) => match priority(&name) {
                        Some(PriorityClass::Safety) if !matches!(&source, Source::Device(d) if d == LOCAL_DEVICE) =>
                        {
                            self.refuse(
                                CtlStatus::NotAuthorized,
                                &table,
                                "max_priority",
                                "safety is reserved for the robot's own machine (local)".into(),
                                "max_priority = \"operator\"".into(),
                            );
                            PriorityClass::Operator
                        }
                        Some(class) => class,
                        None => {
                            self.invalid(
                                &table,
                                "max_priority",
                                format!("{name:?} is not scripted, model, operator or safety"),
                                "max_priority = \"model\"",
                            );
                            PriorityClass::Model
                        }
                    },
                };
                let robot = self
                    .string(&table, t, "robot")
                    .and_then(|r| match ParticipantId::new(r.as_str()) {
                        Ok(r) => Some(r),
                        Err(err) => {
                            self.invalid(&table, "robot", err.to_string(), "name a robot member");
                            None
                        }
                    });
                Role::Controller {
                    source,
                    max_priority,
                    robot,
                }
            }
            _ => {
                self.invalid(
                    &table,
                    "role",
                    "a member's role is robot or controller",
                    "role = \"robot\"",
                );
                return None;
            }
        };
        Some(Member { name: id, role })
    }
}

impl Scene {
    /// Reads and checks a scene file's text.
    ///
    /// # Errors
    /// Every problem found, each with its line and fix.
    pub fn parse(text: &str) -> Result<Self, Refusal> {
        let root: Table = text.parse().map_err(|err: toml::de::Error| Refusal {
            scene: "this scene".into(),
            problems: vec![Problem {
                status: CtlStatus::InvalidRequest,
                line: err.span().map(|s| line_at(text, s.start)),
                key: String::new(),
                what: format!("not TOML: {}", err.message()),
                fix: "write it as TOML, as in the scene design's §2".into(),
            }],
        })?;
        let mut r = Reader {
            text,
            problems: Vec::new(),
        };
        r.keys("", &root, &["scene", "members"]);
        let empty = Table::new();
        let s = match root.get("scene") {
            Some(Value::Table(s)) => s,
            _ => {
                r.invalid(
                    "scene",
                    "",
                    "no [scene] table",
                    "[scene] format = 1, name = \"...\"",
                );
                &empty
            }
        };
        r.keys(
            "scene",
            s,
            &[
                "format", "name", "home", "host", "security", "wait", "arm", "rejoin", "record",
            ],
        );
        if s.get("format").and_then(Value::as_integer) != Some(FORMAT) {
            r.invalid(
                "scene",
                "format",
                "this build reads scene format 1",
                "format = 1",
            );
        }
        let name = r.string("scene", s, "name").unwrap_or_default();
        if name.is_empty() {
            r.invalid(
                "scene",
                "name",
                "the scene has no name",
                "name = \"my-rig\"",
            );
        }
        let home = r.string("scene", s, "home");
        let host = r.string("scene", s, "host");
        let security = r.choice(
            "scene",
            s,
            "security",
            &[("sign", Security::Sign), ("seal", Security::Seal)],
            Security::Sign,
        );
        let wait = r.duration(
            "scene",
            s,
            "wait",
            (Duration::from_secs(1), Duration::from_secs(600)),
            DEFAULT_WAIT,
        );
        let arm = r.choice(
            "scene",
            s,
            "arm",
            &[("auto", Arm::Auto), ("manual", Arm::Manual)],
            Arm::Auto,
        );
        let rejoin = r.duration(
            "scene",
            s,
            "rejoin",
            (Duration::ZERO, Duration::from_secs(3600)),
            DEFAULT_REJOIN,
        );
        let record = match s.get("record") {
            None => false,
            Some(Value::Boolean(b)) => *b,
            Some(_) => {
                r.invalid("scene", "record", "is not true or false", "record = true");
                false
            }
        };
        let mut members = Vec::new();
        match root.get("members") {
            None => {}
            Some(Value::Table(t)) => {
                for (name, value) in t {
                    if let Some(member) = r.member(name, value) {
                        members.push(member);
                    }
                }
            }
            Some(_) => r.invalid(
                "members",
                "",
                "is not a table",
                "[members.<name>] role = ...",
            ),
        }
        for member in &members {
            if let Role::Controller {
                robot: Some(robot), ..
            } = &member.role
                && !members
                    .iter()
                    .any(|m| &m.name == robot && matches!(m.role, Role::Robot { .. }))
            {
                r.invalid(
                    &format!("members.{}", member.name),
                    "robot",
                    format!("{robot} is not a robot member of this scene"),
                    "name a robot member, or drop robot to bind against the view",
                );
            }
        }
        r.problems.sort_by_key(|p| p.line.unwrap_or(usize::MAX));
        if !r.problems.is_empty() {
            return Err(Refusal {
                scene: if name.is_empty() {
                    "this scene".into()
                } else {
                    name
                },
                problems: r.problems,
            });
        }
        Ok(Self {
            name,
            home,
            host,
            security,
            wait,
            arm,
            rejoin,
            record,
            members,
            text: text.to_string(),
        })
    }

    /// The 1-based line of `key` of `table` (`scene`, `members.arm`; an
    /// empty key: the table's header): for the daemon's own checks.
    pub fn line(&self, table: &str, key: &str) -> Option<usize> {
        line_of(&self.text, table, key)
    }

    /// The member called `name`.
    pub fn member(&self, name: &str) -> Option<&Member> {
        self.members.iter().find(|m| m.name.as_str() == name)
    }

    /// The places of its placed robots.
    pub fn places(&self) -> Vec<Place> {
        self.members
            .iter()
            .filter_map(|m| match &m.role {
                Role::Robot { place, .. } => place.clone(),
                Role::Controller { .. } => None,
            })
            .collect()
    }

    /// The scene fingerprint: BLAKE3 over its canonical form (sorted,
    /// defaults filled), 8 bytes.
    pub fn fingerprint(&self) -> Fingerprint {
        let ns = |d: Duration| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX);
        let members: serde_json::Map<String, serde_json::Value> = self
            .members
            .iter()
            .map(|m| {
                let value = match &m.role {
                    Role::Robot {
                        device,
                        place,
                        seal,
                    } => json!({
                        "role": "robot",
                        "device": device,
                        "place": place.as_ref().map(|p| json!({
                            "at": p.at,
                            "from": p.from,
                            "frame": if p.frame.is_empty() { "base" } else { &p.frame },
                            "xyz": p.pose.xyz,
                            "quat_xyzw": p.pose.quat_xyzw,
                        })),
                        "seal": match seal {
                            Seal::Default => json!(false),
                            Seal::All => json!(true),
                            Seal::Parts(parts) => json!(parts.iter().map(PartPath::as_str).collect::<Vec<_>>()),
                        },
                    }),
                    Role::Controller {
                        source,
                        max_priority,
                        robot,
                    } => json!({
                        "role": "controller",
                        "device": match source { Source::Device(d) => Some(d), Source::Serve { .. } => None },
                        "serve": match source { Source::Serve { model, .. } => Some(model), Source::Device(_) => None },
                        "on": match source { Source::Serve { on, .. } => Some(on), Source::Device(_) => None },
                        "max_priority": format!("{max_priority:?}").to_lowercase(),
                        "robot": robot.as_ref().map(ParticipantId::as_str),
                    }),
                };
                (m.name.to_string(), value)
            })
            .collect();
        canonical::fingerprint(&json!({
            "format": FORMAT,
            "name": self.name,
            "home": self.home,
            "host": self.host,
            "security": match self.security { Security::Sign => "sign", Security::Seal => "seal" },
            "wait_ns": ns(self.wait),
            "arm": match self.arm { Arm::Auto => "auto", Arm::Manual => "manual" },
            "rejoin_ns": ns(self.rejoin),
            "record": self.record,
            "members": members,
        }))
    }

    /// What run `run`'s session host carries for it: its identity, its
    /// places and its member table, robots on their pinned devices (`pins`,
    /// rlmeshd's `[robots.*]`; an unpinned robot binds to its first
    /// embodiment's device, `local` unless the caller allows more).
    pub fn host(&self, run: &str, pins: &BTreeMap<ParticipantId, Pin>) -> SceneConfig {
        let members = self
            .members
            .iter()
            .map(|m| {
                let member = match &m.role {
                    Role::Robot { device, .. } => HostMember::Robot {
                        device: pins
                            .get(&m.name)
                            .map(|p| p.device.clone())
                            .or_else(|| device.clone()),
                        // The home daemon names who else may pin it.
                        may_pin: vec![LOCAL_DEVICE.to_string()],
                    },
                    Role::Controller { max_priority, .. } => HostMember::Controller {
                        device: m.device().unwrap_or(LOCAL_DEVICE).to_string(),
                        max_priority: *max_priority,
                    },
                };
                (m.name.clone(), member)
            })
            .collect();
        SceneConfig {
            info: SceneInfo {
                run: run.to_string(),
                name: self.name.clone(),
                fingerprint: self.fingerprint(),
                closing: None,
                rejoin_until_ns: None,
            },
            places: self.places(),
            members,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scene_file_refuses_every_problem_with_its_line_and_fix() {
        let text = r#"[scene]
format = 1
name = "rig"
wait = "2h"
clock = "virtual"

[members.eye-l]
role = "robot"
max_priority = "model"
place = { at = "Eyes", pose = { xyz = [0, 0, 0], quat_xyzw = [0, 0, 0, 1] } }

[members.tracker]
role = "controller"
device = "laptop-josh"
serve = "prebuilt/x"
x_note = "kept"
"#;
        let refusal = Scene::parse(text).expect_err("refused");
        let got: Vec<(Option<usize>, &str)> = refusal
            .problems
            .iter()
            .map(|p| (p.line, p.key.as_str()))
            .collect();
        assert_eq!(
            got,
            [
                (Some(4), "scene.wait"),
                (Some(5), "scene.clock"),
                (Some(9), "members.eye-l.max_priority"),
                (Some(10), "members.eye-l.place"),
                (Some(12), "members.tracker"),
            ]
        );
        let text = refusal.to_string();
        assert!(
            text.contains("  [INVALID_REQUEST] line 5, scene.clock: unknown key \"clock\"\n    fix: remove it: scenes run in wall mode"),
            "{text}"
        );
    }
}
