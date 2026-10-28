//! Per-device authorization (design §4, §12 decisions 7, 8 and 14): what
//! each device may do, by its entry (`max_priority`, part globs, verbs).
//!
//! - An attach above the device's cap is refused (`NOT_AUTHORIZED`), never
//!   clamped; an `observer` cap attaches only as an observer.
//! - A lease needs every part it names covered by the device's `parts`, and
//!   its class within the cap.
//! - Reading a stream (its `k_stream`, a history query, a lockstep reply)
//!   needs `parts` to cover it.
//! - `arm` and `disarm` need their verb; `estop` is every member's.
//! - The `safety` class is `local`'s alone, and `local` (every holder of an
//!   embedded host's address) may do everything.
//!
//! A device with no entry may do nothing.

use alloc::collections::BTreeMap;

use crate::auth::LOCAL_DEVICE;
use crate::prelude::*;
use crate::wire::{CtlError, CtlStatus, ParticipantId, PriorityClass, Role};

/// A verb that needs its own permission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verb {
    /// `Arm`.
    Arm,
    /// `Disarm`.
    Disarm,
}

impl Verb {
    fn name(self) -> &'static str {
        match self {
            Self::Arm => "arm",
            Self::Disarm => "disarm",
        }
    }
}

/// One device's entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceAccess {
    /// The highest class it may attach and lease at; `None`: observer only.
    pub max_priority: Option<PriorityClass>,
    /// Globs over `{robot}/{path}` it may lease and read (`*` one path
    /// segment, `**` any number).
    pub parts: Vec<String>,
    /// The verbs it may send besides every member's.
    pub verbs: Vec<Verb>,
}

impl DeviceAccess {
    /// The defaults at `max_priority`: every part, and `arm`/`disarm` from
    /// `operator` up.
    pub fn new(max_priority: Option<PriorityClass>) -> Self {
        let operator = max_priority.is_some_and(|p| p >= PriorityClass::Operator);
        Self {
            max_priority,
            parts: vec!["**".into()],
            verbs: if operator {
                vec![Verb::Arm, Verb::Disarm]
            } else {
                Vec::new()
            },
        }
    }

    fn covers(&self, robot: &ParticipantId, path: &str) -> bool {
        self.parts
            .iter()
            .any(|glob| glob_match(glob, robot.as_str(), path))
    }
}

/// Every device's entry: [`super::AuthorityConfig::access`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Access {
    devices: BTreeMap<String, DeviceAccess>,
}

fn refused(message: String) -> CtlError {
    CtlError::new(CtlStatus::NotAuthorized, message)
}

fn class_name(class: Option<PriorityClass>) -> &'static str {
    match class {
        None => "observer",
        Some(PriorityClass::Scripted) => "scripted",
        Some(PriorityClass::Model) => "model",
        Some(PriorityClass::Operator) => "operator",
        Some(PriorityClass::Safety) => "safety",
    }
}

impl Access {
    /// Sets `device`'s entry (`local` has none: it may do everything).
    pub fn set(&mut self, device: impl Into<String>, access: DeviceAccess) {
        self.devices.insert(device.into(), access);
    }

    /// `device`'s entry, if it has one.
    pub fn get(&self, device: &str) -> Option<&DeviceAccess> {
        self.devices.get(device)
    }

    fn entry(&self, device: &str) -> Result<Option<&DeviceAccess>, CtlError> {
        if device == LOCAL_DEVICE {
            return Ok(None);
        }
        self.devices
            .get(device)
            .map(Some)
            .ok_or_else(|| refused(format!("device {device} has no entry on this host")))
    }

    /// The highest class `device` may use (`None`: observer only).
    pub fn max_priority(&self, device: &str) -> Option<PriorityClass> {
        match self.entry(device) {
            Ok(None) => Some(PriorityClass::Safety),
            // A non-local entry never reaches `safety`.
            Ok(Some(entry)) => entry.max_priority.map(|p| p.min(PriorityClass::Operator)),
            Err(_) => None,
        }
    }

    fn check_class(&self, device: &str, class: PriorityClass, what: &str) -> Result<(), CtlError> {
        let cap = self.max_priority(device);
        if cap.is_some_and(|cap| class <= cap) {
            return Ok(());
        }
        let hint = if class == PriorityClass::Safety {
            "; safety is reserved for the robot's own machine (local)".to_string()
        } else {
            format!(
                "; on the robot: rlmesh devices allow {device} --max {}",
                class_name(Some(class))
            )
        };
        Err(refused(format!(
            "{device} may {what} at {} at most (asked {}){hint}",
            class_name(cap),
            class_name(Some(class))
        )))
    }

    /// Whether `device` may attach with `roles` at `priority`.
    ///
    /// # Errors
    /// `NOT_AUTHORIZED`.
    pub fn check_attach(
        &self,
        device: &str,
        roles: &[Role],
        priority: PriorityClass,
    ) -> Result<(), CtlError> {
        self.entry(device)?;
        if self.max_priority(device).is_none() {
            return if roles.iter().all(|r| *r == Role::Observer) {
                Ok(())
            } else {
                Err(refused(format!("{device} may attach only as an observer")))
            };
        }
        self.check_class(device, priority, "attach")
    }

    /// Whether `device` may lease `robot`'s part at `path` at `class`.
    ///
    /// # Errors
    /// `NOT_AUTHORIZED`, naming the part.
    pub fn check_lease(
        &self,
        device: &str,
        robot: &ParticipantId,
        path: &str,
        class: PriorityClass,
    ) -> Result<(), CtlError> {
        let Some(entry) = self.entry(device)? else {
            return Ok(());
        };
        if !entry.covers(robot, path) {
            return Err(refused(format!(
                "{device} may not lease {robot}/{path} (its parts: {:?})",
                entry.parts
            )));
        }
        self.check_class(device, class, &format!("lease {robot}/{path}"))
    }

    /// Whether `device` may read `robot`'s stream at `path` (a part path,
    /// or one with `/_applied`).
    pub fn may_read(&self, device: &str, robot: &ParticipantId, path: &str) -> bool {
        let path = path.strip_suffix("/_applied").unwrap_or(path);
        match self.entry(device) {
            Ok(None) => true,
            Ok(Some(entry)) => entry.covers(robot, path),
            Err(_) => false,
        }
    }

    /// Whether `device` may send `verb`.
    ///
    /// # Errors
    /// `NOT_AUTHORIZED`.
    pub fn check_verb(&self, device: &str, verb: Verb) -> Result<(), CtlError> {
        match self.entry(device)? {
            Some(entry) if !entry.verbs.contains(&verb) => Err(refused(format!(
                "{device} may not {}; on the robot: rlmesh devices allow {device} --verbs {}",
                verb.name(),
                verb.name()
            ))),
            _ => Ok(()),
        }
    }
}

/// Whether `glob` covers `{robot}/{path}`, segment by segment: `*` is one
/// segment, `**` any number (none included), anything else itself.
fn glob_match(glob: &str, robot: &str, path: &str) -> bool {
    fn go(glob: &[&str], name: &[&str]) -> bool {
        match glob.split_first() {
            None => name.is_empty(),
            Some((&"**", rest)) => (0..=name.len()).any(|skip| go(rest, &name[skip..])),
            Some((first, rest)) => name
                .split_first()
                .is_some_and(|(n, tail)| (*first == "*" || first == n) && go(rest, tail)),
        }
    }
    let glob: Vec<&str> = glob.split('/').collect();
    let name: Vec<&str> = core::iter::once(robot).chain(path.split('/')).collect();
    go(&glob, &name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pid(id: &str) -> ParticipantId {
        ParticipantId::new(id).expect("id")
    }

    #[test]
    fn globs_attach_leases_reads_and_verbs_follow_the_entry() {
        let mut access = Access::default();
        access.set(
            "laptop",
            DeviceAccess {
                max_priority: Some(PriorityClass::Model),
                parts: vec!["ur10e/arm".into(), "ur10e/wrist/*".into()],
                verbs: Vec::new(),
            },
        );
        access.set("eye", DeviceAccess::new(None));
        access.set("rogue", DeviceAccess::new(Some(PriorityClass::Safety)));
        let ur = pid("ur10e");
        let status = |r: Result<(), CtlError>| r.err().map(|e| e.status);
        let denied = Some(CtlStatus::NotAuthorized);

        // (device, robot, path, may read)
        for (device, robot, path, reads) in [
            ("laptop", "ur10e", "arm", true),
            ("laptop", "ur10e", "arm/_applied", true),
            ("laptop", "ur10e", "wrist/cam", true),
            ("laptop", "ur10e", "wrist/cam/depth", false),
            ("laptop", "ur10e", "base", false),
            ("laptop", "ur5", "arm", false),
            ("eye", "ur5", "a/b/c", true),
            ("local", "ur5", "anything", true),
            ("stranger", "ur10e", "arm", false),
        ] {
            assert_eq!(
                access.may_read(device, &pid(robot), path),
                reads,
                "{device} {robot}/{path}"
            );
        }
        // Attach: within the cap; observers only as observers; no safety
        // but local's; no entry, nothing.
        let ctl = [Role::Controller];
        assert_eq!(
            status(access.check_attach("laptop", &ctl, PriorityClass::Model)),
            None
        );
        assert_eq!(
            status(access.check_attach("laptop", &ctl, PriorityClass::Operator)),
            denied
        );
        assert_eq!(
            status(access.check_attach("eye", &[Role::Observer], PriorityClass::Model)),
            None
        );
        assert_eq!(
            status(access.check_attach("eye", &ctl, PriorityClass::Scripted)),
            denied
        );
        assert_eq!(
            status(access.check_attach("rogue", &ctl, PriorityClass::Safety)),
            denied
        );
        assert_eq!(
            status(access.check_attach("local", &ctl, PriorityClass::Safety)),
            None
        );
        assert_eq!(
            status(access.check_attach("stranger", &ctl, PriorityClass::Scripted)),
            denied
        );
        // Leases: the part and the class.
        assert_eq!(
            status(access.check_lease("laptop", &ur, "arm", PriorityClass::Model)),
            None
        );
        let err = access
            .check_lease("laptop", &ur, "base", PriorityClass::Model)
            .expect_err("outside its parts");
        assert!(err.message.contains("ur10e/base"), "{err}");
        assert_eq!(
            status(access.check_lease("laptop", &ur, "arm", PriorityClass::Operator)),
            denied
        );
        // Verbs: arm and disarm from operator up, by default.
        assert_eq!(status(access.check_verb("laptop", Verb::Arm)), denied);
        assert_eq!(status(access.check_verb("rogue", Verb::Disarm)), None);
        assert_eq!(status(access.check_verb("local", Verb::Arm)), None);
    }
}
