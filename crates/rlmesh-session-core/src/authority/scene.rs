//! A scene's session: its member table, its
//! rejoin window and its closing. Fenced behind the `fleet` feature; its
//! places are an ordinary session's ([`super::AuthorityConfig::places`]).
//!
//! - **Members.** Only the scene's members attach, each from its device: a
//!   robot from its pinned device (`board_id` is checked at declare), a
//!   controller from its member's device up to its cap. Anyone else
//!   attaches only as an observer (`NOT_AUTHORIZED`).
//! - **Rejoin**: a member lost from the run is waited
//!   for until `rejoin_until_ns` ([`SceneConfig::rejoin_ns`] after the
//!   loss); a robot is back once it declared again, the same map as before
//!   (`INCOMPATIBLE` otherwise), and the run arms only once every member is
//!   back, on an explicit arm.
//! - **Closing** ([`super::Authority::close_scene`]), in one revision: the
//!   mode goes DISARMED (ESTOP stays), every lease ends with the reason,
//!   and `scene.closing` (every device view's `closing`) is set.

use alloc::collections::{BTreeMap, BTreeSet};

use super::{Authority, AuthorityEvent, Identity, Mode};
use crate::prelude::*;
use crate::wire::{
    Command, CtlError, CtlStatus, Fingerprint, ParticipantId, PriorityClass, Reason, ReasonCode,
    Role, SceneInfo,
};

/// One member of a scene.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Member {
    /// A robot: it embodies its pinned name.
    Robot {
        /// Its pinned device (`None`: unpinned, so the name binds to its
        /// first embodiment's device).
        device: Option<String>,
        /// While unpinned, the devices whose first embodiment may pin it: `local`, a
        /// board provisioned under its
        /// name, a device with the `arm` verb. Anyone else is refused.
        may_pin: Vec<String>,
    },
    /// A controller (a script, a served model).
    Controller {
        /// The device its processes attach from.
        device: String,
        /// The highest class it attaches and leases at.
        max_priority: PriorityClass,
    },
}

/// What a scene's session carries besides an ordinary session's rules.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneConfig {
    /// The run, the scene's name and fingerprint (`closing` and
    /// `rejoin_until_ns` are the authority's).
    pub info: SceneInfo,
    /// Its members by name.
    pub members: BTreeMap<ParticipantId, Member>,
    /// How long a lost member is waited for (0: the run fails at once).
    pub rejoin_ns: u64,
}

fn refused(message: String) -> CtlError {
    CtlError::new(CtlStatus::NotAuthorized, message)
}

fn class_name(class: PriorityClass) -> &'static str {
    match class {
        PriorityClass::Scripted => "scripted",
        PriorityClass::Model => "model",
        PriorityClass::Operator => "operator",
        PriorityClass::Safety => "safety",
    }
}

impl SceneConfig {
    /// Whether `participant`, on `device`, may attach with `roles` at
    /// `priority`: observers always; otherwise only a member, from its
    /// device, a robot as an embodiment and a controller within its cap.
    ///
    /// # Errors
    /// `NOT_AUTHORIZED`, naming the member and the fix.
    pub fn admit(
        &self,
        participant: &ParticipantId,
        device: &str,
        roles: &[Role],
        priority: PriorityClass,
    ) -> Result<(), CtlError> {
        if roles.iter().all(|r| *r == Role::Observer) {
            return Ok(());
        }
        // The home daemon watches and arms a run as an observer: arming is
        // its device's verb, not a role.
        let scene = &self.info.name;
        let member = self.members.get(participant).ok_or_else(|| {
            refused(format!(
                "{participant} is not a member of scene {scene}: only its members attach, \
                 anyone else as an observer; add [members.{participant}] to the scene"
            ))
        })?;
        let embodies = roles.contains(&Role::Embodiment);
        match member {
            Member::Robot {
                device: pinned,
                may_pin,
            } => {
                if let Some(pinned) = pinned.as_deref().filter(|p| *p != device) {
                    return Err(refused(format!(
                        "{participant} is pinned to {pinned}, not {device}: rlmesh devices move \
                         {participant} --to {device} if it moved"
                    )));
                }
                if pinned.is_none() && !may_pin.iter().any(|d| d == device) {
                    return Err(refused(format!(
                        "{participant} is not pinned, and {device} may not pin it: provision the \
                         board as {participant}, or rlmesh devices allow {device} --verbs arm"
                    )));
                }
                if !embodies {
                    return Err(refused(format!(
                        "{participant} is a robot member of scene {scene}: it attaches as an \
                         embodiment"
                    )));
                }
            }
            Member::Controller {
                device: own,
                max_priority,
            } => {
                if own != device {
                    return Err(refused(format!(
                        "{participant} is scene {scene}'s member from {own}, not {device}"
                    )));
                }
                if embodies {
                    return Err(refused(format!(
                        "{participant} is a controller member of scene {scene}: it embodies no \
                         robot"
                    )));
                }
                if priority > *max_priority {
                    return Err(refused(format!(
                        "{participant} may lease at {} at most in scene {scene} (asked {}); \
                         raise its max_priority in the scene",
                        class_name(*max_priority),
                        class_name(priority)
                    )));
                }
            }
        }
        Ok(())
    }
}

/// A scene run's state in the authority: why it is closing, and the members
/// lost and not back yet (a robot is back once it declared again) with
/// when the run stops waiting for them.
#[derive(Clone, Debug, Default)]
pub(super) struct SceneState {
    closing: Option<Reason>,
    rejoining: BTreeSet<ParticipantId>,
    rejoin_until_ns: Option<i64>,
}

/// The scene's rules inside [`Authority`]; without `fleet` each is a no-op
/// (`super::no_scene`).
impl Authority {
    fn scene_config(&self) -> Option<&SceneConfig> {
        self.config.scene.as_ref()
    }

    pub(super) fn scene_info(&self) -> Option<Box<SceneInfo>> {
        self.scene_config().map(|s| {
            Box::new(SceneInfo {
                closing: self.scene.closing.clone(),
                rejoin_until_ns: self.scene.rejoin_until_ns,
                ..s.info.clone()
            })
        })
    }

    /// A closing scene is over: nothing arms it or
    /// takes a lease in it again, a grant waiting for its keys included.
    pub(super) fn scene_refuses(&self, command: &Command) -> Result<(), CtlError> {
        match &self.scene.closing {
            Some(closing)
                if matches!(
                    command,
                    Command::Arm | Command::Rearm { .. } | Command::GrantLease { .. }
                ) =>
            {
                Err(CtlError::new(
                    CtlStatus::InvalidTransition,
                    format!("the scene's run is closing ({})", closing.code),
                ))
            }
            _ => Ok(()),
        }
    }

    pub(super) fn scene_admit(
        &self,
        from: &ParticipantId,
        sender: &Identity,
        roles: &[Role],
        priority: PriorityClass,
    ) -> Result<(), CtlError> {
        match self.scene_config() {
            Some(scene) => scene.admit(from, &sender.device, roles, priority),
            None => Ok(()),
        }
    }

    /// The id of the run this session hosts.
    pub(super) fn scene_run(&self) -> Option<&str> {
        self.scene_config().map(|s| s.info.run.as_str())
    }

    /// Whether `id` is a member of the scene this session runs.
    pub(super) fn scene_member(&self, id: &ParticipantId) -> bool {
        self.scene_config()
            .is_some_and(|s| s.members.contains_key(id))
    }

    /// Whether `id` is a lost member the run still waits for.
    pub(super) fn scene_rejoining(&self, id: &ParticipantId) -> bool {
        self.scene.rejoining.contains(id)
    }

    /// `id` is back in the run: the window closes once every lost member
    /// is.
    pub(super) fn rejoined(&mut self, id: &ParticipantId) {
        if self.scene.rejoining.remove(id) && self.scene.rejoining.is_empty() {
            self.scene.rejoin_until_ns = None;
        }
    }

    /// The run arms only once every lost member is back.
    pub(super) fn scene_arm_ready(&self) -> Result<(), CtlError> {
        match self.scene.rejoining.first() {
            Some(member) => Err(CtlError::new(
                CtlStatus::NotReady,
                format!("{member} has not rejoined the run: it arms once every member is back"),
            )),
            None => Ok(()),
        }
    }

    /// A member is lost: waited for until the rejoin window closes. Returns
    /// whether `id` is a member.
    pub(super) fn scene_lost(&mut self, id: &ParticipantId) -> bool {
        let Some(rejoin_ns) = self
            .scene_config()
            .filter(|s| s.members.contains_key(id))
            .map(|s| s.rejoin_ns)
        else {
            return false;
        };
        self.scene.rejoining.insert(id.clone());
        let until = self.now_ns.saturating_add_unsigned(rejoin_ns);
        self.scene.rejoin_until_ns.get_or_insert(until);
        true
    }

    /// A scene's robot keeps its map for the run: bindings were made
    /// against it.
    pub(super) fn scene_keeps_map(
        &self,
        from: &ParticipantId,
        fingerprint: &Fingerprint,
    ) -> Result<(), CtlError> {
        match self.contract.get(from).filter(|f| *f != fingerprint) {
            Some(before) if self.scene_config().is_some() => Err(CtlError::new(
                CtlStatus::Incompatible,
                format!(
                    "{from} declared another map (contract {fingerprint}, and this run's is \
                     {before}): flash the map it ran with, or start a new run"
                ),
            )),
            _ => Ok(()),
        }
    }

    /// Closes the scene this session runs with `code`
    /// (`SCENE_ENDED`, or `SCENE_FAILED` with the stage's error in
    /// `detail`), `by` whoever stopped it, in one revision: the mode goes
    /// DISARMED (ESTOP stays), every lease ends with the reason, and
    /// `scene.closing` is set, which tells robots to detach and return to the
    /// lobby. Nothing happens outside a scene or once it is closing.
    pub fn close_scene(
        &mut self,
        code: ReasonCode,
        by: &str,
        detail: &str,
        now_ns: i64,
    ) -> Vec<AuthorityEvent> {
        self.now_ns = now_ns;
        let mut events = Vec::new();
        if self.config.scene.is_none() || self.scene.closing.is_some() {
            return events;
        }
        let why = self.reason(code, by, detail);
        self.revoke_all(code, &why, &mut events);
        if self.mode != Mode::Estop {
            self.set_mode(Mode::Disarmed, why.clone(), &mut events);
        }
        self.scene.closing = Some(why);
        self.revision += 1;
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pid(id: &str) -> ParticipantId {
        ParticipantId::new(id).expect("id")
    }

    #[test]
    fn only_members_attach_from_their_devices() {
        let scene = SceneConfig {
            info: SceneInfo {
                run: "r-7k2m".into(),
                name: "eyeball-rig".into(),
                fingerprint: Fingerprint::default(),
                closing: None,
                rejoin_until_ns: None,
            },
            rejoin_ns: 0,
            members: [
                (
                    pid("eye-l"),
                    Member::Robot {
                        device: Some("eye-l".into()),
                        may_pin: Vec::new(),
                    },
                ),
                (
                    pid("skull"),
                    Member::Robot {
                        device: None,
                        may_pin: vec!["local".into(), "skull".into()],
                    },
                ),
                (
                    pid("tracker"),
                    Member::Controller {
                        device: "laptop-josh".into(),
                        max_priority: PriorityClass::Model,
                    },
                ),
            ]
            .into(),
        };
        let robot = [Role::Embodiment];
        let controller = [Role::Controller];
        let (model, operator) = (PriorityClass::Model, PriorityClass::Operator);
        // (participant, device, roles, priority, admitted, why)
        for (who, device, roles, priority, admitted, why) in [
            ("eye-l", "eye-l", &robot[..], model, true, ""),
            (
                "eye-l",
                "arm-pc",
                &robot[..],
                model,
                false,
                "pinned to eye-l, not arm-pc",
            ),
            ("skull", "local", &robot[..], model, true, ""),
            ("skull", "skull", &robot[..], model, true, ""),
            // Unpinned: a guest without `arm`, or a board under another
            // name, may not take its name.
            (
                "skull",
                "guest",
                &robot[..],
                model,
                false,
                "guest may not pin it",
            ),
            (
                "skull",
                "eye-l",
                &robot[..],
                model,
                false,
                "eye-l may not pin it",
            ),
            ("tracker", "laptop-josh", &controller[..], model, true, ""),
            (
                "tracker",
                "arm-pc",
                &controller[..],
                model,
                false,
                "from laptop-josh, not arm-pc",
            ),
            (
                "tracker",
                "laptop-josh",
                &controller[..],
                operator,
                false,
                "at model at most",
            ),
            (
                "tracker",
                "laptop-josh",
                &robot[..],
                model,
                false,
                "embodies no robot",
            ),
            (
                "eye-l",
                "eye-l",
                &controller[..],
                model,
                false,
                "attaches as an embodiment",
            ),
            (
                "eye-r",
                "local",
                &robot[..],
                model,
                false,
                "not a member of scene eyeball-rig",
            ),
            (
                "rogue",
                "arm-pc",
                &controller[..],
                model,
                false,
                "add [members.rogue]",
            ),
            ("cockpit", "arm-pc", &[Role::Observer][..], model, true, ""),
        ] {
            let got = scene.admit(&pid(who), device, roles, priority);
            match got {
                Ok(()) => assert!(admitted, "{who} from {device} admitted"),
                Err(err) => {
                    assert!(!admitted, "{who} from {device}: {err}");
                    assert_eq!(err.status, CtlStatus::NotAuthorized);
                    assert!(err.message.contains(why), "{who}: {}", err.message);
                }
            }
        }
    }
}
