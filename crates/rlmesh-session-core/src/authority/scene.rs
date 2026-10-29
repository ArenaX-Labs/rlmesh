//! A scene's session (scene design §3.6, §4.2, §5): the places its state
//! carries, its member table and its closing.
//!
//! - **Members.** Only the scene's members attach, each from its device: a
//!   robot from its pinned device (`board_id` is checked at declare), a
//!   controller from its member's device up to its cap. Anyone else
//!   attaches only as an observer (`NOT_AUTHORIZED`).
//! - **Places** go out in every snapshot, so every binder builds the same
//!   view; a declaration whose placed paths collide is refused
//!   (`INCOMPATIBLE`).
//! - **Closing** ([`super::Authority::close_scene`]), in one revision: the
//!   mode goes DISARMED (ESTOP stays), every lease ends with the reason,
//!   and `scene.closing` (every device view's `closing`) is set.

use alloc::collections::BTreeMap;

use crate::prelude::*;
use crate::wire::{CtlError, CtlStatus, ParticipantId, Place, PriorityClass, Role, SceneInfo};

/// One member of a scene.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Member {
    /// A robot: it embodies its pinned name.
    Robot {
        /// Its pinned device (`None`: unpinned, so the name binds to its
        /// first embodiment's device).
        device: Option<String>,
        /// While unpinned, the devices whose first embodiment may pin it
        /// (scene design §3.2): `local`, a board provisioned under its
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
    /// Where each placed robot's parts appear in the view.
    pub places: Vec<Place>,
    /// Its members by name.
    pub members: BTreeMap<ParticipantId, Member>,
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
        let scene = &self.info.name;
        // TODO: admit the home daemon's own `rlmeshd` participant, which
        // arms runs as an operator (scene step 7).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::Fingerprint;

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
            places: Vec::new(),
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
