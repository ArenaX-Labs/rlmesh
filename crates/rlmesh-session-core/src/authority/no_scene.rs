//! Without the `fleet` feature a session runs no scene: each of the scene's
//! rules in [`Authority`] is a no-op (`scene.rs` has them).

use super::{Authority, Identity};
use crate::prelude::*;
use crate::wire::{Command, CtlError, Fingerprint, ParticipantId, PriorityClass, Role, SceneInfo};

#[allow(clippy::unused_self, clippy::unnecessary_wraps)]
impl Authority {
    pub(super) fn scene_info(&self) -> Option<Box<SceneInfo>> {
        None
    }

    pub(super) fn scene_refuses(&self, _: &Command) -> Result<(), CtlError> {
        Ok(())
    }

    pub(super) fn scene_admit(
        &self,
        _: &ParticipantId,
        _: &Identity,
        _: &[Role],
        _: PriorityClass,
    ) -> Result<(), CtlError> {
        Ok(())
    }

    pub(super) fn scene_member(&self, _: &ParticipantId) -> bool {
        false
    }

    pub(super) fn scene_rejoining(&self, _: &ParticipantId) -> bool {
        false
    }

    pub(super) fn rejoined(&mut self, _: &ParticipantId) {}

    pub(super) fn scene_arm_ready(&self) -> Result<(), CtlError> {
        Ok(())
    }

    pub(super) fn scene_lost(&mut self, _: &ParticipantId) -> bool {
        false
    }

    pub(super) fn scene_keeps_map(
        &self,
        _: &ParticipantId,
        _: &Fingerprint,
    ) -> Result<(), CtlError> {
        Ok(())
    }
}
