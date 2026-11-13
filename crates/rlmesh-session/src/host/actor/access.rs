//! Runtime changes of the device table and robot pins, and closing a scene.

use crate::wire::{ParticipantId, Role};
use rlmesh_session_core::auth::{Device, LOCAL_DEVICE, Pin};
use tracing::{info, warn};

use crate::authority::{AuthorityEvent, DeviceAccess};
use crate::error::SessionError;
#[cfg(doc)]
use crate::host::SessionHost;
use crate::host::{HostActor, OsRng};
#[cfg(feature = "fleet")]
use crate::wire::ReasonCode;

impl HostActor {
    /// Changes `device`'s entry: see [`SessionHost::set_access`].
    pub(super) async fn set_access(
        &mut self,
        device: &str,
        entry: DeviceAccess,
    ) -> Result<(), SessionError> {
        if self.authority.config().access.get(device).is_none() {
            return Err(SessionError::Config(format!(
                "device {device} has no entry on this host"
            )));
        }
        let (events, detached) = self.authority.set_access(device, entry, self.now());
        info!(%device, ?detached, "device entry changed");
        self.access_changed(events).await;
        Ok(())
    }

    /// Accepts a new device: see [`SessionHost::admit`].
    pub(super) async fn admit(&mut self, device: Device) -> Result<(), SessionError> {
        if device.id == LOCAL_DEVICE || self.authority.config().access.get(&device.id).is_some() {
            return Err(SessionError::Config(format!(
                "device {} is local or known already",
                device.id
            )));
        }
        self.auth.enroll(device.public, device.id.clone());
        let (events, _) = self
            .authority
            .set_access(&device.id, device.access, self.now());
        info!(device = %device.id, "device admitted");
        self.access_changed(events).await;
        Ok(())
    }

    /// Revokes a device: see [`SessionHost::revoke`].
    pub(super) async fn revoke(&mut self, device: &str) -> Result<(), SessionError> {
        if self.authority.config().access.get(device).is_none() {
            return Err(SessionError::Config(format!(
                "device {device} has no entry on this host"
            )));
        }
        self.auth.unenroll(device);
        let (events, detached) = self.authority.revoke(device, self.now());
        info!(%device, ?detached, "device revoked");
        self.access_changed(events).await;
        Ok(())
    }

    /// The device table changed: the stores see it, and every stream key a
    /// device can no longer read rotates.
    async fn access_changed(&mut self, events: Vec<AuthorityEvent>) {
        let access = self.authority.config().access.clone();
        if let Ok(mut shared) = self.store.access.write() {
            *shared = access.clone();
        }
        self.record(events).await;
        let now = self.now();
        if let Err(err) = self.auth.issuer().reauthorize(&mut OsRng, &access, now) {
            warn!(%err, "stream keys");
        }
    }

    /// Closes the scene: see [`SessionHost::close_scene`].
    #[cfg(feature = "fleet")]
    pub(super) async fn close_scene(
        &mut self,
        code: ReasonCode,
        by: &str,
        detail: &str,
    ) -> Result<(), SessionError> {
        if self.config.scene.is_none() {
            return Err(SessionError::Config("this session runs no scene".into()));
        }
        let events = self.authority.close_scene(code, by, detail, self.now());
        // A disarmed session with no lease changes only `closing`.
        if events.is_empty() {
            self.publish().await;
        } else {
            self.record(events).await;
        }
        Ok(())
    }

    /// Pins a robot name: see [`SessionHost::pin`].
    pub(super) async fn pin(&mut self, robot: &ParticipantId, pin: Option<Pin>) {
        let device = pin.as_ref().map(|p| p.device.clone());
        let before = self.auth.pin(robot, pin);
        let embodied = self
            .authority
            .participant(robot)
            .filter(|p| p.present && p.has_role(Role::Embodiment))
            .map(|p| p.device.clone());
        info!(%robot, ?device, "pin changed");
        if let Some(embodied) = embodied
            && (before.is_some() || device.as_deref() != Some(embodied.as_str()))
        {
            let events = self.authority.participant_lost(robot, self.now());
            self.record(events).await;
        }
    }
}
