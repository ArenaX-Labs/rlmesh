//! The host's side: its identity, and per host instance the device table,
//! the handshakes it answers and which generation is in effect.

use super::credential::Credential;
use super::{HOST_ID_LABEL, KeyIssuer, LOCAL_DEVICE, address, local_keypair};
use crate::authority::Access;
use crate::crypto::StaticKeypair;
use crate::crypto::handshake::{self, IkResponder};
use crate::crypto::lp::hash_labelled;
use crate::crypto::{Rng, RngError, SealReceiver, SealSender, SignedSender};
use crate::prelude::*;
use crate::wire::{
    CtlError, CtlStatus, DatagramOffer, HandshakeAccept, HandshakeRequest, ParticipantId,
};
use crate::wire::{codec, version};
use alloc::collections::BTreeMap;
use zeroize::{Zeroize, Zeroizing};

/// A host's long-lived keys: its static X25519 key, its Ed25519 signing
/// key and, for an embedded host, the local secret its address carries.
pub struct HostIdentity {
    pub(super) noise: StaticKeypair,
    sign: Zeroizing<[u8; 32]>,
    sign_public: [u8; 32],
    local: Zeroizing<[u8; 32]>,
}

impl core::fmt::Debug for HostIdentity {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HostIdentity")
            .field("static", self.noise.public())
            .finish_non_exhaustive()
    }
}

impl HostIdentity {
    /// A new identity from `rng`, as an embedded host draws at start.
    ///
    /// # Errors
    /// [`RngError`].
    pub fn generate(rng: &mut dyn Rng) -> Result<Self, RngError> {
        let noise = StaticKeypair::generate(rng)?;
        let mut sign = Zeroizing::new([0u8; 32]);
        rng.fill_bytes(&mut sign[..])?;
        let mut local = Zeroizing::new([0u8; 32]);
        rng.fill_bytes(&mut local[..])?;
        let sign_public = SignedSender::new(&sign, 0).verifying_key();
        Ok(Self {
            noise,
            sign,
            sign_public,
            local,
        })
    }

    /// The identity of these secrets (its static X25519 private key, its
    /// Ed25519 seed and its local secret), which are wiped in place: what a
    /// launcher hands the host it starts in a container (the
    /// workload spec).
    pub fn from_secrets(noise: &mut [u8; 32], sign: &mut [u8; 32], local: &mut [u8; 32]) -> Self {
        let noise = StaticKeypair::from_private(noise);
        let sign_bytes = Zeroizing::new(*sign);
        let local_bytes = Zeroizing::new(*local);
        sign.zeroize();
        local.zeroize();
        let sign_public = SignedSender::new(&sign_bytes, 0).verifying_key();
        Self {
            noise,
            sign: sign_bytes,
            sign_public,
            local: local_bytes,
        }
    }

    /// The static X25519 public key participants handshake with.
    pub fn static_public(&self) -> &[u8; 32] {
        self.noise.public()
    }

    /// The host id: `rlm-` and 6 hex digits of
    /// `BLAKE2s(lp("rlmesh/v2 host id") ‖ static key ‖ signing key)`. A
    /// name for files and messages only: whoever pairs or is provisioned
    /// keeps the static key itself.
    pub fn host_id(&self) -> String {
        let digest = hash_labelled(HOST_ID_LABEL, &[self.noise.public(), &self.sign_public])
            .unwrap_or_else(|_| unreachable!("a short label"));
        format!("rlm-{:02x}{:02x}{:02x}", digest[0], digest[1], digest[2])
    }

    /// The local secret an embedded host's address carries.
    pub fn local_secret(&self) -> &[u8; 32] {
        &self.local
    }

    /// Copies its static X25519 private key and its Ed25519 seed into
    /// `noise` and `sign`, for storage (rlmeshd's `host.key`); wiping them
    /// is the caller's job.
    pub fn write_secrets(&self, noise: &mut [u8; 32], sign: &mut [u8; 32]) {
        self.noise.write_private(noise);
        sign.copy_from_slice(&self.sign[..]);
    }

    /// The password of this host's gated links
    /// ([`address::link_password`]).
    pub fn link_password(&self) -> Zeroizing<String> {
        address::link_password(&self.local)
    }

    /// The `local` device's static public key.
    pub fn local_public(&self) -> [u8; 32] {
        *local_keypair(&self.local).public()
    }

    /// The credential of a process holding this host's address.
    pub fn local_credential(&self) -> Credential {
        Credential::local(&self.local, *self.noise.public())
    }

    /// The signer of one host instance's broadcasts.
    pub fn signer(&self, host_instance: u64) -> SignedSender {
        SignedSender::new(&self.sign, host_instance)
    }
}

/// A device the host accepts handshakes from besides `local`: its id, its
/// static public key and its entry. An embedded host takes
/// them from its config; rlmeshd will read them from `devices.toml`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device {
    /// Its id (never `local`).
    pub id: String,
    /// Its static X25519 public key.
    pub public: [u8; 32],
    /// What it may do.
    pub access: crate::authority::DeviceAccess,
}

/// A handshake the host answered: keys for one participant generation,
/// in effect once the host accepts its sealed `Attach`.
#[derive(Debug)]
pub struct Candidate {
    /// The participant id it attaches as.
    pub participant: ParticipantId,
    /// The device its static key belongs to.
    pub device: String,
    /// Its generation.
    pub generation: u32,
    /// The id of `k_up`.
    pub up_key_id: u32,
    /// `k_up`: what the host opens.
    pub up: SealReceiver,
    /// `k_down`: what the host seals.
    pub down: SealSender,
}

/// A robot name's pin.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pin {
    /// The device that embodies it, checked at every handshake.
    pub device: String,
    /// The tethered board behind that device, checked at every
    /// `DeclareParts`; `None` leaves the board unchecked (a native board, or
    /// a tethered one pinned by its device alone).
    pub board_id: Option<[u8; 8]>,
}

/// One host instance's handshakes: see the module docs.
#[derive(Debug)]
pub struct HostAuth {
    tenant: String,
    session: String,
    host_instance: u64,
    sign_public: [u8; 32],
    /// Device static public key → device id.
    devices: BTreeMap<[u8; 32], String>,
    /// Participant id → the device it is bound to, for this host instance.
    names: BTreeMap<ParticipantId, String>,
    /// Robot name → its pin (rlmeshd's): the device, checked at
    /// every handshake, and the board, at every declare.
    pins: BTreeMap<ParticipantId, Pin>,
    /// Participant id → the generation in effect.
    current: BTreeMap<ParticipantId, u32>,
    last_generation: u32,
    /// Every key besides the handshake's, and the one key-id counter.
    issuer: KeyIssuer,
    /// The datagram path each handshake offers, if any.
    datagram: Option<DatagramOffer>,
}

/// A handshake that went wrong for a reason the initiator should hear (an
/// unauthenticated hint).
fn refused(status: CtlStatus, message: impl core::fmt::Display) -> CtlError {
    CtlError::new(status, message)
}

impl HostAuth {
    /// Handshakes for one instance of `identity`'s host, serving `session`
    /// of `tenant`. The device table holds `local`, every holder of the
    /// host's address, and whatever [`Self::enroll`] adds.
    pub fn new(identity: &HostIdentity, tenant: &str, session: &str, host_instance: u64) -> Self {
        let mut table = BTreeMap::new();
        table.insert(identity.local_public(), LOCAL_DEVICE.to_string());
        Self {
            tenant: tenant.to_string(),
            session: session.to_string(),
            host_instance,
            sign_public: identity.sign_public,
            devices: table,
            names: BTreeMap::new(),
            pins: BTreeMap::new(),
            current: BTreeMap::new(),
            last_generation: 0,
            issuer: KeyIssuer::default(),
            datagram: None,
        }
    }

    /// Offers `datagram` in every handshake from now on (`None`: offers
    /// nothing, and participants keep every part on the session).
    pub fn offer_datagram(&mut self, datagram: Option<DatagramOffer>) {
        self.datagram = datagram;
    }

    /// The host's key schedule.
    pub fn issuer(&mut self) -> &mut KeyIssuer {
        &mut self.issuer
    }

    /// Accepts handshakes from the device with this static public key.
    pub fn enroll(&mut self, public: [u8; 32], device: impl Into<String>) {
        self.devices.insert(public, device.into());
    }

    /// Refuses `device`'s handshakes from now on (it was revoked). The
    /// names it is bound to stay bound, so they can't be taken.
    pub fn unenroll(&mut self, device: &str) {
        self.devices.retain(|_, id| id != device);
    }

    /// Pins robot name `robot` (`None`: unpins it, and the name binds to
    /// its next embodiment's device again): the pin it had.
    pub fn pin(&mut self, robot: &ParticipantId, pin: Option<Pin>) -> Option<Pin> {
        match pin {
            Some(pin) => {
                self.names.insert(robot.clone(), pin.device.clone());
                self.pins.insert(robot.clone(), pin)
            }
            None => {
                self.names.remove(robot);
                self.pins.remove(robot)
            }
        }
    }

    /// Whether `robot`, embodied by `device`, may declare its parts from
    /// the board `board_id`: its pin names no board, or
    /// this one.
    ///
    /// # Errors
    /// `NOT_AUTHORIZED` for another board, or none.
    pub fn check_board(
        &self,
        robot: &ParticipantId,
        device: &str,
        board_id: Option<&[u8; 8]>,
    ) -> Result<(), CtlError> {
        let Some(want) = self.pins.get(robot).and_then(|p| p.board_id.as_ref()) else {
            return Ok(());
        };
        if board_id == Some(want) {
            return Ok(());
        }
        let hex = |id: &[u8; 8]| id.iter().map(|b| format!("{b:02x}")).collect::<String>();
        let got = board_id.map_or_else(|| "none".to_string(), hex);
        Err(refused(
            CtlStatus::NotAuthorized,
            format!(
                "board on {device} is {got}, expected {robot}'s {}",
                hex(want)
            ),
        ))
    }

    /// The instance these handshakes are bound to.
    pub fn host_instance(&self) -> u64 {
        self.host_instance
    }

    /// Answers msg 1 with the device's cap from `access`: msg 2 and the
    /// candidate, or the refusal to send back
    /// (`INCOMPATIBLE` for a participant on another wire revision,
    /// `UNAUTHENTICATED` for an unknown device or a message that fails to
    /// authenticate, `NOT_AUTHORIZED` for a participant id bound to another
    /// device).
    ///
    /// # Errors
    /// The refusal.
    pub fn handshake(
        &mut self,
        rng: &mut dyn Rng,
        identity: &HostIdentity,
        request: &HandshakeRequest,
        access: &Access,
    ) -> Result<(Vec<u8>, Candidate), CtlError> {
        let participant = &request.participant;
        if let Some(why) =
            version::participant_revision_mismatch(participant.as_str(), request.wire_revision)
        {
            return Err(refused(CtlStatus::Incompatible, why));
        }
        let prologue = handshake::prologue(
            &self.tenant,
            &self.session,
            self.host_instance,
            participant.as_str(),
        )
        .map_err(|_| refused(CtlStatus::InvalidRequest, "a field over 65535 bytes"))?;
        let responder = IkResponder::read_request(rng, &identity.noise, &prologue, &request.noise)
            .map_err(|err| {
                refused(
                    CtlStatus::Unauthenticated,
                    format!(
                        "the handshake failed ({err}): the address is stale or names another \
                         host instance"
                    ),
                )
            })?;
        let device = self
            .devices
            .get(responder.initiator_static())
            .cloned()
            .ok_or_else(|| refused(CtlStatus::Unauthenticated, "this device is not enrolled"))?;
        self.check_name(participant, &device)?;
        let exhausted = || {
            refused(
                CtlStatus::Unsupported,
                "this host instance has issued every generation or key id; restart it",
            )
        };
        let generation = self.last_generation.checked_add(1).ok_or_else(exhausted)?;
        let up_key_id = self.issuer.next_key_id().ok_or_else(exhausted)?;
        let down_key_id = self.issuer.next_key_id().ok_or_else(exhausted)?;
        let accept = HandshakeAccept {
            generation,
            device: device.clone(),
            max_priority: access.max_priority(&device),
            host_sign_key: self.sign_public,
            host_instance: self.host_instance,
            up_key_id,
            down_key_id,
            datagram: self.datagram,
            wire_revision: version::WIRE_REVISION,
        };
        let (msg2, keys) = responder
            .accept(&codec::encode_handshake_accept(&accept))
            .map_err(|err| refused(CtlStatus::Unauthenticated, err))?;
        self.last_generation = generation;
        let (down, up) = keys.into_host(up_key_id, down_key_id);
        Ok((
            msg2,
            Candidate {
                participant: participant.clone(),
                device,
                generation,
                up_key_id,
                up,
                down,
            },
        ))
    }

    fn check_name(&self, participant: &ParticipantId, device: &str) -> Result<(), CtlError> {
        match self
            .pins
            .get(participant)
            .map(|p| &p.device)
            .or_else(|| self.names.get(participant))
        {
            Some(bound) if bound != device => Err(refused(
                CtlStatus::NotAuthorized,
                format!("name {participant} belongs to device {bound}"),
            )),
            _ => Ok(()),
        }
    }

    /// The generation in effect for `participant`, if any.
    pub fn current(&self, participant: &ParticipantId) -> Option<u32> {
        self.current.get(participant).copied()
    }

    /// Whether generation `generation` of `participant` (on `device`) may
    /// take effect: it is above the one in effect, and the name is free or
    /// bound to its device. Checked before its `Attach` is applied.
    ///
    /// # Errors
    /// `UNAUTHENTICATED` for a superseded candidate, `NOT_AUTHORIZED` for a
    /// name bound to another device.
    pub fn may_take_effect(
        &self,
        participant: &ParticipantId,
        device: &str,
        generation: u32,
    ) -> Result<(), CtlError> {
        if self.current(participant).is_some_and(|g| g >= generation) {
            return Err(refused(
                CtlStatus::Unauthenticated,
                format!("generation {generation} of {participant} is superseded"),
            ));
        }
        self.check_name(participant, device)
    }

    /// Generation `generation`'s accepted `Attach` made it the one in
    /// effect, and bound the name to `device` for this host instance. Every
    /// older generation of the participant is superseded from now on.
    pub fn take_effect(&mut self, participant: &ParticipantId, device: &str, generation: u32) {
        self.current.insert(participant.clone(), generation);
        self.names
            .entry(participant.clone())
            .or_insert_with(|| device.to_string());
    }
}
