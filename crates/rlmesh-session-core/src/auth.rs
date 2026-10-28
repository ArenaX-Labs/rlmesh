//! Attaching through the handshake (design §2, §8.3, §8.6), sans-IO:
//!
//! - a participant's [`Credential`] (its device's static key and the host's
//!   static key), which runs the IK handshake ([`Credential::start`],
//!   [`finish`]) and leaves the participant its pairwise keys and the
//!   host's signing key;
//! - a participant's [`Link`] to the host, built from them: its sealed
//!   verbs and heartbeats, the replies and echoes it opens, request ids,
//!   and the host's signed broadcasts it verifies. The async participant
//!   and a board's runtime both drive one, and only do IO around it;
//! - a host's [`HostIdentity`] (its static X25519 key, its Ed25519 key and
//!   the embedded host's local secret) and, per host instance, its
//!   [`HostAuth`]: the device table, the generation and key-id counters,
//!   name → device bindings and which generation of each participant is in
//!   effect;
//! - the host's per-generation [`RequestWindow`]: request ids and the
//!   reply cache that makes a retried verb idempotent;
//! - the keys the host delivers besides the handshake's ([`keyring`] on
//!   each holder, [`issuer`] at the host, §8.4, §8.5): a lease's `k_cmd` and
//!   `k_obs`, a stream's `k_stream`. A [`Link`] holds its participant's
//!   [`Keyring`], installs what the host delivers (`KeyDelivery`, the keys a
//!   reply carries), and seals and opens chunks, samples, history and
//!   lockstep under them.
//!
//! What a participant sends under its keys, the host accepts only from the
//! generation in effect: a handshake makes a candidate, and the candidate
//! takes effect when the host accepts its sealed `Attach`
//! ([`HostAuth::take_effect`]). Randomness is the caller's ([`Rng`]).

use alloc::collections::BTreeMap;

use bytes::Bytes;
use zeroize::{Zeroize, Zeroizing};

pub mod address;
pub mod issuer;
pub mod keyring;

pub use address::AddressKey;
pub use issuer::KeyIssuer;
pub use keyring::{KeyError, Keyring, StreamWindows};

use crate::authority::Access;
use crate::crypto::StaticKeypair;
use crate::crypto::handshake::{self, HandshakeError, IkInitiator, IkResponder};
use crate::crypto::lp::kdf;
use crate::crypto::sealed::{HEADER_LEN, Header, TAG_LEN};
use crate::crypto::{
    Kind, OpenError, Rng, RngError, SealError, SealReceiver, SealSender, SignedError,
    SignedReceiver, SignedSender,
};
use crate::prelude::*;
use crate::wire::codec;
use crate::wire::error::WireError;
use crate::wire::heartbeat::{Heartbeat, HeartbeatEcho};
use crate::wire::tick::VirtualTick;
use crate::wire::{
    ActionChunk, AuthoritySnapshot, Command, CtlError, CtlReply, CtlStatus, Epoch, HandshakeAccept,
    HandshakeRequest, HistoryQuery, KeysAck, LeaseToken, LockstepRequest, LockstepResponse,
    ParticipantId, Request, Sample, SampleBatch, SampleMeta,
};

/// The device of every process holding an embedded host's address.
pub const LOCAL_DEVICE: &str = "local";
/// The `kdf` label of the `local` device's static key.
pub const LOCAL_LABEL: &[u8] = b"rlmesh/v2 local x25519";

/// The `local` device's static private key from an embedded host's local
/// secret: `kdf(local secret, "rlmesh/v2 local x25519")`.
fn local_keypair(secret: &[u8; 32]) -> StaticKeypair {
    let mut private = Zeroizing::new(
        kdf(secret, LOCAL_LABEL, &[]).unwrap_or_else(|_| unreachable!("a short label")),
    );
    StaticKeypair::from_private(&mut private)
}

/// What a participant attaches with: its device's static key and the
/// host's static public key.
#[derive(Clone)]
pub struct Credential {
    device: StaticKeypair,
    host_static: [u8; 32],
}

impl core::fmt::Debug for Credential {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Credential")
            .field("device", &self.device)
            .field("host_static", &self.host_static)
            .finish()
    }
}

impl Credential {
    /// A device's key pair and the host's static public key.
    pub fn new(device: StaticKeypair, host_static: [u8; 32]) -> Self {
        Self {
            device,
            host_static,
        }
    }

    /// The `local` device of an embedded host, from its address: the local
    /// secret and the host's static public key.
    pub fn local(secret: &[u8; 32], host_static: [u8; 32]) -> Self {
        Self::new(local_keypair(secret), host_static)
    }

    /// The device's static public key.
    pub fn device_public(&self) -> &[u8; 32] {
        self.device.public()
    }

    /// The host's static public key.
    pub fn host_static(&self) -> &[u8; 32] {
        &self.host_static
    }

    /// Starts a handshake as `participant` with the host instance
    /// `host_instance` (read from the public `host/state`; a wrong value
    /// only makes the handshake fail): the state and msg 1.
    ///
    /// # Errors
    /// [`HandshakeError`].
    pub fn start(
        &self,
        rng: &mut dyn Rng,
        tenant: &str,
        session: &str,
        host_instance: u64,
        participant: &ParticipantId,
    ) -> Result<(IkInitiator, HandshakeRequest), HandshakeError> {
        let prologue = handshake::prologue(tenant, session, host_instance, participant.as_str())
            .map_err(|_| HandshakeError::PayloadTooLong)?;
        let (initiator, noise) =
            IkInitiator::start(rng, &self.device, &self.host_static, &prologue)?;
        Ok((
            initiator,
            HandshakeRequest {
                participant: participant.clone(),
                noise,
            },
        ))
    }
}

/// A participant's side of a completed handshake.
#[derive(Debug)]
pub struct Attached {
    /// What the host issued.
    pub accept: HandshakeAccept,
    /// `k_up`: ctl requests and heartbeats.
    pub up: SealSender,
    /// `k_down`: ctl replies and heartbeat echoes.
    pub down: SealReceiver,
    /// The host's broadcasts, from the authenticated instance and key.
    pub host: SignedReceiver,
}

/// Why a handshake did not leave the participant attached.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthError {
    /// msg 2 failed to authenticate.
    #[error(transparent)]
    Handshake(#[from] HandshakeError),
    /// msg 2's payload is not a `HandshakeAccept`, or names a bad key.
    #[error("malformed handshake accept: {0}")]
    Malformed(String),
}

/// Reads msg 2: the participant's keys and the host's signing key.
///
/// # Errors
/// [`AuthError`].
pub fn finish(initiator: IkInitiator, msg2: &[u8]) -> Result<Attached, AuthError> {
    let (payload, keys) = initiator.finish(msg2)?;
    let accept = codec::decode_handshake_accept(&payload)
        .map_err(|err| AuthError::Malformed(err.to_string()))?;
    let host = SignedReceiver::new(&accept.host_sign_key, accept.host_instance)
        .map_err(|err| AuthError::Malformed(err.to_string()))?;
    let (up, down) = keys.into_participant(accept.up_key_id, accept.down_key_id);
    Ok(Attached {
        accept,
        up,
        down,
        host,
    })
}

/// Why a [`Link`] refused to send or dropped what it received.
#[derive(Debug, thiserror::Error)]
pub enum LinkError {
    /// Sealing failed (an exhausted counter, a key expression too long).
    #[error(transparent)]
    Seal(#[from] SealError),
    /// A message under `k_down` failed to open: forged, replayed or not
    /// ours.
    #[error(transparent)]
    Open(#[from] OpenError),
    /// A broadcast failed to verify: unsigned, another instance, or not
    /// newer than the last taken.
    #[error(transparent)]
    Signed(#[from] SignedError),
    /// It authenticated but did not decode.
    #[error(transparent)]
    Wire(#[from] WireError),
    /// A delivered key's message: unknown key, out of scope, or none to
    /// send under.
    #[error(transparent)]
    Key(#[from] KeyError),
    /// A history page answering another query.
    #[error("a page for history query {0}, not this one")]
    OtherQuery(u64),
    /// Request `0` is still unanswered, and a new id would put it below the
    /// host's window (§8.6).
    #[error("request {0} is still unanswered: {REQUEST_WINDOW} requests are outstanding")]
    Outstanding(u64),
}

/// What a reply on `ctl` was ([`Link::reply`]).
#[derive(Debug, Clone, PartialEq)]
pub enum ReplyOutcome {
    /// The host's sealed answer to the request.
    Answer(Box<CtlReply>),
    /// The host's sealed answer to another request (an earlier attempt).
    Other(u64),
    /// An unsealed `UNAUTHENTICATED` refusal: only a hint, since anyone can
    /// answer `ctl`; worth reporting when no answer comes.
    Hint(CtlError),
}

/// Verifies a `host/state` broadcast received on `keyexpr` under `host`.
///
/// # Errors
/// [`LinkError`].
pub fn open_state(
    host: &mut SignedReceiver,
    keyexpr: &str,
    msg: &[u8],
) -> Result<AuthoritySnapshot, LinkError> {
    Ok(codec::decode_state(host.verify(keyexpr, msg)?.body)?)
}

/// Verifies a `host/tick` broadcast received on `keyexpr` under `host`.
///
/// # Errors
/// [`LinkError`].
pub fn open_tick(
    host: &mut SignedReceiver,
    keyexpr: &str,
    msg: &[u8],
) -> Result<VirtualTick, LinkError> {
    Ok(codec::decode_tick(host.verify(keyexpr, msg)?.body)?)
}

/// A participant generation's link to the host (design §8.2, §8.6, §8.7):
/// `k_up` for its verbs and heartbeats, `k_down` for the replies and echoes
/// it opens, its request ids, and the host's key, instance and last
/// publication for the broadcasts it verifies. Sans-IO: key expressions
/// come in, sealed bytes go out.
#[derive(Debug)]
pub struct Link {
    participant: ParticipantId,
    accept: HandshakeAccept,
    up: SealSender,
    down: SealReceiver,
    host: SignedReceiver,
    last_request: u64,
    /// Request ids sent and not yet answered: a caller that timed out
    /// leaves its id here, to retry until answered (§8.6).
    unanswered: alloc::collections::BTreeSet<u64>,
    last_history: u64,
    keyring: Keyring,
}

impl Link {
    /// The link a completed handshake as `participant` left.
    pub fn new(participant: ParticipantId, attached: Attached) -> Self {
        Self {
            keyring: Keyring::new(Some(participant.clone()), attached.accept.generation),
            participant,
            accept: attached.accept,
            up: attached.up,
            down: attached.down,
            host: attached.host,
            last_request: 0,
            unanswered: alloc::collections::BTreeSet::new(),
            last_history: 0,
        }
    }

    /// The keys the host delivered to this generation.
    pub fn keyring(&self) -> &Keyring {
        &self.keyring
    }

    /// The keys, for two-phase sealing and opening of large samples
    /// ([`Keyring::begin_sample`], [`Keyring::begin_open_sample`]).
    pub fn keyring_mut(&mut self) -> &mut Keyring {
        &mut self.keyring
    }

    /// Opens a `KeyDelivery` on `keyexpr` (`p/{self}/keys`) in place and
    /// applies it at `now_ns` (this participant's clock): the `KeysAck` to
    /// send.
    ///
    /// # Errors
    /// [`LinkError`]: it is dropped.
    pub fn keys(
        &mut self,
        keyexpr: &str,
        msg: &mut [u8],
        now_ns: i64,
    ) -> Result<KeysAck, LinkError> {
        let opened = self.down.open(keyexpr, msg)?;
        if opened.header.kind != Kind::KeyDelivery {
            return Err(OpenError::WrongKind(opened.header.kind).into());
        }
        let delivery = codec::decode_key_delivery(opened.body)?;
        Ok(self.keyring.apply(&delivery, now_ns))
    }

    /// Takes a newer authority snapshot, at `now_ns`: lease keys whose
    /// token ended go, and every key once a newer generation took over.
    pub fn observe(&mut self, snapshot: &AuthoritySnapshot, now_ns: i64) {
        self.keyring.observe(snapshot, now_ns);
    }

    /// Seals `chunk` for `robot` on `keyexpr` under its lease's `k_cmd`.
    ///
    /// # Errors
    /// [`LinkError::Key`] without that key.
    pub fn seal_chunk(
        &mut self,
        keyexpr: &str,
        robot: &ParticipantId,
        chunk: &ActionChunk,
    ) -> Result<Vec<u8>, LinkError> {
        Ok(self.keyring.seal_chunk(keyexpr, robot, chunk)?)
    }

    /// Opens a chunk to this robot on `keyexpr`: under a `k_cmd` whose part,
    /// token and sender match it.
    ///
    /// # Errors
    /// [`LinkError::Key`]: it is dropped.
    pub fn open_chunk(&mut self, keyexpr: &str, msg: Vec<u8>) -> Result<ActionChunk, LinkError> {
        Ok(self.keyring.open_chunk(keyexpr, msg)?)
    }

    /// Seals one of this robot's samples (an `_applied` entry when
    /// `applied`) on `keyexpr`.
    ///
    /// # Errors
    /// [`LinkError::Key`] before the stream's key arrived.
    pub fn seal_sample(
        &mut self,
        keyexpr: &str,
        sample: &Sample,
        applied: bool,
    ) -> Result<Vec<u8>, LinkError> {
        Ok(self.keyring.seal_sample(keyexpr, sample, applied)?)
    }

    /// Opens a sample of `robot` on `keyexpr`.
    ///
    /// # Errors
    /// [`LinkError::Key`]: it is dropped.
    pub fn open_sample(
        &mut self,
        keyexpr: &str,
        msg: Vec<u8>,
        robot: &ParticipantId,
        applied: bool,
    ) -> Result<Sample, LinkError> {
        Ok(self.keyring.open_sample(keyexpr, msg, robot, applied)?)
    }

    /// Seals a history query on `keyexpr` (`h/{robot}/{path}`) under
    /// `k_up`, with a new request id: the id and the bytes.
    ///
    /// # Errors
    /// [`LinkError::Seal`].
    pub fn history_query(
        &mut self,
        keyexpr: &str,
        query: HistoryQuery,
    ) -> Result<(u64, Vec<u8>), LinkError> {
        let request_id = self.last_history + 1;
        let query = HistoryQuery {
            request_id,
            ..query
        };
        let sealed = self.up.seal(
            Kind::HistoryQuery,
            keyexpr,
            &codec::encode_history_query(&query),
        )?;
        self.last_history = request_id;
        Ok((request_id, sealed))
    }

    /// Opens one reply to history query `request_id`, received on
    /// `history_key`: the host's page under `k_down`, or an event stream's
    /// last sample its robot sealed on `data_key` under the stream's key (a
    /// page of one).
    ///
    /// # Errors
    /// [`LinkError`]: it is dropped.
    pub fn history_page(
        &mut self,
        history_key: &str,
        data_key: &str,
        mut msg: Vec<u8>,
        request_id: u64,
        robot: &ParticipantId,
    ) -> Result<SampleBatch, LinkError> {
        let kind = Header::parse(&msg)?.kind;
        if matches!(kind, Kind::Sample | Kind::Applied) {
            let sample = self.open_sample(data_key, msg, robot, kind == Kind::Applied)?;
            return Ok(SampleBatch {
                channel: sample.channel,
                epoch: sample.epoch,
                publish_ns: sample.publish_ns,
                samples: vec![SampleMeta {
                    seq: sample.seq,
                    capture_ns: sample.capture_ns,
                    tick: sample.tick,
                    applied: sample.applied,
                }],
                payload: sample.payload,
                truncated: false,
                gaps: Vec::new(),
                request_id,
            });
        }
        let opened = self.down.open(history_key, &mut msg)?;
        if opened.header.kind != Kind::HistoryPage {
            return Err(OpenError::WrongKind(opened.header.kind).into());
        }
        let end = msg.len() - TAG_LEN;
        let page = codec::decode_batch(Bytes::from(msg).slice(HEADER_LEN..end))?;
        if page.request_id != request_id {
            return Err(LinkError::OtherQuery(page.request_id));
        }
        Ok(page)
    }

    /// Seals a lockstep request to `robot` under token `token` on
    /// `keyexpr` (`l/{robot}`).
    ///
    /// # Errors
    /// [`LinkError::Key`] without the lease's keys.
    pub fn seal_lockstep_request(
        &mut self,
        keyexpr: &str,
        robot: &ParticipantId,
        token: LeaseToken,
        request: &LockstepRequest,
    ) -> Result<Vec<u8>, LinkError> {
        Ok(self
            .keyring
            .seal_lockstep_request(keyexpr, robot, token, request)?)
    }

    /// Opens a lockstep request to this robot: it and its key's token.
    ///
    /// # Errors
    /// [`LinkError::Key`]: it is dropped.
    pub fn open_lockstep_request(
        &mut self,
        keyexpr: &str,
        msg: Vec<u8>,
    ) -> Result<(LockstepRequest, LeaseToken), LinkError> {
        Ok(self.keyring.open_lockstep_request(keyexpr, msg)?)
    }

    /// Seals this robot's response under token `token`'s `k_obs`.
    ///
    /// # Errors
    /// [`LinkError::Key`] without it.
    pub fn seal_lockstep_response(
        &mut self,
        keyexpr: &str,
        token: LeaseToken,
        response: &LockstepResponse,
    ) -> Result<Vec<u8>, LinkError> {
        Ok(self
            .keyring
            .seal_lockstep_response(keyexpr, token, response)?)
    }

    /// Opens `robot`'s response under token `token`'s `k_obs`.
    ///
    /// # Errors
    /// [`LinkError::Key`]: it is dropped.
    pub fn open_lockstep_response(
        &mut self,
        keyexpr: &str,
        msg: Vec<u8>,
        robot: &ParticipantId,
        token: LeaseToken,
    ) -> Result<LockstepResponse, LinkError> {
        Ok(self
            .keyring
            .open_lockstep_response(keyexpr, msg, robot, token)?)
    }

    /// What the host issued: generation, device, keys' ids, instance.
    pub fn accept(&self) -> &HandshakeAccept {
        &self.accept
    }

    /// A verifier of the host's broadcasts with this link's floor, for a
    /// second subscriber (each keeps its own last `seq`).
    pub fn host_verifier(&self) -> SignedReceiver {
        self.host.clone()
    }

    /// Seals `command`, sent from `epoch`, as a new request on `keyexpr`
    /// (the session's `ctl`): its id and the bytes. The id stays
    /// unanswered until [`Self::reply`] takes its answer.
    ///
    /// # Errors
    /// [`LinkError::Seal`]; [`LinkError::Outstanding`] when an unanswered
    /// id would fall below the host's window.
    pub fn request(
        &mut self,
        keyexpr: &str,
        epoch: Option<Epoch>,
        command: &Command,
    ) -> Result<(u64, Vec<u8>), LinkError> {
        let id = self.last_request + 1;
        if let Some(&oldest) = self.unanswered.first()
            && id >= oldest + REQUEST_WINDOW
        {
            return Err(LinkError::Outstanding(oldest));
        }
        let sealed = self.resend(keyexpr, id, epoch, command)?;
        self.last_request = id;
        self.unanswered.insert(id);
        Ok((id, sealed))
    }

    /// Whether request `request_id` is still unanswered.
    pub fn unanswered(&self, request_id: u64) -> bool {
        self.unanswered.contains(&request_id)
    }

    /// Seals `command` again as request `request_id`, under a fresh `seq`:
    /// a retry, which the host answers from its reply cache (§8.6).
    ///
    /// # Errors
    /// [`LinkError::Seal`].
    pub fn resend(
        &mut self,
        keyexpr: &str,
        request_id: u64,
        epoch: Option<Epoch>,
        command: &Command,
    ) -> Result<Vec<u8>, LinkError> {
        let request = Request {
            from: self.participant.clone(),
            epoch,
            request_id,
            command: command.clone(),
        };
        Ok(self
            .up
            .seal(Kind::CtlRequest, keyexpr, &codec::encode_request(&request))?)
    }

    /// Takes a reply on `keyexpr` to request `request_id`: opened under
    /// `k_down` in place, or, failing that, an unsealed refusal as a hint.
    /// An answer's keys are installed at `now_ns` (this participant's clock).
    ///
    /// # Errors
    /// [`LinkError`]: anything else, which is dropped.
    pub fn reply(
        &mut self,
        keyexpr: &str,
        msg: &mut [u8],
        request_id: u64,
        now_ns: i64,
    ) -> Result<ReplyOutcome, LinkError> {
        let plain = codec::decode_reply(msg);
        match self.down.open(keyexpr, msg) {
            Ok(opened) => {
                let reply = codec::decode_reply(opened.body)?;
                self.unanswered.remove(&reply.request_id);
                Ok(if reply.request_id == request_id {
                    for key in &reply.keys {
                        self.keyring.install(key, now_ns);
                    }
                    self.keyring.observe(&reply.state, now_ns);
                    ReplyOutcome::Answer(Box::new(reply))
                } else {
                    ReplyOutcome::Other(reply.request_id)
                })
            }
            Err(err) => match plain {
                Ok(CtlReply {
                    result: Err(refusal),
                    ..
                }) if refusal.status == CtlStatus::Unauthenticated => {
                    Ok(ReplyOutcome::Hint(refusal))
                }
                _ => Err(err.into()),
            },
        }
    }

    /// Seals a heartbeat on `keyexpr` (the participant's `hb`).
    ///
    /// # Errors
    /// [`LinkError::Seal`].
    pub fn heartbeat(
        &mut self,
        keyexpr: &str,
        heartbeat: &Heartbeat,
    ) -> Result<Vec<u8>, LinkError> {
        Ok(self.up.seal(
            Kind::Heartbeat,
            keyexpr,
            &codec::encode_heartbeat(heartbeat),
        )?)
    }

    /// Opens a heartbeat echo on `keyexpr` (the participant's `hb/echo`) in
    /// place.
    ///
    /// # Errors
    /// [`LinkError`].
    pub fn echo(&mut self, keyexpr: &str, msg: &mut [u8]) -> Result<HeartbeatEcho, LinkError> {
        Ok(codec::decode_echo(self.down.open(keyexpr, msg)?.body)?)
    }

    /// Verifies a `host/state` broadcast on `keyexpr`.
    ///
    /// # Errors
    /// [`LinkError`].
    pub fn state(&mut self, keyexpr: &str, msg: &[u8]) -> Result<AuthoritySnapshot, LinkError> {
        open_state(&mut self.host, keyexpr, msg)
    }

    /// Verifies a `host/tick` broadcast on `keyexpr`.
    ///
    /// # Errors
    /// [`LinkError`].
    pub fn tick(&mut self, keyexpr: &str, msg: &[u8]) -> Result<VirtualTick, LinkError> {
        open_tick(&mut self.host, keyexpr, msg)
    }
}

/// A host's long-lived keys: its static X25519 key, its Ed25519 signing
/// key and, for an embedded host, the local secret its address carries.
pub struct HostIdentity {
    noise: StaticKeypair,
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
    /// launcher hands the host it starts in a container (design §10, the
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

    /// The local secret an embedded host's address carries.
    pub fn local_secret(&self) -> &[u8; 32] {
        &self.local
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
/// static public key and its entry (design §4, §9). An embedded host takes
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
    /// Participant id → the generation in effect.
    current: BTreeMap<ParticipantId, u32>,
    last_generation: u32,
    /// Every key besides the handshake's, and the one key-id counter.
    issuer: KeyIssuer,
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
            current: BTreeMap::new(),
            last_generation: 0,
            issuer: KeyIssuer::default(),
        }
    }

    /// The host's key schedule (§8.4, §8.5).
    pub fn issuer(&mut self) -> &mut KeyIssuer {
        &mut self.issuer
    }

    /// Accepts handshakes from the device with this static public key.
    pub fn enroll(&mut self, public: [u8; 32], device: impl Into<String>) {
        self.devices.insert(public, device.into());
    }

    /// The instance these handshakes are bound to.
    pub fn host_instance(&self) -> u64 {
        self.host_instance
    }

    /// Answers msg 1 with the device's cap from `access`: msg 2 and the
    /// candidate, or the refusal to send back
    /// (`UNAUTHENTICATED` for an unknown device or a message that fails to
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
        match self.names.get(participant) {
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

/// How many request ids the host remembers per participant generation.
pub const REQUEST_WINDOW: u64 = 64;

/// What the host does with a request id ([`RequestWindow::admit`]).
#[derive(Debug, PartialEq, Eq)]
pub enum Admit<'a, R> {
    /// New: apply the verb once, then [`RequestWindow::complete`] it.
    Execute,
    /// A retry of an answered request: send this reply again.
    Done(&'a R),
    /// A retry of a request still executing (a grant waiting for its
    /// robots' keys): it joins that operation and gets its one reply.
    Pending,
    /// Below the window: never applied (`EXPIRED`).
    Expired,
}

/// One participant generation's request ids (§8.6): a window of
/// [`REQUEST_WINDOW`] ids from a floor, each absent, pending or answered
/// (with its reply). The host applies verbs one at a time; only a grant
/// stays pending past its turn, while its robots install its keys.
#[derive(Debug)]
pub struct RequestWindow<R> {
    floor: u64,
    done: BTreeMap<u64, R>,
    pending: alloc::collections::BTreeSet<u64>,
}

impl<R> Default for RequestWindow<R> {
    fn default() -> Self {
        Self {
            floor: 1,
            done: BTreeMap::new(),
            pending: alloc::collections::BTreeSet::new(),
        }
    }
}

impl<R> RequestWindow<R> {
    /// Admits request `id`. An id above the window moves the floor up (and
    /// forgets the replies below it).
    pub fn admit(&mut self, id: u64) -> Admit<'_, R> {
        if id < self.floor {
            return Admit::Expired;
        }
        if id - self.floor >= REQUEST_WINDOW {
            self.floor = id - (REQUEST_WINDOW - 1);
            self.done = self.done.split_off(&self.floor);
        }
        if self.pending.contains(&id) {
            return Admit::Pending;
        }
        match self.done.get(&id) {
            Some(reply) => Admit::Done(reply),
            None => Admit::Execute,
        }
    }

    /// `id` is executing past its turn: retries join it until
    /// [`complete`](Self::complete).
    pub fn begin(&mut self, id: u64) {
        self.pending.insert(id);
    }

    /// Records the reply to `id`, which a retry gets again.
    pub fn complete(&mut self, id: u64, reply: R) {
        self.pending.remove(&id);
        if id >= self.floor {
            self.done.insert(id, reply);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::sealed::Kind;
    use crate::crypto::test_rng::TestRng;

    fn pid(id: &str) -> ParticipantId {
        ParticipantId::new(id).expect("id")
    }

    fn host() -> (HostIdentity, HostAuth) {
        let identity = HostIdentity::generate(&mut TestRng::new(1)).expect("identity");
        let mut auth = HostAuth::new(&identity, "local", "demo", 7);
        let laptop = Credential::local(&[9; 32], [0; 32]);
        auth.enroll(*laptop.device_public(), "laptop-josh");
        (identity, auth)
    }

    fn handshake(
        identity: &HostIdentity,
        auth: &mut HostAuth,
        credential: &Credential,
        name: &str,
    ) -> Result<(Attached, Candidate), CtlError> {
        let mut rng = TestRng::new(3);
        let (initiator, request) = credential
            .start(&mut rng, "local", "demo", 7, &pid(name))
            .expect("start");
        let (msg2, candidate) = auth.handshake(&mut rng, identity, &request, &Access::default())?;
        Ok((finish(initiator, &msg2).expect("finish"), candidate))
    }

    #[test]
    fn a_link_seals_verbs_and_heartbeats_and_takes_only_the_hosts_replies_and_broadcasts() {
        let (identity, mut auth) = host();
        let (attached, mut candidate) =
            handshake(&identity, &mut auth, &identity.local_credential(), "robot")
                .expect("handshake");
        assert_eq!(attached.accept.device, LOCAL_DEVICE);
        assert_eq!(attached.accept.host_instance, 7);
        assert_eq!(attached.accept.generation, candidate.generation);
        let mut link = Link::new(pid("robot"), attached);

        // A verb: the host opens it; its sealed reply answers that id only.
        let (id, mut msg) = link.request("ctl", None, &Command::Arm).expect("seal");
        let opened = candidate.up.open("ctl", &mut msg).expect("open");
        let request = codec::decode_request(opened.body).expect("request");
        assert_eq!((request.request_id, request.command), (1, Command::Arm));
        let snapshot = crate::authority::Authority::new(
            crate::wire::SessionId::named("demo"),
            crate::authority::AuthorityConfig::default(),
        )
        .snapshot(Vec::new(), 0);
        let reply = |request_id| CtlReply {
            result: Ok(()),
            state: snapshot.clone(),
            leases: Vec::new(),
            request_id,
            keys: Vec::new(),
        };
        let mut sealed = candidate
            .down
            .seal(Kind::CtlReply, "ctl", &codec::encode_reply(&reply(id)))
            .expect("seal");
        assert_eq!(
            link.reply("ctl", &mut sealed.clone(), id, 0).expect("ours"),
            ReplyOutcome::Answer(Box::new(reply(id)))
        );
        assert!(
            link.reply("ctl", &mut sealed, id, 0).is_err(),
            "a replay is dropped"
        );
        // A retry keeps its id under a fresh seq.
        let mut again = link.resend("ctl", id, None, &Command::Arm).expect("seal");
        let retried =
            codec::decode_request(candidate.up.open("ctl", &mut again).expect("open").body);
        assert_eq!(retried.expect("request").request_id, id);
        // An unanswered id holds the window: 63 more ids, then no new one
        // until it is answered.
        assert!(!link.unanswered(id), "answered");
        let (lost, _) = link.request("ctl", None, &Command::Arm).expect("seal");
        for _ in 0..63 {
            let (next, _) = link.request("ctl", None, &Command::Arm).expect("seal");
            let mut sealed = candidate
                .down
                .seal(Kind::CtlReply, "ctl", &codec::encode_reply(&reply(next)))
                .expect("seal");
            link.reply("ctl", &mut sealed, next, 0).expect("answered");
        }
        assert!(matches!(
            link.request("ctl", None, &Command::Arm),
            Err(LinkError::Outstanding(id)) if id == lost
        ));
        let mut late = candidate
            .down
            .seal(Kind::CtlReply, "ctl", &codec::encode_reply(&reply(lost)))
            .expect("seal");
        link.reply("ctl", &mut late, lost, 0)
            .expect("the retry's answer");
        assert!(link.request("ctl", None, &Command::Arm).is_ok());
        // Anyone can answer `ctl`: a plain refusal is only a hint, a plain
        // answer nothing.
        let refusal = CtlError::new(CtlStatus::Unauthenticated, "superseded");
        let mut plain = codec::encode_reply(&CtlReply {
            result: Err(refusal.clone()),
            ..reply(id)
        });
        assert_eq!(
            link.reply("ctl", &mut plain, id, 0).expect("a hint"),
            ReplyOutcome::Hint(refusal)
        );
        assert!(
            link.reply("ctl", &mut codec::encode_reply(&reply(id)), id, 0)
                .is_err()
        );

        // Heartbeats up, echoes down.
        let heartbeat = Heartbeat {
            participant: pid("robot"),
            epoch: Epoch(1),
            mode: crate::wire::Mode::Disarmed,
            seq: 1,
            t1_ns: 5,
            previous: None,
            last_seq_seen: Vec::new(),
            last_chunk_accepted: 0,
            last_chunk_applied: 0,
            queue_depth: 0,
            health: crate::wire::heartbeat::Health::Ok,
        };
        let mut hb = link.heartbeat("hb", &heartbeat).expect("seal");
        assert_eq!(
            candidate.up.open("hb", &mut hb).expect("open").header.kind,
            Kind::Heartbeat
        );
        let echo = HeartbeatEcho {
            seq: 1,
            t1_ns: 5,
            t2_ns: 6,
            t3_ns: 7,
            epoch: Epoch(1),
        };
        let mut sealed = candidate
            .down
            .seal(Kind::HeartbeatEcho, "echo", &codec::encode_echo(&echo))
            .expect("seal");
        assert_eq!(link.echo("echo", &mut sealed).expect("echo"), echo);

        // Broadcasts: signed by the host's instance, newer than the last.
        let mut signer = identity.signer(7);
        let body = codec::encode_state(&snapshot);
        let first = signer
            .sign(crate::crypto::SignedKind::AuthorityState, "state", &body)
            .expect("sign");
        let second = signer
            .sign(crate::crypto::SignedKind::AuthorityState, "state", &body)
            .expect("sign");
        let mut verifier = link.host_verifier();
        assert!(link.state("state", &second).is_ok());
        assert!(link.state("state", &first).is_err(), "stale");
        assert!(
            open_state(&mut verifier, "state", &first).is_ok(),
            "its own floor"
        );
        assert!(link.state("state", &body).is_err(), "unsigned");
        let stranger =
            identity
                .signer(8)
                .sign(crate::crypto::SignedKind::AuthorityState, "state", &body);
        assert!(
            link.state("state", &stranger.expect("sign")).is_err(),
            "another instance"
        );
    }

    #[test]
    fn unknown_keys_stale_instances_and_taken_names_are_refused() {
        let (identity, mut auth) = host();
        // A key the host never enrolled.
        let stranger = Credential::local(&[4; 32], *identity.static_public());
        let err = handshake(&identity, &mut auth, &stranger, "robot").expect_err("unknown");
        assert_eq!(err.status, CtlStatus::Unauthenticated);
        // Another host's static key (a stale address).
        let other = HostIdentity::generate(&mut TestRng::new(2)).expect("other");
        let stale = Credential::local(identity.local_secret(), *other.static_public());
        let err = handshake(&identity, &mut auth, &stale, "robot").expect_err("stale");
        assert_eq!(err.status, CtlStatus::Unauthenticated);
        // A name binds to the first device it took effect for.
        let (_, local) =
            handshake(&identity, &mut auth, &identity.local_credential(), "eye-3").expect("local");
        auth.take_effect(&local.participant, &local.device, local.generation);
        let laptop = Credential::local(&[9; 32], *identity.static_public());
        let err = handshake(&identity, &mut auth, &laptop, "eye-3").expect_err("taken");
        assert_eq!(err.status, CtlStatus::NotAuthorized);
        assert!(err.message.contains("belongs to device local"), "{err}");
        handshake(&identity, &mut auth, &laptop, "cockpit").expect("another name is free");
    }

    #[test]
    fn only_a_newer_generation_takes_effect() {
        let (identity, mut auth) = host();
        let local = identity.local_credential();
        let (_, older) = handshake(&identity, &mut auth, &local, "policy").expect("first");
        let (_, newer) = handshake(&identity, &mut auth, &local, "policy").expect("second");
        assert!(newer.generation > older.generation);
        assert!(
            newer.up_key_id > older.up_key_id + 1,
            "key ids are never reused"
        );
        let take = |auth: &HostAuth, c: &Candidate| {
            auth.may_take_effect(&c.participant, &c.device, c.generation)
        };
        take(&auth, &newer).expect("newer");
        auth.take_effect(&newer.participant, &newer.device, newer.generation);
        let err = take(&auth, &older).expect_err("superseded");
        assert_eq!(err.status, CtlStatus::Unauthenticated);
        assert_eq!(auth.current(&pid("policy")), Some(newer.generation));
    }

    #[test]
    fn the_request_window_answers_retries_from_its_cache_and_expires_old_ids() {
        let mut window = RequestWindow::<&str>::default();
        assert_eq!(window.admit(0), Admit::Expired, "ids start at 1");
        assert_eq!(window.admit(1), Admit::Execute);
        window.complete(1, "one");
        assert_eq!(
            window.admit(1),
            Admit::Done(&"one"),
            "a retry is answered again"
        );
        assert_eq!(window.admit(64), Admit::Execute);
        assert_eq!(window.admit(1), Admit::Done(&"one"), "64 ids fit");
        // An id above the window moves the floor: 1 falls out.
        assert_eq!(window.admit(65), Admit::Execute);
        assert_eq!(window.admit(1), Admit::Expired);
        window.complete(65, "sixty-five");
        assert_eq!(window.admit(200), Admit::Execute);
        assert_eq!(window.admit(65), Admit::Expired);
        assert_eq!(window.admit(137), Admit::Execute);
    }
}
