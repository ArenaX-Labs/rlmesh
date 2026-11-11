//! The keys a process holds besides its pairwise ones,
//! sans-IO: what the host delivered to a participant, or what the host
//! keeps to open its members' streams. One [`Keyring`] per participant
//! generation (inside its [`Link`](super::Link)), one in the host.
//!
//! - **Install.** A rotating key (a stream's) is installed only above every
//!   version seen for its slot and above the slot's retire floor; a fixed key
//!   (a lease's) by id. A same-id delivery with `send` switches a held key
//!   to sending without resetting its counters or windows. A participant
//!   never sends under a key whose sender is another participant or
//!   generation.
//! - **Retire.** `retire_through` drops a key and every older version of its
//!   slot, and raises the slot's floor so recovery never brings them back.
//! - **Rotation.** Every version above the floor is accepted until the first
//!   valid message under a newer one, which drops the older versions.
//! - **Discard.** A fixed key goes when its token, seen live in authority
//!   state, ends, or when the token hasn't appeared within
//!   [`UNSEEN_TTL_NS`] of installing it.
//!
//! Opening checks a message's kind against the key's purpose (its
//! receiver's kinds) and then its body against the key's scope: a chunk's
//! part, token and source, a sample's robot and part.

use alloc::collections::{BTreeMap, BTreeSet};

use bytes::Bytes;

use crate::crypto::replay::Replayed;
use crate::crypto::sealed::{HEADER_LEN, Header, TAG_LEN};
use crate::crypto::{
    Key, Kind, KindSet, OpenError, Opening, ReplayWindow, SealError, SealReceiver, SealSender,
    Sealing,
};
use crate::prelude::*;
use crate::wire::codec;
use crate::wire::error::WireError;
use crate::wire::{
    ActionChunk, AuthorityView, ChannelId, EstopSignal, IssuedKey, KeyDelivery, KeyPurpose,
    KeysAck, LeaseToken, LockstepRequest, LockstepResponse, ParticipantId, Sample, Security,
};

/// How long a fixed key waits for its token to appear in authority state
/// before it is discarded.
pub const UNSEEN_TTL_NS: i64 = 4_000_000_000;

/// The kinds a key of `purpose` carries.
pub fn kinds(purpose: KeyPurpose) -> KindSet {
    match purpose {
        KeyPurpose::Command => KindSet::of(&[Kind::ActionChunk, Kind::LockstepRequest]),
        KeyPurpose::LockstepReply => KindSet::of(&[Kind::LockstepReply]),
        KeyPurpose::Stream => KindSet::of(&[Kind::Sample, Kind::Applied]),
        KeyPurpose::Estop => KindSet::of(&[Kind::EstopSignal]),
    }
}

/// Everything about a held key but its bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scope {
    /// Its id.
    pub key_id: u32,
    /// What it is for.
    pub purpose: KeyPurpose,
    /// Who may send under it (`None`: the host).
    pub sender: Option<ParticipantId>,
    /// The robot it concerns.
    pub robot: Option<ParticipantId>,
    /// The parts it covers.
    pub channels: Vec<ChannelId>,
    /// A lease key's token.
    pub token: LeaseToken,
    /// A rotating key's version.
    pub version: u32,
    /// The level of its configured kinds.
    pub security: Security,
    /// The sender's generation.
    pub generation: u32,
}

impl Scope {
    fn of(key: &IssuedKey) -> Self {
        Self {
            key_id: key.key_id,
            purpose: key.purpose,
            sender: key.sender.clone(),
            robot: key.robot.clone(),
            channels: key.channels.clone(),
            token: key.token,
            version: key.version,
            security: key.security,
            generation: key.generation,
        }
    }

    fn slot(&self) -> Slot {
        Slot {
            purpose: self.purpose,
            robot: self.robot.clone(),
            channels: self.channels.clone(),
        }
    }

    fn in_slot(&self, slot: &Slot) -> bool {
        self.purpose == slot.purpose && self.robot == slot.robot && self.channels == slot.channels
    }
}

/// A rotating key's slot.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Slot {
    purpose: KeyPurpose,
    robot: Option<ParticipantId>,
    channels: Vec<ChannelId>,
}

/// What a participant remembers of a slot.
#[derive(Clone, Copy, Debug, Default)]
struct Mark {
    /// The highest version seen.
    seen: u32,
    /// Versions at or below this are retired.
    floor: u32,
}

struct Held {
    scope: Scope,
    key: Key,
    rx: SealReceiver,
    tx: Option<SealSender>,
    installed_ns: i64,
    /// Its token was in authority state.
    live: bool,
    /// A message under it opened (a rotating key retires older versions
    /// at its first).
    opened: bool,
}

/// One subscription's replay windows over a stream, by key and kind, and
/// the newest key version it accepted (the first valid
/// message under a newer version restarts the stream's sample high-water).
#[derive(Debug, Default)]
pub struct StreamWindows {
    windows: BTreeMap<(u32, u8), ReplayWindow>,
    version: u32,
}

impl StreamWindows {
    fn check(&self, header: &Header) -> Result<(), Replayed> {
        self.windows
            .get(&(header.key_id, header.kind.as_u8()))
            .map_or(Ok(()), |w| w.check(header.seq))
    }

    /// Records a verified message under a key of `version`: whether it is
    /// the first under a newer version (the sender restarted, so its `seq`s
    /// did too).
    ///
    /// # Errors
    /// [`Replayed`]: a second copy verified meanwhile.
    pub fn commit(&mut self, header: &Header, version: u32) -> Result<bool, Replayed> {
        self.windows
            .entry((header.key_id, header.kind.as_u8()))
            .or_default()
            .insert(header.seq)?;
        let restarted = version > self.version && self.version > 0;
        if version > self.version {
            self.windows.retain(|(id, _), _| *id == header.key_id);
            self.version = version;
        }
        Ok(restarted)
    }
}

/// Why a key-scoped message was refused or not sealed.
#[derive(Debug, thiserror::Error)]
pub enum KeyError {
    /// Sealed under a key this process doesn't hold: drop it, and recover
    /// with `GetKeys` if the key should exist.
    #[error("key {0} is not held")]
    Unknown(u32),
    /// No held key may send this.
    #[error("no key to send {0}")]
    NoKey(&'static str),
    /// It opened, but its body is outside the key's scope.
    #[error("outside its key's scope: {0}")]
    Scope(&'static str),
    /// It failed to open.
    #[error(transparent)]
    Open(#[from] OpenError),
    /// It failed to seal.
    #[error(transparent)]
    Seal(#[from] SealError),
    /// It opened but did not decode.
    #[error(transparent)]
    Wire(#[from] WireError),
}

/// A process's delivered keys: see the module docs.
#[derive(Default)]
pub struct Keyring {
    me: Option<ParticipantId>,
    generation: u32,
    held: BTreeMap<u32, Held>,
    marks: BTreeMap<Slot, Mark>,
    /// Fixed keys this process dropped: never installed again, so a late
    /// delivery or recovery reply never restarts their counters (a reused
    /// nonce) or replay windows.
    // TODO: grows by a few ids per lease for the life of a generation;
    // bound it if boards run sessions long enough to feel it.
    gone: BTreeSet<u32>,
    /// The newest snapshot observed: (revision, host_ns).
    observed: Option<(u64, i64)>,
}

impl core::fmt::Debug for Keyring {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Keyring")
            .field("me", &self.me)
            .field("generation", &self.generation)
            .field("held", &self.held.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl Keyring {
    /// The keys of generation `generation` of `me` (`None`: the host,
    /// which only receives).
    pub fn new(me: Option<ParticipantId>, generation: u32) -> Self {
        Self {
            me,
            generation,
            ..Self::default()
        }
    }

    /// Applies a delivery at `now_ns` (this process's clock): what to
    /// acknowledge.
    pub fn apply(&mut self, delivery: &KeyDelivery, now_ns: i64) -> KeysAck {
        let mut ack = KeysAck::default();
        for key in &delivery.keys {
            if let Some(sending) = self.install(key, now_ns) {
                ack.installed.push(key.key_id);
                if sending {
                    ack.sending.push(key.key_id);
                }
            }
        }
        for id in &delivery.retire_through {
            self.retire(*id);
            ack.retired.push(*id);
        }
        ack
    }

    /// Installs `key` (or switches a held one to sending): whether it now
    /// sends under it, or `None` when it is refused (another sender's key
    /// to send, a retired or superseded version).
    pub fn install(&mut self, key: &IssuedKey, now_ns: i64) -> Option<bool> {
        if key.send && (key.sender != self.me || key.generation != self.generation) {
            return None;
        }
        if !key.purpose.rotates() && self.gone.contains(&key.key_id) {
            return None;
        }
        if let Some(held) = self.held.get_mut(&key.key_id) {
            if held.scope != Scope::of(key) {
                return None;
            }
            if key.send && held.tx.is_none() {
                if !held.key.ct_eq(&Key::from_bytes(key.key.0)) {
                    return None;
                }
                held.tx = Some(SealSender::new(
                    key.key_id,
                    Key::from_bytes(key.key.0),
                    kinds(key.purpose),
                    key.security,
                ));
            }
            return Some(held.tx.is_some());
        }
        let scope = Scope::of(key);
        if key.purpose.rotates() {
            let mark = self.marks.entry(scope.slot()).or_default();
            if key.version <= mark.floor || key.version <= mark.seen {
                return None;
            }
            mark.seen = key.version;
        }
        let new = |bytes: [u8; 32]| Key::from_bytes(bytes);
        let tx = key
            .send
            .then(|| SealSender::new(key.key_id, new(key.key.0), kinds(key.purpose), key.security));
        self.held.insert(
            key.key_id,
            Held {
                rx: SealReceiver::new(key.key_id, new(key.key.0), kinds(key.purpose), key.security),
                key: new(key.key.0),
                tx,
                installed_ns: now_ns,
                live: false,
                opened: false,
                scope,
            },
        );
        Some(key.send)
    }

    /// Drops key `key_id` and, for a rotating key, every older version of
    /// its slot (which recovery never reinstalls).
    pub fn retire(&mut self, key_id: u32) {
        let Some(held) = self.held.remove(&key_id) else {
            return;
        };
        if !held.scope.purpose.rotates() {
            self.gone.insert(key_id);
        }
        if held.scope.purpose.rotates() {
            let slot = held.scope.slot();
            let mark = self.marks.entry(slot.clone()).or_default();
            mark.floor = mark.floor.max(held.scope.version);
            let floor = mark.floor;
            self.held
                .retain(|_, h| !(h.scope.in_slot(&slot) && h.scope.version <= floor));
        }
    }

    /// Takes an authority snapshot or a device view, if newer than the
    /// last: drops lease
    /// keys whose token ended (or never appeared within [`UNSEEN_TTL_NS`]),
    /// and every key once a newer generation of this participant took over.
    /// An older snapshot (a delayed broadcast, a cached reply) changes
    /// nothing.
    pub fn observe(&mut self, snapshot: &impl AuthorityView, now_ns: i64) {
        let order = (snapshot.revision(), snapshot.host_ns());
        if self.observed.is_some_and(|last| order <= last) {
            return;
        }
        self.observed = Some(order);
        if let Some(me) = &self.me
            && snapshot.generation(me).is_some_and(|g| g > self.generation)
        {
            let gone = &mut self.gone;
            gone.extend(
                self.held
                    .iter()
                    .filter(|(_, h)| !h.scope.purpose.rotates())
                    .map(|(id, _)| *id),
            );
            self.held.clear();
            return;
        }
        // Inserted one by one: `collect` sorts first, with a 4 KB stack buffer.
        let mut live = BTreeSet::new();
        for lease in snapshot.leases() {
            live.insert(lease.token);
        }
        let gone = &mut self.gone;
        self.held.retain(|id, held| {
            if held.scope.purpose.rotates() {
                return true;
            }
            // A `k_estop` lives while its sender's generation does.
            let current = if held.scope.purpose == KeyPurpose::Estop {
                let sender = held.scope.sender.as_ref();
                sender
                    .and_then(|s| snapshot.generation(s))
                    .is_some_and(|g| g == held.scope.generation)
            } else {
                live.contains(&held.scope.token)
            };
            if current {
                held.live = true;
                return true;
            }
            let keep = !held.live && now_ns.saturating_sub(held.installed_ns) < UNSEEN_TTL_NS;
            if !keep {
                gone.insert(*id);
            }
            keep
        });
    }

    /// Whether key `key_id` is held.
    pub fn holds(&self, key_id: u32) -> bool {
        self.held.contains_key(&key_id)
    }

    /// The scopes of every held key, by id.
    pub fn scopes(&self) -> impl Iterator<Item = (&Scope, bool)> {
        self.held.values().map(|h| (&h.scope, h.tx.is_some()))
    }

    /// Whether this process may send `channel`'s samples (it holds that
    /// stream's key with `send`).
    pub fn sends_stream(&self, channel: ChannelId) -> bool {
        self.held.values().any(|h| {
            h.tx.is_some()
                && h.scope.purpose == KeyPurpose::Stream
                && h.scope.channels.contains(&channel)
        })
    }

    /// Takes the next `seq` of `kind` under the newest sending key `pick`
    /// accepts (the lowest id among equals).
    fn sender(
        &mut self,
        kind: Kind,
        what: &'static str,
        pick: impl Fn(&Scope) -> bool,
    ) -> Result<Sealing, KeyError> {
        let mut best: Option<(u32, u32)> = None;
        for (id, held) in &self.held {
            if held.tx.is_some()
                && pick(&held.scope)
                && best.is_none_or(|(version, _)| held.scope.version > version)
            {
                best = Some((held.scope.version, *id));
            }
        }
        let (_, id) = best.ok_or(KeyError::NoKey(what))?;
        let tx = self
            .held
            .get_mut(&id)
            .and_then(|h| h.tx.as_mut())
            .ok_or(KeyError::NoKey(what))?;
        Ok(tx.begin(kind)?)
    }

    /// Seals `body` as `kind` on `keyexpr` (see [`Self::sender`]).
    fn seal(
        &mut self,
        kind: Kind,
        keyexpr: &str,
        body: &[u8],
        what: &'static str,
        pick: impl Fn(&Scope) -> bool,
    ) -> Result<Vec<u8>, KeyError> {
        let sealing = self.sender(kind, what, pick)?;
        let mut out = Vec::with_capacity(crate::crypto::sealed::OVERHEAD + body.len());
        sealing.finish_to(keyexpr, body, &mut out)?;
        Ok(out)
    }

    /// Checks `msg` (which must be `kind`) against the key its header
    /// names and that key's replay window: what to verify, and the key's
    /// scope.
    fn opening(&self, msg: &[u8], kind: Kind) -> Result<(Opening, &Scope), KeyError> {
        let header = Header::parse(msg)?;
        if header.kind != kind {
            return Err(OpenError::WrongKind(kind).into());
        }
        let held = self
            .held
            .get(&header.key_id)
            .ok_or(KeyError::Unknown(header.key_id))?;
        Ok((held.rx.begin(msg)?, &held.scope))
    }

    /// Records the `seq` of a message an [`Opening`] verified. A rotating
    /// key's first message drops its older versions.
    ///
    /// # Errors
    /// [`KeyError`]: a second copy, or a key retired meanwhile.
    pub fn commit_open(&mut self, header: &Header) -> Result<(), KeyError> {
        let id = header.key_id;
        self.held
            .get_mut(&id)
            .ok_or(KeyError::Unknown(id))?
            .rx
            .commit(header)?;
        self.opened(header)
    }

    /// A message under `header`'s key verified (its replay window kept
    /// elsewhere: [`StreamWindows`]): a rotating key's first drops its
    /// older versions.
    ///
    /// # Errors
    /// [`KeyError::Unknown`] for a key retired meanwhile.
    pub fn opened(&mut self, header: &Header) -> Result<(), KeyError> {
        let id = header.key_id;
        let held = self.held.get_mut(&id).ok_or(KeyError::Unknown(id))?;
        if !held.opened {
            held.opened = true;
            if held.scope.purpose.rotates() {
                let (slot, version) = (held.scope.slot(), held.scope.version);
                let mark = self.marks.entry(slot.clone()).or_default();
                mark.floor = mark.floor.max(version.saturating_sub(1));
                self.held
                    .retain(|_, h| !(h.scope.in_slot(&slot) && h.scope.version < version));
            }
        }
        Ok(())
    }

    /// Takes the next `seq` for one of this robot's samples of `channel`
    /// (an `_applied` entry when `applied`), to seal without the caller's
    /// lock: the two-phase [`Self::seal_sample`].
    ///
    /// # Errors
    /// [`KeyError::NoKey`] before the stream's key reached this process.
    pub fn begin_sample(&mut self, channel: ChannelId, applied: bool) -> Result<Sealing, KeyError> {
        let me = self.me.clone();
        let kind = if applied { Kind::Applied } else { Kind::Sample };
        self.sender(kind, "this stream", |s| {
            s.purpose == KeyPurpose::Stream && s.robot == me && s.channels.contains(&channel)
        })
    }

    /// Checks a sample of `robot`'s stream `channel` against its stream
    /// key and replay window, to verify without the caller's lock (then
    /// [`Self::commit_open`]): the two-phase [`Self::open_sample`].
    ///
    /// # Errors
    /// [`KeyError`]: it is dropped.
    pub fn begin_open_sample(
        &self,
        msg: &[u8],
        robot: &ParticipantId,
        channel: ChannelId,
        applied: bool,
    ) -> Result<Opening, KeyError> {
        let kind = if applied { Kind::Applied } else { Kind::Sample };
        let (opening, scope) = self.opening(msg, kind)?;
        if scope.purpose != KeyPurpose::Stream
            || scope.robot.as_ref() != Some(robot)
            || !scope.channels.contains(&channel)
        {
            return Err(KeyError::Scope("not this stream's key"));
        }
        Ok(opening)
    }

    /// [`Self::begin_open_sample`] for one subscription, which keeps its
    /// own replay windows (`windows`): several subscriptions to one stream
    /// in one process each get every sample, and each refuses a repeat.
    /// Also returns the key's version.
    ///
    /// # Errors
    /// [`KeyError`]: it is dropped.
    pub fn begin_open_stream(
        &self,
        msg: &[u8],
        robot: &ParticipantId,
        channel: ChannelId,
        applied: bool,
        windows: &StreamWindows,
    ) -> Result<(Opening, u32), KeyError> {
        let kind = if applied { Kind::Applied } else { Kind::Sample };
        let header = Header::parse(msg)?;
        if header.kind != kind {
            return Err(OpenError::WrongKind(kind).into());
        }
        let held = self
            .held
            .get(&header.key_id)
            .ok_or(KeyError::Unknown(header.key_id))?;
        let scope = &held.scope;
        if scope.purpose != KeyPurpose::Stream
            || scope.robot.as_ref() != Some(robot)
            || !scope.channels.contains(&channel)
        {
            return Err(KeyError::Scope("not this stream's key"));
        }
        windows.check(&header).map_err(OpenError::from)?;
        Ok((held.rx.begin_unwindowed(msg)?, scope.version))
    }

    /// Seals `chunk` for `robot`'s edge on `keyexpr` (`c/{robot}/{path}`),
    /// under the `k_cmd` of its token for its part.
    ///
    /// # Errors
    /// [`KeyError::NoKey`] without that key.
    pub fn seal_chunk(
        &mut self,
        keyexpr: &str,
        robot: &ParticipantId,
        chunk: &ActionChunk,
    ) -> Result<Vec<u8>, KeyError> {
        let body = codec::encode_chunk(chunk)?.to_vec();
        self.seal(Kind::ActionChunk, keyexpr, &body, "this chunk", |s| {
            s.purpose == KeyPurpose::Command
                && s.robot.as_ref() == Some(robot)
                && s.token == chunk.lease_token
                && s.channels.contains(&chunk.channel)
        })
    }

    /// Opens a chunk to this robot received on `keyexpr`: under a `k_cmd`
    /// for this robot whose part, token and sender match the chunk's.
    ///
    /// # Errors
    /// [`KeyError`]: it is dropped.
    pub fn open_chunk(&mut self, keyexpr: &str, msg: Vec<u8>) -> Result<ActionChunk, KeyError> {
        let me = self.me.clone();
        let (body, scope) = self.open_body(keyexpr, msg, Kind::ActionChunk)?;
        let chunk = codec::decode_chunk(body)?;
        if scope.purpose != KeyPurpose::Command || scope.robot != me {
            return Err(KeyError::Scope("not a command key for this robot"));
        }
        if !scope.channels.contains(&chunk.channel)
            || scope.token != chunk.lease_token
            || scope.sender.as_ref() != Some(&chunk.source)
        {
            return Err(KeyError::Scope("the chunk's part, token or source"));
        }
        Ok(chunk)
    }

    /// Seals one of this robot's samples on `keyexpr` (`d/{robot}/{path}`,
    /// `_applied` for an applied entry) under its stream's newest sending
    /// key.
    ///
    /// # Errors
    /// [`KeyError::NoKey`] before the stream's key reached this process.
    pub fn seal_sample(
        &mut self,
        keyexpr: &str,
        sample: &Sample,
        applied: bool,
    ) -> Result<Vec<u8>, KeyError> {
        let me = self.me.clone();
        let body = codec::encode_sample(sample).to_vec();
        let kind = if applied { Kind::Applied } else { Kind::Sample };
        self.seal(kind, keyexpr, &body, "this stream", |s| {
            s.purpose == KeyPurpose::Stream && s.robot == me && s.channels.contains(&sample.channel)
        })
    }

    /// Opens a sample of `robot` received on `keyexpr`, an applied entry
    /// when `applied`: under a stream key of that robot covering its part.
    ///
    /// # Errors
    /// [`KeyError`]: it is dropped.
    pub fn open_sample(
        &mut self,
        keyexpr: &str,
        msg: Vec<u8>,
        robot: &ParticipantId,
        applied: bool,
    ) -> Result<Sample, KeyError> {
        let kind = if applied { Kind::Applied } else { Kind::Sample };
        let (body, scope) = self.open_body(keyexpr, msg, kind)?;
        let sample = codec::decode_sample(body)?;
        if scope.purpose != KeyPurpose::Stream
            || scope.robot.as_ref() != Some(robot)
            || !scope.channels.contains(&sample.channel)
        {
            return Err(KeyError::Scope("not this robot's stream key"));
        }
        Ok(sample)
    }

    /// Seals a lockstep request to `robot` under token `token`'s `k_cmd`
    /// for that robot (the one with the lowest id).
    ///
    /// # Errors
    /// [`KeyError::NoKey`] without a lease key.
    pub fn seal_lockstep_request(
        &mut self,
        keyexpr: &str,
        robot: &ParticipantId,
        token: LeaseToken,
        request: &LockstepRequest,
    ) -> Result<Vec<u8>, KeyError> {
        let body = codec::encode_lockstep_request(request);
        self.seal(Kind::LockstepRequest, keyexpr, &body, "lockstep", |s| {
            s.purpose == KeyPurpose::Command && s.robot.as_ref() == Some(robot) && s.token == token
        })
    }

    /// Opens a lockstep request to this robot: the request and the token
    /// of the key it came under, which the robot checks against its view.
    ///
    /// # Errors
    /// [`KeyError`]: it is dropped.
    pub fn open_lockstep_request(
        &mut self,
        keyexpr: &str,
        msg: Vec<u8>,
    ) -> Result<(LockstepRequest, LeaseToken), KeyError> {
        let me = self.me.clone();
        let (body, scope) = self.open_body(keyexpr, msg, Kind::LockstepRequest)?;
        if scope.purpose != KeyPurpose::Command || scope.robot != me {
            return Err(KeyError::Scope("not a command key for this robot"));
        }
        let token = scope.token;
        Ok((codec::decode_lockstep_request(body)?, token))
    }

    /// Seals this robot's lockstep response under token `token`'s `k_obs`.
    ///
    /// # Errors
    /// [`KeyError::NoKey`] without it.
    pub fn seal_lockstep_response(
        &mut self,
        keyexpr: &str,
        token: LeaseToken,
        response: &LockstepResponse,
    ) -> Result<Vec<u8>, KeyError> {
        let me = self.me.clone();
        let body = codec::encode_lockstep_response(response);
        self.seal(Kind::LockstepReply, keyexpr, &body, "lockstep", |s| {
            s.purpose == KeyPurpose::LockstepReply && s.robot == me && s.token == token
        })
    }

    /// Opens `robot`'s lockstep response under token `token`'s `k_obs`.
    ///
    /// # Errors
    /// [`KeyError`]: it is dropped.
    pub fn open_lockstep_response(
        &mut self,
        keyexpr: &str,
        msg: Vec<u8>,
        robot: &ParticipantId,
        token: LeaseToken,
    ) -> Result<LockstepResponse, KeyError> {
        let (body, scope) = self.open_body(keyexpr, msg, Kind::LockstepReply)?;
        if scope.purpose != KeyPurpose::LockstepReply
            || scope.robot.as_ref() != Some(robot)
            || scope.token != token
        {
            return Err(KeyError::Scope("not this lease's reply key"));
        }
        Ok(codec::decode_lockstep_response(body)?)
    }

    /// Opens `msg` (which must be `kind`) and returns its body, sliced from
    /// the buffer without a copy, and its key's scope.
    /// Seals this participant's e-stop signal on `keyexpr`
    /// (`estop/{me}`) under its generation's `k_estop`.
    ///
    /// # Errors
    /// [`KeyError::NoKey`] without it: this generation isn't attached.
    pub fn seal_estop(&mut self, keyexpr: &str, signal: &EstopSignal) -> Result<Vec<u8>, KeyError> {
        let (me, generation) = (self.me.clone(), self.generation);
        let body = codec::encode_estop_signal(signal);
        self.seal(Kind::EstopSignal, keyexpr, &body, "an e-stop signal", |s| {
            s.purpose == KeyPurpose::Estop && s.sender == me && s.generation == generation
        })
    }

    /// Opens `by`'s e-stop signal received on `keyexpr`: under the
    /// `k_estop` of `by`'s current generation, and no one else's.
    ///
    /// # Errors
    /// [`KeyError`]: it is dropped (a non-member, a forgery, an ended
    /// generation, or a key this edge doesn't hold yet).
    pub fn open_estop(
        &mut self,
        keyexpr: &str,
        msg: Vec<u8>,
        by: &ParticipantId,
    ) -> Result<EstopSignal, KeyError> {
        let (body, scope) = self.open_body(keyexpr, msg, Kind::EstopSignal)?;
        if scope.purpose != KeyPurpose::Estop || scope.sender.as_ref() != Some(by) {
            return Err(KeyError::Scope("not the sender's e-stop key"));
        }
        Ok(codec::decode_estop_signal(&body)?)
    }

    fn open_body(
        &mut self,
        keyexpr: &str,
        mut msg: Vec<u8>,
        kind: Kind,
    ) -> Result<(Bytes, Scope), KeyError> {
        let (opening, scope) = self.opening(&msg, kind)?;
        let scope = scope.clone();
        opening.finish(keyexpr, &mut msg)?;
        self.commit_open(opening.header())?;
        let end = msg.len() - TAG_LEN;
        Ok((Bytes::from(msg).slice(HEADER_LEN..end), scope))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{Epoch, Fingerprint, KeyBytes};

    fn pid(id: &str) -> ParticipantId {
        ParticipantId::new(id).expect("id")
    }

    fn stream_key(id: u32, version: u32, send: bool, security: Security) -> IssuedKey {
        IssuedKey {
            key_id: id,
            purpose: KeyPurpose::Stream,
            sender: Some(pid("arm")),
            robot: Some(pid("arm")),
            channels: vec![ChannelId(3), ChannelId(4)],
            token: LeaseToken(0),
            version,
            key: KeyBytes([id as u8; 32]),
            send,
            security,
            generation: 1,
        }
    }

    fn sample(seq: u64) -> Sample {
        Sample {
            channel: ChannelId(3),
            epoch: Epoch(1),
            seq,
            capture_ns: 0,
            publish_ns: 0,
            tick: 0,
            payload: Bytes::from_static(b"wrench!!"),
            applied: None,
        }
    }

    const KEY: &str = "rlmesh/v2/acme/s/demo/d/arm/ft";

    #[test]
    fn a_reader_without_the_key_cannot_read_a_sealed_part_and_a_signed_one_still_verifies() {
        for security in [Security::Mac, Security::Seal] {
            let mut robot = Keyring::new(Some(pid("arm")), 1);
            let mut reader = Keyring::new(Some(pid("policy")), 1);
            robot.install(&stream_key(9, 1, true, security), 0);
            reader.install(&stream_key(9, 1, false, security), 0);
            let sealed = robot.seal_sample(KEY, &sample(1), false).expect("seal");
            let clear = sealed.windows(8).any(|w| w == b"wrench!!");
            assert_eq!(clear, security == Security::Mac, "{security:?}");
            // The keyed reader opens and verifies it either way.
            let opened = reader.open_sample(KEY, sealed.clone(), &pid("arm"), false);
            assert_eq!(opened.expect("open").payload, sample(1).payload);
            // A member without the key can't open (or forge) it.
            let mut stranger = Keyring::new(Some(pid("cockpit")), 1);
            assert!(matches!(
                stranger.open_sample(KEY, sealed.clone(), &pid("arm"), false),
                Err(KeyError::Unknown(9))
            ));
            // A changed body fails to verify at the keyed reader.
            let mut forged = robot.seal_sample(KEY, &sample(2), false).expect("seal");
            let at = forged.len() - TAG_LEN - 1;
            forged[at] ^= 1;
            assert!(matches!(
                reader.open_sample(KEY, forged, &pid("arm"), false),
                Err(KeyError::Open(OpenError::Auth))
            ));
            // A stream key never opens another robot's part.
            let other = robot.seal_sample(KEY, &sample(3), false).expect("seal");
            assert!(matches!(
                reader.open_sample(KEY, other, &pid("gripper"), false),
                Err(KeyError::Scope(_))
            ));
        }
    }

    #[test]
    fn rotation_installs_above_the_floor_then_the_first_newer_message_and_retire_drop_the_old() {
        let mut robot_v1 = Keyring::new(Some(pid("arm")), 1);
        let mut reader = Keyring::new(Some(pid("policy")), 1);
        robot_v1.install(&stream_key(9, 1, true, Security::Mac), 0);
        reader.install(&stream_key(9, 1, false, Security::Mac), 0);
        let old = robot_v1.seal_sample(KEY, &sample(1), false).expect("seal");
        // The robot's new generation gets version 2; the reader installs it
        // first, and a send key for another generation is refused.
        let mut robot_v2 = Keyring::new(Some(pid("arm")), 2);
        let v2 = |send| IssuedKey {
            generation: 2,
            ..stream_key(12, 2, send, Security::Mac)
        };
        assert_eq!(reader.install(&v2(false), 0), Some(false));
        assert_eq!(
            robot_v1.install(&v2(true), 0),
            None,
            "another generation's key"
        );
        assert_eq!(robot_v2.install(&v2(true), 0), Some(true));
        // Both versions open until the first message under v2...
        assert!(
            reader
                .open_sample(KEY, old.clone(), &pid("arm"), false)
                .is_ok()
        );
        let new = robot_v2.seal_sample(KEY, &sample(1), false).expect("seal");
        assert!(reader.open_sample(KEY, new, &pid("arm"), false).is_ok());
        // ...which drops v1, as `retire_through = [9]` does.
        let late = robot_v1.seal_sample(KEY, &sample(2), false).expect("seal");
        assert!(matches!(
            reader.open_sample(KEY, late, &pid("arm"), false),
            Err(KeyError::Unknown(9))
        ));
        let ack = reader.apply(
            &KeyDelivery {
                keys: Vec::new(),
                retire_through: vec![9],
            },
            0,
        );
        assert_eq!(ack.retired, vec![9]);
        // Recovery never reinstalls a retired or superseded version.
        assert_eq!(
            reader.install(&stream_key(9, 1, false, Security::Mac), 0),
            None
        );
        assert!(reader.holds(12));
    }

    #[test]
    fn a_lease_key_goes_when_its_token_ends_or_never_appears() {
        let command = |id, token| IssuedKey {
            key_id: id,
            purpose: KeyPurpose::Command,
            sender: Some(pid("policy")),
            robot: Some(pid("arm")),
            channels: vec![ChannelId(3)],
            token: LeaseToken(token),
            version: 0,
            key: KeyBytes([id as u8; 32]),
            send: false,
            security: Security::Mac,
            generation: 1,
        };
        let mut edge = Keyring::new(Some(pid("arm")), 1);
        edge.install(&command(20, 5), 0);
        edge.install(&command(21, 6), 0);
        let mut snapshot = crate::authority::Authority::new(
            crate::wire::SessionId::named("demo"),
            crate::authority::AuthorityConfig::default(),
        )
        .snapshot(Vec::new(), 0);
        snapshot.leases.push(crate::wire::Lease {
            channel: ChannelId(3),
            holder: pid("policy"),
            priority: crate::wire::PriorityClass::Model,
            token: LeaseToken(5),
            ttl_ns: 1,
            granted_ns: 0,
            expires_ns: 1,
            binding: Fingerprint::default(),
            key_id: 20,
        });
        edge.observe(&snapshot, 1);
        assert!(edge.holds(20) && edge.holds(21), "21 may still commit");
        snapshot.host_ns += 1;
        edge.observe(&snapshot, UNSEEN_TTL_NS);
        assert!(edge.holds(20) && !edge.holds(21), "21 never appeared");
        // A delayed, older snapshot without the lease drops nothing.
        let mut stale = snapshot.clone();
        stale.leases.clear();
        stale.host_ns -= 1;
        edge.observe(&stale, UNSEEN_TTL_NS);
        assert!(edge.holds(20));
        snapshot.leases.clear();
        snapshot.host_ns += 1;
        edge.observe(&snapshot, UNSEEN_TTL_NS);
        assert!(!edge.holds(20), "20's token ended");
    }

    #[test]
    fn each_subscription_gets_every_sample_once_and_a_newer_key_restarts_it() {
        let mut robot = Keyring::new(Some(pid("arm")), 1);
        let mut reader = Keyring::new(Some(pid("policy")), 1);
        robot.install(&stream_key(9, 1, true, Security::Mac), 0);
        reader.install(&stream_key(9, 1, false, Security::Mac), 0);
        let msg = robot.seal_sample(KEY, &sample(1), false).expect("seal");
        let mut subscriptions = [StreamWindows::default(), StreamWindows::default()];
        let open = |reader: &mut Keyring, windows: &mut StreamWindows, msg: &[u8]| {
            let (opening, version) =
                reader.begin_open_stream(msg, &pid("arm"), ChannelId(3), false, windows)?;
            opening.finish(KEY, &mut msg.to_vec())?;
            let restarted = windows
                .commit(opening.header(), version)
                .map_err(OpenError::from)?;
            reader.opened(opening.header())?;
            Ok::<_, KeyError>(restarted)
        };
        for windows in &mut subscriptions {
            assert!(!open(&mut reader, windows, &msg).expect("each gets it"));
            assert!(open(&mut reader, windows, &msg).is_err(), "a repeat");
        }
        // The robot's next generation, its seqs from 1 under version 2.
        let v2 = IssuedKey {
            generation: 2,
            ..stream_key(12, 2, true, Security::Mac)
        };
        let mut restarted = Keyring::new(Some(pid("arm")), 2);
        restarted.install(&v2, 0);
        reader.install(&IssuedKey { send: false, ..v2 }, 0);
        let msg = restarted.seal_sample(KEY, &sample(1), false).expect("seal");
        assert!(open(&mut reader, &mut subscriptions[0], &msg).expect("newer key"));
    }

    #[test]
    fn a_delayed_recovery_never_brings_back_an_ended_lease_key() {
        // The holder sends under token 5's key...
        let key = IssuedKey {
            key_id: 30,
            purpose: KeyPurpose::Command,
            sender: Some(pid("policy")),
            robot: Some(pid("arm")),
            channels: vec![ChannelId(3)],
            token: LeaseToken(5),
            version: 0,
            key: KeyBytes([30; 32]),
            send: true,
            security: Security::Seal,
            generation: 1,
        };
        let mut holder = Keyring::new(Some(pid("policy")), 1);
        assert_eq!(holder.install(&key, 0), Some(true));
        let chunk = ActionChunk {
            channel: ChannelId(3),
            epoch: Epoch(1),
            lease_token: LeaseToken(5),
            seq: 1,
            source: pid("policy"),
            binding: Fingerprint::default(),
            based_on: Vec::new(),
            t0_ns: 0,
            dt_ns: 1,
            replace_from_ns: 0,
            valid_until_ns: 1,
            count: 1,
            actions: Bytes::from_static(b"a"),
            publish_ns: 0,
            tick: 0,
        };
        let first = holder.seal_chunk(KEY, &pid("arm"), &chunk).expect("seal");
        let mut snapshot = crate::authority::Authority::new(
            crate::wire::SessionId::named("demo"),
            crate::authority::AuthorityConfig::default(),
        )
        .snapshot(Vec::new(), 0);
        let lease = crate::wire::Lease {
            channel: ChannelId(3),
            holder: pid("policy"),
            priority: crate::wire::PriorityClass::Model,
            token: LeaseToken(5),
            ttl_ns: 1,
            granted_ns: 0,
            expires_ns: 1,
            binding: Fingerprint::default(),
            key_id: 30,
        };
        snapshot.leases.push(lease);
        holder.observe(&snapshot, 1);
        // ...the token ends...
        snapshot.leases.clear();
        snapshot.revision += 1;
        holder.observe(&snapshot, 2);
        assert!(!holder.holds(30));
        // ...and a GetKeys reply from before that lands: refused, so no
        // second message ever goes out under nonce 1.
        assert_eq!(holder.install(&key, 3), None);
        assert!(matches!(
            holder.seal_chunk(KEY, &pid("arm"), &chunk),
            Err(KeyError::NoKey(_))
        ));
        assert_eq!(Header::parse(&first).expect("header").seq, 1);
    }

    /// A lease key on part 3.
    fn lease_key(
        id: u32,
        purpose: KeyPurpose,
        sender: &str,
        robot: &str,
        token: u64,
        send: bool,
    ) -> IssuedKey {
        IssuedKey {
            key_id: id,
            purpose,
            sender: Some(pid(sender)),
            robot: Some(pid(robot)),
            channels: vec![ChannelId(3)],
            token: LeaseToken(token),
            version: 0,
            key: KeyBytes([id as u8; 32]),
            send,
            security: Security::Mac,
            generation: 1,
        }
    }

    fn estop_key(send: bool) -> IssuedKey {
        IssuedKey {
            robot: None,
            channels: Vec::new(),
            ..lease_key(40, KeyPurpose::Estop, "policy", "policy", 0, send)
        }
    }

    fn chunk(token: u64, channel: ChannelId, source: &str) -> ActionChunk {
        ActionChunk {
            channel,
            epoch: Epoch(1),
            lease_token: LeaseToken(token),
            seq: 1,
            source: pid(source),
            binding: Fingerprint::default(),
            based_on: Vec::new(),
            t0_ns: 0,
            dt_ns: 1,
            replace_from_ns: 0,
            valid_until_ns: 1,
            count: 1,
            actions: Bytes::from_static(b"a"),
            publish_ns: 0,
            tick: 0,
        }
    }

    /// Revision `revision` of a view with these members and lease tokens.
    fn view(revision: u64, members: &[(&str, u32)], tokens: &[u64]) -> crate::wire::DeviceView {
        let state = crate::authority::Authority::new(
            crate::wire::SessionId::named("demo"),
            crate::authority::AuthorityConfig::default(),
        )
        .snapshot(Vec::new(), 0);
        let mut view = crate::wire::DeviceView::of(&state, &pid("arm"));
        view.revision = revision;
        view.members = members
            .iter()
            .map(|(id, generation)| crate::wire::view::Member {
                id: pid(id),
                generation: *generation,
            })
            .collect();
        view.leases = tokens
            .iter()
            .map(|token| crate::wire::Lease {
                channel: ChannelId(3),
                holder: pid("policy"),
                priority: crate::wire::PriorityClass::Model,
                token: LeaseToken(*token),
                ttl_ns: 1,
                granted_ns: 0,
                expires_ns: 1,
                binding: Fingerprint::default(),
                key_id: 0,
            })
            .collect();
        view
    }

    fn seq(sealed: &[u8]) -> u64 {
        Header::parse(sealed).expect("header").seq
    }

    #[test]
    fn retiring_a_version_leaves_another_robots_slot_alone() {
        let mut reader = Keyring::new(Some(pid("policy")), 1);
        let other = IssuedKey {
            robot: Some(pid("gripper")),
            ..stream_key(10, 1, false, Security::Mac)
        };
        reader.install(&stream_key(9, 1, false, Security::Mac), 0);
        reader.install(&other, 0);
        reader.retire(9);
        assert!(!reader.holds(9));
        assert!(reader.holds(10), "the same parts of another robot");
    }

    #[test]
    fn a_stream_repeat_is_refused_before_its_tag_is_checked() {
        let mut robot = Keyring::new(Some(pid("arm")), 1);
        let mut reader = Keyring::new(Some(pid("policy")), 1);
        robot.install(&stream_key(9, 1, true, Security::Mac), 0);
        reader.install(&stream_key(9, 1, false, Security::Mac), 0);
        let msg = robot.seal_sample(KEY, &sample(1), false).expect("seal");
        let mut windows = StreamWindows::default();
        let (opening, version) = reader
            .begin_open_stream(&msg, &pid("arm"), ChannelId(3), false, &windows)
            .expect("the first");
        windows.commit(opening.header(), version).expect("commit");
        assert!(matches!(
            reader.begin_open_stream(&msg, &pid("arm"), ChannelId(3), false, &windows),
            Err(KeyError::Open(OpenError::Replayed(_)))
        ));
    }

    #[test]
    fn a_second_key_of_the_same_version_keeps_the_first_keys_replay_window() {
        let header = |key_id| Header {
            encrypted: false,
            kind: Kind::Sample,
            key_id,
            seq: 1,
        };
        let mut windows = StreamWindows::default();
        assert_eq!(windows.commit(&header(9), 1), Ok(false));
        assert_eq!(windows.commit(&header(10), 1), Ok(false));
        assert!(windows.check(&header(9)).is_err(), "still a repeat");
        // A newer version does drop it.
        assert_eq!(windows.commit(&header(11), 2), Ok(true));
        assert!(windows.check(&header(9)).is_ok());
    }

    #[test]
    fn redelivering_a_held_key_keeps_its_sender_and_counters() {
        let key = lease_key(30, KeyPurpose::Command, "policy", "arm", 5, true);
        let mut holder = Keyring::new(Some(pid("policy")), 1);
        assert_eq!(holder.install(&key, 0), Some(true));
        let good = chunk(5, ChannelId(3), "policy");
        let first = holder.seal_chunk(KEY, &pid("arm"), &good).expect("seal");
        assert_eq!(holder.install(&key, 1), Some(true), "a repeat delivery");
        let second = holder.seal_chunk(KEY, &pid("arm"), &good).expect("seal");
        assert_eq!((seq(&first), seq(&second)), (1, 2), "no nonce is reused");
    }

    #[test]
    fn a_held_key_switches_to_sending_only_when_told_and_only_with_the_same_bytes() {
        let receiving = lease_key(30, KeyPurpose::Command, "policy", "arm", 5, false);
        let sending = IssuedKey {
            send: true,
            ..receiving.clone()
        };
        let mut holder = Keyring::new(Some(pid("policy")), 1);
        assert_eq!(holder.install(&receiving, 0), Some(false));
        assert_eq!(
            holder.install(&receiving, 1),
            Some(false),
            "still receiving"
        );
        let forged = IssuedKey {
            key: KeyBytes([99; 32]),
            ..sending.clone()
        };
        assert_eq!(holder.install(&forged, 2), None, "other bytes");
        assert_eq!(holder.install(&sending, 3), Some(true));
    }

    #[test]
    fn a_delivery_that_changes_a_held_keys_scope_is_refused() {
        let key = lease_key(30, KeyPurpose::Command, "policy", "arm", 5, false);
        let moved = IssuedKey {
            token: LeaseToken(6),
            ..key.clone()
        };
        let mut edge = Keyring::new(Some(pid("arm")), 1);
        edge.install(&key, 0);
        assert_eq!(edge.install(&moved, 1), None);
    }

    #[test]
    fn a_version_below_the_newest_seen_is_refused_though_above_the_floor() {
        let mut reader = Keyring::new(Some(pid("policy")), 1);
        assert_eq!(
            reader.install(&stream_key(12, 3, false, Security::Mac), 0),
            Some(false)
        );
        assert_eq!(
            reader.install(&stream_key(11, 2, false, Security::Mac), 0),
            None
        );
        assert!(!reader.holds(11));
    }

    #[test]
    fn a_retired_lease_key_is_never_installed_again() {
        let key = lease_key(20, KeyPurpose::Command, "policy", "arm", 5, false);
        let mut edge = Keyring::new(Some(pid("arm")), 1);
        edge.install(&key, 0);
        edge.retire(20);
        assert!(!edge.holds(20));
        assert_eq!(edge.install(&key, 1), None);
    }

    #[test]
    fn only_a_newer_generation_of_this_participant_ends_its_keys_and_for_good() {
        let command = lease_key(20, KeyPurpose::Command, "policy", "arm", 5, false);
        let mut edge = Keyring::new(Some(pid("arm")), 2);
        edge.install(&command, 0);
        edge.install(&stream_key(9, 1, false, Security::Mac), 0);
        edge.observe(&view(1, &[("arm", 1)], &[5]), 1);
        assert!(edge.holds(20) && edge.holds(9), "an older generation");
        edge.observe(&view(2, &[("arm", 2)], &[5]), 2);
        assert!(edge.holds(20) && edge.holds(9), "this generation");
        edge.observe(&view(3, &[("arm", 3)], &[5]), 3);
        assert!(!edge.holds(20) && !edge.holds(9), "a newer one");
        assert_eq!(edge.install(&command, 4), None, "never brought back");
    }

    #[test]
    fn an_estop_key_lives_exactly_while_its_senders_generation_does() {
        let mut edge = Keyring::new(Some(pid("arm")), 1);
        edge.install(&estop_key(false), 0);
        edge.observe(&view(1, &[("policy", 1)], &[]), UNSEEN_TTL_NS);
        assert!(edge.holds(40), "the sender's generation is in effect");
        edge.observe(&view(2, &[("policy", 2)], &[]), UNSEEN_TTL_NS);
        assert!(!edge.holds(40), "it was superseded");
    }

    #[test]
    fn sends_stream_needs_a_sending_stream_key_covering_the_part() {
        let mut robot = Keyring::new(Some(pid("arm")), 1);
        assert!(!robot.sends_stream(ChannelId(3)), "no key");
        robot.install(&stream_key(9, 1, false, Security::Mac), 0);
        assert!(!robot.sends_stream(ChannelId(3)), "only receiving");
        robot.install(
            &IssuedKey {
                channels: vec![ChannelId(5)],
                ..lease_key(30, KeyPurpose::Command, "arm", "arm", 5, true)
            },
            0,
        );
        assert!(!robot.sends_stream(ChannelId(5)), "not a stream key");
        robot.install(&stream_key(9, 1, true, Security::Mac), 0);
        assert!(robot.sends_stream(ChannelId(3)));
        assert!(!robot.sends_stream(ChannelId(6)), "another part");
    }

    #[test]
    fn the_lowest_id_among_equal_versions_is_the_one_sent_under() {
        let mut holder = Keyring::new(Some(pid("policy")), 1);
        for id in [21, 20, 22] {
            holder.install(
                &lease_key(id, KeyPurpose::Command, "policy", "arm", 5, true),
                0,
            );
        }
        let request = LockstepRequest {
            request_id: 1,
            op: crate::wire::LockstepOp::Probe,
        };
        let sealed = holder
            .seal_lockstep_request(KEY, &pid("arm"), LeaseToken(5), &request)
            .expect("seal");
        assert_eq!(Header::parse(&sealed).expect("header").key_id, 20);
    }

    #[test]
    fn a_two_phase_open_checks_the_stream_key_against_its_robot_and_part() {
        let mut robot = Keyring::new(Some(pid("arm")), 1);
        let mut reader = Keyring::new(Some(pid("policy")), 1);
        robot.install(&stream_key(9, 1, true, Security::Mac), 0);
        reader.install(&stream_key(9, 1, false, Security::Mac), 0);
        let msg = robot.seal_sample(KEY, &sample(1), false).expect("seal");
        let windows = StreamWindows::default();
        let sample_of =
            |who: &str, channel| reader.begin_open_sample(&msg, &pid(who), channel, false);
        let stream_of = |who: &str, channel| {
            reader.begin_open_stream(&msg, &pid(who), channel, false, &windows)
        };
        assert!(sample_of("arm", ChannelId(3)).is_ok());
        assert!(stream_of("arm", ChannelId(4)).is_ok());
        for (who, channel) in [("gripper", ChannelId(3)), ("arm", ChannelId(5))] {
            assert!(
                matches!(sample_of(who, channel), Err(KeyError::Scope(_))),
                "{who} {channel}"
            );
            assert!(
                matches!(stream_of(who, channel), Err(KeyError::Scope(_))),
                "{who} {channel}"
            );
        }
    }

    #[test]
    fn a_chunk_is_sealed_only_under_the_command_key_of_its_robot_token_and_part() {
        let mut holder = Keyring::new(Some(pid("policy")), 1);
        holder.install(
            &lease_key(30, KeyPurpose::Command, "policy", "arm", 5, true),
            0,
        );
        let good = chunk(5, ChannelId(3), "policy");
        assert!(holder.seal_chunk(KEY, &pid("arm"), &good).is_ok());
        for (robot, bad) in [
            ("gripper", good.clone()),
            ("arm", chunk(6, ChannelId(3), "policy")),
            ("arm", chunk(5, ChannelId(4), "policy")),
        ] {
            assert!(
                matches!(
                    holder.seal_chunk(KEY, &pid(robot), &bad),
                    Err(KeyError::NoKey(_))
                ),
                "{robot} {bad:?}"
            );
        }
    }

    #[test]
    fn a_chunk_opens_only_under_a_key_for_this_robot_matching_its_part_token_and_source() {
        let key = lease_key(30, KeyPurpose::Command, "policy", "arm", 5, false);
        let mut edge = Keyring::new(Some(pid("arm")), 1);
        let mut other = Keyring::new(Some(pid("gripper")), 1);
        edge.install(&key, 0);
        other.install(&key, 0);
        // Whoever has the bytes can seal anything: the scope is the check.
        let mut forger = SealSender::new(
            30,
            Key::from_bytes([30; 32]),
            kinds(KeyPurpose::Command),
            Security::Mac,
        );
        let mut seal = |chunk: &ActionChunk| {
            let body = codec::encode_chunk(chunk).expect("encode").to_vec();
            forger.seal(Kind::ActionChunk, KEY, &body).expect("seal")
        };
        let good = chunk(5, ChannelId(3), "policy");
        assert_eq!(edge.open_chunk(KEY, seal(&good)).expect("in scope"), good);
        assert!(
            matches!(other.open_chunk(KEY, seal(&good)), Err(KeyError::Scope(_))),
            "another robot's key"
        );
        for bad in [
            chunk(5, ChannelId(4), "policy"),
            chunk(6, ChannelId(3), "policy"),
            chunk(5, ChannelId(3), "mallory"),
        ] {
            assert!(
                matches!(edge.open_chunk(KEY, seal(&bad)), Err(KeyError::Scope(_))),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn a_sample_is_sealed_only_under_a_stream_key_of_this_robot() {
        let mut robot = Keyring::new(Some(pid("arm")), 1);
        let another = IssuedKey {
            robot: Some(pid("gripper")),
            ..stream_key(9, 1, true, Security::Mac)
        };
        assert_eq!(robot.install(&another, 0), Some(true));
        assert!(matches!(
            robot.seal_sample(KEY, &sample(1), false),
            Err(KeyError::NoKey(_))
        ));
    }

    #[test]
    fn a_lockstep_request_travels_under_its_robots_command_key_of_that_token() {
        let key = |send| lease_key(30, KeyPurpose::Command, "policy", "arm", 5, send);
        let mut holder = Keyring::new(Some(pid("policy")), 1);
        let mut edge = Keyring::new(Some(pid("arm")), 1);
        let mut other = Keyring::new(Some(pid("gripper")), 1);
        holder.install(&key(true), 0);
        edge.install(&key(false), 0);
        other.install(&key(false), 0);
        let request = LockstepRequest {
            request_id: 1,
            op: crate::wire::LockstepOp::Probe,
        };
        let sealed = holder
            .seal_lockstep_request(KEY, &pid("arm"), LeaseToken(5), &request)
            .expect("seal");
        assert_eq!(
            edge.open_lockstep_request(KEY, sealed.clone())
                .expect("open"),
            (request.clone(), LeaseToken(5))
        );
        assert!(
            matches!(
                other.open_lockstep_request(KEY, sealed),
                Err(KeyError::Scope(_))
            ),
            "another robot's key"
        );
        for (robot, token) in [("gripper", 5), ("arm", 6)] {
            assert!(
                matches!(
                    holder.seal_lockstep_request(KEY, &pid(robot), LeaseToken(token), &request),
                    Err(KeyError::NoKey(_))
                ),
                "{robot} {token}"
            );
        }
    }

    #[test]
    fn a_lockstep_response_travels_under_this_robots_reply_key_of_that_token() {
        let key = |send| lease_key(31, KeyPurpose::LockstepReply, "arm", "arm", 5, send);
        let mut edge = Keyring::new(Some(pid("arm")), 1);
        let mut holder = Keyring::new(Some(pid("policy")), 1);
        edge.install(&key(true), 0);
        edge.install(
            &lease_key(32, KeyPurpose::LockstepReply, "arm", "gripper", 7, true),
            0,
        );
        holder.install(&key(false), 0);
        let response = LockstepResponse {
            request_id: 1,
            result: crate::wire::LockstepResult::Probe,
        };
        let mut seal = || {
            edge.seal_lockstep_response(KEY, LeaseToken(5), &response)
                .expect("seal")
        };
        let (good, robot, token) = (seal(), seal(), seal());
        assert_eq!(
            holder
                .open_lockstep_response(KEY, good, &pid("arm"), LeaseToken(5))
                .expect("open"),
            response
        );
        assert!(matches!(
            holder.open_lockstep_response(KEY, robot, &pid("gripper"), LeaseToken(5)),
            Err(KeyError::Scope(_))
        ));
        assert!(matches!(
            holder.open_lockstep_response(KEY, token, &pid("arm"), LeaseToken(6)),
            Err(KeyError::Scope(_))
        ));
        // Key 32 names another robot, and no key names token 6.
        for token in [6, 7] {
            assert!(
                matches!(
                    edge.seal_lockstep_response(KEY, LeaseToken(token), &response),
                    Err(KeyError::NoKey(_))
                ),
                "{token}"
            );
        }
    }

    #[test]
    fn an_estop_signal_opens_only_as_coming_from_its_keys_sender() {
        let mut member = Keyring::new(Some(pid("policy")), 1);
        let mut edge = Keyring::new(Some(pid("arm")), 1);
        member.install(&estop_key(true), 0);
        edge.install(&estop_key(false), 0);
        let signal = EstopSignal {
            reason: "stop".into(),
            sent_ns: 1,
        };
        let good = member.seal_estop(KEY, &signal).expect("seal");
        assert_eq!(
            edge.open_estop(KEY, good, &pid("policy")).expect("open"),
            signal
        );
        let forged = member.seal_estop(KEY, &signal).expect("seal");
        assert!(matches!(
            edge.open_estop(KEY, forged, &pid("mallory")),
            Err(KeyError::Scope(_))
        ));
    }

    #[test]
    fn a_keyrings_debug_output_never_shows_a_keys_bytes() {
        let secret = [0xa7; 32];
        let key = IssuedKey {
            key: KeyBytes(secret),
            ..stream_key(9, 1, true, Security::Mac)
        };
        let mut ring = Keyring::new(Some(pid("arm")), 1);
        ring.install(&key, 0);
        crate::crypto::assert_redacted(&ring, &secret);
        crate::crypto::assert_redacted(&key, &secret);
        crate::crypto::assert_redacted(&KeyBytes(secret), &secret);
    }
}
