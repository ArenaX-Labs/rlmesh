//! The keys a process holds besides its pairwise ones (design §8.4, §8.5),
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
    ActionChunk, AuthoritySnapshot, ChannelId, IssuedKey, KeyDelivery, KeyPurpose, KeysAck,
    LeaseToken, LockstepRequest, LockstepResponse, ParticipantId, Sample, Security,
};

/// How long a fixed key waits for its token to appear in authority state
/// before it is discarded (§8.5).
pub const UNSEEN_TTL_NS: i64 = 4_000_000_000;

/// The kinds a key of `purpose` carries.
pub fn kinds(purpose: KeyPurpose) -> KindSet {
    match purpose {
        KeyPurpose::Command => KindSet::of(&[Kind::ActionChunk, Kind::LockstepRequest]),
        KeyPurpose::LockstepReply => KindSet::of(&[Kind::LockstepReply]),
        KeyPurpose::Stream => KindSet::of(&[Kind::Sample, Kind::Applied]),
        KeyPurpose::SessionStream => KindSet::of(&[Kind::HostSessionStream]),
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
/// the newest key version it accepted (design §8.5: the first valid
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

    /// Takes an authority snapshot, if newer than the last: drops lease
    /// keys whose token ended (or never appeared within [`UNSEEN_TTL_NS`]),
    /// and every key once a newer generation of this participant took over.
    /// An older snapshot (a delayed broadcast, a cached reply) changes
    /// nothing.
    pub fn observe(&mut self, snapshot: &AuthoritySnapshot, now_ns: i64) {
        let order = (snapshot.revision, snapshot.host_ns);
        if self.observed.is_some_and(|last| order <= last) {
            return;
        }
        self.observed = Some(order);
        if let Some(me) = &self.me
            && snapshot
                .roster
                .iter()
                .any(|p| &p.id == me && p.generation > self.generation)
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
        let live: BTreeSet<LeaseToken> = snapshot.leases.iter().map(|l| l.token).collect();
        let gone = &mut self.gone;
        self.held.retain(|id, held| {
            if held.scope.purpose.rotates() || held.scope.purpose == KeyPurpose::Estop {
                return true;
            }
            if live.contains(&held.scope.token) {
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
        for security in [Security::Sign, Security::Seal] {
            let mut robot = Keyring::new(Some(pid("arm")), 1);
            let mut reader = Keyring::new(Some(pid("policy")), 1);
            robot.install(&stream_key(9, 1, true, security), 0);
            reader.install(&stream_key(9, 1, false, security), 0);
            let sealed = robot.seal_sample(KEY, &sample(1), false).expect("seal");
            let clear = sealed.windows(8).any(|w| w == b"wrench!!");
            assert_eq!(clear, security == Security::Sign, "{security:?}");
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
        robot_v1.install(&stream_key(9, 1, true, Security::Sign), 0);
        reader.install(&stream_key(9, 1, false, Security::Sign), 0);
        let old = robot_v1.seal_sample(KEY, &sample(1), false).expect("seal");
        // The robot's new generation gets version 2; the reader installs it
        // first, and a send key for another generation is refused.
        let mut robot_v2 = Keyring::new(Some(pid("arm")), 2);
        let v2 = |send| IssuedKey {
            generation: 2,
            ..stream_key(12, 2, send, Security::Sign)
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
            reader.install(&stream_key(9, 1, false, Security::Sign), 0),
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
            security: Security::Sign,
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
        robot.install(&stream_key(9, 1, true, Security::Sign), 0);
        reader.install(&stream_key(9, 1, false, Security::Sign), 0);
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
            ..stream_key(12, 2, true, Security::Sign)
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
}
