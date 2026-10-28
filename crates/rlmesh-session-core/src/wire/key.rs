//! Delivered keys (design §8.4, §8.5): what the host draws, scopes and
//! delivers to a participant, and what the participant acknowledges. The
//! key bytes travel only inside sealed messages (a `KeyDelivery` under the
//! recipient's `k_down`, a `CtlReply`); authority state never carries one.

#[allow(unused_imports)]
use crate::prelude::*;

use zeroize::Zeroize;

pub use crate::crypto::Security;
use crate::wire::ids::{ChannelId, LeaseToken, ParticipantId};

/// What a key is for, and so which kinds it carries (§8.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KeyPurpose {
    /// `k_cmd`: chunks and lockstep requests under one token to one robot,
    /// for parts of one transport priority.
    Command,
    /// `k_obs`: one robot's lockstep replies under one token.
    LockstepReply,
    /// `k_stream`: one part's samples and `_applied` entries.
    Stream,
    /// `k_session_stream`: one host session stream.
    SessionStream,
    /// `k_estop`: one participant generation's e-stop signals.
    Estop,
}

impl KeyPurpose {
    /// Whether keys of this purpose rotate (they belong to a slot that
    /// outlives a key) rather than live and die with a token or generation.
    pub fn rotates(self) -> bool {
        matches!(self, Self::Stream | Self::SessionStream)
    }
}

/// A key's 32 bytes, wiped on drop and never printed.
#[derive(Clone, PartialEq, Eq)]
pub struct KeyBytes(pub [u8; 32]);

impl Drop for KeyBytes {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl core::fmt::Debug for KeyBytes {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("KeyBytes(..)")
    }
}

/// One key with its scope (`Key` on the wire). A receiver uses it only for
/// its purpose, on the paths its scope names, from that sender generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IssuedKey {
    /// Its id: one counter per host instance, never reused.
    pub key_id: u32,
    /// What it is for.
    pub purpose: KeyPurpose,
    /// The only participant that may send under it (`None`: the host).
    pub sender: Option<ParticipantId>,
    /// The robot it concerns (commands, lockstep replies, streams).
    pub robot: Option<ParticipantId>,
    /// The parts it covers: a command key's parts, a stream key's part and
    /// its applied stream.
    pub channels: Vec<ChannelId>,
    /// The lease token of a command or lockstep reply key.
    pub token: LeaseToken,
    /// Rises per slot of a rotating key; 0 for a fixed one.
    pub version: u32,
    /// The bytes.
    pub key: KeyBytes,
    /// The recipient sends under it.
    pub send: bool,
    /// The level of its configured kinds.
    pub security: Security,
    /// The sender's generation (0 for a host-sent key).
    pub generation: u32,
}

/// Keys to install and keys to retire, for one participant (§8.5).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeyDelivery {
    /// Install these (or switch one already held to sending).
    pub keys: Vec<IssuedKey>,
    /// Drop these and, for a rotating key, every older version of its slot.
    pub retire_through: Vec<u32>,
}

impl KeyDelivery {
    /// Whether it carries nothing.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty() && self.retire_through.is_empty()
    }
}

/// What a delivery changed at its recipient: the `KeysAck` verb.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeysAck {
    /// Keys now held.
    pub installed: Vec<u32>,
    /// Keys now sent under.
    pub sending: Vec<u32>,
    /// The `retire_through` ids handled.
    pub retired: Vec<u32>,
}

impl KeysAck {
    /// Whether it acknowledges nothing.
    pub fn is_empty(&self) -> bool {
        self.installed.is_empty() && self.sending.is_empty() && self.retired.is_empty()
    }
}
