//! Machine pairing's messages (design §3, §8.8): what a laptop and rlmeshd
//! exchange on [`PAIR_KEY`]. The XX messages and the sealed bodies are
//! opaque here; [`crate::auth::pairing`] runs the exchange.

#[allow(unused_imports)]
use crate::prelude::*;

use crate::wire::authority::CtlError;

/// rlmeshd's pairing queryable, and the key expression its `Sealed`
/// messages are bound to.
pub const PAIR_KEY: &str = "rlmeshd/pair";

/// One pairing packet, laptop → rlmeshd.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairRequest {
    /// The attempt, 16 random bytes the laptop drew.
    pub exchange: [u8; 16],
    /// What it carries.
    pub step: PairStep,
}

/// What a [`PairRequest`] carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PairStep {
    /// XX msg 1.
    Xx1(Vec<u8>),
    /// XX msg 3.
    Xx3(Vec<u8>),
    /// A `Sealed` `PairNonce` or `PairWait`.
    Sealed(Vec<u8>),
}

/// rlmeshd's answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PairReply {
    /// XX msg 2.
    Xx2(Vec<u8>),
    /// A `Sealed` `PairCommit`, `PairReveal` or `PairResult`.
    Sealed(Vec<u8>),
    /// An unauthenticated refusal: busy, expired or unknown.
    Error(CtlError),
}

/// `PairNonce`: the laptop's `Ni`, and what it pairs as.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairNonce {
    /// `Ni`.
    pub ni: [u8; 32],
    /// The machine's name (its host name), from which rlmeshd makes its
    /// device id.
    pub name: String,
    /// What it is (`machine`).
    pub kind: String,
}

/// What an approved machine stores (`PairResult.ok`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairInfo {
    /// Its device id in rlmeshd's table.
    pub device: String,
    /// The host's static X25519 key.
    pub host_key: [u8; 32],
    /// The host id (`rlm-3f9a2c`).
    pub host_id: String,
    /// The host's base addresses, `host:port`.
    pub addresses: Vec<String>,
}
