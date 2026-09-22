//! The relay seam: what the driver does with a payload bound for a served peer.
//!
//! The runtime is the interpreter between two peers that never talk to each
//! other, so every payload it relays is checked against the *target* leg's
//! [`PeerCeiling`]. A [`RelayPolicy`] decides whether it goes out as-is, goes
//! out converted with an advisory, or is refused. The OSS default,
//! [`RefusingRelayPolicy`], never converts.

use std::sync::Arc;

use prost::bytes::Bytes;
use rlmesh_proto::spaces::v1::{SpaceSpec, space_spec::Spec};
use rlmesh_spaces::{Advisory, DType};

use crate::spec::PeerCeiling;

/// Which way a relayed payload travels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Leg {
    /// The env contract or an observation, on its way to the model.
    EnvToModel,
    /// An action, on its way to the env.
    ModelToEnv,
}

impl Leg {
    /// The peer this leg delivers to.
    pub fn target(self) -> &'static str {
        match self {
            Leg::EnvToModel => "model",
            Leg::ModelToEnv => "env",
        }
    }
}

/// One relayed payload, for a [`RelayPolicy`] to check against the target
/// leg's [`PeerCeiling`]. Capability gating is not checked here: it happens
/// where a feature is emitted.
#[derive(Debug, Clone, PartialEq)]
pub struct PayloadFacts {
    /// The space typing `leaves`, in leaf order. For the env contract, a
    /// two-element Tuple of the observation space then the action space.
    pub space: Arc<SpaceSpec>,
    /// Summed leaf bytes, not the framed wire size; the encoded contract's
    /// length for the env contract.
    pub byte_len: usize,
    /// The payload's leaves, for a policy that converts them; empty for the
    /// env contract, which is checked before any leaf flows.
    pub leaves: Vec<Bytes>,
}

impl PayloadFacts {
    /// The facts of `leaves` typed by `space`.
    pub fn new(space: Arc<SpaceSpec>, leaves: Vec<Bytes>) -> Self {
        Self {
            space,
            byte_len: leaves.iter().map(Bytes::len).sum(),
            leaves,
        }
    }

    /// Why `ceiling` cannot carry this payload, or `None` when it can.
    pub fn exceeds(&self, ceiling: &PeerCeiling) -> Option<String> {
        if let Some(dtype) = dtype_outside(&self.space, ceiling) {
            return Some(format!("dtype {} is outside its ceiling", dtype.name()));
        }
        if self.byte_len > ceiling.max_message_size {
            return Some(format!(
                "{} bytes exceeds its {}-byte message cap",
                self.byte_len, ceiling.max_message_size
            ));
        }
        None
    }
}

/// The first leaf dtype of `space` that `ceiling` cannot decode.
fn dtype_outside(space: &SpaceSpec, ceiling: &PeerCeiling) -> Option<DType> {
    match &space.spec {
        Some(Spec::Dict(dict)) => dict
            .spaces
            .iter()
            .find_map(|space| dtype_outside(space, ceiling)),
        Some(Spec::Tuple(tuple)) => tuple
            .spaces
            .iter()
            .find_map(|space| dtype_outside(space, ceiling)),
        // A value no DType names never decoded into this spec: the contract
        // decode at the handshake refuses it first.
        _ => DType::try_from(space.dtype)
            .ok()
            .filter(|dtype| !ceiling.dtypes.contains(dtype)),
    }
}

/// A policy's verdict on one relayed payload.
#[derive(Debug, Clone, PartialEq)]
pub enum RelayDecision {
    /// Relay the payload unchanged.
    Forward,
    /// Relay `leaves` in its place, recording `advisory` on the report.
    /// `space` types the converted leaves when their layout changed; the
    /// emitted hook event then carries it in place of the original leg's space.
    ///
    /// On the env contract (empty `leaves`) only `advisory` takes effect.
    Convert {
        leaves: Vec<Bytes>,
        space: Option<Arc<SpaceSpec>>,
        advisory: Advisory,
    },
    /// Fail the route with this reason.
    Refuse(String),
}

/// Decides what reaches a served peer that may not decode a payload as sent.
///
/// Consulted for the env contract at session start and for every observation
/// and action after the transform hooks, only on a leg whose ceiling the spec
/// carries (an in-process leg has none). Install one with
/// [`RuntimeDriver::with_relay_policy`](crate::RuntimeDriver::with_relay_policy);
/// the default is [`RefusingRelayPolicy`].
///
/// The contract call is advisory-only: the driver records a
/// [`RelayDecision::Convert`]'s advisory but relays nothing, so the policy
/// owner must down-convert the contract it sends the model at resolve itself.
pub trait RelayPolicy: Send + Sync {
    fn reconcile(&self, leg: Leg, ceiling: &PeerCeiling, payload: &PayloadFacts) -> RelayDecision;
}

/// The OSS relay policy: forward what the target can decode, refuse the rest.
#[derive(Debug, Default)]
pub struct RefusingRelayPolicy;

impl RelayPolicy for RefusingRelayPolicy {
    fn reconcile(&self, _leg: Leg, ceiling: &PeerCeiling, payload: &PayloadFacts) -> RelayDecision {
        payload
            .exceeds(ceiling)
            .map_or(RelayDecision::Forward, RelayDecision::Refuse)
    }
}
