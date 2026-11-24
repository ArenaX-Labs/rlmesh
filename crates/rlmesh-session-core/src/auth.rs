//! The participant attach handshake and sealed link, sans-IO.
//!
//! - A participant's [`Credential`] runs the IK handshake ([`Credential::start`],
//!   [`finish`]) and yields its pairwise keys and the host's signing key.
//! - Its [`Link`] to the host seals verbs and heartbeats, opens replies and echoes, assigns
//!   request ids, verifies the host's signed broadcasts and holds the participant's
//!   [`Keyring`] for chunks, samples, history and lockstep.
//! - A host's [`HostIdentity`] and, per host instance, its [`HostAuth`]: the device table,
//!   name bindings and which generation of each participant is in effect.
//! - [`RequestWindow`]: per-generation request ids and the reply cache that makes a retried
//!   verb idempotent.
//! - [`keyring`] and [`issuer`]: the lease and stream keys the host delivers besides the
//!   handshake's.
//! - [`pairing`]: machine pairing's exchanges.
//!
//! The host accepts traffic only from the generation in effect: a handshake makes a
//! candidate, which takes effect when the host accepts its sealed `Attach`
//! ([`HostAuth::take_effect`]). Randomness is the caller's ([`Rng`]).

use zeroize::Zeroizing;

pub mod address;
mod credential;
mod host;
pub mod issuer;
pub mod keyring;
mod link;
pub mod pairing;
mod window;

pub use address::AddressKey;
pub use credential::{Attached, AuthError, Credential, finish};
pub use host::{Candidate, Device, HostAuth, HostIdentity, Pin};
pub use issuer::KeyIssuer;
pub use keyring::{KeyError, Keyring, StreamWindows};
pub use link::{Link, LinkError, ReplyOutcome, open_health, open_state, open_tick, open_view};
pub use window::{Admit, REQUEST_WINDOW, RequestWindow};

#[cfg(doc)]
use crate::crypto::Rng;
use crate::crypto::StaticKeypair;
use crate::crypto::lp::kdf;

/// The device of every process holding an embedded host's address.
pub const LOCAL_DEVICE: &str = "local";
/// The `kdf` label of the `local` device's static key.
pub const LOCAL_LABEL: &[u8] = b"rlmesh/v2 local x25519";
/// The label of a host id's hash ([`HostIdentity::host_id`]).
pub const HOST_ID_LABEL: &[u8] = b"rlmesh/v2 host id";

fn local_keypair(secret: &[u8; 32]) -> StaticKeypair {
    let mut private = Zeroizing::new(
        kdf(secret, LOCAL_LABEL, &[]).unwrap_or_else(|_| unreachable!("a short label")),
    );
    StaticKeypair::from_private(&mut private)
}

#[cfg(test)]
mod tests;
