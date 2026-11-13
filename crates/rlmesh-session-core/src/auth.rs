//! Attaching through the handshake, sans-IO:
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
//!   each holder, [`issuer`] at the host): a lease's `k_cmd` and
//!   `k_obs`, a stream's `k_stream`. A [`Link`] holds its participant's
//!   [`Keyring`], installs what the host delivers (`KeyDelivery`, the keys a
//!   reply carries), and seals and opens chunks, samples, history and
//!   lockstep under them;
//! - machine pairing's exchanges ([`pairing`]): rlmeshd's pending
//!   exchanges and the owner's approval, and the laptop's side.
//!
//! What a participant sends under its keys, the host accepts only from the
//! generation in effect: a handshake makes a candidate, and the candidate
//! takes effect when the host accepts its sealed `Attach`
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

/// The `local` device's static private key from an embedded host's local
/// secret: `kdf(local secret, "rlmesh/v2 local x25519")`.
fn local_keypair(secret: &[u8; 32]) -> StaticKeypair {
    let mut private = Zeroizing::new(
        kdf(secret, LOCAL_LABEL, &[]).unwrap_or_else(|_| unreachable!("a short label")),
    );
    StaticKeypair::from_private(&mut private)
}

#[cfg(test)]
mod tests;
