//! Protocol identity and the compatibility rule.
//!
//! | Layer | Identifier | Rule |
//! |---|---|---|
//! | Generation | [`GENERATION`]: the key prefix `rlmesh/v2` and the protobuf package `rlmesh.session.v2` | A breaking change is a new generation on new keys; generations never share keys, so they never see each other's traffic. |
//! | Schema | protobuf | Additive only. Unknown fields are ignored; unknown enum values are refused where safety depends on them (`Mode`, `PriorityClass`, `EdgeMode`, dtypes) and ignored elsewhere. |
//! | Protocol minor | [`PROTOCOL_MINOR`], announced in `Attach` and `AuthorityState` | Monotonic, 0 at v0.2. Any minor talks to any other. New behaviour is a **feature string** (`"merge.blend"`). |
//! | Features | `Attach.supports/requires`, `AuthorityState.supports`, `PartSpec.requires` | Attach fails with `CtlStatus::Incompatible` if the host lacks a feature the participant requires. |
//! | Kind table | [`SPEC_VERSION`], in `DeclareParts` | New kinds and optional settings are additive. A host accepts a robot on any table from 1 up to its own, and refuses a newer one. |
//! | Wire revision | [`WIRE_REVISION`], in `HandshakeRequest` and `HandshakeAccept` | Pre-release only: bumped on any wire-breaking change before 0.2.0 freezes the protocol, frozen there. A participant refuses a host on another revision, or one that sends none (an earlier 0.2 pre-release). A host refuses a participant on another revision, and admits one that sends none (a board flashed before revisions were sent). Both refuse with `CtlStatus::Incompatible`, naming both revisions. |

#[allow(unused_imports)]
use crate::prelude::*;

/// The protocol generation: the `v2` of the key prefix.
pub const GENERATION: u32 = 2;

/// This build's protocol minor (0 at v0.2).
pub const PROTOCOL_MINOR: u32 = 0;

/// The kind table version a declaration is written against.
pub const SPEC_VERSION: u32 = 1;

/// This build's session wire revision: which transport and wire a 0.2
/// pre-release speaks, so two builds of one version that cannot talk say so
/// instead of timing out. 1: one port, which every traffic class dials.
/// Bumped on any wire-breaking change before 0.2.0 freezes the protocol;
/// frozen at 0.2.0. 0 on the wire is absent: a build from before the field,
/// whose host listened on four ports.
pub const WIRE_REVISION: u32 = 1;

/// What each wire revision was, by number.
const REVISIONS: &[&str] = &["four ports: an earlier 0.2 pre-release", "one port"];

/// What wire revision `revision` is (`"one port"`).
pub fn wire_revision_name(revision: u32) -> &'static str {
    usize::try_from(revision)
        .ok()
        .and_then(|r| REVISIONS.get(r))
        .copied()
        .unwrap_or("a later pre-release")
}

/// Why a participant refuses the host `host` (`"the host at 10.0.0.2:7500"`)
/// whose handshake accept carries `revision`, or `None` when it is this
/// build's.
pub fn host_revision_mismatch(host: &str, revision: u32) -> Option<String> {
    (revision != WIRE_REVISION).then(|| {
        format!(
            "{host} speaks session wire revision {revision} ({}); this build speaks revision \
             {WIRE_REVISION} ({}); rebuild both from the same release",
            wire_revision_name(revision),
            wire_revision_name(WIRE_REVISION),
        )
    })
}

/// Why a host refuses `participant`'s hello carrying `revision`, or `None`
/// when it is this build's or absent (0: a board flashed before revisions
/// were sent, which still talks to this host).
pub fn participant_revision_mismatch(participant: &str, revision: u32) -> Option<String> {
    (revision != 0 && revision != WIRE_REVISION).then(|| {
        format!(
            "participant {participant} speaks session wire revision {revision} ({}); this host \
             speaks revision {WIRE_REVISION} ({}); rebuild both from the same release",
            wire_revision_name(revision),
            wire_revision_name(WIRE_REVISION),
        )
    })
}

/// A robot's `Attach.supports` asks for its device view, which a host
/// listing it publishes on `host/view/{robot}`.
pub const VIEW_DEVICE: &str = "view.device";

/// A robot's `Attach.supports` asks for its device view sealed under its
/// own `k_down` on `p/{robot}/view` as well (kind 20): a board opens that at
/// a symmetric cipher's cost instead of checking a signature on each view.
/// A host listing it publishes both; the signed one stays for every other
/// reader and for robots that don't ask.
pub const VIEW_DEVICE_SEALED: &str = "view.device.sealed";

/// A robot's `Attach.supports` asks for its device view without its parts
/// (`DeviceView.parts` empty), except in the answer to its
/// `DeclareParts`: it declared them, takes the host's normalised form once
/// from that answer, and reads nothing of them in a view after. A host
/// listing it sends such a robot each view smaller and cheaper to decode;
/// one that does not sends the parts, which the robot skips unread.
pub const VIEW_DEVICE_LEAN: &str = "view.device.lean";

/// A robot's `Attach.supports` takes its realtime parts over the datagram
/// path its host offered in the handshake (`HandshakeAccept.datagram`): the
/// host lists their routes in its device view, and relays its chunks and
/// samples between those routes and their keys. A robot that does not list
/// it, or a host that offered nothing, keeps every part on the session.
pub const LINK_DATAGRAM: &str = "link.datagram";

/// A host listing it may carry a scene's places in its authority state,
/// and policies that name no robot bind against their view ([`crate::spec::view`]). A
/// binder that lacks it refuses to bind in a
/// session whose state carries places, unless given `robot=`: binding robot
/// by robot there would read the wrong paths.
pub const SCENE_PLACES: &str = "scene.places";

/// A hand-back: a release that names the participant its lease preempted
/// (`ReleaseLease.to`), which takes the parts again
/// (`LeaseEnd.to`). A host listing it checks the hand-back and records
/// it on the lease's end; one that does not would read a plain release, so
/// a holder asks only a host that lists it. A participant's
/// `Attach.supports` says it resumes a part handed back to it; the host
/// refuses a hand-back to one that does not list it, which would stay
/// parked.
pub const LEASE_HANDBACK: &str = "lease.handback";

/// A robot's `Attach.supports` declares that it runs on the lockstep
/// clock: it advances one step per `l/{robot}` request and waits between
/// them. The roster carries it (`ParticipantInfo.supports`), so every
/// participant learns a robot's clock from the authority state, before the
/// robot serves a step. A robot that does not list it is on the wall clock.
/// The host takes no part in it, so it is not a host feature.
pub const CLOCK_LOCKSTEP: &str = "clock.lockstep";

/// Features this build implements as a host: behaviour added in later
/// minors. `auth.sealed`: participants attach through the handshake and
/// seal their control traffic; [`VIEW_DEVICE`]; [`VIEW_DEVICE_SEALED`];
/// [`VIEW_DEVICE_LEAN`]; [`SCENE_PLACES`]; [`LEASE_HANDBACK`].
pub const FEATURES: &[&str] = &[
    "auth.sealed",
    VIEW_DEVICE,
    VIEW_DEVICE_SEALED,
    VIEW_DEVICE_LEAN,
    SCENE_PLACES,
    LEASE_HANDBACK,
];

/// The features in `required` that `supported` lacks (empty: compatible).
pub fn missing_features<'a>(required: &'a [String], supported: &[String]) -> Vec<&'a str> {
    required
        .iter()
        .filter(|feature| !supported.contains(feature))
        .map(String::as_str)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wire_revision_mismatch_names_both_revisions() {
        assert_eq!(
            host_revision_mismatch("the host at 10.0.0.2:7500", WIRE_REVISION),
            None
        );
        assert_eq!(
            host_revision_mismatch("the host at 10.0.0.2:7500", 0).as_deref(),
            Some(
                "the host at 10.0.0.2:7500 speaks session wire revision 0 (four ports: an earlier \
                 0.2 pre-release); this build speaks revision 1 (one port); rebuild both from the \
                 same release"
            )
        );
        // A host admits a participant that sends no revision, and refuses
        // any other.
        assert_eq!(participant_revision_mismatch("eye-3", 0), None);
        assert_eq!(participant_revision_mismatch("eye-3", WIRE_REVISION), None);
        assert_eq!(
            participant_revision_mismatch("eye-3", 2).as_deref(),
            Some(
                "participant eye-3 speaks session wire revision 2 (a later pre-release); this \
                 host speaks revision 1 (one port); rebuild both from the same release"
            )
        );
    }

    #[test]
    fn a_participant_requiring_an_unsupported_feature_is_incompatible() {
        let supported = vec!["frame.fixed".to_string()];
        assert!(missing_features(&["frame.fixed".into()], &supported).is_empty());
        assert_eq!(
            missing_features(&["merge.blend".into(), "frame.fixed".into()], &supported),
            vec!["merge.blend"]
        );
        assert_eq!(
            FEATURES,
            [
                "auth.sealed",
                "view.device",
                "view.device.sealed",
                "view.device.lean",
                "scene.places",
                "lease.handback"
            ]
        );
    }
}
