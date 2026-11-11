//! Protocol identity and the compatibility rule.
//!
//! | Layer | Identifier | Rule |
//! |---|---|---|
//! | Generation | [`GENERATION`]: the key prefix `rlmesh/v2` and the protobuf package `rlmesh.session.v2` | A breaking change is a new generation on new keys; generations never share keys, so they never see each other's traffic. |
//! | Schema | protobuf | Additive only. Unknown fields are ignored; unknown enum values are refused where safety depends on them (`Mode`, `PriorityClass`, `EdgeMode`, dtypes) and ignored elsewhere. |
//! | Protocol minor | [`PROTOCOL_MINOR`], announced in `Attach` and `AuthorityState` | Monotonic, 0 at v0.2. Any minor talks to any other. New behaviour is a **feature string** (`"merge.blend"`). |
//! | Features | `Attach.supports/requires`, `AuthorityState.supports`, `PartSpec.requires` | Attach fails with `CtlStatus::Incompatible` if the host lacks a feature the participant requires. |
//! | Kind table | [`SPEC_VERSION`], in `DeclareParts` | New kinds and optional settings are additive. A host accepts a robot on any table from 1 up to its own, and refuses a newer one. |

#[allow(unused_imports)]
use crate::prelude::*;

/// The protocol generation: the `v2` of the key prefix.
pub const GENERATION: u32 = 2;

/// This build's protocol minor (0 at v0.2).
pub const PROTOCOL_MINOR: u32 = 0;

/// The kind table version a declaration is written against.
pub const SPEC_VERSION: u32 = 1;

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

/// Features this build implements as a host: behaviour added in later
/// minors. `auth.sealed`: participants attach through the handshake and
/// seal their control traffic; [`VIEW_DEVICE`]; [`VIEW_DEVICE_SEALED`];
/// [`VIEW_DEVICE_LEAN`]; [`SCENE_PLACES`].
pub const FEATURES: &[&str] = &[
    "auth.sealed",
    VIEW_DEVICE,
    VIEW_DEVICE_SEALED,
    VIEW_DEVICE_LEAN,
    SCENE_PLACES,
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
                "scene.places"
            ]
        );
    }
}
