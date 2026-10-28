//! Protocol identity and the compatibility rule (§10.7).
//!
//! | Layer | Identifier | Rule |
//! |---|---|---|
//! | Generation | [`GENERATION`]: the key prefix `rlmesh/v2` and the protobuf package `rlmesh.session.v2` | A breaking change is a new generation on new keys; generations never share keys, so they never see each other's traffic. |
//! | Schema | protobuf | Additive only. Unknown fields are ignored; unknown enum values are refused where safety depends on them (`Mode`, `PriorityClass`, `EdgeMode`, dtypes) and ignored elsewhere. |
//! | Protocol minor | [`PROTOCOL_MINOR`], announced in `Attach` and `AuthorityState` | Monotonic, 0 at v0.2. Any minor talks to any other. New behaviour is a **feature string** (`"merge.blend"`). |
//! | Features | `Attach.supports/requires`, `AuthorityState.supports`, `PartSpec.requires` | Attach fails with `CtlStatus::Incompatible` if the host lacks a feature the participant requires. |
//! | Kind table | [`SPEC_VERSION`], in `DeclareParts` | New kinds and optional settings are additive. |

#[allow(unused_imports)]
use crate::prelude::*;

/// The protocol generation: the `v2` of the key prefix.
pub const GENERATION: u32 = 2;

/// This build's protocol minor (0 at v0.2).
pub const PROTOCOL_MINOR: u32 = 0;

/// The kind table version a declaration is written against.
pub const SPEC_VERSION: u32 = 1;

/// A robot's `Attach.supports` asks for its device view, which a host
/// listing it publishes on `host/view/{robot}` (scene design §3.4).
pub const VIEW_DEVICE: &str = "view.device";

/// Features this build implements as a host: behaviour added in later
/// minors. `auth.sealed`: participants attach through the handshake and
/// seal their control traffic (design §8.8); [`VIEW_DEVICE`].
pub const FEATURES: &[&str] = &["auth.sealed", VIEW_DEVICE];

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
        assert_eq!(FEATURES, ["auth.sealed", "view.device"]);
    }
}
