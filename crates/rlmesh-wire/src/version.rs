//! Protocol identity and the compatibility rule.
//!
//! | Layer | Identifier | Rule |
//! |---|---|---|
//! | Generation | [`GENERATION`]: the key prefix `rlmesh/v2` and the protobuf package `rlmesh.session.v2` | A breaking change is a new generation on new keys; generations never share keys, so they never see each other's traffic. |
//! | Protocol minor | [`PROTOCOL_MINOR`], announced in `Attach` and `AuthorityState` | Monotonic, additive only. Any minor talks to any other. A feature added in a minor is a **capability** string; the host lists the ones the session supports, and a participant that requires one the host lacks is refused at attach (`CtlStatus::Incompatible`). |
//! | Control schema | protobuf | New fields, verbs and enum values only; never renumber or retype. Unknown fields are ignored; unknown enum values are refused where safety depends on them. |
//! | Tensor payloads | the channel declaration's layout | A sample's leaves are raw tensors in the layout declared once per channel; a new encoding is a capability. |

/// The protocol generation: the `v2` of the key prefix.
pub const GENERATION: u32 = 2;

/// This build's protocol minor (0 at v0.2).
pub const PROTOCOL_MINOR: u32 = 0;

/// Capabilities this build supports as a host: features added in later
/// minors. Empty at minor 0.
pub const CAPABILITIES: &[&str] = &[];

/// The capabilities in `required` that `supported` lacks (empty: compatible).
pub fn missing_capabilities<'a>(required: &'a [String], supported: &[String]) -> Vec<&'a str> {
    required
        .iter()
        .filter(|cap| !supported.contains(cap))
        .map(String::as_str)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_participant_requiring_an_unsupported_capability_is_incompatible() {
        let supported = vec!["frame.batch".to_string()];
        assert!(missing_capabilities(&["frame.batch".into()], &supported).is_empty());
        assert_eq!(
            missing_capabilities(&["merge.blend".into(), "frame.batch".into()], &supported),
            vec!["merge.blend"]
        );
        assert!(CAPABILITIES.is_empty(), "minor 0 adds no capability");
    }
}
