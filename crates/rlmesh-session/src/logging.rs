//! What a tracing subscriber must not record.
//!
//! Zenoh logs its whole config at `DEBUG` when a session opens (target
//! `zenoh::api::session`), and a gated host's config, and every config that
//! links to one, holds the link password a session's key derives in plain
//! text: whoever reads that log can link to the host. Cap that target at
//! `INFO` in every subscriber, whatever level the user asked for. With
//! `tracing-subscriber`, add [`may_record`] as a global filter layer:
//!
//! ```no_run
//! use tracing_subscriber::prelude::*;
//! tracing_subscriber::registry()
//!     .with(tracing_subscriber::EnvFilter::from_default_env())
//!     .with(tracing_subscriber::filter::filter_fn(rlmesh_session::logging::may_record))
//!     .with(tracing_subscriber::fmt::layer())
//!     .init();
//! ```
//!
//! A directive such as `zenoh::api::session=info` alone is not enough: a
//! span directive (`[span]=trace`) enables events of any target inside it.

use tracing::{Level, Metadata};

/// The targets (with their submodules) that may log a secret below `INFO`:
/// Zenoh's session open (its config) and scouting (the same config).
pub const CAPPED_TARGETS: &[&str] = &["zenoh::api::session", "zenoh::api::scouting"];

/// Whether a subscriber may record `metadata`'s event or span: anything at
/// `INFO` or above, and below it anything outside [`CAPPED_TARGETS`].
#[must_use]
pub fn may_record(metadata: &Metadata<'_>) -> bool {
    *metadata.level() <= Level::INFO || !is_capped(metadata.target())
}

fn is_capped(target: &str) -> bool {
    CAPPED_TARGETS.iter().any(|capped| {
        target
            .strip_prefix(capped)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with("::"))
    })
}

#[cfg(test)]
mod tests {
    use super::is_capped;

    #[test]
    fn caps_the_config_targets_and_their_submodules_only() {
        for target in ["zenoh::api::session", "zenoh::api::scouting::x"] {
            assert!(is_capped(target), "{target}");
        }
        for target in ["zenoh::api::sessions", "zenoh::net", "rlmesh_session"] {
            assert!(!is_capped(target), "{target}");
        }
    }
}
