//! Workflow editions: the row of behavioural defaults a participant runs under.
//!
//! An edition is declared in source and never negotiated or overridden at deploy time. A
//! participant runs at its type's declaration
//! ([`Policy::workflow_edition`](crate::Policy::workflow_edition),
//! [`EnvFactory::workflow_edition`](crate::EnvFactory::workflow_edition)), else this build's
//! [`current_workflow_edition`]. A plain env declares nothing.
//!
//! A blank value declares nothing. An edition this build does not know is the ValueError class.

use crate::{Error, Result};

/// The bare `YYYY.MM` workflow edition this build runs at when nothing declares
/// one: the value to paste into a declaration.
pub fn current_workflow_edition() -> &'static str {
    rlmesh_proto::CURRENT_WORKFLOW_EDITION
}

/// Refuse an edition this build does not know, naming the editions it knows
/// and the remedy; return it trimmed.
///
/// # Errors
/// [`Error::InvalidArgument`] for an edition this build does not know.
pub fn validate_workflow_edition(edition: &str) -> Result<String> {
    rlmesh_proto::Edition::parse(edition)
        .map(|edition| edition.name().to_string())
        .map_err(|unknown| Error::invalid_argument(unknown.to_string()))
}

/// The edition a participant runs at: its in-source declaration `declared`,
/// else [`current_workflow_edition`]. A blank declaration declares nothing.
///
/// # Errors
/// [`Error::InvalidArgument`] for an edition this build does not know.
pub fn resolve_workflow_edition(declared: Option<&str>) -> Result<String> {
    validate_workflow_edition(
        declared
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(rlmesh_proto::CURRENT_WORKFLOW_EDITION),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_declaration_else_the_current_edition() {
        let current = current_workflow_edition();
        for declared in [None, Some("  "), Some(" 2026.10 ")] {
            assert_eq!(resolve_workflow_edition(declared).unwrap(), current);
        }
        let err = resolve_workflow_edition(Some("2099.01")).unwrap_err();
        assert!(err.to_string().contains("\"2099.01\""), "{err}");
    }

    #[test]
    fn an_unknown_edition_is_a_value_error() {
        let err = validate_workflow_edition("2099.01").unwrap_err();
        assert_eq!(err.class(), crate::ErrorClass::ValueError);
        assert!(
            err.to_string().contains("upgrade the rlmesh runtime"),
            "{err}"
        );
    }
}
