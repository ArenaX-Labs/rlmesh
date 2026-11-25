//! Workflow editions: the row of behavioural defaults a participant runs under.
//!
//! An edition is declared locally and never negotiated. A participant's edition is, highest first:
//!
//! 1. the `RLMESH_WORKFLOW_EDITION` environment variable;
//! 2. its in-source declaration ([`Policy::workflow_edition`](crate::Policy::workflow_edition),
//!    [`EnvFactory::workflow_edition`](crate::EnvFactory::workflow_edition));
//! 3. this build's [`current_workflow_edition`].
//!
//! A blank value declares nothing. An edition this build does not know is the ValueError class.

use crate::{Error, Result};

pub(crate) const WORKFLOW_EDITION_ENV_VAR: &str = "RLMESH_WORKFLOW_EDITION";

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

fn nonblank(value: Option<&str>) -> Option<&str> {
    value.filter(|value| !value.trim().is_empty())
}

/// The edition a participant runs at: the environment variable, then its
/// in-source declaration `declared`, then [`current_workflow_edition`].
///
/// # Errors
/// [`Error::InvalidArgument`] for an edition this build does not know.
pub fn resolve_workflow_edition(declared: Option<&str>) -> Result<String> {
    resolve_served(None, declared)
}

/// [`resolve_workflow_edition`] for a served env or model: a non-blank
/// `ServeOptions.workflow_edition` `option` stands above the type's `declared`.
pub(crate) fn resolve_served(option: Option<&str>, declared: Option<&str>) -> Result<String> {
    let environment = std::env::var(WORKFLOW_EDITION_ENV_VAR).ok();
    resolve_in(environment.as_deref(), &[option, declared])
}

fn resolve_in(environment: Option<&str>, declarations: &[Option<&str>]) -> Result<String> {
    let edition = std::iter::once(environment)
        .chain(declarations.iter().copied())
        .find_map(nonblank)
        .unwrap_or(rlmesh_proto::CURRENT_WORKFLOW_EDITION);
    validate_workflow_edition(edition)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_environment_then_the_option_then_the_declaration_then_the_current_edition() {
        let current = current_workflow_edition();
        let resolve = |environment, option, declared| {
            resolve_in(environment, &[option, declared]).map_err(|err| err.to_string())
        };
        let unknown = Some("2099.01");
        for (environment, option, declared) in [
            (None, None, None),
            (None, Some("  "), None),
            (None, None, Some(" 2026.10 ")),
            (None, Some(" "), Some("2026.10")),
            (Some(current), unknown, unknown),
            (Some(" "), Some(current), unknown),
            (None, Some(current), unknown),
        ] {
            assert_eq!(
                resolve(environment, option, declared).as_deref(),
                Ok(current),
                "{environment:?} {option:?} {declared:?}"
            );
        }
        for (environment, option, declared) in [
            (unknown, Some(current), Some(current)),
            (None, unknown, Some(current)),
            (None, Some(" "), unknown),
        ] {
            let err = resolve(environment, option, declared).unwrap_err();
            assert!(err.contains("\"2099.01\""), "{err}");
        }
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
