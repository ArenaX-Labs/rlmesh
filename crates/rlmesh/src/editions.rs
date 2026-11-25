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
    let environment = std::env::var(WORKFLOW_EDITION_ENV_VAR).ok();
    let edition = nonblank(environment.as_deref())
        .or(nonblank(declared))
        .unwrap_or(rlmesh_proto::CURRENT_WORKFLOW_EDITION);
    validate_workflow_edition(edition)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::warnings::tests::HANDLER_LOCK;

    #[allow(unsafe_code)]
    fn with_env(value: Option<&str>, body: impl FnOnce()) {
        let _guard = HANDLER_LOCK.lock().unwrap_or_else(|err| err.into_inner());
        let previous = std::env::var(WORKFLOW_EDITION_ENV_VAR).ok();
        // SAFETY: serialized by HANDLER_LOCK; no other test thread reads it.
        unsafe {
            match value {
                Some(value) => std::env::set_var(WORKFLOW_EDITION_ENV_VAR, value),
                None => std::env::remove_var(WORKFLOW_EDITION_ENV_VAR),
            }
        }
        body();
        unsafe {
            match previous {
                Some(value) => std::env::set_var(WORKFLOW_EDITION_ENV_VAR, value),
                None => std::env::remove_var(WORKFLOW_EDITION_ENV_VAR),
            }
        }
    }

    #[test]
    fn the_environment_then_the_declaration_then_the_current_edition() {
        let current = current_workflow_edition();
        with_env(None, || {
            assert_eq!(resolve_workflow_edition(None).unwrap(), current);
            assert_eq!(resolve_workflow_edition(Some("  ")).unwrap(), current);
            assert_eq!(
                resolve_workflow_edition(Some(&format!(" {current} "))).unwrap(),
                current
            );
        });
        with_env(Some(current), || {
            assert_eq!(resolve_workflow_edition(Some("2099.01")).unwrap(), current);
        });
        with_env(Some(" "), || {
            assert_eq!(resolve_workflow_edition(Some(current)).unwrap(), current);
        });
        with_env(Some("2099.01"), || {
            let err = resolve_workflow_edition(Some(current)).unwrap_err();
            assert!(err.to_string().contains("\"2099.01\""), "{err}");
        });
    }

    #[test]
    fn an_unknown_edition_is_a_value_error() {
        with_env(None, || {
            let err = resolve_workflow_edition(Some("2099.01")).unwrap_err();
            assert_eq!(err.class(), crate::ErrorClass::ValueError);
            assert!(
                err.to_string().contains("upgrade the rlmesh runtime"),
                "{err}"
            );
        });
    }
}
