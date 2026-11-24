//! Workflow editions: which declaration a participant runs at.
//!
//! A participant states the edition it was authored against and keeps it until its author bumps
//! it. The first declaration found wins:
//!
//! 1. the call's own value (its options struct's `workflow_edition`);
//! 2. the `RLMESH_WORKFLOW_EDITION` environment variable;
//! 3. the served peer's option;
//! 4. the type's declaration;
//!
//! then none: the participant floats to this build's newest edition, and a describe envelope marks
//! the float (`workflow_edition_floats`). A blank value declares nothing and falls through; a
//! set-but-blank environment variable stops resolution and floats deliberately. The winner is
//! trimmed and validated; an edition this build cannot run is the ValueError class. No project
//! file is read.

use crate::{Error, Result};

pub(crate) const WORKFLOW_EDITION_ENV_VAR: &str = "RLMESH_WORKFLOW_EDITION";

/// The bare `YYYY.MM` workflow edition this build runs at: the value to
/// paste into a declaration, the same on every build of that edition.
///
/// A bare base selects whichever spelling of it both sides offer: the sealed name on a release,
/// this build's cohort
/// ([`BuildInfo::workflow_edition`](crate::BuildInfo::workflow_edition)) on a prerelease or local
/// build.
pub fn current_workflow_edition() -> &'static str {
    rlmesh_proto::WORKFLOW_EDITION_BASE
}

pub(crate) fn offer(declared: Option<&str>) -> (String, Vec<String>) {
    (
        rlmesh_proto::declared_workflow_edition(declared).to_string(),
        rlmesh_proto::supported_workflow_editions(),
    )
}

pub(crate) fn negotiate_env(
    info: &crate::lockstep::GymEnvInfo,
    declared: Option<&str>,
) -> std::result::Result<(rlmesh_proto::SessionOffer, String, rlmesh_proto::Edition), String> {
    let env = rlmesh_proto::SessionOffer {
        editions: info.supported_workflow_editions.clone(),
        preferred: Some(info.workflow_edition.trim().to_string()).filter(|w| !w.is_empty()),
    };
    let selected = rlmesh_proto::negotiate_workflow_edition(
        &env,
        &rlmesh_proto::SessionOffer::this_build(declared),
    )
    .map_err(|refusal| {
        format!(
            "no mutual workflow edition with the env: {refusal}{}",
            refusal.hint()
        )
    })?;
    let base = rlmesh_proto::parse_retained_edition(&selected)?;
    Ok((env, selected, base))
}

/// The declarations one participant could resolve from (rungs 1, 3 and 4).
#[derive(Clone, Copy, Debug, Default)]
pub struct EditionSources<'a> {
    /// The call's own value.
    pub call: Option<&'a str>,
    /// The served peer's option.
    pub option: Option<&'a str>,
    /// The type's declaration.
    pub declared: Option<&'a str>,
}

/// What the resolver settled on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EditionResolution {
    /// A declaration won, validated and trimmed.
    Declared(String),
    /// A set-but-blank `RLMESH_WORKFLOW_EDITION`: floats deliberately.
    FloatsDeliberately,
    /// Nothing declared one: floats.
    Undeclared,
}

impl EditionResolution {
    /// The declared edition, if any.
    pub fn into_declared(self) -> Option<String> {
        match self {
            EditionResolution::Declared(edition) => Some(edition),
            _ => None,
        }
    }
}

/// Refuse an edition this build cannot run a session at, with the ValueError
/// class naming it and the editions this build offers; return it trimmed.
pub fn validate_workflow_edition(edition: &str) -> Result<String> {
    rlmesh_proto::parse_declared_edition(edition)
        .map(|_| edition.trim().to_string())
        .map_err(Error::invalid_argument)
}

fn nonblank(value: Option<&str>) -> Option<&str> {
    value.filter(|value| !value.trim().is_empty())
}

/// Resolve the call, environment, peer and type declarations.
pub fn resolve_rungs(sources: EditionSources<'_>) -> Result<EditionResolution> {
    if let Some(call) = nonblank(sources.call) {
        return validate_workflow_edition(call).map(EditionResolution::Declared);
    }
    let environment = std::env::var(WORKFLOW_EDITION_ENV_VAR).ok();
    if environment
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Ok(EditionResolution::FloatsDeliberately);
    }
    for candidate in [environment.as_deref(), sources.option, sources.declared] {
        if let Some(candidate) = nonblank(candidate) {
            return validate_workflow_edition(candidate).map(EditionResolution::Declared);
        }
    }
    Ok(EditionResolution::Undeclared)
}

pub(crate) fn resolve_native(sources: EditionSources<'_>) -> Result<Option<String>> {
    resolve_rungs(sources).map(EditionResolution::into_declared)
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

    fn sources<'a>(
        call: Option<&'a str>,
        option: Option<&'a str>,
        declared: Option<&'a str>,
    ) -> EditionSources<'a> {
        EditionSources {
            call,
            option,
            declared,
        }
    }

    #[test]
    fn rungs_resolve_highest_first() {
        let base = current_workflow_edition();
        let padded = format!("  {base}  ");
        with_env(None, || {
            let declared = |s| resolve_rungs(s).unwrap();
            assert_eq!(
                declared(sources(Some(&padded), Some("bogus"), Some("bogus"))),
                EditionResolution::Declared(base.to_string())
            );
            assert_eq!(
                declared(sources(Some("  "), Some(base), Some("bogus"))),
                EditionResolution::Declared(base.to_string())
            );
            assert_eq!(
                declared(sources(None, None, Some(base))),
                EditionResolution::Declared(base.to_string())
            );
            assert_eq!(
                declared(sources(None, Some(""), None)),
                EditionResolution::Undeclared
            );
        });
        with_env(Some(base), || {
            assert_eq!(
                resolve_rungs(sources(None, Some("bogus"), Some("bogus"))).unwrap(),
                EditionResolution::Declared(base.to_string())
            );
        });
        with_env(Some(" "), || {
            assert_eq!(
                resolve_rungs(sources(None, Some(base), Some(base))).unwrap(),
                EditionResolution::FloatsDeliberately
            );
            assert_eq!(
                resolve_rungs(sources(Some(base), None, None)).unwrap(),
                EditionResolution::Declared(base.to_string())
            );
        });
    }

    #[test]
    fn a_refused_edition_is_a_value_error() {
        with_env(None, || {
            let err = resolve_rungs(sources(None, None, Some("1999.01"))).unwrap_err();
            assert_eq!(err.class(), crate::ErrorClass::ValueError);
        });
    }
}
