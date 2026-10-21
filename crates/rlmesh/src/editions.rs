//! Workflow editions: which declaration a participant runs at.
//!
//! A workflow edition is a sticky declaration: a participant states the
//! edition it was authored against and keeps it until its author bumps it.
//! One resolver settles which declaration wins, highest first:
//!
//! 1. the call's own value (its options struct's `workflow_edition`);
//! 2. the `RLMESH_WORKFLOW_EDITION` environment variable;
//! 3. the served peer's option;
//! 4. the type's declaration;
//!
//! then none: the participant floats to this build's newest edition, and an
//! authored participant says so once per process as a
//! [`WorkflowEditionWarning`](crate::WarningCategory::WorkflowEditionWarning).
//! A blank value declares nothing and falls through; a set-but-blank
//! environment variable stops resolution, floats, and does not warn. The
//! winner is validated and trimmed; an edition this build cannot run is the
//! ValueError class. No project file is read.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::warnings::{WarningCategory, warn};
use crate::{Error, Result};

/// The environment variable holding the process-wide declaration.
pub(crate) const WORKFLOW_EDITION_ENV_VAR: &str = "RLMESH_WORKFLOW_EDITION";

/// The bare `YYYY.MM` workflow edition this build runs at: the value to
/// paste into a declaration, the same on every build of that edition.
///
/// A bare base selects whichever spelling of it both sides offer: the sealed
/// name on a release, this build's cohort (what
/// [`BuildInfo::workflow_edition`](crate::BuildInfo::workflow_edition)
/// reports) on a prerelease or local build.
pub fn current_workflow_edition() -> &'static str {
    rlmesh_proto::WORKFLOW_EDITION_BASE
}

/// What an env declaring `declared` offers a handshake: the edition it runs
/// (this build's current when it declares none) and every edition this
/// build runs.
#[cfg(feature = "lockstep")]
pub(crate) fn env_offer(declared: Option<&str>) -> (String, Vec<String>) {
    (
        rlmesh_proto::declared_workflow_edition(declared).to_string(),
        rlmesh_proto::supported_workflow_editions(),
    )
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
    /// Nothing declared one: floats, and an authored participant warns.
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

/// Resolve the call, environment, peer and type declarations, without
/// warning.
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

static WARNED: AtomicBool = AtomicBool::new(false);

/// Resolve a native participant's edition; an `authored` participant that
/// floats warns once per process (the handler's failure is this call's).
pub(crate) fn resolve_native(
    sources: EditionSources<'_>,
    authored: bool,
) -> Result<Option<String>> {
    let resolution = resolve_rungs(sources)?;
    if resolution == EditionResolution::Undeclared
        && authored
        && !WARNED.swap(true, Ordering::SeqCst)
    {
        warn_floating()?;
    }
    Ok(resolution.into_declared())
}

/// The native float warning.
fn warn_floating() -> Result<()> {
    let edition = current_workflow_edition();
    warn(
        WarningCategory::WorkflowEditionWarning,
        format!(
            "no workflow edition declared: floating to '{edition}', which moves when rlmesh \
             upgrades. Pin it with workflow_edition \"{edition}\" on the type or in the call's \
             options; {WORKFLOW_EDITION_ENV_VAR}='' floats deliberately."
        ),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::warnings::{Warning, set_warning_handler, tests::HANDLER_LOCK};

    /// Runs `body` with `RLMESH_WORKFLOW_EDITION` set to `value` (or unset),
    /// serialized with every other test touching the variable or the handler.
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
            // The variable outranks the served peer's option and the type.
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
            // A call value still wins over the deliberate float.
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

    #[test]
    fn an_authored_float_warns_once() {
        let seen = Arc::new(Mutex::new(Vec::<Warning>::new()));
        with_env(None, || {
            WARNED.store(false, Ordering::SeqCst);
            let sink = seen.clone();
            set_warning_handler(Some(Box::new(move |warning: &Warning| {
                sink.lock().unwrap().push(warning.clone());
                Ok(())
            })));
            assert_eq!(
                resolve_native(EditionSources::default(), false).unwrap(),
                None
            );
            assert!(
                seen.lock().unwrap().is_empty(),
                "an unauthored float is quiet"
            );
            assert_eq!(
                resolve_native(EditionSources::default(), true).unwrap(),
                None
            );
            assert_eq!(
                resolve_native(EditionSources::default(), true).unwrap(),
                None
            );
            set_warning_handler(None);
        });
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].category, WarningCategory::WorkflowEditionWarning);
        let edition = current_workflow_edition();
        assert_eq!(
            seen[0].message,
            format!(
                "no workflow edition declared: floating to '{edition}', which moves when rlmesh \
                 upgrades. Pin it with workflow_edition \"{edition}\" on the type or in the \
                 call's options; RLMESH_WORKFLOW_EDITION='' floats deliberately."
            )
        );
    }
}
