use std::fmt;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::SandboxError;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EnvironmentSourceRef {
    Gym(GymSourceRef),
    Hf(HfSourceRef),
    Recipe(RecipeSourceRef),
}

/// Where a recipe document came from, and therefore how much its build phase is
/// trusted. `Installed` is set only when the Python registry resolved the name
/// from a local `register()` or an installed `rlmesh.recipes` entry point (the
/// pip-install-is-consent path); `Remote` is set for any document handed in as
/// data from an untrusted source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecipeProvenance {
    Installed,
    Remote,
}

/// A recipe handed to the sandbox already-structured (the Python registry
/// resolves a name to a recipe before `sandbox_start_env`; the document is the
/// recipe's canonical JSON).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecipeSourceRef {
    pub name: String,
    pub document: serde_json::Value,
    pub provenance: RecipeProvenance,
}

impl EnvironmentSourceRef {
    /// Parse a sandbox source reference (`gym://...`, `hf://...`, or a bare
    /// gymnasium env id).
    pub fn parse(value: &str) -> std::result::Result<Self, SandboxError> {
        Self::parse_inner(value).map_err(SandboxError::invalid_source)
    }

    fn parse_inner(value: &str) -> Result<Self> {
        let value = value.trim();
        if value.is_empty() {
            bail!("sandbox source must not be empty");
        }

        if let Some(rest) = value.strip_prefix("gym://") {
            return Self::parse_gym(rest);
        }

        if let Some(rest) = value.strip_prefix("hf://") {
            return Ok(Self::Hf(HfSourceRef::parse(rest)?));
        }

        if value.contains("://") {
            bail!("unsupported sandbox source '{value}'");
        }

        Self::parse_gym(value)
    }

    fn parse_gym(env_id: &str) -> Result<Self> {
        let env_id = env_id.trim();
        if env_id.is_empty() {
            bail!("gym source must include an environment id");
        }
        Ok(Self::Gym(GymSourceRef {
            env_id: env_id.to_string(),
        }))
    }

    pub fn slug(&self) -> String {
        match self {
            Self::Gym(source) => sanitize_slug(&source.env_id),
            Self::Hf(source) => {
                let mut value = source.repo.replace('/', "-");
                if let Some(suite) = &source.suite {
                    value.push('-');
                    value.push_str(suite);
                }
                if let Some(task) = &source.task {
                    value.push('-');
                    value.push_str(task);
                }
                sanitize_slug(&value)
            }
            Self::Recipe(source) => sanitize_slug(&source.name),
        }
    }
}

impl fmt::Display for EnvironmentSourceRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gym(source) => write!(f, "gym://{}", source.env_id),
            Self::Hf(source) => {
                write!(f, "hf://{}", source.repo)?;
                if let Some(revision) = &source.revision {
                    write!(f, "@{revision}")?;
                }
                if let Some(suite) = &source.suite {
                    write!(f, ":{suite}")?;
                }
                if let Some(task) = &source.task {
                    write!(f, "/{task}")?;
                }
                Ok(())
            }
            Self::Recipe(source) => write!(f, "recipe://{}", source.name),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GymSourceRef {
    pub env_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HfSourceRef {
    pub repo: String,
    pub revision: Option<String>,
    pub suite: Option<String>,
    pub task: Option<String>,
}

impl HfSourceRef {
    fn parse(value: &str) -> Result<Self> {
        let value = value.trim();
        if value.is_empty() {
            bail!("hugging face source must include org/repo");
        }

        let (repo_and_revision, suite, task) = match value.rsplit_once(':') {
            Some((left, right)) if !left.is_empty() && !right.is_empty() => {
                let (suite, task) = parse_selector(right)?;
                (left, Some(suite), task)
            }
            _ => (value, None, None),
        };

        let (repo, revision) = match repo_and_revision.rsplit_once('@') {
            Some((left, right)) if !left.is_empty() && !right.is_empty() => {
                (left, Some(validate_ref_part("revision", right)?))
            }
            Some(_) => bail!("hugging face revision must look like @revision"),
            None => (repo_and_revision, None),
        };

        validate_hf_repo(repo)?;

        Ok(Self {
            repo: repo.to_string(),
            revision,
            suite,
            task,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum ResolvedEnvironmentSourceRef {
    Gym(GymSourceRef),
    Hf(ResolvedHfSourceRef),
    Recipe(RecipeSourceRef),
}

impl ResolvedEnvironmentSourceRef {
    pub(crate) fn slug(&self) -> String {
        match self {
            Self::Gym(source) => sanitize_slug(&source.env_id),
            Self::Hf(source) => {
                let mut value = source.repo.replace('/', "-");
                if let Some(suite) = &source.suite {
                    value.push('-');
                    value.push_str(suite);
                }
                if let Some(task) = &source.task {
                    value.push('-');
                    value.push_str(task);
                }
                sanitize_slug(&value)
            }
            Self::Recipe(source) => sanitize_slug(&source.name),
        }
    }
}

impl fmt::Display for ResolvedEnvironmentSourceRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gym(source) => write!(f, "gym://{}", source.env_id),
            Self::Hf(source) => {
                write!(f, "hf://{}@{}", source.repo, source.resolved_revision)?;
                if let Some(suite) = &source.suite {
                    write!(f, ":{suite}")?;
                }
                if let Some(task) = &source.task {
                    write!(f, "/{task}")?;
                }
                Ok(())
            }
            Self::Recipe(source) => write!(f, "recipe://{}", source.name),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ResolvedHfSourceRef {
    pub repo: String,
    pub resolved_revision: String,
    pub suite: Option<String>,
    pub task: Option<String>,
}

pub fn sanitize_slug(value: &str) -> String {
    let mut slug = String::new();
    let mut prev_dash = false;
    for ch in value.chars() {
        let next = match ch {
            'a'..='z' | '0'..='9' => ch,
            'A'..='Z' => ch.to_ascii_lowercase(),
            _ => '-',
        };

        if next == '-' {
            if prev_dash {
                continue;
            }
            prev_dash = true;
            slug.push(next);
        } else {
            prev_dash = false;
            slug.push(next);
        }
    }

    let slug = slug.trim_matches('-');
    if slug.is_empty() {
        "env".to_string()
    } else {
        slug.to_string()
    }
}

fn validate_hf_repo(repo: &str) -> Result<()> {
    let mut parts = repo.split('/');
    let Some(owner) = parts.next() else {
        bail!("hugging face sources must look like hf://org/repo[@revision][:suite[/task]]");
    };
    let Some(name) = parts.next() else {
        bail!("hugging face sources must look like hf://org/repo[@revision][:suite[/task]]");
    };
    if parts.next().is_some() || owner.is_empty() || name.is_empty() {
        bail!("hugging face sources must look like hf://org/repo[@revision][:suite[/task]]");
    }
    validate_hf_repo_part("owner", owner)?;
    validate_hf_repo_part("repo", name)?;
    Ok(())
}

fn parse_selector(value: &str) -> Result<(String, Option<String>)> {
    let (suite, task) = match value.split_once('/') {
        Some((suite, task)) if !suite.is_empty() && !task.is_empty() && !task.contains('/') => (
            validate_ref_part("suite", suite)?,
            Some(validate_ref_part("task", task)?),
        ),
        Some(_) => bail!("hugging face selector must look like :suite or :suite/task"),
        None => (validate_ref_part("suite", value)?, None),
    };
    Ok((suite, task))
}

fn validate_hf_repo_part(label: &str, value: &str) -> Result<()> {
    validate_ref_part(label, value)?;
    if value.starts_with(['-', '.']) || value.ends_with(['-', '.']) {
        bail!("hugging face {label} must not start or end with '-' or '.'");
    }
    if value.contains("--") || value.contains("..") {
        bail!("hugging face {label} must not contain '--' or '..'");
    }
    if !value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
    {
        bail!("hugging face {label} may only contain ASCII letters, digits, '-', '_', and '.'");
    }
    Ok(())
}

fn validate_ref_part(label: &str, value: &str) -> Result<String> {
    let value = value.trim();
    if value.is_empty() {
        bail!("{label} must not be empty");
    }
    if value.contains(char::is_whitespace) {
        bail!("{label} must not contain whitespace");
    }
    // Reject leading '-' so the value can never be reparsed as a CLI option
    // when it is later handed to git (e.g. as a revision passed to ls-remote).
    if value.starts_with('-') {
        bail!("{label} must not start with '-'");
    }
    Ok(value.to_string())
}

#[cfg(test)]
mod tests {
    use super::{EnvironmentSourceRef, HfSourceRef, sanitize_slug};

    #[test]
    fn parses_plain_gym_sources() {
        let source = EnvironmentSourceRef::parse("CartPole-v1").unwrap();
        match source {
            EnvironmentSourceRef::Gym(source) => assert_eq!(source.env_id, "CartPole-v1"),
            _ => panic!("expected gym"),
        }
    }

    #[test]
    fn parses_gym_scheme_sources() {
        let source = EnvironmentSourceRef::parse("gym://CartPole-v1").unwrap();
        assert_eq!(source.to_string(), "gym://CartPole-v1");
    }

    #[test]
    fn parses_hf_sources() {
        let source = HfSourceRef::parse("org/repo@main:suite_1").unwrap();
        assert_eq!(source.repo, "org/repo");
        assert_eq!(source.revision.as_deref(), Some("main"));
        assert_eq!(source.suite.as_deref(), Some("suite_1"));
        assert_eq!(source.task, None);
    }

    #[test]
    fn parses_hf_sources_with_suite_and_task() {
        let source = HfSourceRef::parse("org/repo@main:suite_1/0").unwrap();
        assert_eq!(source.repo, "org/repo");
        assert_eq!(source.revision.as_deref(), Some("main"));
        assert_eq!(source.suite.as_deref(), Some("suite_1"));
        assert_eq!(source.task.as_deref(), Some("0"));
    }

    #[test]
    fn parses_hf_source_refs() {
        let source = EnvironmentSourceRef::parse("hf://org/repo").unwrap();
        assert_eq!(source.to_string(), "hf://org/repo");

        let source = EnvironmentSourceRef::parse("hf://org/repo@main:suite_1/0").unwrap();
        assert_eq!(source.to_string(), "hf://org/repo@main:suite_1/0");
    }

    #[test]
    fn hf_slug_includes_suite_and_task() {
        let source = EnvironmentSourceRef::parse("hf://org/repo@main:suite_1/0").unwrap();
        assert_eq!(source.slug(), "org-repo-suite-1-0");
    }

    #[test]
    fn rejects_malformed_hf_selectors() {
        let err = EnvironmentSourceRef::parse("hf://org/repo@main:suite/").unwrap_err();
        assert!(err.to_string().contains(":suite/task"));

        let err = EnvironmentSourceRef::parse("hf://org/repo@main:suite/task/extra").unwrap_err();
        assert!(err.to_string().contains(":suite/task"));
    }

    #[test]
    fn rejects_invalid_hf_sources() {
        let err = EnvironmentSourceRef::parse("hf://org").unwrap_err();
        assert!(err.to_string().contains("hf://org/repo"));
    }

    #[test]
    fn rejects_suspicious_hf_repo_parts() {
        let err = EnvironmentSourceRef::parse("hf://org/repo?x=1").unwrap_err();
        assert!(err.to_string().contains("may only contain ASCII"));

        let err = EnvironmentSourceRef::parse("hf://org/..repo").unwrap_err();
        assert!(err.to_string().contains("must not start or end"));
    }

    #[test]
    fn slug_sanitizes_input() {
        assert_eq!(sanitize_slug("sai_mujoco:Franka"), "sai-mujoco-franka");
    }
}
