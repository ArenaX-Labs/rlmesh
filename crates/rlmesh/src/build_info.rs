//! Build identity and release information.

use std::fmt;

/// This crate's version, in SemVer (`0.1.0-rc.12`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Where a build's cohort came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BuildSource {
    /// `"release"`: a release build.
    Release,
    /// `"package"`: a published crate, or a checkout without git.
    Package,
    /// `"git"`: a source build stamped from its commit.
    Git,
}

impl BuildSource {
    /// The canonical spelling (`"release"`, `"package"`, `"git"`).
    pub fn as_str(self) -> &'static str {
        match self {
            BuildSource::Release => "release",
            BuildSource::Package => "package",
            BuildSource::Git => "git",
        }
    }

    fn of(source: &str) -> BuildSource {
        match source {
            "release" => BuildSource::Release,
            "git" => BuildSource::Git,
            _ => BuildSource::Package,
        }
    }
}

impl fmt::Display for BuildSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The identity of this build. Every field is fixed at compile time.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct BuildInfo {
    /// The package version the build was made as ([`VERSION`]).
    pub version: &'static str,
    /// The wire protocol generation this build speaks (`rlmesh-wire-v1`).
    pub protocol_generation: &'static str,
    /// The exact workflow edition spelling this build advertises: the sealed
    /// `YYYY.MM` base on a release, `YYYY.MM-<cohort>` on a prerelease or
    /// source build.
    pub workflow_edition: &'static str,
    /// The sealed `YYYY.MM` base of `workflow_edition`
    /// ([`current_workflow_edition`](crate::current_workflow_edition)).
    pub workflow_edition_base: &'static str,
    /// Every workflow edition this build can drive a session at, newest
    /// first.
    pub supported_workflow_editions: &'static [&'static str],
    /// The cohort that spells `workflow_edition`: `stable` on a sealed
    /// release, the prerelease version on a prerelease, `dev.<git>` on a
    /// source build.
    pub build_cohort: &'static str,
    /// Where the cohort came from.
    pub build_source: BuildSource,
    /// The git state a source build was stamped from (the short sha,
    /// suffixed `.dirty.<fingerprint>` for uncommitted changes), or `None`.
    pub git: Option<&'static str>,
}

/// This build's identity.
pub fn build_info() -> BuildInfo {
    BuildInfo {
        version: VERSION,
        protocol_generation: rlmesh_proto::PROTOCOL_GENERATION,
        workflow_edition: rlmesh_proto::CURRENT_WORKFLOW_EDITION,
        workflow_edition_base: rlmesh_proto::WORKFLOW_EDITION_BASE,
        supported_workflow_editions: rlmesh_proto::SUPPORTED_WORKFLOW_EDITIONS,
        build_cohort: rlmesh_proto::BUILD_COHORT,
        build_source: BuildSource::of(rlmesh_proto::BUILD_SOURCE),
        git: rlmesh_proto::BUILD_COHORT.strip_prefix("dev."),
    }
}

/// Text representation: `BuildInfo(version="0.1.0", ..., git=None)`.
impl fmt::Display for BuildInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BuildInfo(version={:?}, protocol_generation={:?}, workflow_edition={:?}, \
             workflow_edition_base={:?}, build_cohort={:?}, build_source={:?}, git=",
            self.version,
            self.protocol_generation,
            self.workflow_edition,
            self.workflow_edition_base,
            self.build_cohort,
            self.build_source.as_str(),
        )?;
        match self.git {
            Some(git) => write!(f, "{git:?})"),
            None => f.write_str("None)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_follow_the_build() {
        let info = build_info();
        assert_eq!(info.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(info.protocol_generation, "rlmesh-wire-v1");
        assert_eq!(
            info.workflow_edition_base,
            crate::current_workflow_edition()
        );
        assert_eq!(info.supported_workflow_editions[0], info.workflow_edition);
        assert!(
            info.workflow_edition
                .starts_with(info.workflow_edition_base)
        );
        assert_eq!(info.git.is_some(), info.build_cohort.starts_with("dev."));
        assert_eq!(
            info.build_source.as_str(),
            rlmesh_proto::BUILD_SOURCE,
            "a closed set"
        );
    }

    #[test]
    fn display_is_pythons_repr() {
        let info = build_info();
        let text = info.to_string();
        assert!(
            text.starts_with(&format!(
                "BuildInfo(version={:?}, protocol_generation=\"rlmesh-wire-v1\", ",
                info.version
            )),
            "{text}"
        );
        let git = info
            .git
            .map_or("None".to_string(), |git| format!("{git:?}"));
        assert!(
            text.ends_with(&format!(
                "build_source={:?}, git={git})",
                info.build_source.as_str()
            )),
            "{text}"
        );
    }
}
