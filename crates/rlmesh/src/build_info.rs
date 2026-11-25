//! Build identity and release information.

use std::fmt;

/// This crate's version, in SemVer (`0.1.0-rc.12`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// How a build was made.
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
    /// The wire protocol generation this build speaks (`rlmesh-realtime-v1`).
    pub protocol_generation: &'static str,
    /// The session wire revision this build speaks: a 0.2 pre-release's wire
    /// may break before 0.2.0 freezes it, and two builds on different
    /// revisions cannot talk.
    pub session_wire_revision: u32,
    /// The bare `YYYY.MM` workflow edition this build runs at when nothing
    /// declares one ([`current_workflow_edition`](crate::current_workflow_edition)),
    /// the same on every build of a release line.
    pub workflow_edition: &'static str,
    /// How the build was made.
    pub build_source: BuildSource,
    /// The short git commit a source build was stamped from, or `None`.
    pub git: Option<&'static str>,
}

/// This build's identity.
pub fn build_info() -> BuildInfo {
    BuildInfo {
        version: VERSION,
        protocol_generation: rlmesh_proto::PROTOCOL_GENERATION,
        session_wire_revision: rlmesh_session::wire::version::WIRE_REVISION,
        workflow_edition: rlmesh_proto::CURRENT_WORKFLOW_EDITION,
        build_source: BuildSource::of(rlmesh_proto::BUILD_SOURCE),
        git: Some(rlmesh_proto::BUILD_GIT).filter(|git| !git.is_empty()),
    }
}

impl fmt::Display for BuildInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BuildInfo(version={:?}, protocol_generation={:?}, session_wire_revision={}, \
             workflow_edition={:?}, build_source={:?}, git=",
            self.version,
            self.protocol_generation,
            self.session_wire_revision,
            self.workflow_edition,
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
        assert_eq!(info.protocol_generation, "rlmesh-realtime-v1");
        assert_eq!(
            info.session_wire_revision,
            rlmesh_session::wire::version::WIRE_REVISION
        );
        assert_eq!(info.workflow_edition, crate::current_workflow_edition());
        assert_eq!(
            rlmesh_proto::Edition::parse(info.workflow_edition),
            Ok(rlmesh_proto::Edition::current()),
            "every build, release or not, reports the bare edition"
        );
        assert_eq!(info.git.is_some(), info.build_source == BuildSource::Git);
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
                "BuildInfo(version={:?}, protocol_generation=\"rlmesh-realtime-v1\", \
                 session_wire_revision={}, ",
                info.version, info.session_wire_revision
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
