//! What `rlmesh --version` and `rlmesh version` say about this build: what kind
//! it is, its workflow edition and its session wire revision.

use std::sync::LazyLock;

use rlmesh::BuildSource;

/// This build's `--version` text, after the `rlmesh` clap prints first.
pub(crate) static VERSION_LINE: LazyLock<String> = LazyLock::new(|| {
    let info = rlmesh::build_info();
    version_line(
        info.version,
        info.build_source,
        info.git,
        info.workflow_edition,
        info.session_wire_revision,
    )
});

/// `0.2.0 (dev build of 1a2b3c4d5e6f) · edition 2026.10 · session wire 2`.
pub(crate) fn version_line(
    version: &str,
    source: BuildSource,
    git: Option<&str>,
    edition: &str,
    wire_revision: u32,
) -> String {
    format!(
        "{version} ({}) · edition {edition} · session wire {wire_revision}",
        build_kind(source, git)
    )
}

/// What kind of build this is: a release build, a dev build of a commit
/// (`RLMESH_RELEASE_BUILD=1` makes a release build from a checkout), or a
/// package build. Builds talk to each other when their session wire revisions
/// match, whatever their kind.
pub(crate) fn build_kind(source: BuildSource, git: Option<&str>) -> String {
    match (source, git) {
        (BuildSource::Release, _) => "release build".to_owned(),
        (BuildSource::Git, Some(git)) => format!("dev build of {git}"),
        (BuildSource::Git, None) => "dev build".to_owned(),
        _ => "package build".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_version_line_says_what_kind_of_build_this_is() {
        assert_eq!(
            version_line(
                "0.2.0-alpha.1",
                BuildSource::Git,
                Some("1a2b3c4d5e6f"),
                "2026.10",
                2,
            ),
            "0.2.0-alpha.1 (dev build of 1a2b3c4d5e6f) · edition 2026.10 · session wire 2"
        );
        assert_eq!(
            version_line("0.2.0", BuildSource::Release, None, "2026.10", 2),
            "0.2.0 (release build) · edition 2026.10 · session wire 2"
        );
        assert_eq!(
            build_kind(BuildSource::Package, None),
            "package build".to_owned()
        );
    }

    #[test]
    fn this_build_is_described_by_its_own_identity() {
        let info = rlmesh::build_info();
        assert!(VERSION_LINE.starts_with(&format!("{} (", info.version)));
        assert!(
            VERSION_LINE.contains(&format!("edition {} ·", rlmesh::current_workflow_edition()))
        );
        assert!(VERSION_LINE.ends_with(&format!(
            "session wire {}",
            rlmesh_session::wire::version::WIRE_REVISION
        )));
        if let Some(git) = info.git {
            assert!(
                VERSION_LINE.contains(&format!("dev build of {git}")),
                "{}",
                *VERSION_LINE
            );
        }
    }
}
