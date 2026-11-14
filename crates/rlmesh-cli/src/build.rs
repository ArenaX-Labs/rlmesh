//! What `rlmesh --version` and `rlmesh version` say about this build: what
//! kind it is and so whom it can talk to, its workflow editions and its
//! session wire revision.

use std::sync::LazyLock;

use rlmesh::BuildSource;

/// This build's `--version` text, after the `rlmesh` clap prints first.
pub(crate) static VERSION_LINE: LazyLock<String> = LazyLock::new(|| {
    let info = rlmesh::build_info();
    version_line(
        info.version,
        info.build_source,
        info.git,
        info.supported_workflow_editions,
        info.session_wire_revision,
    )
});

/// `0.2.0 (dev build of 1a2b3c4d5e6f: talks only to the same commit) ·
/// edition 2026.10-dev.1a2b3c4d5e6f · session wire 1`.
pub(crate) fn version_line(
    version: &str,
    source: BuildSource,
    git: Option<&str>,
    editions: &[&str],
    wire_revision: u32,
) -> String {
    let editions = match editions {
        [edition] => format!("edition {edition}"),
        editions => format!("editions {}", editions.join(", ")),
    };
    format!(
        "{version} ({}) · {editions} · session wire {wire_revision}",
        build_kind(source, git)
    )
}

/// What kind of build this is, and so whom it can talk to: a release build
/// talks to any build that shares an edition, a dev build only to its own
/// commit (`RLMESH_RELEASE_BUILD=1` makes a release build from a checkout).
pub(crate) fn build_kind(source: BuildSource, git: Option<&str>) -> String {
    match (source, git) {
        (BuildSource::Release, _) => "release build".to_owned(),
        (BuildSource::Git, Some(git)) => match git.split_once(".dirty.") {
            Some((commit, _)) => format!(
                "dev build of {commit} with uncommitted changes: talks only to the same tree"
            ),
            None => format!("dev build of {git}: talks only to the same commit"),
        },
        (BuildSource::Git, None) => "dev build: talks only to the same commit".to_owned(),
        // A published crate, or a checkout without git: the release's editions.
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
                &["2026.10-dev.1a2b3c4d5e6f"],
                1,
            ),
            "0.2.0-alpha.1 (dev build of 1a2b3c4d5e6f: talks only to the same commit) · \
             edition 2026.10-dev.1a2b3c4d5e6f · session wire 1"
        );
        assert_eq!(
            version_line(
                "0.2.0-alpha.1",
                BuildSource::Git,
                Some("1a2b3c4d5e6f.dirty.00ff00ff00ff00ff"),
                &["2026.10-dev.1a2b3c4d5e6f.dirty.00ff00ff00ff00ff"],
                1,
            ),
            "0.2.0-alpha.1 (dev build of 1a2b3c4d5e6f with uncommitted changes: talks only \
             to the same tree) · edition 2026.10-dev.1a2b3c4d5e6f.dirty.00ff00ff00ff00ff · \
             session wire 1"
        );
        assert_eq!(
            version_line(
                "0.2.0",
                BuildSource::Release,
                None,
                &["2026.10", "2026.06"],
                2,
            ),
            "0.2.0 (release build) · editions 2026.10, 2026.06 · session wire 2"
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
        assert!(VERSION_LINE.contains(info.workflow_edition));
        assert!(VERSION_LINE.ends_with(&format!(
            "session wire {}",
            rlmesh_session::wire::version::WIRE_REVISION
        )));
        if let Some(git) = info.git {
            let commit = git.split_once(".dirty.").map_or(git, |(commit, _)| commit);
            assert!(
                VERSION_LINE.contains(&format!("dev build of {commit}")),
                "{}",
                *VERSION_LINE
            );
        }
    }
}
