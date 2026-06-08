use rlmesh_proto::{CURRENT_WORKFLOW_EDITION, SUPPORTED_WORKFLOW_EDITIONS};

pub(crate) fn fallback_workflow_edition(
    peer_supported_editions: &[String],
) -> Option<&'static str> {
    SUPPORTED_WORKFLOW_EDITIONS
        .iter()
        .copied()
        .filter(|edition| *edition != CURRENT_WORKFLOW_EDITION)
        .find(|edition| {
            peer_supported_editions
                .iter()
                .any(|supported| supported.trim() == *edition)
        })
}

#[cfg(test)]
mod tests {
    use rlmesh_proto::LEGACY_WORKFLOW_EDITION_2026;

    use super::fallback_workflow_edition;

    #[test]
    fn fallback_selects_mutual_legacy_edition() {
        assert_eq!(
            fallback_workflow_edition(&[LEGACY_WORKFLOW_EDITION_2026.to_string()]),
            Some(LEGACY_WORKFLOW_EDITION_2026)
        );
    }

    #[test]
    fn fallback_ignores_current_and_unknown_editions() {
        assert_eq!(
            fallback_workflow_edition(&["2026.06".to_string(), "2099.01".to_string()]),
            None
        );
    }
}
