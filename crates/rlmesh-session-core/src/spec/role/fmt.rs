#[allow(unused_imports)]
use crate::prelude::*;

use alloc::collections::BTreeMap;

use crate::spec::role::input::Accept;
use crate::spec::role::roles::registry::is_known_role;
use crate::spec::rotation::Rotation;

pub(crate) fn quoted(value: &str) -> String {
    format!("{value:?}")
}

pub(crate) fn quoted_encoding(value: Option<Rotation>) -> String {
    match value {
        Some(encoding) => format!("{:?}", encoding.name()),
        None => "None".to_owned(),
    }
}

pub(crate) fn quoted_accept_set(value: Option<&Accept<Rotation>>) -> String {
    match value {
        None => "None".to_owned(),
        Some(set) => match set.names.as_slice() {
            [only] => format!("{only:?}"),
            names => format!("{names:?}"),
        },
    }
}

pub(crate) fn number(value: f64) -> String {
    format!("{value:?}")
}

pub(crate) fn quoted_range(range: (f64, f64)) -> String {
    format!("{range:?}")
}

pub(crate) fn quoted_keys<V>(map: &BTreeMap<String, V>) -> String {
    let keys: Vec<&String> = map.keys().collect();
    format!("{keys:?}")
}

pub(crate) fn quoted_leaf_keys<V>(
    map: &BTreeMap<(String, Option<String>, Option<String>), V>,
) -> String {
    let keys: Vec<String> = map
        .keys()
        .map(|(role, part, provenance)| {
            let mut key = role.clone();
            for suffix in [part, provenance].into_iter().flatten() {
                key.push('#');
                key.push_str(suffix);
            }
            key
        })
        .collect();
    format!("{keys:?}")
}

pub(crate) fn missing_role_hint<'a>(
    missing: &str,
    offered: impl IntoIterator<Item = &'a str>,
) -> String {
    let (kind, name) = split_role(missing);
    let words: Vec<&str> = name.split('_').filter(|w| !w.is_empty()).collect();
    let mut offered: Vec<&str> = offered.into_iter().collect();
    offered.sort_unstable();
    offered.dedup();
    let mut candidates: Vec<(usize, &str)> = offered
        .iter()
        .copied()
        .filter(|role| *role != missing)
        .filter_map(|role| {
            let (other_kind, other_name) = split_role(role);
            if kind != other_kind && kind != "x" && other_kind != "x" {
                return None;
            }
            if is_known_role(missing) && is_known_role(role) {
                return None;
            }
            let distance = edit_distance(name, other_name);
            let shares_word = other_name.split('_').any(|w| words.contains(&w));
            let prefix = other_name.starts_with(name) || name.starts_with(other_name);
            (shares_word || prefix || distance <= 2).then_some((distance, role))
        })
        .collect();
    candidates.sort();
    let mut hint = String::new();
    if !candidates.is_empty() {
        let names: Vec<String> = candidates.iter().take(3).map(|(_, r)| quoted(r)).collect();
        hint.push_str(&format!("; did you mean {}?", names.join(" or ")));
    }
    let escape = missing.starts_with("x/") || offered.iter().any(|role| role.starts_with("x/"));
    if escape {
        hint.push_str(
            " (an x/ role binds only to the identical string: the env and the model \
             must spell it the same way)",
        );
    }
    hint
}

fn split_role(role: &str) -> (&str, &str) {
    role.split_once('/').unwrap_or(("", role))
}

fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (above + 1)
                .min(row[j] + 1)
                .min(diagonal + usize::from(ca != *cb));
            diagonal = above;
        }
    }
    row[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hint_suggests_a_shared_word_and_explains_x_roles() {
        let hint = missing_role_hint("x/goal_pos", ["x/goal_body", "x/sinkage", "x/yaw_rate"]);
        assert_eq!(
            hint,
            "; did you mean \"x/goal_body\"? (an x/ role binds only to the identical \
             string: the env and the model must spell it the same way)"
        );
    }

    #[test]
    fn hint_matches_a_name_that_extends_the_missing_one() {
        let hint = missing_role_hint("x/steer", ["x/steering", "x/throttle"]);
        assert!(hint.starts_with("; did you mean \"x/steering\"?"), "{hint}");
    }

    #[test]
    fn hint_points_an_x_role_at_its_registered_twin() {
        let hint = missing_role_hint("proprio/base_rot", ["x/base_rot", "proprio/eef_pos"]);
        assert!(hint.starts_with("; did you mean \"x/base_rot\"?"), "{hint}");
    }

    #[test]
    fn hint_keeps_kinds_apart_and_is_empty_without_a_lead() {
        assert_eq!(missing_role_hint("action/gripper", ["proprio/gripper"]), "");
        assert_eq!(missing_role_hint("proprio/eef_pos", ["image/primary"]), "");
    }

    #[test]
    fn hint_never_swaps_one_registered_quantity_for_another() {
        assert_eq!(
            missing_role_hint("proprio/joint_pos", ["proprio/joint_vel"]),
            ""
        );
        assert_eq!(
            missing_role_hint(
                "proprio/joint_pos",
                ["proprio/joint_vel", "proprio/joint_angles"]
            ),
            "; did you mean \"proprio/joint_angles\"?"
        );
    }

    #[test]
    fn edit_distance_counts_single_edits() {
        assert_eq!(edit_distance("steer", "steer"), 0);
        assert_eq!(edit_distance("steer", "steering"), 3);
        assert_eq!(edit_distance("place_pos", "target_pos"), 5);
    }
}
