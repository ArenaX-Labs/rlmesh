//! The parts vocabulary: the physical places a role can repeat at.
//!
//! A `part` is an identity key on a leaf, not a per-robot name: `left_arm` is
//! where a second `proprio/eef_pos` lives, never "the Franka". The list grows
//! the way roles do, when a real env and model pair needs the slot. Like the
//! role registry this is data only: it validates names, it never supplies one.

pub const LEFT_ARM: &str = "left_arm";
pub const RIGHT_ARM: &str = "right_arm";
/// The second of a pair when no side is meant: the legacy `_2` roles alias to
/// this part (see [`ALIASES`](super::registry::ALIASES)).
pub const ARM_2: &str = "arm_2";
pub const HEAD: &str = "head";
pub const TORSO: &str = "torso";
pub const BASE: &str = "base";
pub const LEFT_LEG: &str = "left_leg";
pub const RIGHT_LEG: &str = "right_leg";

/// Every registered part, for consumers exporting the vocabulary.
pub const PARTS: [&str; 8] = [
    LEFT_ARM, RIGHT_ARM, ARM_2, HEAD, TORSO, BASE, LEFT_LEG, RIGHT_LEG,
];

/// Whether `part` is a registered part.
pub fn is_known_part(part: &str) -> bool {
    PARTS.contains(&part)
}

/// Whether `part` is in the reserved `x/` escape namespace: intentionally
/// outside the registry, never nudged and never rejected at the publish gate.
pub fn is_escape_part(part: &str) -> bool {
    part.starts_with("x/")
}

/// Whether `part` needs no blessing: registered or an `x/` escape. Anything
/// else is ad-hoc -- it draws the authoring nudge and the strict publish tier
/// rejects it, exactly as an ad-hoc role does.
pub fn is_sanctioned_part(part: &str) -> bool {
    is_escape_part(part) || is_known_part(part)
}

#[cfg(test)]
mod tests {
    use super::{PARTS, is_escape_part, is_known_part, is_sanctioned_part};

    #[test]
    fn registered_escape_and_ad_hoc_parts_are_distinguished() {
        assert!(is_known_part("left_arm") && is_sanctioned_part("left_arm"));
        assert!(!is_known_part("x/tail") && is_escape_part("x/tail"));
        assert!(is_sanctioned_part("x/tail"));
        assert!(!is_sanctioned_part("franka"));
        assert!(!is_known_part("2"));
    }

    #[test]
    fn the_registered_parts_are_the_amended_eight() {
        assert_eq!(
            PARTS,
            [
                "left_arm",
                "right_arm",
                "arm_2",
                "head",
                "torso",
                "base",
                "left_leg",
                "right_leg",
            ]
        );
    }
}
