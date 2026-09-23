//! Embodiment profiles: the shipped joint-label tuples an env or model writes
//! `labels=` from.
//!
//! A profile is data, not a wire type and not a resolver input: an env writes
//! `labels=embodiments::UNITREE_GO2.joints`, a model writes the same tuple or
//! a subset, and the wire still carries plain strings. The resolver never
//! consults this table; two sides agree on strings or they do not. What the
//! table is for is the label lint (a tuple that matches no profile as a set
//! draws the same nudge an ad-hoc role does, and the strict publish tier
//! refuses it) and for authors, who should not retype twelve joint names.
//!
//! Labels are the vendor URDF joint names verbatim (`FR_hip_joint`,
//! `panda_joint1`, `shoulder_pan_joint`), which is what Isaac Lab, MuJoCo
//! and ROS descriptions expose, so a label written from a robot's own config
//! matches a profile without renaming.

/// A named embodiment: the parts it has and its joint labels in the
/// embodiment's canonical order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbodimentProfile {
    /// The profile's stable name, e.g. `"unitree_go2"`.
    pub name: &'static str,
    /// The registered parts this body has (see [`parts`](super::parts)).
    pub parts: &'static [&'static str],
    /// Joint labels in canonical order (the vendor SDK's motor order).
    pub joints: &'static [&'static str],
}

/// Unitree Go2, 12 joints in SDK motor order: FR, FL, RR, RL, each hip,
/// thigh, calf. Isaac Lab and legged_gym read the MJCF in FL, FR, RL, RR
/// order; a model trained there names the same labels in that order and the
/// resolver derives the permutation.
pub const UNITREE_GO2: EmbodimentProfile = EmbodimentProfile {
    name: "unitree_go2",
    parts: &[super::parts::BASE],
    joints: &[
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ],
};

/// Unitree G1, the 29-DoF configuration: two 6-DoF legs, a 3-DoF waist and
/// two 7-DoF arms, in the Unitree SDK order.
pub const UNITREE_G1_29DOF: EmbodimentProfile = EmbodimentProfile {
    name: "unitree_g1_29dof",
    parts: &[
        super::parts::LEFT_LEG,
        super::parts::RIGHT_LEG,
        super::parts::TORSO,
        super::parts::LEFT_ARM,
        super::parts::RIGHT_ARM,
        super::parts::HEAD,
    ],
    joints: &[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
};

/// Franka Emika Panda, 7 arm joints as the `franka_ros` URDF names them.
pub const FRANKA_PANDA: EmbodimentProfile = EmbodimentProfile {
    name: "franka_panda",
    parts: &[],
    joints: &[
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ],
};

/// Universal Robots UR5e, 6 arm joints as the `ur_description` URDF names
/// them, base to wrist. A gripper (Robotiq or other) is not part of the arm:
/// it is its own `proprio/gripper` leaf.
pub const UR5E: EmbodimentProfile = EmbodimentProfile {
    name: "ur5e",
    parts: &[],
    joints: &[
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ],
};

/// Every shipped profile, for consumers exporting the table.
pub const PROFILES: [&EmbodimentProfile; 4] =
    [&UNITREE_GO2, &UNITREE_G1_29DOF, &FRANKA_PANDA, &UR5E];

/// The shipped profile whose joints cover every one of `labels` (order-free;
/// a model may name fewer joints than the body has), or `None`.
pub fn matching_profile(labels: &[String]) -> Option<&'static EmbodimentProfile> {
    PROFILES.iter().copied().find(|profile| {
        labels
            .iter()
            .all(|label| profile.joints.contains(&label.as_str()))
    })
}

/// The shipped profile sharing the most labels with `labels`, with the
/// overlap count, for the lint's hint. `None` when no profile shares any.
pub fn closest_profile(labels: &[String]) -> Option<(&'static EmbodimentProfile, usize)> {
    PROFILES
        .iter()
        .copied()
        .map(|profile| {
            let overlap = labels
                .iter()
                .filter(|label| profile.joints.contains(&label.as_str()))
                .count();
            (profile, overlap)
        })
        .filter(|(_, overlap)| *overlap > 0)
        .max_by_key(|(_, overlap)| *overlap)
}

#[cfg(test)]
mod tests {
    use super::{
        FRANKA_PANDA, PROFILES, UNITREE_G1_29DOF, UNITREE_GO2, UR5E, closest_profile,
        matching_profile,
    };

    fn owned(labels: &[&str]) -> Vec<String> {
        labels.iter().map(|label| (*label).to_owned()).collect()
    }

    #[test]
    fn the_four_profiles_carry_their_joint_counts_and_unique_labels() {
        assert_eq!(UNITREE_GO2.joints.len(), 12);
        assert_eq!(UNITREE_G1_29DOF.joints.len(), 29);
        assert_eq!(FRANKA_PANDA.joints.len(), 7);
        assert_eq!(UR5E.joints.len(), 6);
        assert_eq!(UR5E.joints[0], "shoulder_pan_joint");
        assert_eq!(UR5E.joints[5], "wrist_3_joint");
        assert!(UR5E.parts.is_empty());
        for profile in PROFILES {
            let mut joints = profile.joints.to_vec();
            joints.sort_unstable();
            joints.dedup();
            assert_eq!(
                joints.len(),
                profile.joints.len(),
                "{} repeats a label",
                profile.name
            );
            for part in profile.parts {
                assert!(
                    super::super::parts::is_known_part(part),
                    "{} part {part}",
                    profile.name
                );
            }
        }
        assert_eq!(UNITREE_GO2.joints[0], "FR_hip_joint");
        assert_eq!(UNITREE_GO2.joints[3], "FL_hip_joint");
        assert_eq!(UNITREE_G1_29DOF.joints[12], "waist_yaw_joint");
    }

    #[test]
    fn a_set_or_subset_matches_and_a_stray_label_names_the_closest() {
        let isaac = owned(&[
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ]);
        assert_eq!(
            matching_profile(&isaac).map(|p| p.name),
            Some("unitree_go2")
        );
        assert_eq!(
            matching_profile(&owned(&["left_knee_joint", "right_knee_joint"])).map(|p| p.name),
            Some("unitree_g1_29dof")
        );
        let stray = owned(&["FR_hip_joint", "FR_thigh_joint", "FR_shin"]);
        assert_eq!(matching_profile(&stray), None);
        let (closest, overlap) = closest_profile(&stray).expect("shares two");
        assert_eq!((closest.name, overlap), ("unitree_go2", 2));
        assert_eq!(closest_profile(&owned(&["j0", "j1"])), None);
    }
}
