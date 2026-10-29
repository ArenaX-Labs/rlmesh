//! Cross-robot geometry (scene design §5): each place's pose is data, the
//! static pose of one of its robot's frames in the scene frame `world`
//! (REP-103, z up). The binder never transforms a value (§5.2); a policy
//! that needs another robot's frame composes two poses with [`between`].

#[allow(unused_imports)]
use crate::prelude::*;

use crate::wire::Pose;

fn hamilton(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

fn conjugate([x, y, z, w]: [f64; 4]) -> [f64; 4] {
    [-x, -y, -z, w]
}

/// `v` rotated by the unit quaternion `q`.
fn rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let [x, y, z, _] = hamilton(hamilton(q, [v[0], v[1], v[2], 0.0]), conjugate(q));
    [x, y, z]
}

/// The static rigid transform from place `a`'s frame to place `b`'s:
/// `T_b←a = T_world←b⁻¹ · T_world←a`, in 64 bits. A point `p` in `a`'s
/// frame is `rotate(q, p) + xyz` in `b`'s. The quaternion is unit length
/// with w ≥ 0, as every quaternion a conversion produces (§5.3).
pub fn between(a: &Pose, b: &Pose) -> Pose {
    let unit = |q: [f64; 4]| {
        let n = libm::sqrt(q.iter().map(|c| c * c).sum());
        q.map(|c| c / n)
    };
    let (qa, qb) = (unit(a.quat_xyzw), unit(b.quat_xyzw));
    let inverse = conjugate(qb);
    let delta = [0, 1, 2].map(|i| a.xyz[i] - b.xyz[i]);
    let q = unit(hamilton(inverse, qa));
    Pose {
        xyz: rotate(inverse, delta),
        quat_xyzw: if q[3] < 0.0 { q.map(|c| -c) } else { q },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::FRAC_1_SQRT_2;

    #[test]
    fn between_composes_two_world_poses() {
        // The scene design's §2 rig: eye-l unrotated, the arm turned 90°
        // about z. In the arm's frame eye-l's frame is turned -90°, and
        // its origin, (-0.365, 0.2, 0.3) from the arm in world, reads
        // (0.2, 0.365, 0.3).
        let eye = Pose {
            xyz: [0.035, 0.0, 0.30],
            quat_xyzw: [0.0, 0.0, 0.0, 1.0],
        };
        let arm = Pose {
            xyz: [0.40, -0.20, 0.0],
            quat_xyzw: [0.0, 0.0, FRAC_1_SQRT_2, FRAC_1_SQRT_2],
        };
        let t = between(&eye, &arm);
        let want = Pose {
            xyz: [0.2, 0.365, 0.3],
            quat_xyzw: [0.0, 0.0, -FRAC_1_SQRT_2, FRAC_1_SQRT_2],
        };
        for (got, want) in t
            .xyz
            .iter()
            .zip(want.xyz)
            .chain(t.quat_xyzw.iter().zip(want.quat_xyzw))
        {
            assert!((got - want).abs() < 1e-9, "{t:?}");
        }
        // Back again (eye-l's frame is world's), and a frame to itself is
        // the identity.
        let back = between(&arm, &eye);
        assert!(
            (back.xyz[0] - 0.365).abs() < 1e-9 && (back.xyz[1] + 0.2).abs() < 1e-9,
            "{back:?}"
        );
        let same = between(&arm, &arm);
        assert!(
            same.xyz.iter().all(|c| c.abs() < 1e-12) && (same.quat_xyzw[3] - 1.0).abs() < 1e-12
        );
    }
}
