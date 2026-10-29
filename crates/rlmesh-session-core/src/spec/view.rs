//! A scene's view (scene design §5): its placed robots' parts in one path
//! tree, which a policy binds against when the session carries places and
//! the run names no robot.
//!
//! - A place puts robot R's parts at or below `from` (default its root) at
//!   `at`: R's `{from}/x` is the view's `{at}/x`, and keeps its robot,
//!   channel and frame. Robots without a place are outside the view.
//! - Its pose is the static pose of R's `frame` (default `base`) in the
//!   scene frame `world`: data for [`crate::spec::frames::between`], never
//!   a transform the binder applies.
//! - v0.2 refuses a second place for one robot, a place on another robot's
//!   frame (every pose's parent is `world`), and, once maps are declared,
//!   two parts at one view path or one over §2.1's limits.

#[allow(unused_imports)]
use crate::prelude::*;

use crate::wire::path::{MAX_SEGMENT_LEN, is_identifier};
use crate::wire::{PartPath, PartSpec, Place};

/// The view's name: a binding against it targets `scene`.
pub const VIEW: &str = "scene";

/// How far a place's quaternion may sit from unit length.
const UNIT_TOLERANCE: f64 = 1e-6;

/// Whether `place` is well formed (scene design §2): `at` a path with no
/// `_` segment, `from` empty or a path, `frame` empty or a frame name, and
/// its pose finite metres and a unit quaternion.
///
/// # Errors
/// What is wrong and how to fix it.
pub fn check_place(place: &Place) -> Result<(), String> {
    let robot = &place.robot;
    PartPath::new(place.at.as_str())
        .map_err(|e| format!("{robot}'s place at {:?}: {}", place.at, e.reason))?;
    if !place.from.is_empty() {
        PartPath::new(place.from.as_str())
            .map_err(|e| format!("{robot}'s place from {:?}: {}", place.from, e.reason))?;
    }
    if !place.frame.is_empty() && !is_identifier(&place.frame, MAX_SEGMENT_LEN) {
        return Err(format!(
            "{robot}'s place frame {:?} is not a frame name ([a-z][a-z0-9_]*): a place poses one \
             of {robot}'s own frames in world, never another robot's (v0.2); a camera on another \
             robot's moving part belongs in that robot's own map",
            place.frame
        ));
    }
    if !place.pose.xyz.iter().all(|c| c.is_finite()) {
        return Err(format!("{robot}'s place pose.xyz must be 3 finite metres"));
    }
    let norm = libm::sqrt(place.pose.quat_xyzw.iter().map(|c| c * c).sum());
    if norm.is_nan() || (norm - 1.0).abs() > UNIT_TOLERANCE {
        return Err(format!(
            "{robot}'s place pose.quat_xyzw has norm {norm}, not 1 (within 1e-6): give a unit \
             quaternion, x, y, z, then w"
        ));
    }
    Ok(())
}

/// Whether a place table is well formed: each place ([`check_place`]),
/// and one per robot.
///
/// # Errors
/// The first problem.
pub fn check(places: &[Place]) -> Result<(), String> {
    for (i, place) in places.iter().enumerate() {
        check_place(place)?;
        if let Some(first) = places[..i].iter().find(|p| p.robot == place.robot) {
            return Err(format!(
                "{} is placed twice (at {:?} and {:?}): a robot has one place; place a subtree \
                 with from and bind the rest by its own paths",
                place.robot, first.at, place.at
            ));
        }
    }
    Ok(())
}

/// The view's parts: every placed robot's parts at or below its `from`,
/// renamed to their view paths. Channel ids, robots and frames stay.
///
/// # Errors
/// A malformed table ([`check`]), two parts at one view path, or a view
/// path over §2.1's limits, naming both parts.
pub fn parts(parts: &[PartSpec], places: &[Place]) -> Result<Vec<PartSpec>, String> {
    check(places)?;
    let mut view: Vec<PartSpec> = Vec::new();
    for place in places {
        for part in parts.iter().filter(|p| p.owner == place.robot) {
            let path = part.path.as_str();
            let below = if place.from.is_empty() {
                Some(path)
            } else if path == place.from {
                Some("")
            } else {
                path.strip_prefix(place.from.as_str())
                    .and_then(|rest| rest.strip_prefix('/'))
            };
            let Some(below) = below else { continue };
            let placed = if below.is_empty() {
                place.at.clone()
            } else {
                format!("{}/{below}", place.at)
            };
            let renamed = PartPath::new(placed.as_str()).map_err(|e| {
                format!(
                    "{}:{path} placed at {:?} is {placed}: {} (spec §2.1); place it higher \
                     or under a shorter path",
                    part.owner, place.at, e.reason
                )
            })?;
            if let Some(other) = view.iter().find(|p| p.path == renamed) {
                let original = parts
                    .iter()
                    .find(|p| p.channel == other.channel)
                    .map_or("?", |p| p.path.as_str());
                return Err(format!(
                    "placed paths collide at {placed}: {}:{original} and {}:{path}; place them \
                     apart",
                    other.owner, part.owner
                ));
            }
            let mut part = part.clone();
            part.path = renamed;
            view.push(part);
        }
    }
    Ok(view)
}
