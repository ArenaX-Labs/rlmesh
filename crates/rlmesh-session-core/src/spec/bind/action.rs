//! Binding an action entry, or dropping an absent optional one.

use super::steps::field_range;
use super::table::ActionRow;
use super::table::Status;
use super::{Binder, Slot, joint_count};
use crate::prelude::*;

use core::time::Duration;

use crate::wire::part::{DType, MAX_CHUNK_BYTES, PartSpec};

use crate::spec::explain::{hz, ms, num};
use crate::spec::kinds::{Kind, quantity};
use crate::spec::plan::{Delivered, Delivery, Order, Plan, Step};
use crate::spec::policy::{Action, Allow};
use crate::spec::projection::Projection;
use crate::spec::rotation::Rotation;
use crate::spec::units::Quantity;

impl<'a> Binder<'a> {
    fn grid(&self, action: &Action, replan: f64) -> (f64, bool, Duration) {
        let grid = action.hz.unwrap_or(replan);
        let dt = Duration::from_nanos((1e9 / grid).round() as u64);
        (grid, action.hz.is_some(), dt)
    }

    pub(super) fn dropped(&mut self, slot: &Slot<'_>, action: &Action, replan: f64) -> ActionRow {
        let (grid_hz, grid_declared, dt) = self.grid(action, replan);
        ActionRow {
            entry: slot.entry.name.clone(),
            kind: slot.entry.kind_name.clone(),
            part: None,
            status: Status::Dropped,
            plan: Plan {
                steps: vec![Step::Project(Projection::DropAction)],
                ..Plan::default()
            },
            delivers: Delivery::Array {
                dtype: DType::F32,
                shape: vec![action.chunk],
                concat: Vec::new(),
            },
            grid_hz,
            grid_declared,
            dt,
            chunk: action.chunk,
            valid: Duration::ZERO,
            t0: action.t0,
        }
    }

    /// Step 6 for one action.
    pub(super) fn action(
        &mut self,
        slot: &Slot<'a>,
        action: &Action,
        part: &'a PartSpec,
        status: Status,
        replan: f64,
    ) -> Option<ActionRow> {
        let kind = Kind::parse(&part.kind).unwrap_or(Kind::Custom);
        let path = self.part_name(part);
        let declared = self.declared(part).path.to_string();
        if self.view.is_some() {
            if !self.places.iter().any(|p| p.robot == part.owner) {
                self.incompatible(
                    slot,
                    format!(
                        "{}: {} is not placed in this scene's view.",
                        slot.name(),
                        part.owner
                    ),
                    vec![format!(
                        "    fix: place {} in the scene, or bind it on its own: {}",
                        part.owner,
                        self.run.with_robot(part.owner.as_str())
                    )],
                );
                return None;
            }
        } else if slot.robot != self.robot {
            self.incompatible(
                slot,
                format!(
                    "{}: targets {}, but the policy's actions target {}.",
                    slot.name(),
                    slot.robot,
                    self.robot
                ),
                vec!["    All of a policy's action entries target one robot in v0.2.".into()],
            );
            return None;
        }
        let Some(command) = &part.command else {
            let mode = action.command.clone().unwrap_or_else(|| {
                kind.command_modes()
                    .first()
                    .map_or("value", |m| *m)
                    .to_string()
            });
            let reference = action
                .reference
                .as_ref()
                .map_or_else(String::new, |r| format!(" (reference {r})"));
            let mut body = vec![format!(
                "      {path} is a {} part without a command.",
                part.kind
            )];
            if kind == Kind::Pose {
                body.push(format!(
                    "    RLMesh does not do IK (scope rule). Declare {declared:?} with \"command\": {mode:?}"
                ));
                body.push("    and serve it from your robot or controller.".into());
            } else {
                body.push(format!(
                    "    Declare {declared:?} with \"command\": {mode:?} if the robot accepts commands there."
                ));
            }
            self.incompatible(
                slot,
                format!(
                    "{} wants {} command {mode:?}{reference} at {path:?}:",
                    slot.name(),
                    part.kind
                ),
                body,
            );
            return None;
        };
        let before = self.problems.len();
        if let Some(mode) = &action.command
            && mode != &command.mode
        {
            self.incompatible(
                slot,
                format!(
                    "{} wants {} command {mode:?} at {path:?}:",
                    slot.name(),
                    part.kind
                ),
                vec![
                    format!("      {path} accepts {:?}.", command.mode),
                    "    RLMesh does not convert between command modes.".into(),
                ],
            );
            return None;
        }
        if let Some(reference) = &action.reference
            && reference != &command.reference
        {
            self.incompatible(
                slot,
                format!(
                    "{}: the model's deltas are relative to the {reference} pose; {path} composes them on the {}.",
                    slot.name(),
                    if command.reference.is_empty() { "absolute" } else { &command.reference }
                ),
                Vec::new(),
            );
            return None;
        }
        self.frames(slot, part, None);
        let mut plan = Plan::default();
        let mut fields = command.fields.clone();
        // Declared affine first (actions apply it before the conversions).
        let elements = if fields
            .iter()
            .all(|f| f.shape.len() == 1 && f.dtype == fields[0].dtype)
        {
            Some(fields.iter().map(|f| u64::from(f.shape[0])).sum::<u64>())
        } else {
            None
        };
        match self.affine(slot, elements) {
            Ok(Some(step)) => plan.steps.push(step),
            Ok(None) => {}
            Err(()) => return None,
        }
        // Rescale: the gripper's open/closed, or a declared range.
        if kind == Kind::Gripper {
            let (open, closed) = (action.open.unwrap_or(1.0), action.closed.unwrap_or(0.0));
            if (open, closed) != (1.0, 0.0) {
                if !self.allowed(slot.entry, Allow::Rescale) {
                    self.not_allowed(
                        slot,
                        format!(
                            "the model's gripper is closed at {} and open at {}; {path} is 0 closed … 1 open",
                            num(closed),
                            num(open)
                        ),
                        Allow::Rescale,
                    );
                    return None;
                }
                plan.steps.push(Step::Rescale {
                    from: (closed, open),
                    to: (0.0, 1.0),
                });
            }
        }
        if let Some(from) = slot.entry.range {
            let to = fields
                .iter()
                .map(|f| field_range(part, kind, &f.name, &f.unit))
                .collect::<Option<Vec<_>>>()
                .and_then(|r| {
                    r.first()
                        .copied()
                        .filter(|first| r.iter().all(|x| x == first))
                });
            let Some(to) = to else {
                self.incompatible(
                    slot,
                    format!(
                        "{}: the model emits range {}…{}, but {path} declares no range.",
                        slot.name(),
                        num(from.0),
                        num(from.1)
                    ),
                    Vec::new(),
                );
                return None;
            };
            if from != to {
                if !self.allowed(slot.entry, Allow::Rescale) {
                    self.not_allowed(
                        slot,
                        format!(
                            "the model emits {}…{}, {path} takes {}…{}",
                            num(from.0),
                            num(from.1),
                            num(to.0),
                            num(to.1)
                        ),
                        Allow::Rescale,
                    );
                    return None;
                }
                plan.steps.push(Step::Rescale { from, to });
            }
        }
        // Rotation: the model's encoding to the part's.
        if let Some(from) = slot.entry.rotation {
            let Some(i) = fields.iter().position(|f| !f.rotation.is_empty()) else {
                self.incompatible(
                    slot,
                    format!(
                        "{}: rotation {from}, but {path} takes no rotation.",
                        slot.name()
                    ),
                    Vec::new(),
                );
                return None;
            };
            let to = Rotation::from_name(&fields[i].rotation).unwrap_or(Rotation::QuatXyzw);
            match self.rotation_step(slot, &fields[i].name.clone(), from, to, &path) {
                Ok(Some(step)) => {
                    fields[i].shape = vec![from.width()];
                    plan.steps.push(step);
                }
                Ok(None) => {}
                Err(()) => return None,
            }
        }
        // Units: the model's (SI unless declared) to the part's.
        for (field, _) in &slot.entry.units {
            if !fields.iter().any(|f| &f.name == field) {
                self.incompatible(
                    slot,
                    format!(
                        "{}: units on {field}, which {path} does not take.",
                        slot.name()
                    ),
                    Vec::new(),
                );
                return None;
            }
        }
        for f in &fields {
            let q = quantity(kind, &f.name, &f.unit);
            if q == Quantity::None {
                continue;
            }
            let from = slot
                .entry
                .units
                .iter()
                .find(|(n, _)| n == &f.name)
                .map(|(_, u)| u.clone())
                .unwrap_or_else(|| {
                    if q == Quantity::Custom {
                        f.unit.clone()
                    } else {
                        q.si().to_string()
                    }
                });
            match self.unit_step(slot, part, kind, &f.name, &from, &f.unit, &path) {
                Ok(Some(step)) => plan.steps.push(step),
                Ok(None) => {}
                Err(()) => return None,
            }
        }
        // Names.
        let mut count = None;
        if kind.is_joint_set() {
            let n = joint_count(&fields);
            let order = self.names(slot, part, n, &path)?;
            if let Order::Select(picked) = &order {
                count = Some(picked.len() as u32);
            }
            plan.steps.push(Step::Names(order));
        }
        if let Some(schema) = &slot.entry.schema {
            let matches = schema.len() == fields.len()
                && schema
                    .iter()
                    .zip(&fields)
                    .all(|(a, b)| a.name == b.name && a.shape == b.shape);
            if !matches {
                self.incompatible(
                    slot,
                    format!(
                        "{}: {path}'s command fields do not match the declared schema.",
                        slot.name()
                    ),
                    vec![
                        "    A custom kind binds only by exact name and a matching fields schema."
                            .into(),
                    ],
                );
                return None;
            }
        }
        // Timing.
        let (grid_hz, grid_declared, dt) = self.grid(action, replan);
        if command.is_delta() && grid_hz > part.control_hz {
            self.incompatible(
                slot,
                format!(
                    "{}: a delta every {} ms ({} Hz) is faster than {path}'s control rate of {} Hz.",
                    slot.name(),
                    ms(dt),
                    hz(grid_hz),
                    hz(part.control_hz)
                ),
                vec!["    At most one delta is dispatched per control tick.".into()],
            );
            return None;
        }
        let span = dt * action.chunk;
        let valid = action.ttl.map_or(span, |ttl| ttl.min(span));
        // A part that can't hold its target (a velocity, a torque; not a
        // delta, which keeps its target) runs its safe behaviour whenever
        // no action is due: each chunk must outlast the replan period, or
        // the robot stutters between chunks and goes safe after hold_max.
        let period = Duration::from_secs_f64(1.0 / replan);
        if !command.hold_last && !command.is_delta() && valid <= period {
            let needed = period.as_nanos() / dt.as_nanos().max(1) + 2;
            self.incompatible(
                slot,
                format!(
                    "{}: a chunk covers {} ms, no longer than the replan period of {} ms, and {path} can't hold its target between chunks.",
                    slot.name(),
                    ms(valid),
                    ms(period)
                ),
                vec![format!(
                    "    fix: \"chunk\": {needed} (or more) on {:?}, so each chunk outlasts the next replan.",
                    slot.entry.name
                )],
            );
            return None;
        }
        let stride = command.layout().ok().and_then(|l| l.stride()).unwrap_or(0);
        if (action.chunk as usize).saturating_mul(stride) > MAX_CHUNK_BYTES {
            self.incompatible(
                slot,
                format!(
                    "{}: {} actions of {stride} bytes is over the 4 MiB chunk cap.",
                    slot.name(),
                    action.chunk
                ),
                Vec::new(),
            );
            return None;
        }
        if valid > command.max_action_age {
            self.warnings.push(format!(
                "{}: a chunk covers {} ms, but {path}'s edge plays at most {} ms of it (max_action_age_ms); the rest is never dispatched.",
                slot.name(),
                ms(valid),
                ms(command.max_action_age)
            ));
        }
        if self.problems.len() > before {
            return None;
        }
        let delivered: Vec<Delivered> = fields
            .iter()
            .map(|f| Delivered {
                name: f.name.clone(),
                dtype: f.dtype,
                shape: match count {
                    Some(n) => vec![n],
                    None => f.shape.clone(),
                },
            })
            .collect();
        Some(ActionRow {
            entry: slot.entry.name.clone(),
            kind: slot.entry.kind_name.clone(),
            part: Some(part.clone()),
            status,
            plan,
            delivers: Delivery::of(delivered, Some(action.chunk)),
            grid_hz,
            grid_declared,
            dt,
            chunk: action.chunk,
            valid,
            t0: action.t0,
        })
    }
}
