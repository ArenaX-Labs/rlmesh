//! Compatibility steps shared by inputs and actions: frames, names, units,
//! rotations, casts and affine maps.

use super::problem::{Code, NamesClash, Problem};
use super::table::Dir;
use super::{Binder, Slot, cast};
use crate::prelude::*;

use crate::wire::part::{DType, PartSpec};
use serde_json::Value;

use crate::spec::kinds::{Kind, quantity};
use crate::spec::normalise::attrs;
use crate::spec::plan::{Order, Step};
use crate::spec::policy::{Allow, Names};
use crate::spec::rotation::{self, Rotation, RotationChange};
use crate::spec::units::{self, UnitChange};

/// `width_m`, when the part declares it.
fn width_m(part: &PartSpec) -> Option<(f64, f64)> {
    let value: Value = serde_json::from_str(part.attrs.get(attrs::WIDTH_M)?).ok()?;
    Some((value.get(0)?.as_f64()?, value.get(1)?.as_f64()?))
}

/// A field's declared range: a gripper's 0…1 (or `width_m` in metres), or
/// the part's `range` for it.
pub(super) fn field_range(
    part: &PartSpec,
    kind: Kind,
    field: &str,
    unit: &str,
) -> Option<(f64, f64)> {
    if kind == Kind::Gripper {
        return if unit == "m" {
            width_m(part)
        } else {
            Some((0.0, 1.0))
        };
    }
    let ranges: Value = serde_json::from_str(part.attrs.get(attrs::RANGE)?).ok()?;
    let r = ranges.get(field)?;
    Some((r.get(0)?.as_f64()?, r.get(1)?.as_f64()?))
}

impl<'a> Binder<'a> {
    /// Frame identities: `frame`, pose `child`, wrench `point`.
    pub(super) fn frames(
        &mut self,
        slot: &Slot<'_>,
        part: &PartSpec,
        point: Option<&String>,
    ) -> bool {
        let kind = Kind::parse(&part.kind).unwrap_or(Kind::Custom);
        let path = part.path.as_str();
        let checks = [
            (
                "frame",
                slot.entry.frame.as_ref(),
                Some(part.frame.as_str()),
            ),
            (
                "child",
                slot.entry.child.as_ref(),
                part.attrs.get(attrs::CHILD).map(String::as_str),
            ),
            (
                "point",
                point,
                Some(
                    part.attrs
                        .get(attrs::POINT)
                        .map_or(part.frame.as_str(), String::as_str),
                ),
            ),
        ];
        let mut ok = true;
        for (key, want, have) in checks {
            let Some(want) = want else { continue };
            let have = have.filter(|h| !h.is_empty());
            if have == Some(want.as_str()) {
                continue;
            }
            ok = false;
            let found = match (key, have) {
                (_, None) => format!("{path} has no {key}."),
                ("frame", Some(h)) => format!("{path} is in frame {h}."),
                (_, Some(h)) => format!("{path} has {key} {h}."),
            };
            let have = have.unwrap_or("its own");
            self.incompatible(
                slot,
                format!("{} wants {key} {want}; {found}", slot.name()),
                vec![
                    format!(
                        "    RLMesh does not transform frames. Publish the {} in {want} (as {path:?},",
                        kind.value_noun()
                    ),
                    format!(
                        "    {key} {want:?}), or use a policy trained on {have}-{key} {}.",
                        kind.values_noun()
                    ),
                ],
            );
        }
        ok
    }

    /// Joint names: the order the model's elements take from the
    /// robot's, or a refusal.
    pub(super) fn names(
        &mut self,
        slot: &Slot<'_>,
        part: &PartSpec,
        n: u32,
        path: &str,
    ) -> Option<Order> {
        let entry = slot.entry;
        let positional = self.options.positional.contains(&entry.name);
        let model_count = |names: &Names| match names {
            Names::List(list) => Some(list.len() as u32),
            _ => entry.count,
        };
        let count_mismatch = |this: &mut Self, count: u32| {
            this.incompatible(
                slot,
                format!(
                    "{}: the model has {count} joints; {path} has {n}.",
                    slot.name()
                ),
                Vec::new(),
            );
        };
        if positional || entry.names == Names::Positional {
            let Some(count) = model_count(&entry.names) else {
                self.incompatible(
                    slot,
                    format!("{}: positional binding needs a count.", slot.name()),
                    Vec::new(),
                );
                return None;
            };
            if count != n {
                count_mismatch(self, count);
                return None;
            }
            return Some(Order::Positional);
        }
        match &entry.names {
            Names::Positional => Some(Order::Positional),
            Names::Robot => match entry.count {
                Some(count) if count != n => {
                    count_mismatch(self, count);
                    None
                }
                Some(_) => Some(Order::Positional),
                None => Some(Order::Robot),
            },
            Names::List(model) => {
                if part.names.is_empty() {
                    self.incompatible(
                        slot,
                        format!(
                            "{}: {path} declares a count, not joint names, so the model's names cannot be matched.",
                            slot.name()
                        ),
                        vec![format!(
                            "    fix: if index i means the same joint on both, bind it positionally:\n         {}",
                            self.run.with_positional([entry.name.as_str()])
                        )],
                    );
                    return None;
                }
                let index: Vec<Option<usize>> = model
                    .iter()
                    .map(|m| part.names.iter().position(|r| r == m))
                    .collect();
                if index.iter().all(Option::is_none) {
                    self.problems.push(Problem {
                        code: Code::Incompatible,
                        dir: slot.dir,
                        entry: entry.name.clone(),
                        head: String::new(),
                        body: Vec::new(),
                        names: Some(NamesClash {
                            part: path.to_string(),
                            model: model.clone(),
                            robot: part.names.clone(),
                        }),
                    });
                    return None;
                }
                let missing: Vec<&String> = model
                    .iter()
                    .zip(&index)
                    .filter(|(_, i)| i.is_none())
                    .map(|(m, _)| m)
                    .collect();
                if !missing.is_empty() {
                    let missing: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();
                    self.incompatible(
                        slot,
                        format!(
                            "{}: {path} has no joint named {}.",
                            slot.name(),
                            missing.join(", ")
                        ),
                        Vec::new(),
                    );
                    return None;
                }
                let index: Vec<usize> = index.into_iter().flatten().collect();
                if index.len() < part.names.len() {
                    if slot.dir == Dir::Out {
                        self.incompatible(
                            slot,
                            format!(
                                "{}: the model commands {} of {path}'s {} joints.",
                                slot.name(),
                                index.len(),
                                part.names.len()
                            ),
                            vec!["    RLMesh does not command a subset of a part yet: command every joint from the model, or declare the subset as its own part on the robot.".into()],
                        );
                        return None;
                    }
                    if !self.allowed(entry, Allow::Select) {
                        self.not_allowed(
                            slot,
                            format!(
                                "{path} has {} joints, the model names {} of them",
                                part.names.len(),
                                index.len()
                            ),
                            Allow::Select,
                        );
                        return None;
                    }
                    return Some(Order::Select(index));
                }
                if index.iter().enumerate().all(|(i, j)| i == *j) {
                    Some(Order::Identical)
                } else {
                    Some(Order::Reorder(index))
                }
            }
        }
    }

    /// Units for one field: the step, or a refusal. `to` is the target unit,
    /// `from` the source's.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn unit_step(
        &mut self,
        slot: &Slot<'_>,
        part: &PartSpec,
        kind: Kind,
        field: &str,
        from: &str,
        to: &str,
        path: &str,
    ) -> Result<Option<Step>, ()> {
        let q = quantity(kind, field, from);
        if from == to {
            return Ok(None);
        }
        if !q.accepts(to) {
            self.incompatible(
                slot,
                format!(
                    "{}: {to:?} is not a {q} unit ({path} publishes {field} in {from}).",
                    slot.name()
                ),
                Vec::new(),
            );
            return Err(());
        }
        match units::change(q, from, to) {
            UnitChange::Same => Ok(None),
            UnitChange::Auto(factor) => Ok(Some(Step::Units {
                field: field.into(),
                from: from.into(),
                to: to.into(),
                factor,
                auto: true,
            })),
            UnitChange::Allowed(factor) => {
                if !self.allowed(slot.entry, Allow::Units) {
                    self.not_allowed(
                        slot,
                        format!("{path} publishes {field} in {from}, the model wants {to}"),
                        Allow::Units,
                    );
                    return Err(());
                }
                if kind == Kind::Depth && from == "mm" {
                    return Ok(Some(Step::DepthMetres));
                }
                Ok(Some(Step::Units {
                    field: field.into(),
                    from: from.into(),
                    to: to.into(),
                    factor,
                    auto: false,
                }))
            }
            UnitChange::GripperWidth { to_normalized } => {
                let Some((low, high)) = width_m(part) else {
                    self.incompatible(
                        slot,
                        format!(
                            "{}: {path} declares no width_m, so metres cannot be mapped.",
                            slot.name()
                        ),
                        Vec::new(),
                    );
                    return Err(());
                };
                if !self.allowed(slot.entry, Allow::Units) {
                    self.not_allowed(
                        slot,
                        format!("{path} publishes pos in {from}, the model wants {to}"),
                        Allow::Units,
                    );
                    return Err(());
                }
                Ok(Some(Step::GripperWidth {
                    to_normalized,
                    low,
                    high,
                }))
            }
            UnitChange::Incompatible => {
                self.incompatible(
                    slot,
                    format!(
                        "{}: {path} publishes {field} in {from}; the model wants {to}, and {q} units compare by equality.",
                        slot.name()
                    ),
                    Vec::new(),
                );
                Err(())
            }
        }
    }

    /// A rotation re-encoding for one field.
    pub(super) fn rotation_step(
        &mut self,
        slot: &Slot<'_>,
        field: &str,
        from: Rotation,
        to: Rotation,
        path: &str,
    ) -> Result<Option<Step>, ()> {
        match rotation::change(from, to) {
            RotationChange::Same => Ok(None),
            RotationChange::Auto => Ok(Some(Step::Rotation {
                field: field.into(),
                from,
                to,
                auto: true,
            })),
            RotationChange::Allowed => {
                if !self.allowed(slot.entry, Allow::Rotation) {
                    let what = match slot.dir {
                        Dir::In => {
                            format!("{path} publishes {field} ({from}), the model wants {to}")
                        }
                        Dir::Out => format!("the model emits {field} as {from}, {path} takes {to}"),
                    };
                    self.not_allowed(slot, what, Allow::Rotation);
                    return Err(());
                }
                Ok(Some(Step::Rotation {
                    field: field.into(),
                    from,
                    to,
                    auto: false,
                }))
            }
            RotationChange::Incompatible => {
                self.incompatible(
                    slot,
                    format!(
                        "{}: nothing converts from {from} to {to} ({from} discards yaw).",
                        slot.name()
                    ),
                    Vec::new(),
                );
                Err(())
            }
        }
    }

    /// A cast to `to`, automatic when exact.
    pub(super) fn cast_step(
        &mut self,
        slot: &Slot<'_>,
        from: DType,
        to: DType,
        path: &str,
    ) -> Result<Option<Step>, ()> {
        match cast(from, to) {
            Some(true) if from == to => Ok(None),
            Some(true) => Ok(Some(Step::Cast {
                from,
                to,
                exact: true,
            })),
            Some(false) => {
                if !self.allowed(slot.entry, Allow::Dtype) {
                    self.not_allowed(
                        slot,
                        format!("{path} publishes {from}, the model wants {to} (not exact)"),
                        Allow::Dtype,
                    );
                    return Err(());
                }
                Ok(Some(Step::Cast {
                    from,
                    to,
                    exact: false,
                }))
            }
            None => {
                self.incompatible(
                    slot,
                    format!(
                        "{}: {path} publishes {from}; no cast to {to} exists.",
                        slot.name()
                    ),
                    Vec::new(),
                );
                Err(())
            }
        }
    }

    /// The affine's arity: one value, or one per element.
    pub(super) fn affine(
        &mut self,
        slot: &Slot<'_>,
        elements: Option<u64>,
    ) -> Result<Option<Step>, ()> {
        let (scale, offset) = (&slot.entry.scale, &slot.entry.offset);
        if scale.is_none() && offset.is_none() {
            return Ok(None);
        }
        for (key, values) in [("scale", scale), ("offset", offset)] {
            if let Some(values) = values
                && values.len() > 1
                && Some(values.len() as u64) != elements
            {
                self.incompatible(
                    slot,
                    format!(
                        "{}: {key} has {} values for {} elements.",
                        slot.name(),
                        values.len(),
                        elements.map_or_else(|| "per-field".to_string(), |e| e.to_string())
                    ),
                    Vec::new(),
                );
                return Err(());
            }
        }
        Ok(Some(Step::Affine {
            scale: scale.clone().unwrap_or_default(),
            offset: offset.clone().unwrap_or_default(),
        }))
    }
}
