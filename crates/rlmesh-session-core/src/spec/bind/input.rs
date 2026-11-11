//! Binding an input entry, or blanking an absent optional one.

use super::steps::field_range;
use super::table::{InputRow, Status};
use super::{Binder, Slot, joint_count};
use crate::prelude::*;

use crate::wire::part::{DType, Encoding, FieldSpec, PartSpec};

use crate::spec::contract;
use crate::spec::convert;
use crate::spec::explain::{hz, num};
use crate::spec::kinds::{Kind, quantity};
use crate::spec::plan::{Delivered, Delivery, Order, Plan, Step};
use crate::spec::policy::{Allow, ImageFormat, ImageLayout, Input, Names, Source};
use crate::spec::rotation::Rotation;
use crate::spec::units::Quantity;

/// How far `history_hz` may sit from the source rate.
const HISTORY_HZ_TOLERANCE: f64 = 0.01;

impl<'a> Binder<'a> {
    /// Step 6 for one input.
    pub(super) fn input(
        &mut self,
        slot: &Slot<'a>,
        input: &Input,
        part: &'a PartSpec,
        status: Status,
        replan: f64,
    ) -> Option<InputRow> {
        let kind = Kind::parse(&part.kind).unwrap_or(Kind::Custom);
        let applied = input.source == Source::Applied;
        let path = if applied {
            part.path.applied()
        } else {
            part.path.to_string()
        };
        let before = self.problems.len();
        let source_fields: Vec<FieldSpec> = if applied {
            match &part.command {
                Some(command) => command.fields.clone(),
                None => {
                    self.incompatible(
                        slot,
                        format!(
                            "{}: {} has no command, so no applied stream.",
                            slot.name(),
                            part.path
                        ),
                        Vec::new(),
                    );
                    return None;
                }
            }
        } else {
            part.fields.clone()
        };
        self.frames(slot, part, input.point.as_ref());
        let mut plan = Plan::default();
        if part.encoding != Encoding::Raw {
            plan.steps.push(Step::Decode);
        }
        let mut delivered: Vec<Delivered>;
        let mut elements: Option<u64> = None;
        if kind.is_image() {
            delivered = self.image(slot, input, part, kind, &source_fields[0], &path, &mut plan)?;
        } else {
            // Fields, in the model's order.
            let wanted: Vec<String> = match (&input.fields, &slot.entry.schema) {
                (_, Some(schema)) => schema.iter().map(|f| f.name.clone()).collect(),
                (Some(fields), None) => fields.clone(),
                (None, None) => source_fields.iter().map(|f| f.name.clone()).collect(),
            };
            let mut fields: Vec<FieldSpec> = Vec::new();
            for name in &wanted {
                match source_fields.iter().find(|f| &f.name == name) {
                    Some(f) => fields.push(f.clone()),
                    None => {
                        let have: Vec<&str> =
                            source_fields.iter().map(|f| f.name.as_str()).collect();
                        self.incompatible(
                            slot,
                            format!(
                                "{}: {path} publishes {}; the model wants {name}.",
                                slot.name(),
                                have.join(", ")
                            ),
                            Vec::new(),
                        );
                        return None;
                    }
                }
            }
            if let Some(schema) = &slot.entry.schema {
                // `fields` is already in the schema's order, by name.
                let matches = schema.len() == source_fields.len()
                    && schema
                        .iter()
                        .zip(&fields)
                        .all(|(a, b)| a.name == b.name && a.shape == b.shape);
                if !matches {
                    self.incompatible(
                        slot,
                        format!(
                            "{}: {path}'s fields do not match the declared schema of {}.",
                            slot.name(),
                            part.kind
                        ),
                        vec!["    A custom kind binds only by exact name and a matching fields schema.".into()],
                    );
                    return None;
                }
            }
            if source_fields.len() > 1 {
                plan.steps.push(Step::Fields(wanted.clone()));
            }
            // Names.
            let mut count = None;
            if kind.is_joint_set() {
                let n = joint_count(&source_fields);
                let order = self.names(slot, part, n, &path)?;
                if let Order::Select(picked) = &order {
                    count = Some(picked.len() as u32);
                }
                plan.steps.push(Step::Names(order));
            }
            // Units.
            for (field, _) in &slot.entry.units {
                if !fields.iter().any(|f| &f.name == field) {
                    self.incompatible(
                        slot,
                        format!(
                            "{}: units on {field}, which it does not deliver.",
                            slot.name()
                        ),
                        Vec::new(),
                    );
                    return None;
                }
            }
            for f in &mut fields {
                let q = quantity(kind, &f.name, &f.unit);
                let target = slot
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
                if q == Quantity::None {
                    continue;
                }
                match self.unit_step(slot, part, kind, &f.name, &f.unit, &target, &path) {
                    Ok(Some(step)) => {
                        if !f.dtype.is_float() {
                            f.dtype = DType::F32;
                        }
                        f.unit = target;
                        plan.steps.push(step);
                    }
                    Ok(None) => {}
                    Err(()) => return None,
                }
            }
            // Rotation.
            let rotation_fields: Vec<usize> = fields
                .iter()
                .enumerate()
                .filter(|(_, f)| !f.rotation.is_empty())
                .map(|(i, _)| i)
                .collect();
            if let Some(to) = slot.entry.rotation {
                if rotation_fields.is_empty() {
                    self.incompatible(
                        slot,
                        format!(
                            "{}: rotation {to}, but it delivers no rotation field.",
                            slot.name()
                        ),
                        Vec::new(),
                    );
                    return None;
                }
                for i in rotation_fields {
                    let from =
                        Rotation::from_name(&fields[i].rotation).unwrap_or(Rotation::QuatXyzw);
                    match self.rotation_step(slot, &fields[i].name.clone(), from, to, &path) {
                        Ok(Some(step)) => {
                            fields[i].rotation = to.name().into();
                            fields[i].shape = vec![to.width()];
                            plan.steps.push(step);
                        }
                        Ok(None) => {}
                        Err(()) => return None,
                    }
                }
            }
            // Tactile and custom shapes.
            if let Some(shape) = &input.shape
                && fields.iter().any(|f| &f.shape != shape)
            {
                self.incompatible(
                    slot,
                    format!(
                        "{}: {path} has shape {:?}; the model wants {shape:?}.",
                        slot.name(),
                        fields[0].shape
                    ),
                    vec!["    RLMesh does not reshape.".into()],
                );
                return None;
            }
            // dtype (from the schema, for custom kinds).
            let targets: Vec<Option<DType>> = match &slot.entry.schema {
                Some(schema) => schema.iter().map(|s| Some(s.dtype)).collect(),
                None => vec![input.dtype; fields.len()],
            };
            for (f, target) in fields.iter_mut().zip(targets) {
                let Some(to) = target else { continue };
                if f.dtype == DType::Utf8 {
                    continue;
                }
                match self.cast_step(slot, f.dtype, to, &path) {
                    Ok(Some(step)) => {
                        f.dtype = to;
                        plan.steps.push(step);
                    }
                    Ok(None) => {}
                    Err(()) => return None,
                }
            }
            // A declared range maps through rescale.
            if let Some(to) = slot.entry.range {
                let from = fields
                    .iter()
                    .map(|f| field_range(part, kind, &f.name, &f.unit))
                    .collect::<Option<Vec<_>>>()
                    .and_then(|r| {
                        r.first()
                            .copied()
                            .filter(|first| r.iter().all(|x| x == first))
                    });
                let Some(from) = from else {
                    self.incompatible(
                        slot,
                        format!(
                            "{}: the model wants range {}…{}, but {path} declares no range for {}.",
                            slot.name(),
                            num(to.0),
                            num(to.1),
                            wanted.join(", ")
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
                                "{path} ranges {}…{}, the model wants {}…{}",
                                num(from.0),
                                num(from.1),
                                num(to.0),
                                num(to.1)
                            ),
                            Allow::Rescale,
                        );
                        return None;
                    }
                    for f in &mut fields {
                        if !f.dtype.is_float() {
                            f.dtype = DType::F32;
                        }
                    }
                    plan.steps.push(Step::Rescale { from, to });
                }
            }
            if let Some(n) = count {
                for f in &mut fields {
                    if !f.shape.is_empty() {
                        f.shape[0] = n;
                    }
                }
            }
            delivered = fields
                .iter()
                .map(|f| Delivered {
                    name: f.name.clone(),
                    dtype: f.dtype,
                    shape: f.shape.clone(),
                })
                .collect();
            elements = if delivered.len() == 1
                || delivered
                    .iter()
                    .all(|d| d.shape.len() == 1 && d.dtype == delivered[0].dtype)
            {
                Some(
                    delivered
                        .iter()
                        .map(|d| d.shape.iter().map(|x| u64::from(*x)).product::<u64>())
                        .sum(),
                )
            } else {
                None
            };
        }
        if kind.is_image() {
            // Per-channel affines are the common image normalisation.
            elements = delivered
                .first()
                .and_then(|d| match input.layout {
                    ImageLayout::Hwc => d.shape.last(),
                    ImageLayout::Chw => d.shape.first(),
                })
                .map(|c| u64::from(*c));
        }
        match self.affine(slot, elements) {
            Ok(Some(step)) => {
                for d in &mut delivered {
                    if !d.dtype.is_float() {
                        d.dtype = DType::F32;
                    }
                }
                plan.steps.push(step);
            }
            Ok(None) => {}
            Err(()) => return None,
        }
        // History.
        let source_hz = contract::source_hz(part, applied);
        if input.history > 1 {
            if let Some(trained) = input.history_hz
                && (trained - source_hz).abs() > HISTORY_HZ_TOLERANCE * source_hz
            {
                self.incompatible(
                    slot,
                    format!(
                        "{}: the window was trained at {} Hz; {path} publishes at {} Hz.",
                        slot.name(),
                        hz(trained),
                        hz(source_hz)
                    ),
                    vec!["    Resampling a window to another rate is Later.".into()],
                );
                return None;
            }
            plan.steps.push(Step::Window(input.history));
        }
        let allows = self.allows(slot.entry);
        if input.history > 1 && allows.has(Allow::Warmup) {
            plan.notes.push("warmup at episode start".into());
        }
        if allows.has(Allow::Hold) {
            plan.notes.push("hold when stale".into());
        }
        if self.problems.len() > before {
            return None;
        }
        let (max_age, explicit) = contract::freshness(part, applied, replan, input.max_age);
        let delivers = Delivery::of(delivered, (input.history > 1).then_some(input.history));
        Some(InputRow {
            entry: slot.entry.name.clone(),
            kind: slot.entry.kind_name.clone(),
            part: Some(part.clone()),
            channel: if applied {
                part.applied_channel
            } else {
                Some(part.channel)
            },
            applied,
            status,
            plan,
            delivers,
            history: input.history,
            source_hz,
            max_age,
            max_age_explicit: explicit,
            required: slot.required(),
            hold: allows.has(Allow::Hold),
            warmup: allows.has(Allow::Warmup),
        })
    }

    /// An absent optional input bound to a constant, when its target is
    /// fully declared.
    pub(super) fn blank(&mut self, slot: &Slot<'_>, input: &Input) -> Option<InputRow> {
        let kind = slot.entry.kind;
        let undeclared = |what: &str| format!("{}: cannot blank it without {what}.", slot.name());
        // A black frame through the declared preprocessing: range, then the
        // affine (per channel when it has one value per channel).
        let mut fill = vec![match (kind, slot.entry.range) {
            (Kind::Camera, Some((low, _))) => low,
            _ => 0.0,
        }];
        if kind == Kind::Camera && (slot.entry.scale.is_some() || slot.entry.offset.is_some()) {
            let n = [&slot.entry.scale, &slot.entry.offset]
                .iter()
                .filter_map(|v| v.as_ref().map(Vec::len))
                .max()
                .unwrap_or(1);
            let (scale, offset) = (
                slot.entry.scale.as_deref().unwrap_or_default(),
                slot.entry.offset.as_deref().unwrap_or_default(),
            );
            fill = (0..n)
                .map(|i| convert::affine(fill[0], i, scale, offset))
                .collect();
        }
        let (what, delivered) = match kind {
            Kind::Camera => {
                let Some((h, w)) = input.size else {
                    self.incompatible(slot, undeclared("a size"), Vec::new());
                    return None;
                };
                let c = if input.format == ImageFormat::Mono8 {
                    1
                } else {
                    3
                };
                let shape = match input.layout {
                    ImageLayout::Hwc => vec![h, w, c],
                    ImageLayout::Chw => vec![c, h, w],
                };
                let dtype = if slot.entry.range.is_some() {
                    input.dtype.unwrap_or(DType::F32)
                } else {
                    input.dtype.unwrap_or(DType::U8)
                };
                (
                    format!("u8 {w}x{h}x{c} black, then declared preprocessing"),
                    vec![Delivered {
                        name: "image".into(),
                        dtype,
                        shape,
                    }],
                )
            }
            Kind::Text => (
                "\"\"".to_string(),
                vec![Delivered {
                    name: "text".into(),
                    dtype: DType::Utf8,
                    shape: Vec::new(),
                }],
            ),
            Kind::Joints | Kind::Hand => {
                let count = match &slot.entry.names {
                    Names::List(l) => Some(l.len() as u32),
                    _ => slot.entry.count,
                };
                let (Some(n), Some(fields)) = (count, &input.fields) else {
                    self.incompatible(slot, undeclared("names or count, and fields"), Vec::new());
                    return None;
                };
                let dtype = input.dtype.unwrap_or(DType::F32);
                (
                    format!("zeros {} [{n}] per field", crate::spec::plan::short(dtype)),
                    fields
                        .iter()
                        .map(|f| Delivered {
                            name: f.clone(),
                            dtype,
                            shape: vec![n],
                        })
                        .collect(),
                )
            }
            _ => {
                let (Some(shape), Some(dtype)) = (&input.shape, input.dtype) else {
                    self.incompatible(slot, undeclared("a shape and a dtype"), Vec::new());
                    return None;
                };
                let names = input
                    .fields
                    .clone()
                    .unwrap_or_else(|| vec![slot.entry.kind_name.replace('/', "_")]);
                (
                    format!("zeros {} {shape:?}", crate::spec::plan::short(dtype)),
                    names
                        .into_iter()
                        .map(|name| Delivered {
                            name,
                            dtype,
                            shape: shape.clone(),
                        })
                        .collect(),
                )
            }
        };
        Some(InputRow {
            entry: slot.entry.name.clone(),
            kind: slot.entry.kind_name.clone(),
            part: None,
            channel: None,
            applied: false,
            status: Status::Blank,
            plan: Plan {
                steps: vec![Step::Blank { what, fill }],
                notes: Vec::new(),
            },
            delivers: Delivery::of(delivered, (input.history > 1).then_some(input.history)),
            history: input.history,
            source_hz: 0.0,
            max_age: None,
            max_age_explicit: false,
            required: false,
            hold: false,
            warmup: false,
        })
    }
}
