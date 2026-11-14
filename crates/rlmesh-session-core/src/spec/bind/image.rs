//! Binding an image input.

use super::problem::{Code, problem};
use super::{Binder, Slot};
use crate::prelude::*;

use crate::wire::part::{DType, FieldSpec, PartSpec};

use crate::spec::kinds::Kind;
use crate::spec::normalise::attrs;
use crate::spec::plan::{Delivered, Fit, Plan, Size, Step};
use crate::spec::policy::{Allow, ImageFormat, ImageLayout, Input, Resample};

impl<'a> Binder<'a> {
    /// The image pipeline: decode → rotate180 → declared crop →
    /// aspect fit + resize → bgr → layout → dtype or range.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn image(
        &mut self,
        slot: &Slot<'_>,
        input: &Input,
        part: &PartSpec,
        kind: Kind,
        field: &FieldSpec,
        path: &str,
        plan: &mut Plan,
    ) -> Option<Vec<Delivered>> {
        if let Some(fields) = &input.fields
            && fields.iter().any(|f| f != &field.name)
        {
            self.incompatible(
                slot,
                format!(
                    "{}: {path} publishes {}; fields are not selectable on an image.",
                    slot.name(),
                    field.name
                ),
                Vec::new(),
            );
            return None;
        }
        let (height, width) = (field.shape[0], field.shape[1]);
        let channels = field.shape.get(2).copied();
        let mut dtype = field.dtype;
        // Units (depth mm → m).
        if kind == Kind::Depth {
            let target = slot
                .entry
                .units
                .iter()
                .find(|(n, _)| n == "depth")
                .map_or("m", |(_, u)| u.as_str())
                .to_string();
            match self.unit_step(slot, part, kind, "depth", &field.unit, &target, path) {
                Ok(Some(step)) => {
                    dtype = if dtype.is_float() { dtype } else { DType::F32 };
                    plan.steps.push(step);
                }
                Ok(None) => {}
                Err(()) => return None,
            }
        }
        if kind == Kind::Camera {
            let source_format = if channels == Some(1) {
                ImageFormat::Mono8
            } else {
                ImageFormat::Rgb8
            };
            if (input.format == ImageFormat::Mono8) != (source_format == ImageFormat::Mono8) {
                self.incompatible(
                    slot,
                    format!(
                        "{}: {path} is {}, the model wants {}.",
                        slot.name(),
                        source_format.name(),
                        input.format.name()
                    ),
                    vec!["    RLMesh does not convert between mono and RGB yet: convert in the model, or declare the other format on the robot.".into()],
                );
                return None;
            }
        } else if input.layout == ImageLayout::Chw {
            self.incompatible(
                slot,
                format!("{}: a depth image has no channels to lay out.", slot.name()),
                Vec::new(),
            );
            return None;
        }
        // rotate180.
        let upside_down = part
            .attrs
            .get(attrs::MOUNT)
            .is_some_and(|m| m == "upside_down");
        if upside_down != input.upside_down {
            if !self.allowed(slot.entry, Allow::Rotate180) {
                let what = if upside_down {
                    format!("{path} is mounted upside down, the model was trained upright")
                } else {
                    format!("{path} is upright, the model was trained on frames rotated 180°")
                };
                self.not_allowed(slot, what, Allow::Rotate180);
                return None;
            }
            plan.steps.push(Step::Rotate180);
        }
        // Declared crop.
        let mut size = Size { width, height };
        if let Some(fraction) = input.crop {
            let cut = |side: u32| ((f64::from(side) * fraction).round() as u32).clamp(1, side);
            let to = Size {
                width: cut(size.width),
                height: cut(size.height),
            };
            plan.steps.push(Step::CenterCrop {
                fraction,
                from: size,
                to,
            });
            size = to;
        }
        // Aspect fit and resize, one step (v1 `fit` semantics).
        let target = input.size.map_or(size, |(h, w)| Size {
            width: w,
            height: h,
        });
        if target != size {
            let allows = self.allows(slot.entry);
            let fit_needed = !size.same_aspect(target);
            let fit = [Fit::Crop, Fit::Pad, Fit::Stretch]
                .into_iter()
                .find(|f| fit_needed && allows.has(f.allow()));
            let (sx, sy) = (
                f64::from(target.width) / f64::from(size.width),
                f64::from(target.height) / f64::from(size.height),
            );
            let enlarges = match fit {
                Some(Fit::Crop) => sx.max(sy) > 1.0,
                Some(Fit::Pad) => sx.min(sy) > 1.0,
                _ => sx > 1.0 || sy > 1.0,
            };
            let missing_fit = fit_needed && fit.is_none();
            let missing_resize = !allows.has(Allow::Resize);
            let what = format!(
                "{path} is {size} ({}), the model wants {target} ({})",
                size.aspect(),
                target.aspect()
            );
            if missing_fit || missing_resize {
                let (needs, fix) = match (missing_fit, missing_resize) {
                    (true, true) => (
                        "resize, plus one of crop | pad | stretch to change the aspect ratio",
                        "add \"resize\" and \"pad\" (or \"crop\") to allow",
                    ),
                    (true, false) => (
                        "one of crop | pad | stretch to change the aspect ratio",
                        "add \"pad\" (or \"crop\") to allow",
                    ),
                    _ => ("resize", "add \"resize\" to allow"),
                };
                self.problems.push(problem(
                    Code::NotAllowed,
                    slot,
                    format!("{}: {what}", slot.name()),
                    vec![format!("    needs: {needs}"), format!("    fix: {fix}")],
                ));
                return None;
            }
            if enlarges && !allows.has(Allow::Upscale) {
                self.problems.push(problem(
                    Code::NotAllowed,
                    slot,
                    format!("{}: {what}, which enlarges it", slot.name()),
                    vec![
                        "    needs: upscale".into(),
                        "    fix: add \"upscale\" to allow".into(),
                    ],
                ));
                return None;
            }
            plan.steps.push(Step::Resize {
                kernel: if kind == Kind::Depth {
                    Resample::Nearest
                } else {
                    input.resample
                },
                from: size,
                to: target,
                fit,
            });
            size = target;
        }
        // bgr.
        if input.format == ImageFormat::Bgr8 {
            if !self.allowed(slot.entry, Allow::Bgr) {
                self.not_allowed(
                    slot,
                    format!("{path} is rgb8, the model wants bgr8"),
                    Allow::Bgr,
                );
                return None;
            }
            plan.steps.push(Step::Bgr);
        }
        if input.layout == ImageLayout::Chw {
            plan.steps.push(Step::Chw);
        }
        // Range normalisation, or a cast.
        if let Some((low, high)) = slot.entry.range {
            if kind == Kind::Depth {
                self.incompatible(
                    slot,
                    format!(
                        "{}: range normalises u8 images; depth has no such range.",
                        slot.name()
                    ),
                    Vec::new(),
                );
                return None;
            }
            let to = input.dtype.unwrap_or(DType::F32);
            if !to.is_float() {
                self.incompatible(
                    slot,
                    format!(
                        "{}: range normalisation produces floats, not {to}.",
                        slot.name()
                    ),
                    Vec::new(),
                );
                return None;
            }
            plan.steps.push(Step::Normalise {
                low,
                high,
                dtype: to,
            });
            dtype = to;
        } else if let Some(to) = input.dtype {
            match self.cast_step(slot, dtype, to, path) {
                Ok(Some(step)) => {
                    plan.steps.push(step);
                    dtype = to;
                }
                Ok(None) => {}
                Err(()) => return None,
            }
        }
        let shape = match (kind, channels, input.layout) {
            (Kind::Depth, _, _) => vec![size.height, size.width],
            (_, Some(c), ImageLayout::Hwc) => vec![size.height, size.width, c],
            (_, Some(c), ImageLayout::Chw) => vec![c, size.height, size.width],
            _ => vec![size.height, size.width],
        };
        Some(vec![Delivered {
            name: field.name.clone(),
            dtype,
            shape,
        }])
    }
}
