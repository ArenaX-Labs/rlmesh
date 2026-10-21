//! What a bound entry does to its samples: the ordered conversion steps of
//! §7.4 and the shape `predict` sees.
//!
//! Inputs apply decode → fields → reorder/select → units → rotation →
//! rotate180 → declared crop → aspect fit + resize → bgr → layout → dtype
//! or range normalisation → declared scale/offset → history window. Actions
//! apply declared scale/offset → rescale → rotation → units → reorder, and
//! the edge then validates and clamps. Automatic steps (§7.1) are marked
//! `auto`.

use std::fmt;

use crate::wire::part::DType;

use crate::spec::explain::num;
use crate::spec::policy::{Allow, Resample};
use crate::spec::rotation::Rotation;

/// An image size, `width × height`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Size {
    /// Columns.
    pub width: u32,
    /// Rows.
    pub height: u32,
}

impl Size {
    /// `w × h` reduced: `4:3`.
    pub fn aspect(self) -> String {
        let g = gcd(self.width, self.height).max(1);
        format!("{}:{}", self.width / g, self.height / g)
    }

    /// Whether `other` has the same aspect ratio.
    pub fn same_aspect(self, other: Self) -> bool {
        u64::from(self.width) * u64::from(other.height)
            == u64::from(self.height) * u64::from(other.width)
    }
}

impl fmt::Display for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// How an image is fitted to another aspect ratio (§7.2, v1 `fit`
/// semantics).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fit {
    /// Scale uniformly to cover the target, then center-crop it.
    Crop,
    /// Scale uniformly to fit inside the target, then letterbox with zeros
    /// (depth: NaN), as `resize_with_pad` does.
    Pad,
    /// Scale each axis to the target (distorting).
    Stretch,
}

impl Fit {
    /// The conversion that allows it.
    pub fn allow(self) -> Allow {
        match self {
            Self::Crop => Allow::Crop,
            Self::Pad => Allow::Pad,
            Self::Stretch => Allow::Stretch,
        }
    }
}

/// How joint elements are matched.
#[derive(Clone, Debug, PartialEq)]
pub enum Order {
    /// No names declared: the robot's order and count.
    Robot,
    /// The model's names are the robot's, in the robot's order.
    Identical,
    /// Permuted by names (automatic): `permutation[i]` is the robot index of
    /// the model's element i.
    Reorder(Vec<usize>),
    /// The model's names out of a larger set (`allow: select`).
    Select(Vec<usize>),
    /// By index (`names: "positional"` or a bare `count`).
    Positional,
}

/// One conversion step.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    /// JPEG to pixels (always: the robot chose the encoding).
    Decode,
    /// Which fields, in the model's order.
    Fields(Vec<String>),
    /// Joint matching.
    Names(Order),
    /// `to = from × factor`.
    Units {
        /// The field.
        field: String,
        /// Source unit.
        from: String,
        /// Target unit.
        to: String,
        /// The factor.
        factor: f64,
        /// Automatic (angles) rather than allowed.
        auto: bool,
    },
    /// A gripper in metres to normalised over `width_m`, or back.
    GripperWidth {
        /// Metres to normalised.
        to_normalized: bool,
        /// The span.
        low: f64,
        /// The span.
        high: f64,
    },
    /// Depth millimetres (uint16) to metres (float32), `0 → NaN`.
    DepthMetres,
    /// A rotation re-encoding.
    Rotation {
        /// The field.
        field: String,
        /// Source encoding.
        from: Rotation,
        /// Target encoding.
        to: Rotation,
        /// Quaternion order (automatic).
        auto: bool,
    },
    /// Rotate the image 180°.
    Rotate180,
    /// The policy's training-time center crop (always applied).
    CenterCrop {
        /// Side fraction.
        fraction: f64,
        /// Before.
        from: Size,
        /// After.
        to: Size,
    },
    /// Resample to the target, through an aspect fit when the aspect
    /// ratios differ (one step, §7.4 "aspect fit + resize").
    Resize {
        /// The kernel.
        kernel: Resample,
        /// The source size.
        from: Size,
        /// The target size.
        to: Size,
        /// The aspect fit (`None`: the same aspect ratio).
        fit: Option<Fit>,
    },
    /// Swap R and B.
    Bgr,
    /// HWC to CHW (automatic).
    Chw,
    /// A cast.
    Cast {
        /// Source dtype.
        from: DType,
        /// Target dtype.
        to: DType,
        /// Exact (no `allow` needed) rather than allowed.
        exact: bool,
    },
    /// u8 `[0, 255]` linearly into `[low, high]`.
    Normalise {
        /// Range.
        low: f64,
        /// Range.
        high: f64,
        /// The result dtype.
        dtype: DType,
    },
    /// A linear map between ranges.
    Rescale {
        /// From.
        from: (f64, f64),
        /// To.
        to: (f64, f64),
    },
    /// Declared affine `x * scale + offset` ([`convert::affine`]).
    ///
    /// [`convert::affine`]: crate::spec::convert::affine
    Affine {
        /// Scale values (empty: 1).
        scale: Vec<f64>,
        /// Offset values (empty: 0).
        offset: Vec<f64>,
    },
    /// A window of the last `n` samples by seq.
    Window(u32),
    /// A constant instead of an absent part.
    Blank {
        /// What it is, for the table.
        what: String,
        /// The value each element takes: one, or one per image channel
        /// (a black frame through the entry's declared preprocessing).
        fill: Vec<f64>,
    },
}

impl Step {
    /// An automatic step (§7.1), shown with `auto`.
    pub fn is_auto(&self) -> bool {
        match self {
            Self::Names(Order::Reorder(_)) | Self::Chw => true,
            Self::Units { auto, .. } | Self::Rotation { auto, .. } => *auto,
            _ => false,
        }
    }
}

fn values(v: &[f64]) -> String {
    match v {
        [one] => num(*one),
        many => format!("{} values", many.len()),
    }
}

impl fmt::Display for Step {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let auto = if self.is_auto() { "auto " } else { "" };
        match self {
            Self::Decode => f.write_str("decode(jpeg)"),
            Self::Fields(fields) => write!(f, "fields {}", fields.join(", ")),
            Self::Names(Order::Robot) => f.write_str("robot order"),
            Self::Names(Order::Identical) => f.write_str("names identical"),
            Self::Names(Order::Reorder(_)) => write!(f, "{auto}reorder(by names)"),
            Self::Names(Order::Select(picked)) => write!(f, "select({} by names)", picked.len()),
            Self::Names(Order::Positional) => f.write_str("positional"),
            Self::Units {
                field, from, to, ..
            } => write!(f, "{auto}units({field} {from} → {to})"),
            Self::GripperWidth {
                to_normalized,
                low,
                high,
            } => {
                let span = format!("{}…{} m", num(*low), num(*high));
                if *to_normalized {
                    write!(f, "units(pos {span} → 0…1)")
                } else {
                    write!(f, "units(pos 0…1 → {span})")
                }
            }
            Self::DepthMetres => f.write_str("units(depth mm → m, 0 → NaN)"),
            Self::Rotation {
                field, from, to, ..
            } => write!(f, "{auto}rotation({field} {from} → {to})"),
            Self::Rotate180 => f.write_str("rotate180"),
            Self::CenterCrop { fraction, from, to } => {
                write!(f, "crop({} of {from} → {to})", num(*fraction))
            }
            Self::Resize {
                kernel,
                to,
                fit: None,
                ..
            } => write!(f, "resize({}, {to})", kernel.name()),
            Self::Resize {
                kernel,
                from,
                to,
                fit: Some(fit),
            } => {
                let name = match fit {
                    Fit::Crop => "crop",
                    Fit::Pad => "pad",
                    Fit::Stretch => "stretch",
                };
                write!(f, "{name}({from} → {to}, {})", kernel.name())
            }
            Self::Bgr => f.write_str("bgr"),
            Self::Chw => write!(f, "{auto}layout(chw)"),
            Self::Cast { from, to, exact } => write!(
                f,
                "{}cast({} → {})",
                if *exact { "auto " } else { "" },
                short(*from),
                short(*to)
            ),
            Self::Normalise { low, high, dtype } => write!(
                f,
                "normalise(u8 → {}…{} {})",
                num(*low),
                num(*high),
                short(*dtype)
            ),
            Self::Rescale { from, to } => write!(
                f,
                "rescale({}…{} → {}…{})",
                num(from.0),
                num(from.1),
                num(to.0),
                num(to.1)
            ),
            Self::Affine { scale, offset } => {
                let mut parts = Vec::new();
                if !scale.is_empty() {
                    parts.push(format!("scale({})", values(scale)));
                }
                if !offset.is_empty() {
                    parts.push(format!("offset({})", values(offset)));
                }
                f.write_str(&parts.join(" → "))
            }
            Self::Window(n) => write!(f, "window({n} by seq)"),
            Self::Blank { what, .. } => write!(f, "blank({what})"),
        }
    }
}

/// The short dtype name tables use (`f32`, `u8`).
pub fn short(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "bool",
        DType::U8 => "u8",
        DType::U16 => "u16",
        DType::U32 => "u32",
        DType::U64 => "u64",
        DType::I8 => "i8",
        DType::I16 => "i16",
        DType::I32 => "i32",
        DType::I64 => "i64",
        DType::F16 => "f16",
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::Utf8 => "str",
    }
}

/// One delivered field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Delivered {
    /// Its name.
    pub name: String,
    /// Its dtype.
    pub dtype: DType,
    /// Its shape (an action's or window's leading axis included).
    pub shape: Vec<u32>,
}

/// What `predict` receives for an input, or returns for an action (§6.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Delivery {
    /// One array: a single field, or 1-D fields of one dtype concatenated
    /// in order.
    Array {
        /// Its dtype.
        dtype: DType,
        /// Its shape.
        shape: Vec<u32>,
        /// The concatenated fields, when more than one.
        concat: Vec<String>,
    },
    /// A dict keyed by field name.
    Dict(Vec<Delivered>),
    /// A `str`.
    Text,
}

impl Delivery {
    /// The delivery of `fields` (each already in its delivered dtype and
    /// shape), with a leading axis of `lead` when it is not `None`.
    pub fn of(fields: Vec<Delivered>, lead: Option<u32>) -> Self {
        if let [field] = fields.as_slice()
            && field.dtype == DType::Utf8
        {
            return Self::Text;
        }
        let with_lead = |mut shape: Vec<u32>| {
            if let Some(n) = lead {
                shape.insert(0, n);
            }
            shape
        };
        match fields.as_slice() {
            [field] => Self::Array {
                dtype: field.dtype,
                shape: with_lead(field.shape.clone()),
                concat: Vec::new(),
            },
            [first, ..]
                if fields
                    .iter()
                    .all(|f| f.shape.len() == 1 && f.dtype == first.dtype) =>
            {
                Self::Array {
                    dtype: first.dtype,
                    shape: with_lead(vec![fields.iter().map(|f| f.shape[0]).sum()]),
                    concat: fields.iter().map(|f| f.name.clone()).collect(),
                }
            }
            _ => Self::Dict(
                fields
                    .into_iter()
                    .map(|f| Delivered {
                        shape: with_lead(f.shape),
                        ..f
                    })
                    .collect(),
            ),
        }
    }
}

fn shape(shape: &[u32]) -> String {
    format!(
        "[{}]",
        shape
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

impl fmt::Display for Delivery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Array {
                dtype, shape: s, ..
            } => write!(f, "{} {}", short(*dtype), shape(s)),
            Self::Dict(fields) => {
                let items: Vec<String> = fields
                    .iter()
                    .map(|d| format!("{}: {} {}", d.name, short(d.dtype), shape(&d.shape)))
                    .collect();
                write!(f, "{{{}}}", items.join(", "))
            }
            Self::Text => f.write_str("str"),
        }
    }
}

/// An entry's plan: its steps, in order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    /// The steps.
    pub steps: Vec<Step>,
    /// Notes shown after them (`names identical`, `warmup at episode
    /// start`).
    pub notes: Vec<String>,
}

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let steps: Vec<String> = self
            .steps
            .iter()
            .filter(|s| !matches!(s, Step::Names(Order::Identical | Order::Robot)))
            .map(ToString::to_string)
            .collect();
        if steps.is_empty() {
            f.write_str("none")?;
        } else {
            f.write_str(&steps.join(" → "))?;
        }
        let identical = self
            .steps
            .iter()
            .any(|s| matches!(s, Step::Names(Order::Identical)));
        if identical {
            f.write_str("; names identical")?;
        }
        for note in &self.notes {
            write!(f, "; {note}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deliveries_concatenate_1d_fields_of_one_dtype() {
        let f = |name: &str, dtype, shape: &[u32]| Delivered {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
        };
        let pos_vel = Delivery::of(
            vec![f("pos", DType::F32, &[6]), f("vel", DType::F32, &[6])],
            None,
        );
        assert_eq!(pos_vel.to_string(), "f32 [12]");
        let window = Delivery::of(
            vec![f("force", DType::F32, &[3]), f("torque", DType::F32, &[3])],
            Some(50),
        );
        assert_eq!(window.to_string(), "f32 [50, 6]");
        let mixed = Delivery::of(
            vec![f("pos", DType::F32, &[3]), f("rot", DType::F64, &[4])],
            None,
        );
        assert_eq!(mixed.to_string(), "{pos: f32 [3], rot: f64 [4]}");
        assert_eq!(
            Delivery::of(vec![f("text", DType::Utf8, &[])], None),
            Delivery::Text
        );
    }

    #[test]
    fn plans_read_like_the_spec() {
        let plan = Plan {
            steps: vec![
                Step::Decode,
                Step::Resize {
                    kernel: Resample::Bilinear,
                    from: Size {
                        width: 1280,
                        height: 720,
                    },
                    to: Size {
                        width: 224,
                        height: 224,
                    },
                    fit: Some(Fit::Pad),
                },
            ],
            notes: Vec::new(),
        };
        assert_eq!(
            plan.to_string(),
            "decode(jpeg) → pad(1280x720 → 224x224, bilinear)"
        );
        let state = Plan {
            steps: vec![
                Step::Fields(vec!["pos".into()]),
                Step::Names(Order::Identical),
            ],
            notes: Vec::new(),
        };
        assert_eq!(state.to_string(), "fields pos; names identical");
        assert_eq!(Plan::default().to_string(), "none");
        assert_eq!(
            Size {
                width: 1280,
                height: 720
            }
            .aspect(),
            "16:9"
        );
    }
}
