//! Per-key checks (rates, durations, names, ranges) and the field
//! builders they feed: schemas, optional fields, units and limits.

use super::decl::{FieldDecl, FieldsDecl, LimitDecl, Ordered};
use super::{MAX_HZ, MAX_MS, MAX_NAME_LEN};
use crate::prelude::*;

use alloc::collections::BTreeMap;
use core::fmt;
use core::time::Duration;

use crate::wire::part::{DType, FieldLimit, FieldSpec};
use crate::wire::path::is_identifier;
use crate::wire::{CtlError, CtlStatus};

use crate::spec::kinds::Kind;
use crate::spec::rotation::Rotation;
use crate::spec::units::Quantity;

pub(super) fn invalid(path: &str, message: impl fmt::Display) -> CtlError {
    CtlError::new(CtlStatus::InvalidRequest, format!("{path}: {message}"))
}

/// Validates a rate in Hz.
pub(super) fn rate(path: &str, key: &str, hz: f64) -> Result<f64, CtlError> {
    if hz.is_finite() && hz > 0.0 && hz <= MAX_HZ {
        Ok(hz)
    } else {
        Err(invalid(path, format!("{key} {hz} is not in (0, 10000]")))
    }
}

/// Validates a duration in ms; `zero_ok` allows 0.
pub(super) fn millis(path: &str, key: &str, ms: f64, zero_ok: bool) -> Result<Duration, CtlError> {
    let ok = ms.is_finite() && ms <= MAX_MS && (ms > 0.0 || (zero_ok && ms == 0.0));
    if ok {
        Ok(Duration::from_secs_f64(ms / 1e3))
    } else {
        Err(invalid(path, format!("{key} {ms} is out of range")))
    }
}

pub(super) fn name(path: &str, key: &str, value: &str) -> Result<String, CtlError> {
    if is_identifier(value, MAX_NAME_LEN) {
        Ok(value.to_string())
    } else {
        Err(invalid(
            path,
            format!("{key} {value:?} must be [a-z][a-z0-9_]*"),
        ))
    }
}

pub(super) fn pair(path: &str, key: &str, [low, high]: [f64; 2]) -> Result<(), CtlError> {
    if low.is_finite() && high.is_finite() && low < high {
        Ok(())
    } else {
        Err(invalid(path, format!("{key}: finite, low < high")))
    }
}

/// A custom kind's field schema, in payload order.
pub(super) fn schema(path: &str, decl: Ordered<FieldDecl>) -> Result<Vec<FieldSpec>, CtlError> {
    decl.0
        .into_iter()
        .map(|(field, FieldDecl(dtype, shape))| {
            let field = name(path, "field", &field)?;
            let dtype = DType::from_name(&dtype)
                .ok_or_else(|| invalid(path, format!("field {field:?} dtype {dtype:?}")))?;
            Ok(FieldSpec::new(field, dtype, &shape))
        })
        .collect()
}

/// The float dtype a part declares (`float32` by default).
pub(super) fn float_dtype(path: &str, dtype: Option<&str>) -> Result<DType, CtlError> {
    match dtype {
        None | Some("float32") => Ok(DType::F32),
        Some("float64") => Ok(DType::F64),
        Some(other) => Err(invalid(
            path,
            format!("dtype {other:?}: float32 or float64"),
        )),
    }
}

/// Which optional fields a joint set or IMU publishes, in table order.
pub(super) fn optional_fields(
    path: &str,
    kind: Kind,
    decl: Option<FieldsDecl>,
    default: &[&'static str],
) -> Result<Vec<&'static str>, CtlError> {
    let table = kind.optional_fields();
    let listed = match decl {
        None => return Ok(default.to_vec()),
        Some(FieldsDecl::Names(names)) => names,
        Some(FieldsDecl::Schema(_)) => {
            return Err(invalid(
                path,
                format!("{kind} fields is a list of {}", table.join(", ")),
            ));
        }
    };
    if listed.is_empty() {
        return Err(invalid(path, "fields is empty"));
    }
    for (i, field) in listed.iter().enumerate() {
        if !table.contains(&field.as_str()) || listed[..i].contains(field) {
            return Err(invalid(
                path,
                format!(
                    "fields: {field:?} is not one of {} or is repeated",
                    table.join(", ")
                ),
            ));
        }
    }
    Ok(table
        .iter()
        .copied()
        .filter(|f| listed.iter().any(|l| l == f))
        .collect())
}

/// A field with its SI (or declared) unit.
pub(super) fn field(name: &str, dtype: DType, shape: &[u32], unit: &str) -> FieldSpec {
    let mut spec = FieldSpec::new(name, dtype, shape);
    spec.unit = unit.to_string();
    spec
}

pub(super) fn rotation_field(name: &str, dtype: DType, rotation: Rotation) -> FieldSpec {
    let mut spec = FieldSpec::new(name, dtype, &[rotation.width()]);
    spec.rotation = rotation.name().to_string();
    spec
}

/// Applies a part's declared `units` to its fields (published and command),
/// checking each against the field's quantity.
pub(super) fn apply_units(
    path: &str,
    kind: Kind,
    units: &[(String, String)],
    fields: &mut [&mut FieldSpec],
    quantities: &dyn Fn(&str) -> Quantity,
    width_m: bool,
) -> Result<(), CtlError> {
    for (key, unit) in units {
        let targets: Vec<&mut &mut FieldSpec> =
            fields.iter_mut().filter(|f| &f.name == key).collect();
        if targets.is_empty() {
            return Err(invalid(path, format!("units on unknown field {key:?}")));
        }
        let quantity = quantities(key);
        if !quantity.accepts(unit) {
            return Err(invalid(
                path,
                format!(
                    "units {unit:?} on {key:?} ({quantity}): one of {}",
                    match quantity.alternatives() {
                        alts if alts.is_empty() => "no unit".to_string(),
                        alts => alts.join(", "),
                    }
                ),
            ));
        }
        if kind == Kind::Gripper && unit == "m" && !width_m {
            return Err(invalid(
                path,
                "a gripper publishes metres only with width_m",
            ));
        }
        for target in targets {
            target.unit.clone_from(unit);
        }
    }
    Ok(())
}

pub(super) fn limits(
    path: &str,
    delta: bool,
    fields: &[FieldSpec],
    decl: BTreeMap<String, LimitDecl>,
) -> Result<Vec<FieldLimit>, CtlError> {
    decl.into_iter()
        .map(|(field, limit)| {
            let spec = fields.iter().find(|f| f.name == field).ok_or_else(|| {
                invalid(path, format!("limits on {field:?}, not a command field"))
            })?;
            if !spec.dtype.is_float() || spec.dtype == DType::F16 {
                return Err(invalid(
                    path,
                    format!("limits on {field:?}: only on float32 and float64 fields"),
                ));
            }
            let elements = spec.elements().unwrap_or(0);
            let pairs = match limit {
                LimitDecl::Norm { max_norm } => {
                    if !delta || spec.shape.len() != 1 {
                        return Err(invalid(
                            path,
                            format!("max_norm on {field:?}: only on a delta vector field"),
                        ));
                    }
                    if !(max_norm.is_finite() && max_norm > 0.0) {
                        return Err(invalid(path, format!("max_norm {max_norm} on {field:?}")));
                    }
                    return Ok(FieldLimit {
                        field,
                        low: Vec::new(),
                        high: Vec::new(),
                        max_norm: Some(max_norm),
                    });
                }
                LimitDecl::Pair(pair) => vec![pair],
                LimitDecl::Pairs(pairs) => {
                    if pairs.len() as u64 != elements {
                        return Err(invalid(
                            path,
                            format!(
                                "limits on {field:?}: {} pairs for {elements} elements",
                                pairs.len()
                            ),
                        ));
                    }
                    pairs
                }
            };
            if pairs
                .iter()
                .any(|[low, high]| !(low.is_finite() && high.is_finite() && low <= high))
            {
                return Err(invalid(
                    path,
                    format!("limits on {field:?}: finite, low <= high"),
                ));
            }
            Ok(FieldLimit {
                field,
                low: pairs.iter().map(|p| p[0]).collect(),
                high: pairs.iter().map(|p| p[1]).collect(),
                max_norm: None,
            })
        })
        .collect()
}
