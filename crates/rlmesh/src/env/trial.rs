//! Python's `rlmesh.trial_index`.

use crate::{Error, InfoDict, Result, Value};

/// The reserved `trial_index` reset option, or `None` when absent: Python's
/// `rlmesh.trial_index(options, *, lane=None)`.
///
/// The ordinal of the episode a `reset` starts, 0-based and walked in order by
/// the runtime. Delivered only to an env that declared `"trial_index"` in
/// [`EnvFactory::reset_options`](crate::EnvFactory::reset_options). A reset
/// covering one lane carries a bare integer; a reset covering several lanes
/// of a vector env carries a list, one ordinal per lane, read with `lane`
/// (the position among the lanes this reset covers).
///
/// Integers are returned as is (negatives too), an integral float in
/// `[-2^63, 2^63)` is converted, and a bool or anything else is `None`. A
/// list read without `lane`, or a `lane` outside the lanes it covers, is the
/// ValueError class.
pub fn trial_index(options: Option<&InfoDict>, lane: Option<usize>) -> Result<Option<i64>> {
    let Some(value) = options.and_then(|options| options.get("trial_index")) else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let (values, lane) = match value {
        Value::List(items) | Value::Tuple(items) => {
            let Some(lane) = lane else {
                return Err(Error::invalid_argument(format!(
                    "trial_index carries one ordinal per lane ({} lanes); pass lane= to read \
                     one, e.g. rlmesh.trial_index(options, lane=0)",
                    items.len()
                )));
            };
            (items.as_slice(), lane)
        }
        other => (std::slice::from_ref(other), lane.unwrap_or(0)),
    };
    let Some(value) = values.get(lane) else {
        return Err(Error::invalid_argument(format!(
            "lane={lane} is outside the {} lane(s) trial_index covers",
            values.len()
        )));
    };
    Ok(as_ordinal(value))
}

/// `value` as an integer ordinal (an integer, or an integral float that fits
/// `i64`), else `None`.
fn as_ordinal(value: &Value) -> Option<i64> {
    const LIMIT: f64 = 9_223_372_036_854_775_808.0; // 2^63
    match value {
        Value::Int(int) => Some(*int),
        Value::Float(float) if float.fract() == 0.0 && (-LIMIT..LIMIT).contains(float) => {
            Some(*float as i64)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ErrorClass, Metadata};

    fn options(value: Value) -> Metadata {
        let mut options = Metadata::new();
        options.insert("trial_index", value);
        options
    }

    #[test]
    fn reads_python_s_ordinals() {
        assert_eq!(trial_index(None, None).unwrap(), None);
        assert_eq!(trial_index(Some(&Metadata::new()), None).unwrap(), None);
        assert_eq!(
            trial_index(Some(&options(Value::Int(-3))), None).unwrap(),
            Some(-3)
        );
        assert_eq!(
            trial_index(Some(&options(Value::Int(3))), Some(0)).unwrap(),
            Some(3)
        );
        assert_eq!(
            trial_index(Some(&options(Value::Float(4.0))), None).unwrap(),
            Some(4)
        );
        for none in [
            Value::Float(4.5),
            Value::Float(1e19),
            Value::Bool(true),
            Value::Text("1".into()),
            Value::None,
        ] {
            assert_eq!(trial_index(Some(&options(none)), None).unwrap(), None);
        }
        let lanes = options(Value::List(vec![Value::Int(7), Value::Int(8)]));
        assert_eq!(trial_index(Some(&lanes), Some(1)).unwrap(), Some(8));
        let err = trial_index(Some(&lanes), None).unwrap_err();
        assert_eq!(err.class(), ErrorClass::ValueError);
        assert!(err.to_string().contains("(2 lanes)"), "{err}");
        let err = trial_index(Some(&lanes), Some(2)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "lane=2 is outside the 2 lane(s) trial_index covers"
        );
        let err = trial_index(Some(&options(Value::Int(1))), Some(1)).unwrap_err();
        assert_eq!(err.class(), ErrorClass::ValueError);
    }
}
