use super::*;
use crate::{Error, Metadata, Result, Value};
use serde::{Deserialize, Serialize};
use serde_json::{Value as Json, json};

pub(crate) mod set {
    use super::*;
    /// Python’s `serialize` adapter operation.
    pub fn serialize<S: serde::Serializer, E: Vocabulary>(
        v: &[Tolerant<E>],
        s: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        if v.len() == 1 {
            v[0].serialize(s)
        } else {
            v.serialize(s)
        }
    }
    /// Python’s `deserialize` adapter operation.
    pub fn deserialize<'de, D: serde::Deserializer<'de>, E: Vocabulary>(
        d: D,
    ) -> std::result::Result<Vec<Tolerant<E>>, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Set<T> {
            One(T),
            Many(Vec<T>),
        }
        Ok(match Set::<Tolerant<E>>::deserialize(d)? {
            Set::One(x) => vec![x],
            Set::Many(v) => v,
        })
    }
}
pub(crate) mod accept {
    use super::*;
    /// Python’s `serialize` adapter operation.
    pub fn serialize<S: serde::Serializer, E: Vocabulary>(
        v: &Option<Vec<Tolerant<E>>>,
        s: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        match v {
            None => s.serialize_none(),
            Some(v) => set::serialize(v, s),
        }
    }
    /// Python’s `deserialize` adapter operation.
    pub fn deserialize<'de, D: serde::Deserializer<'de>, E: Vocabulary>(
        d: D,
    ) -> std::result::Result<Option<Vec<Tolerant<E>>>, D::Error> {
        let j = Option::<Json>::deserialize(d)?;
        j.map(|j| set::deserialize(j).map_err(serde::de::Error::custom))
            .transpose()
    }
}
fn err(e: impl std::fmt::Display) -> Error {
    Error::invalid_argument(e.to_string())
}
pub(crate) fn leaf_json<T: Serialize>(v: &T) -> Result<Json> {
    let mut j = serde_json::to_value(v).map_err(err)?;
    axis_to_wire(&mut j);
    Ok(j)
}
fn axis_to_wire(j: &mut Json) {
    for key in ["scale", "offset"] {
        if j[key].is_array() {
            let value = j
                .as_object_mut()
                .expect("adapter leaves serialize as objects")
                .remove(key)
                .expect("array axis field is present");
            j[format!("axis_{key}")] = value;
        }
    }
}
fn axis_from_wire(j: &mut Json) {
    for key in ["scale", "offset"] {
        if let Some(v) = j
            .as_object_mut()
            .and_then(|m| m.remove(&format!("axis_{key}")))
        {
            j[key] = v;
        }
    }
}
pub(crate) fn state_part(s: &State) -> Result<Json> {
    let mut j = leaf_json(s)?;
    for key in ["pad_to", "dtype", "reshape", "container", "clip"] {
        j.as_object_mut()
            .expect("adapter leaves serialize as objects")
            .remove(key);
    }
    Ok(j)
}
pub(crate) fn state_json(s: &State) -> Result<Json> {
    Ok(
        json!({"components":[state_part(s)?],"pad_to":s.pad_to,"dtype":s.dtype,"reshape":s.reshape,"container":s.container,"clip":s.clip}),
    )
}
pub(crate) fn concat_json(s: &Concat) -> Result<Json> {
    Ok(
        json!({"components":s.parts.iter().map(|p|match p {ConcatPart::Role(s)=>Ok(Json::String(s.clone())),ConcatPart::State(s)=>state_part(s),ConcatPart::Constant(c)=>leaf_json(c)}).collect::<Result<Vec<_>>>()?,"pad_to":s.pad_to,"dtype":s.dtype,"reshape":s.reshape,"container":s.container,"clip":s.clip}),
    )
}
impl Serialize for ConcatPart {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Role(v) => v.serialize(s),
            Self::State(v) => v.serialize(s),
            Self::Constant(v) => v.serialize(s),
        }
    }
}
impl<'de> Deserialize<'de> for ConcatPart {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let j = Json::deserialize(d)?;
        if let Some(s) = j.as_str() {
            return Ok(Self::Role(s.into()));
        }
        if j.get("role").is_some() {
            serde_json::from_value(j)
                .map(Self::State)
                .map_err(serde::de::Error::custom)
        } else {
            serde_json::from_value(j)
                .map(Self::Constant)
                .map_err(serde::de::Error::custom)
        }
    }
}
pub(crate) fn action_json(a: &Action) -> Result<Json> {
    Ok(
        json!({"components":a.components.iter().map(leaf_json).collect::<Result<Vec<_>>>()?,"clip":a.clip}),
    )
}
fn decode<T: serde::de::DeserializeOwned>(mut j: Json) -> Result<T> {
    axis_from_wire(&mut j);
    serde_json::from_value(j).map_err(err)
}
pub(crate) fn action_from(j: Json) -> Result<Action> {
    Ok(Action {
        components: j["components"]
            .as_array()
            .ok_or_else(|| err("action components missing"))?
            .iter()
            .cloned()
            .map(decode)
            .collect::<Result<_>>()?,
        clip: serde_json::from_value(j["clip"].clone()).map_err(err)?,
    })
}
pub(crate) fn input_from(j: Json) -> Result<InputNode> {
    if let Some(v) = j.as_array() {
        return Ok(InputNode::tuple(
            v.iter()
                .cloned()
                .map(input_from)
                .collect::<Result<Vec<_>>>()?,
        ));
    }
    let mut m = j
        .as_object()
        .ok_or_else(|| err("expected input tree"))?
        .clone();
    let Some(kind) = m.remove("type") else {
        return Ok(InputNode::Dict(
            m.into_iter()
                .map(|(k, v)| Ok((k, input_from(v)?)))
                .collect::<Result<_>>()?,
        ));
    };
    let j = Json::Object(m);
    Ok(match kind.as_str() {
        Some("image") => decode::<Image>(j)?.into(),
        Some("text") => decode::<Text>(j)?.into(),
        Some("custom") => {
            let entry = j["transform"]
                .as_str()
                .ok_or_else(|| err("missing custom transform"))?;
            if entry.starts_with("host:") {
                return Err(err("custom input carries a host-placeholder transform"));
            }
            Custom {
                transform: None,
                entrypoint: Some(entry.into()),
            }
            .into()
        }
        Some("state") => {
            let parts = j["components"]
                .as_array()
                .ok_or_else(|| err("state components missing"))?
                .iter()
                .map(|v| {
                    if let Some(s) = v.as_str() {
                        return Ok(ConcatPart::Role(s.into()));
                    }
                    if v["role"].is_null() {
                        return decode::<Constant>(v.clone()).map(ConcatPart::Constant);
                    }
                    decode::<State>(v.clone()).map(ConcatPart::State)
                })
                .collect::<Result<Vec<_>>>()?;
            let mut container = j.as_object().expect("state input is an object").clone();
            container.remove("components");
            if parts.len() == 1 && matches!(&parts[0], ConcatPart::State(_)) {
                let ConcatPart::State(mut s) = parts
                    .into_iter()
                    .next()
                    .expect("one parameterized part was checked above")
                else {
                    unreachable!()
                };
                let c: Concat = decode(Json::Object(container))?;
                s.pad_to = c.pad_to;
                s.dtype = c.dtype;
                s.reshape = c.reshape;
                s.container = c.container;
                s.clip = c.clip;
                s.into()
            } else {
                let mut c: Concat = decode(Json::Object(container))?;
                c.parts = parts;
                c.into()
            }
        }
        _ => return Err(err("unknown input kind")),
    })
}
pub(crate) fn obs_from(j: Json) -> Result<ObsNode> {
    if let Some(v) = j.as_array() {
        return Ok(ObsNode::tuple(
            v.iter()
                .cloned()
                .map(obs_from)
                .collect::<Result<Vec<_>>>()?,
        ));
    }
    let mut m = j
        .as_object()
        .ok_or_else(|| err("expected observation tree"))?
        .clone();
    let Some(kind) = m.remove("type") else {
        return Ok(ObsNode::Dict(
            m.into_iter()
                .map(|(k, v)| Ok((k, obs_from(v)?)))
                .collect::<Result<_>>()?,
        ));
    };
    let j = Json::Object(m);
    Ok(match kind.as_str() {
        Some("image") => decode::<ImageTag>(j)?.into(),
        Some("state") => decode::<StateTag>(j)?.into(),
        Some("text") => decode::<TextTag>(j)?.into(),
        Some("split") => decode::<Split>(j)?.into(),
        _ => return Err(err("unknown observation kind")),
    })
}
pub(crate) fn metadata(j: Json) -> Result<Metadata> {
    match crate::specs::json_to_value(j) {
        Value::Dict(d) => Ok(d),
        _ => Err(err("expected a metadata dictionary")),
    }
}
/// One JSON formatter across the facade and Python: sorted UTF-8 keys, compact
/// separators, serde's finite numeric spelling, and signed zero retained.
pub(crate) fn format_json(j: &Json) -> Result<String> {
    serde_json::to_string(&sorted(j.clone())).map_err(err)
}
fn sorted(j: Json) -> Json {
    match j {
        Json::Object(m) => {
            let mut pairs: Vec<_> = m.into_iter().collect();
            pairs.sort_by(|a, b| a.0.cmp(&b.0));
            Json::Object(pairs.into_iter().map(|(k, v)| (k, sorted(v))).collect())
        }
        Json::Array(v) => Json::Array(v.into_iter().map(sorted).collect()),
        v => v,
    }
}

pub(crate) fn metadata_json(m: &Metadata) -> Result<String> {
    Ok(format!(
        "{{{}}}",
        m.iter()
            .map(|(k, v)| Ok(format!(
                "{}:{}",
                serde_json::to_string(k).map_err(err)?,
                value_json(v)?
            )))
            .collect::<Result<Vec<_>>>()?
            .join(",")
    ))
}
pub(crate) fn value_json(v: &Value) -> Result<String> {
    match v {
        Value::Float(v) if !v.is_finite() => Err(err("adapter JSON numbers must be finite")),
        Value::Bytes(_) | Value::Tensor(_) => {
            Err(err("adapter JSON cannot contain bytes or tensors"))
        }
        Value::Dict(m) => metadata_json(m),
        Value::List(v) | Value::Tuple(v) => Ok(format!(
            "[{}]",
            v.iter()
                .map(value_json)
                .collect::<Result<Vec<_>>>()?
                .join(",")
        )),
        _ => serde_json::to_string(&crate::specs::value_to_json(v)?).map_err(err),
    }
}
