use super::*;
use crate::{Error, Result};
use serde_json::Value as Json;
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum ModelLeaf {
    Image(Image),
    State(State),
    Concat(Concat),
    Text(Text),
    Custom(Custom),
}
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum InputNode {
    Leaf(ModelLeaf),
    Dict(Vec<(String, InputNode)>),
    Tuple(Vec<InputNode>),
}
impl InputNode {
    pub fn dict<K: Into<String>>(pairs: impl IntoIterator<Item = (K, Self)>) -> Self {
        Self::Dict(pairs.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }
    pub fn tuple(items: impl IntoIterator<Item = Self>) -> Self {
        Self::Tuple(items.into_iter().collect())
    }
    pub(crate) fn validate_keys(&self) -> Result<()> {
        match self {
            Self::Leaf(_) => Ok(()),
            Self::Tuple(v) => {
                for n in v {
                    n.validate_keys()?
                }
                Ok(())
            }
            Self::Dict(v) => {
                let mut seen = std::collections::HashSet::new();
                for (k, n) in v {
                    if k == "type" || !seen.insert(k) {
                        return Err(Error::invalid_argument(format!(
                            "duplicate or reserved Dict key: {k}"
                        )));
                    }
                    n.validate_keys()?
                }
                Ok(())
            }
        }
    }
}
impl PartialEq for InputNode {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Leaf(a), Self::Leaf(b)) => a == b,
            (Self::Tuple(a), Self::Tuple(b)) => a == b,
            (Self::Dict(a), Self::Dict(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .all(|(k, v)| b.iter().any(|(bk, bv)| k == bk && v == bv))
            }
            _ => false,
        }
    }
}
impl From<ModelLeaf> for InputNode {
    fn from(v: ModelLeaf) -> Self {
        Self::Leaf(v)
    }
}
impl fmt::Display for InputNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use super::repr::Repr;
        match self {
            Self::Leaf(v) => write!(f, "{v}"),
            Self::Tuple(v) => f.write_str(&v.repr()),
            Self::Dict(v) => write!(
                f,
                "{{{}}}",
                v.iter()
                    .map(|(k, v)| format!("{}: {v}", k.repr()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}
impl fmt::Display for ModelLeaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Image(v) => write!(f, "{v}"),
            Self::State(v) => write!(f, "{v}"),
            Self::Concat(v) => write!(f, "{v}"),
            Self::Text(v) => write!(f, "{v}"),
            Self::Custom(v) => write!(f, "{v}"),
        }
    }
}
impl From<Image> for InputNode {
    fn from(v: Image) -> Self {
        Self::Leaf(ModelLeaf::Image(v))
    }
}
impl From<State> for InputNode {
    fn from(v: State) -> Self {
        Self::Leaf(ModelLeaf::State(v))
    }
}
impl From<Concat> for InputNode {
    fn from(v: Concat) -> Self {
        Self::Leaf(ModelLeaf::Concat(v))
    }
}
impl From<Text> for InputNode {
    fn from(v: Text) -> Self {
        Self::Leaf(ModelLeaf::Text(v))
    }
}
impl From<Custom> for InputNode {
    fn from(v: Custom) -> Self {
        Self::Leaf(ModelLeaf::Custom(v))
    }
}
impl InputNode {
    pub(crate) fn wire_json(&self) -> Result<Json> {
        match self {
            Self::Dict(v) => Ok(Json::Object(
                v.iter()
                    .map(|(k, v)| Ok((k.clone(), v.wire_json()?)))
                    .collect::<Result<_>>()?,
            )),
            Self::Tuple(v) => Ok(Json::Array(
                v.iter().map(Self::wire_json).collect::<Result<_>>()?,
            )),
            Self::Leaf(v) => match v {
                ModelLeaf::Image(v) => {
                    let mut j = codec::leaf_json(v)?;
                    j["type"] = Json::from("image");
                    Ok(j)
                }
                ModelLeaf::State(v) => {
                    let mut j = codec::state_json(v)?;
                    j["type"] = Json::from("state");
                    Ok(j)
                }
                ModelLeaf::Concat(v) => {
                    let mut j = codec::concat_json(v)?;
                    j["type"] = Json::from("state");
                    Ok(j)
                }
                ModelLeaf::Text(v) => {
                    let mut j = codec::leaf_json(v)?;
                    j["type"] = Json::from("text");
                    Ok(j)
                }
                ModelLeaf::Custom(v) => {
                    let mut j = serde_json::json!({"transform":v.entrypoint.as_deref().unwrap_or("host:local")});
                    j["type"] = Json::from("custom");
                    Ok(j)
                }
            },
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum ObsLeaf {
    ImageTag(ImageTag),
    StateTag(StateTag),
    TextTag(TextTag),
    Split(Split),
}
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum ObsNode {
    Leaf(ObsLeaf),
    Dict(Vec<(String, ObsNode)>),
    Tuple(Vec<ObsNode>),
}
impl ObsNode {
    pub fn dict<K: Into<String>>(pairs: impl IntoIterator<Item = (K, Self)>) -> Self {
        Self::Dict(pairs.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }
    pub fn tuple(items: impl IntoIterator<Item = Self>) -> Self {
        Self::Tuple(items.into_iter().collect())
    }
    pub(crate) fn validate_keys(&self) -> Result<()> {
        match self {
            Self::Leaf(_) => Ok(()),
            Self::Tuple(v) => {
                for n in v {
                    n.validate_keys()?
                }
                Ok(())
            }
            Self::Dict(v) => {
                let mut seen = std::collections::HashSet::new();
                for (k, n) in v {
                    if k == "type" || !seen.insert(k) {
                        return Err(Error::invalid_argument(format!(
                            "duplicate or reserved Dict key: {k}"
                        )));
                    }
                    n.validate_keys()?
                }
                Ok(())
            }
        }
    }
}
impl PartialEq for ObsNode {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Leaf(a), Self::Leaf(b)) => a == b,
            (Self::Tuple(a), Self::Tuple(b)) => a == b,
            (Self::Dict(a), Self::Dict(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .all(|(k, v)| b.iter().any(|(bk, bv)| k == bk && v == bv))
            }
            _ => false,
        }
    }
}
impl From<ObsLeaf> for ObsNode {
    fn from(v: ObsLeaf) -> Self {
        Self::Leaf(v)
    }
}
impl fmt::Display for ObsNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use super::repr::Repr;
        match self {
            Self::Leaf(v) => write!(f, "{v}"),
            Self::Tuple(v) => f.write_str(&v.repr()),
            Self::Dict(v) => write!(
                f,
                "{{{}}}",
                v.iter()
                    .map(|(k, v)| format!("{}: {v}", k.repr()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}
impl fmt::Display for ObsLeaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ImageTag(v) => write!(f, "{v}"),
            Self::StateTag(v) => write!(f, "{v}"),
            Self::TextTag(v) => write!(f, "{v}"),
            Self::Split(v) => write!(f, "{v}"),
        }
    }
}
impl From<ImageTag> for ObsNode {
    fn from(v: ImageTag) -> Self {
        Self::Leaf(ObsLeaf::ImageTag(v))
    }
}
impl From<StateTag> for ObsNode {
    fn from(v: StateTag) -> Self {
        Self::Leaf(ObsLeaf::StateTag(v))
    }
}
impl From<TextTag> for ObsNode {
    fn from(v: TextTag) -> Self {
        Self::Leaf(ObsLeaf::TextTag(v))
    }
}
impl From<Split> for ObsNode {
    fn from(v: Split) -> Self {
        Self::Leaf(ObsLeaf::Split(v))
    }
}
impl ObsNode {
    pub(crate) fn wire_json(&self) -> Result<Json> {
        match self {
            Self::Dict(v) => Ok(Json::Object(
                v.iter()
                    .map(|(k, v)| Ok((k.clone(), v.wire_json()?)))
                    .collect::<Result<_>>()?,
            )),
            Self::Tuple(v) => Ok(Json::Array(
                v.iter().map(Self::wire_json).collect::<Result<_>>()?,
            )),
            Self::Leaf(v) => match v {
                ObsLeaf::ImageTag(v) => {
                    let mut j = codec::leaf_json(v)?;
                    j["type"] = Json::from("image");
                    Ok(j)
                }
                ObsLeaf::StateTag(v) => {
                    let mut j = codec::leaf_json(v)?;
                    j["type"] = Json::from("state");
                    Ok(j)
                }
                ObsLeaf::TextTag(v) => {
                    let mut j = codec::leaf_json(v)?;
                    j["type"] = Json::from("text");
                    Ok(j)
                }
                ObsLeaf::Split(v) => {
                    let mut j = codec::leaf_json(v)?;
                    j["type"] = Json::from("split");
                    Ok(j)
                }
            },
        }
    }
}
