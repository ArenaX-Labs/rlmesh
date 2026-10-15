use super::*;
use crate::{Error, Result};
fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::invalid_argument(message))
    }
}
fn labels(v: &Option<Vec<String>>) -> Result<()> {
    if let Some(v) = v {
        require(!v.is_empty(), "labels must not be empty")?;
        let mut seen = std::collections::HashSet::new();
        require(
            v.iter().all(|s| seen.insert(s)),
            "labels must be distinct strings",
        )?
    }
    Ok(())
}
fn axes(v: &Option<AxisValue>) -> Result<()> {
    if let Some(AxisValue::PerAxis(v)) = v {
        require(!v.is_empty(), "per-axis values must not be empty")?
    }
    Ok(())
}
fn accept<E: Vocabulary>(_v: &Option<Vec<Tolerant<E>>>) -> Result<()> {
    Ok(())
}
pub(crate) fn custom(c: &CustomEncoding) -> Result<()> {
    require(
        c.from_base.is_some() || c.to_base.is_some(),
        "CustomEncoding needs at least one of from_base or to_base",
    )?;
    require(
        !matches!(
            (&c.from_base, &c.to_base),
            (
                Some(EncodingArm::Entrypoint(_)),
                Some(EncodingArm::Transform(_))
            ) | (
                Some(EncodingArm::Transform(_)),
                Some(EncodingArm::Entrypoint(_))
            )
        ),
        "CustomEncoding arms must agree: both callables or both entrypoints",
    )
}
fn state(s: &State) -> Result<()> {
    require(
        s.dim.is_none() || s.index.is_none(),
        "State: set dim or index, not both",
    )?;
    require(
        s.fill == 0.0 || s.optional || s.source == StateSource::Action,
        "State: fill applies only to an optional part",
    )?;
    if s.source == StateSource::Action {
        require(
            s.encoding.is_none()
                && s.index.is_none()
                && s.range.is_none()
                && !s.optional
                && s.post_rotate.is_none()
                && s.scale.is_none()
                && s.offset.is_none()
                && s.frame.is_none()
                && s.provenance.is_none(),
            "State: source='action' carries only part/labels/dim/fill",
        )?
    }
    if s.post_rotate.is_some() {
        require(
            matches!(s.encoding, Some(StateEncoding::Native(_))),
            "State: post_rotate needs a native rotation encoding",
        )?
    }
    match &s.encoding {
        Some(StateEncoding::Custom(c)) => custom(c)?,
        Some(StateEncoding::Native(v)) => accept(&Some(v.clone()))?,
        _ => {}
    }
    axes(&s.scale)?;
    axes(&s.offset)?;
    labels(&s.labels)?;
    if let Some(v) = &s.labels {
        require(s.index.is_none(), "State: set labels or index, not both")?;
        require(
            s.dim.is_none_or(|d| d as usize == v.len()),
            "State: dim disagrees with labels",
        )?
    }
    Ok(())
}
pub(crate) fn action(a: &Action) -> Result<()> {
    require(
        !a.components.is_empty(),
        "Action needs at least one actuator",
    )?;
    for c in &a.components {
        require(c.dim >= 1, "Actuator: dim must be >= 1")?;
        axes(&c.scale)?;
        axes(&c.offset)?;
        labels(&c.labels)?;
        if c.role.is_none() {
            require(
                c.encoding.is_none()
                    && c.range.is_none()
                    && !c.binary
                    && c.scale.is_none()
                    && c.offset.is_none()
                    && !c.invert
                    && c.threshold.is_none()
                    && !c.clip
                    && !c.optional
                    && c.frame.is_none()
                    && c.reference.is_none()
                    && c.part.is_none()
                    && c.labels.is_none(),
                "a role-less Actuator carries only dim and fill",
            )?;
            continue;
        }
        require(
            c.labels.as_ref().is_none_or(|l| l.len() == c.dim as usize),
            "Actuator: dim disagrees with labels",
        )?;
        require(
            c.fill == 0.0 || c.optional,
            "Actuator: fill applies only to an opaque or optional actuator",
        )?;
        require(
            !c.clip || c.range.is_some(),
            "Actuator: clip=True needs range",
        )?;
        if let Some(ActionEncoding::Custom(e)) = &c.encoding {
            custom(e)?;
            require(
                c.dim == e.width(),
                "Actuator: CustomEncoding dim must keep its base width",
            )?
        }
    }
    Ok(())
}
pub(crate) fn input(n: &InputNode) -> Result<()> {
    n.validate_keys()?;
    match n {
        InputNode::Dict(v) => {
            for (_, n) in v {
                input(n)?
            }
        }
        InputNode::Tuple(v) => {
            for n in v {
                input(n)?
            }
        }
        InputNode::Leaf(l) => match l {
            ModelLeaf::Image(i) => {
                require(
                    i.crop.is_none() || i.crop_area.is_none(),
                    "Image: pass crop or crop_area, not both",
                )?;
                for v in [i.crop, i.crop_area].into_iter().flatten() {
                    require(
                        v > 0.0 && v <= 1.0,
                        "Image crop must be a fraction in (0, 1]",
                    )?
                }
                require(
                    i.jpeg_quality.is_none_or(|q| (1..=100).contains(&q)),
                    "Image jpeg_quality must be between 1 and 100",
                )?;
                require(
                    i.render
                        .is_none_or(|(h, w)| (1..=4096).contains(&h) && (1..=4096).contains(&w)),
                    "Image render axes must be between 1 and 4096",
                )?;
                require(
                    i.fill.is_none() || i.optional,
                    "Image fill only applies to an optional camera",
                )?;
                if let Normalize::Range(a, b) = i.normalize {
                    require(
                        a.is_finite() && b.is_finite() && a <= b,
                        "Image normalize range must be finite and ordered",
                    )?
                }
            }
            ModelLeaf::State(s) => state(s)?,
            ModelLeaf::Concat(c) => {
                require(!c.parts.is_empty(), "Concat needs at least one part")?;
                require(
                    c.parts
                        .iter()
                        .any(|p| !matches!(p, ConcatPart::Constant(_))),
                    "a state input of only constant parts reads nothing from the env",
                )?;
                for p in &c.parts {
                    if let ConcatPart::State(s) = p {
                        state(s)?;
                        require(
                            s.pad_to.is_none()
                                && s.dtype.as_str() == "float32"
                                && s.reshape.is_none()
                                && s.container == StateContainer::Array
                                && s.clip.is_none(),
                            "a State used as a Concat part must keep its container fields at defaults",
                        )?
                    }
                }
            }
            ModelLeaf::Custom(c) => require(
                c.transform.is_some() != c.entrypoint.is_some(),
                "Custom: set exactly one of transform or entrypoint",
            )?,
            ModelLeaf::Text(_) => {}
        },
    }
    Ok(())
}
pub(crate) fn obs(n: &ObsNode) -> Result<()> {
    n.validate_keys()?;
    match n {
        ObsNode::Dict(v) => {
            for (_, n) in v {
                obs(n)?
            }
        }
        ObsNode::Tuple(v) => {
            for n in v {
                obs(n)?
            }
        }
        ObsNode::Leaf(l) => match l {
            ObsLeaf::StateTag(s) => {
                accept(&s.encoding)?;
                labels(&s.labels)?
            }
            ObsLeaf::Split(s) => {
                require(!s.fields.is_empty(), "Split needs at least one Field")?;
                for f in &s.fields {
                    require(f.dim >= 1, "Field: dim must be >= 1")?;
                    accept(&f.encoding)?;
                    labels(&f.labels)?;
                    require(
                        f.labels.as_ref().is_none_or(|l| l.len() == f.dim as usize),
                        "Field: dim disagrees with labels",
                    )?
                }
            }
            _ => {}
        },
    }
    Ok(())
}
pub(crate) fn identities(n: &ObsNode, out: &mut Vec<(u8, String, Option<String>, Option<String>)>) {
    match n {
        ObsNode::Dict(v) => {
            for (_, n) in v {
                identities(n, out)
            }
        }
        ObsNode::Tuple(v) => {
            for n in v {
                identities(n, out)
            }
        }
        ObsNode::Leaf(l) => match l {
            ObsLeaf::ImageTag(t) => out.push((0, t.role.clone(), t.part.clone(), None)),
            ObsLeaf::TextTag(t) => out.push((2, t.role.clone(), None, None)),
            ObsLeaf::StateTag(t) => out.push((
                1,
                t.role.clone(),
                t.part.clone(),
                t.provenance.as_ref().map(ToString::to_string),
            )),
            ObsLeaf::Split(s) => {
                for f in &s.fields {
                    if let Some(r) = &f.role {
                        out.push((
                            1,
                            r.clone(),
                            f.part.clone(),
                            f.provenance.as_ref().map(ToString::to_string),
                        ))
                    }
                }
            }
        },
    }
}
