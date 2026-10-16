use super::*;
use std::fmt;
pub(crate) trait Repr {
    fn repr(&self) -> String;
}
pub(crate) fn dataclass(
    f: &mut fmt::Formatter<'_>,
    name: &str,
    fields: &[(&str, String)],
) -> fmt::Result {
    write!(
        f,
        "{name}({})",
        fields
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}
pub(crate) fn quote(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::from(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => {
                out.push('\\');
                out.push(c)
            }
            c if c.is_control() => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(q);
    out
}
impl Repr for String {
    fn repr(&self) -> String {
        quote(self)
    }
}
impl Repr for bool {
    fn repr(&self) -> String {
        if *self { "True" } else { "False" }.into()
    }
}
macro_rules! integers {($($t:ty),*)=>{$(impl Repr for $t {fn repr(&self)->String{self.to_string()}})*}}
integers!(u8, u32, i32, i64, u64);
impl Repr for f64 {
    fn repr(&self) -> String {
        let s = format!("{self:?}");
        if let Some((m, e)) = s.split_once('e') {
            let e: i32 = e.parse().unwrap_or(0);
            format!("{m}e{e:+03}")
        } else {
            s
        }
    }
}
impl<T: Repr> Repr for Option<T> {
    fn repr(&self) -> String {
        self.as_ref().map_or_else(|| "None".into(), Repr::repr)
    }
}
impl<T: Repr> Repr for Vec<T> {
    fn repr(&self) -> String {
        let body = self.iter().map(Repr::repr).collect::<Vec<_>>().join(", ");
        format!("({body}{})", if self.len() == 1 { "," } else { "" })
    }
}
impl<A: Repr, B: Repr> Repr for (A, B) {
    fn repr(&self) -> String {
        format!("({}, {})", self.0.repr(), self.1.repr())
    }
}
impl<E: Vocabulary> Repr for Tolerant<E> {
    fn repr(&self) -> String {
        quote(self.as_str())
    }
}
pub(crate) fn accept<E: Vocabulary>(v: &Option<Vec<Tolerant<E>>>) -> String {
    match v {
        Some(v) if v.len() == 1 => v[0].repr(),
        _ => v.repr(),
    }
}
macro_rules! display_repr {($($t:ty),*)=>{$(impl Repr for $t {fn repr(&self)->String{self.to_string()}})*}}
display_repr!(
    Image,
    State,
    Concat,
    Constant,
    Text,
    Actuator,
    Action,
    ImageTag,
    StateTag,
    TextTag,
    Field,
    Split,
    ObservationRoles,
    Rotation,
    Custom,
    CustomEncoding,
    InputNode,
    ObsNode,
    ConcatPart,
    AxisValue,
    Normalize,
    StateEncoding,
    ActionEncoding,
    EncodingArm
);
macro_rules! vocab_repr {($($t:ty),*)=>{$(impl Repr for $t {fn repr(&self)->String{quote(self.as_str())}})*}}
vocab_repr!(
    ImageLayout,
    FitMode,
    Frame,
    Reference,
    Provenance,
    Resample,
    RotationEncoding,
    StackPad,
    CropMode,
    ChannelOrder,
    StateContainer,
    TextContainer,
    StateSource
);
impl Repr for ObsTransform {
    fn repr(&self) -> String {
        format!("{self:?}")
    }
}
impl Repr for RotationTransform {
    fn repr(&self) -> String {
        format!("{self:?}")
    }
}
macro_rules! repr_fields {($name:ident,$($field:ident),*)=>{impl fmt::Display for $name {fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result {dataclass(f,stringify!($name),&[$((stringify!($field),self.$field.repr())),*])}}}}
repr_fields!(Rotation, encoding, value);
repr_fields!(Custom, transform, entrypoint);
repr_fields!(CustomEncoding, base, from_base, to_base, name);
impl fmt::Display for EncodingArm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&match self {
            Self::Entrypoint(s) => s.repr(),
            Self::Transform(t) => t.repr(),
        })
    }
}
impl fmt::Display for AxisValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&match self {
            Self::Scalar(x) => x.repr(),
            Self::PerAxis(xs) => xs.repr(),
        })
    }
}
impl fmt::Display for Normalize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&match self {
            Self::Off => "False".into(),
            Self::Unit => "True".into(),
            Self::Range(a, b) => (*a, *b).repr(),
        })
    }
}
impl fmt::Display for StateEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Custom(c) => write!(f, "{c}"),
            Self::Native(v) => f.write_str(&accept(&Some(v.clone()))),
        }
    }
}
impl fmt::Display for ActionEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Custom(c) => write!(f, "{c}"),
            Self::Native(v) => f.write_str(&v.repr()),
        }
    }
}
impl fmt::Display for ConcatPart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Role(s) => f.write_str(&s.repr()),
            Self::State(s) => write!(f, "{s}"),
            Self::Constant(c) => write!(f, "{c}"),
        }
    }
}
