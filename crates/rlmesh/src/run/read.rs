//! Role-addressed readers and render-source discovery.
use crate::{EnvContract, Error, Metadata, Result, Value, adapters as a};
use a::AdapterBase;
use std::{
    fmt,
    hash::{Hash, Hasher},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
/// A role or a single-role model input leaf to extract from an observation.
#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum ReadItem {
    /// Read this role or typed leaf.
    Role(String),
    /// Read this role or typed leaf.
    Image(a::Image),
    /// Read this role or typed leaf.
    State(a::State),
    /// Read this role or typed leaf.
    Text(a::Text),
}
impl From<&str> for ReadItem {
    fn from(s: &str) -> Self {
        Self::Role(s.into())
    }
}
impl From<String> for ReadItem {
    fn from(s: String) -> Self {
        Self::Role(s)
    }
}
macro_rules! item {($($t:ident),*)=>{$(impl From<a::$t> for ReadItem{fn from(v:a::$t)->Self{Self::$t(v)}})*}}
item!(Image, State, Text);
/// A shared observation reader, resolved once against a connected contract.
#[derive(Clone)]
pub struct Reader {
    adapter: a::Adapter,
    roles: Vec<String>,
}
impl Reader {
    /// Construct a reader with its resolved adapter and roles.
    pub fn new(adapter: a::Adapter, roles: Vec<String>) -> Self {
        Self { adapter, roles }
    }
    /// Roles extracted by this reader, in declaration order.
    pub fn roles(&self) -> &[String] {
        &self.roles
    }
    /// Invoke this callable.
    pub fn call(&self, observation: &Value) -> Result<Metadata> {
        self.adapter
            .clone()
            .transform_obs(observation.clone())?
            .as_dict()
            .cloned()
    }
}
fn find(n: &a::ObsNode, role: &str) -> Option<a::InputNode> {
    match n {
        a::ObsNode::Dict(v) => v.iter().find_map(|(_, n)| find(n, role)),
        a::ObsNode::Tuple(v) => v.iter().find_map(|n| find(n, role)),
        a::ObsNode::Leaf(l) => match l {
            a::ObsLeaf::ImageTag(t) if t.role == role => Some(
                a::Image {
                    layout: t.layout,
                    ..a::Image::new(role)
                }
                .into(),
            ),
            a::ObsLeaf::StateTag(t) if t.role == role => Some(a::State::new(role).into()),
            a::ObsLeaf::TextTag(t) if t.role == role => Some(a::Text::new(role).into()),
            a::ObsLeaf::Split(s) if s.fields.iter().any(|f| f.role.as_deref() == Some(role)) => {
                Some(a::State::new(role).into())
            }
            _ => None,
        },
    }
}
pub(crate) fn reader(contract: &EnvContract, items: &[ReadItem]) -> Result<Reader> {
    if items.is_empty() {
        return Err(Error::Type("reader() needs at least one item".into()));
    }
    let tags = a::EnvTags::from_metadata(contract.metadata().unwrap_or(&Metadata::new()))?
        .ok_or_else(|| {
            Error::adapter_resolution(
                "the env publishes no adapter tags, so there are no roles to read",
            )
        })?;
    let mut pairs = Vec::new();
    let mut roles = Vec::new();
    for item in items {
        let (role, leaf) = match item {
            ReadItem::Role(r) => (
                r.clone(),
                find(tags.observation(), r).ok_or_else(|| {
                    Error::adapter_resolution(format!(
                        "the env declares no observation role {r:?} to read"
                    ))
                })?,
            ),
            ReadItem::Image(i) => (i.role.clone(), i.clone().into()),
            ReadItem::State(s) => (s.role.clone(), s.clone().into()),
            ReadItem::Text(t) => (t.role.clone(), t.clone().into()),
        };
        if roles.contains(&role) {
            return Err(Error::adapter_resolution("a role is read more than once"));
        }
        roles.push(role.clone());
        pairs.push((role, leaf));
    }
    let mut output = tags.action().clone();
    output.clip = None;
    for c in &mut output.components {
        c.clip = false;
    }
    let spec = a::ModelSpec::new(a::InputNode::dict(pairs), output).map_err(a::resolution_error)?;
    Ok(Reader::new(
        a::resolve_from_contract(contract, &spec, Default::default())?,
        roles,
    ))
}
pub(crate) fn image_roles(contract: Option<&EnvContract>) -> Result<Vec<String>> {
    let Some(c) = contract else {
        return Ok(Vec::new());
    };
    Ok(
        a::EnvTags::from_metadata(c.metadata().unwrap_or(&Metadata::new()))?
            .map_or_else(Vec::new, |t| t.observation_roles().images),
    )
}
/// An observation-bound read callback carried by `StepEvent`.
pub type ReadFn = Arc<dyn Fn(&ReadItem) -> Result<Value> + Send + Sync>;
pub(crate) fn read_fn(context: Arc<dyn super::RunContext>, observation: Value) -> ReadFn {
    Arc::new(move |item| context.read(&observation, item))
}
static NEXT_RENDER: AtomicU64 = AtomicU64::new(1);
/// A borrowed render callable with construction identity.
pub struct RenderFn<'a> {
    id: u64,
    call: Box<dyn FnMut() -> Result<Value> + 'a>,
}
impl<'a> RenderFn<'a> {
    /// Wrap a render callback with a unique identity.
    pub fn new(f: impl FnMut() -> Result<Value> + 'a) -> Self {
        Self {
            id: NEXT_RENDER.fetch_add(1, Ordering::Relaxed),
            call: Box::new(f),
        }
    }
    /// Invoke this callable.
    pub fn call(&mut self) -> Result<Value> {
        (self.call)()
    }
}
impl PartialEq for RenderFn<'_> {
    fn eq(&self, o: &Self) -> bool {
        self.id == o.id
    }
}
impl Eq for RenderFn<'_> {}
impl Hash for RenderFn<'_> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.id.hash(h)
    }
}
impl fmt::Debug for RenderFn<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<function>")
    }
}
/// Image roles and the render source reachable through one env or context.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct FrameSources<'a> {
    /// Image roles reachable through this environment, in declaration order.
    pub roles: Vec<String>,
    /// The render source's label, when a render source is available.
    pub render_label: Option<String>,
    /// The render callback, when available.
    pub render: Option<RenderFn<'a>>,
}
impl FrameSources<'_> {
    /// Every source label, with render first.
    pub fn labels(&self) -> Vec<String> {
        self.render_label
            .iter()
            .cloned()
            .chain(self.roles.clone())
            .collect()
    }
    /// The first image role, or the render label when there is none.
    pub fn default_label(&self) -> Option<String> {
        self.roles
            .first()
            .cloned()
            .or_else(|| self.render_label.clone())
    }
}
impl fmt::Display for FrameSources<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::adapters::repr::Repr;
        write!(
            f,
            "FrameSources(roles={}, render_label={}, render={})",
            self.roles.repr(),
            self.render_label.repr(),
            self.render
                .as_ref()
                .map_or_else(|| "None".into(), |r| format!("{r:?}"))
        )
    }
}
pub(crate) fn render_label(contract: &EnvContract, roles: &[String]) -> Option<String> {
    contract
        .render_mode()
        .filter(|m| m.to_ascii_lowercase().contains("rgb"))
        .map(|_| {
            if roles.iter().any(|r| r == "render") {
                "render()".into()
            } else {
                "render".into()
            }
        })
}

struct Dispatch<'a> {
    id: u64,
    render: Option<&'a mut (dyn FnMut() -> Result<Value> + 'a)>,
}
impl<'a, 'b: 'a> scoped_tls_hkt::ReborrowMut<'a> for Dispatch<'b> {
    type Result = Dispatch<'a>;
    fn reborrow_mut(&'a mut self) -> Self::Result {
        Dispatch {
            id: self.id,
            render: match &mut self.render {
                Some(r) => Some(&mut **r),
                None => None,
            },
        }
    }
}
scoped_tls_hkt::scoped_thread_local!(static mut DISPATCH: for<'a> Dispatch<'a>);
pub(crate) fn dispatch<R>(
    id: u64,
    env: Option<&mut super::drive::LocalEnv<'_>>,
    call: impl FnOnce() -> R,
) -> R {
    match env {
        Some(env) => {
            let mut render = || match env {
                super::drive::LocalEnv::Env(e) => e.render().map(|v| v.unwrap_or(Value::None)),
                super::drive::LocalEnv::Vector(e) => e.render().map(|v| v.unwrap_or(Value::None)),
            };
            DISPATCH.set(
                Dispatch {
                    id,
                    render: Some(&mut render),
                },
                call,
            )
        }
        None => DISPATCH.set(Dispatch { id, render: None }, call),
    }
}
pub(crate) fn has_local_render(id: u64) -> bool {
    DISPATCH.is_set() && DISPATCH.with(|s| s.id == id && s.render.is_some())
}
pub(crate) fn local_render(id: u64) -> Result<Value> {
    if !has_local_render(id) {
        return Err(Error::Runtime(
            "local render is reachable only on the thread dispatching a hook".into(),
        ));
    }
    DISPATCH.with(|s| {
        s.render
            .expect("the local renderer is reachable in this dispatch")()
    })
}
