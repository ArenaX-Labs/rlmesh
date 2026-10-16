use super::core::{
    ApplyError, CustomTransform, EncodingTransform, FrameBuffers, NodePath, ObsPlan, PathSeg,
};
use super::*;
use crate::ext::TensorExt;
use crate::model::bridge::{from_adapter, to_adapter};
use crate::{Error, Result, Value, spaces::SpaceSpec};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, OnceLock, RwLock},
};

static ENTRYPOINTS: OnceLock<RwLock<BTreeMap<String, ObsTransform>>> = OnceLock::new();
pub(crate) fn register(name: &str, transform: ObsTransform) -> Result<()> {
    if name.is_empty() {
        return Err(Error::invalid_argument("entrypoint name must not be empty"));
    }
    ENTRYPOINTS
        .get_or_init(Default::default)
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .insert(name.into(), transform);
    Ok(())
}
fn callback_error(error: Error) -> ApplyError {
    ApplyError {
        message: error.to_string(),
        source: Some(Arc::new(error)),
    }
}
#[derive(Clone, Default)]
pub(crate) struct Customs(BTreeMap<String, ObsTransform>);
impl CustomTransform for Customs {
    fn apply(
        &self,
        _key: &str,
        entrypoint: &str,
        raw: &BTreeMap<String, core::Value>,
    ) -> std::result::Result<Option<core::Value>, ApplyError> {
        let run = || {
            let f = self
                .0
                .get(entrypoint)
                .ok_or_else(|| Error::invalid_argument("missing custom transform"))?;
            let raw = from_adapter(core::Value::Map(raw.clone()))?;
            to_adapter(&f.call(&raw)?).map(Some)
        };
        run().map_err(callback_error)
    }
}
#[derive(Clone)]
struct ObsShim {
    path: NodePath,
    offset: usize,
    width: usize,
    total: usize,
    dtype: crate::DType,
    transform: RotationTransform,
}
#[derive(Clone)]
struct ActShim {
    offset: usize,
    width: usize,
    transform: RotationTransform,
}
#[derive(Clone, Default)]
pub(crate) struct Encodings {
    obs: Vec<ObsShim>,
    action: Vec<ActShim>,
}
fn numeric(value: &Value) -> Result<Vec<f64>> {
    match value {
        Value::Tensor(t) => {
            if t.ndim() != 1 {
                return Err(Error::invalid_argument(
                    "custom encoding requires a flat 1-D numeric value",
                ));
            }
            rlmesh_spaces::scalar::decode_scalars(&t.tobytes(), t.dtype().to_core())
                .map(|scalars| {
                    scalars
                        .into_iter()
                        .map(|s| s.to_f64(t.dtype().to_core()))
                        .collect()
                })
                .map_err(|e| Error::invalid_argument(e.to_string()))
        }
        Value::List(v) | Value::Tuple(v) => v
            .iter()
            .map(|v| match v {
                Value::Int(x) => Ok(*x as f64),
                Value::Float(x) => Ok(*x),
                Value::Bool(b) => Ok(f64::from(*b)),
                _ => Err(Error::invalid_argument(
                    "custom encoding requires numeric values",
                )),
            })
            .collect(),
        _ => Err(Error::invalid_argument(
            "custom encoding requires a flat 1-D numeric value",
        )),
    }
}
fn vector(v: &[f64], dtype: crate::DType) -> Result<Value> {
    // Casting through the adapter's numeric pipeline keeps all twelve dtypes
    // identical to declarative state output (including integer narrowing).
    let bytes = rlmesh_spaces::scalar::encode_scalars(
        &v.iter()
            .copied()
            .map(rlmesh_spaces::scalar::Scalar::Float)
            .collect::<Vec<_>>(),
        dtype.to_core(),
    )
    .map_err(|e| Error::invalid_argument(e.to_string()))?;
    crate::Tensor::from_vec(bytes, &[v.len()], dtype).map(Value::Tensor)
}
fn tree_mut<'a>(v: &'a mut core::Value, path: &[PathSeg]) -> Result<&'a mut core::Value> {
    if let Some((head, tail)) = path.split_first() {
        match (head, v) {
            (PathSeg::Key(k), core::Value::Map(m)) => tree_mut(
                m.get_mut(k)
                    .ok_or_else(|| Error::invalid_argument("missing custom encoding placement"))?,
                tail,
            ),
            (PathSeg::Index(i), core::Value::List(v)) => tree_mut(
                v.get_mut(*i)
                    .ok_or_else(|| Error::invalid_argument("missing custom encoding placement"))?,
                tail,
            ),
            _ => Err(Error::invalid_argument("invalid custom encoding placement")),
        }
    } else {
        Ok(v)
    }
}
fn converted(transform: &RotationTransform, input: Value, width: usize) -> Result<Vec<f64>> {
    let out = numeric(&transform.call(&input)?)?;
    if out.len() != width {
        return Err(Error::invalid_argument(format!(
            "custom encoding must return a flat width-{width} vector"
        )));
    }
    Ok(out)
}
impl EncodingTransform for Encodings {
    fn repack_obs(&self, payload: &mut core::Value) -> std::result::Result<(), ApplyError> {
        let mut run = || {
            for s in &self.obs {
                let value = tree_mut(payload, &s.path.0)?;
                let mut data = numeric(&from_adapter(value.clone())?)?;
                if data.len() != s.total {
                    return Err(Error::invalid_argument(
                        "custom encoding expected the resolved flat state width",
                    ));
                }
                let input = vector(&data[s.offset..s.offset + s.width], s.dtype)?;
                let out = converted(&s.transform, input, s.width)?;
                data[s.offset..s.offset + s.width].copy_from_slice(&out);
                *value = to_adapter(&vector(&data, s.dtype)?)?;
            }
            Ok(())
        };
        run().map_err(callback_error)
    }
    fn repack_action(&self, action: &mut core::Value) -> std::result::Result<(), ApplyError> {
        let mut run = || {
            if self.action.is_empty() {
                return Ok(());
            }
            if matches!(action, core::Value::Map(_)) {
                return Err(Error::Type(
                    "a CustomEncoding requires a flat array action, not a mapping".into(),
                ));
            }
            let mut data = numeric(&from_adapter(action.clone())?)?;
            for s in &self.action {
                if data.len() < s.offset + s.width {
                    return Err(Error::invalid_argument(
                        "custom action encoding slice exceeds the action width",
                    ));
                }
                let input = vector(&data[s.offset..s.offset + s.width], crate::DType::Float64)?;
                let out = converted(&s.transform, input, s.width)?;
                data[s.offset..s.offset + s.width].copy_from_slice(&out);
            }
            *action = to_adapter(&vector(&data, crate::DType::Float64)?)?;
            Ok(())
        };
        run().map_err(callback_error)
    }
}
/// Keywords for adapter resolution. Build with `..Default::default()`; adding
/// an options field is a minor change by project policy.
#[derive(Clone, Debug)]
pub struct ResolveOptions {
    /// The `trust_entrypoints` field, with Python’s meaning and default.
    pub trust_entrypoints: bool,
    /// The `check_inverse` field, with Python’s meaning and default.
    pub check_inverse: bool,
}
impl Default for ResolveOptions {
    fn default() -> Self {
        Self {
            trust_entrypoints: false,
            check_inverse: true,
        }
    }
}
impl ResolveOptions {
    /// Set whether registered custom-input entrypoints may be used.
    pub fn trust_entrypoints(mut self, v: bool) -> Self {
        self.trust_entrypoints = v;
        self
    }
    /// Set whether distinct two-arm custom encodings are inverse-checked.
    pub fn check_inverse(mut self, v: bool) -> Self {
        self.check_inverse = v;
        self
    }
}
fn arm(arm: &Option<EncodingArm>, side: &str) -> Result<RotationTransform> {
    match arm {
        Some(EncodingArm::Transform(t)) => Ok(t.clone()),
        Some(EncodingArm::Entrypoint(_)) => Err(Error::adapter_resolution(
            "custom encoding entrypoint arms cannot be executed",
        )),
        None => Err(Error::adapter_resolution(format!(
            "custom encoding needs {side}"
        ))),
    }
}
fn float64_array(value: &Value) -> Result<Value> {
    if let Value::Tensor(t) = value {
        let data = rlmesh_spaces::scalar::decode_scalars(&t.tobytes(), t.dtype().to_core())
            .map_err(|e| Error::invalid_argument(e.to_string()))?
            .into_iter()
            .map(|s| s.to_f64(t.dtype().to_core()))
            .collect::<Vec<_>>();
        let bytes = data
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>();
        return crate::Tensor::new(&bytes, t.shape(), crate::DType::Float64).map(Value::Tensor);
    }
    fn flatten(value: &Value, shape: &mut Vec<usize>, data: &mut Vec<f64>) -> Result<()> {
        match value {
            Value::List(v) | Value::Tuple(v) => {
                shape.push(v.len());
                let mut child_shape = None;
                for child in v {
                    let mut dims = Vec::new();
                    flatten(child, &mut dims, data)?;
                    if child_shape.as_ref().is_some_and(|s| s != &dims) {
                        return Err(Error::invalid_argument(
                            "inverse-check intermediate array is ragged",
                        ));
                    }
                    child_shape = Some(dims)
                }
                if let Some(dims) = child_shape {
                    shape.extend(dims)
                }
                Ok(())
            }
            Value::Int(v) => {
                data.push(*v as f64);
                Ok(())
            }
            Value::Float(v) => {
                data.push(*v);
                Ok(())
            }
            Value::Bool(v) => {
                data.push(f64::from(*v));
                Ok(())
            }
            _ => Err(Error::invalid_argument(
                "inverse-check intermediate value must be numeric",
            )),
        }
    }
    let mut shape = Vec::new();
    let mut data = Vec::new();
    flatten(value, &mut shape, &mut data)?;
    crate::Tensor::new(
        &data
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect::<Vec<_>>(),
        &shape,
        crate::DType::Float64,
    )
    .map(Value::Tensor)
}

fn check_inverse(encodings: &[CustomEncoding]) -> Result<()> {
    let mut seen = Vec::new();
    for c in encodings {
        if seen.contains(&c) {
            continue;
        }
        seen.push(c);
        let (Some(EncodingArm::Transform(from)), Some(EncodingArm::Transform(to))) =
            (&c.from_base, &c.to_base)
        else {
            continue;
        };
        let probe: Vec<f64> = match c.base {
            RotationEncoding::QuatXyzw => vec![0., 0., 0., 1.],
            RotationEncoding::QuatWxyz => vec![1., 0., 0., 0.],
            RotationEncoding::Rot6d => vec![1., 0., 0., 0., 1., 0.],
            RotationEncoding::Rot6dRowmajor => vec![1., 0., 0., 1., 0., 0.],
            RotationEncoding::AxisAngle | RotationEncoding::EulerXyz => vec![0.1, -0.2, 0.3],
            _ => continue,
        };
        let result = (|| {
            let input = vector(&probe, crate::DType::Float64)?;
            let mid = from.call(&input)?;
            let mid = float64_array(&mid)?;
            numeric(&to.call(&mid)?)
        })();
        let output = result.map_err(|e| {
            Error::adapter_resolution(format!(
                "CustomEncoding {:?} raised during the inverse self-check: {e}",
                c.name
            ))
        })?;
        if output.len() != probe.len()
            || output
                .iter()
                .zip(&probe)
                .any(|(r, p)| !(r - p).abs().le(&(1e-5 + 1e-5 * p.abs())))
        {
            return Err(Error::adapter_resolution(format!(
                "CustomEncoding {:?} failed its inverse self-check",
                c.name
            )));
        }
    }
    Ok(())
}
pub(crate) struct ResolvedParts {
    /// The `plan` field, with Python’s meaning and default.
    pub plan: core::ResolvedAdapter,
    /// The `customs` field, with Python’s meaning and default.
    pub customs: Customs,
    /// The `encodings` field, with Python’s meaning and default.
    pub encodings: Encodings,
    /// The `input` field, with Python’s meaning and default.
    pub input: InputNode,
}
fn walk<'a>(n: &'a InputNode, path: NodePath, out: &mut Vec<(NodePath, &'a ModelLeaf)>) {
    match n {
        InputNode::Leaf(l) => out.push((path, l)),
        InputNode::Dict(v) => {
            for (k, n) in v {
                walk(n, path.push_key(k), out)
            }
        }
        InputNode::Tuple(v) => {
            for (i, n) in v.iter().enumerate() {
                walk(n, path.push_index(i), out)
            }
        }
    }
}
fn core_input_mut<'a>(n: &'a mut core::InputNode, path: &[PathSeg]) -> &'a mut core::InputNode {
    if let Some((h, t)) = path.split_first() {
        match (h, n) {
            (PathSeg::Key(k), core::InputNode::Dict(m)) => core_input_mut(
                m.get_mut(k)
                    .expect("lowering retains every authored Dict placement"),
                t,
            ),
            (PathSeg::Index(i), core::InputNode::Tuple(v)) => core_input_mut(&mut v[*i], t),
            _ => unreachable!(),
        }
    } else {
        n
    }
}
pub(crate) fn resolve_parts(
    tags: &core::EnvTags,
    observation: &SpaceSpec,
    action: &SpaceSpec,
    spec: &ModelSpec,
    options: ResolveOptions,
) -> Result<ResolvedParts> {
    let run = || {
        let mut wire = spec.wire()?;
        let mut customs = Customs::default();
        let mut leaves = Vec::new();
        walk(spec.input(), NodePath::root(), &mut leaves);
        let mut all_encodings = Vec::new();
        for (path, leaf) in &leaves {
            if let ModelLeaf::Custom(c) = leaf {
                let transform = if let Some(t) = &c.transform {
                    t.clone()
                } else {
                    let name = c.entrypoint.as_deref().unwrap_or("");
                    if !options.trust_entrypoints {
                        return Err(Error::adapter_resolution(
                            "custom input entrypoint requires trust_entrypoints=True",
                        ));
                    }
                    ENTRYPOINTS
                        .get_or_init(Default::default)
                        .read()
                        .unwrap_or_else(|e| e.into_inner())
                        .get(name)
                        .cloned()
                        .ok_or_else(|| {
                            Error::adapter_resolution(format!(
                                "unregistered custom input entrypoint {name:?}"
                            ))
                        })?
                };
                let token = format!("host:{}", customs.0.len());
                customs.0.insert(token.clone(), transform);
                let core::InputNode::Leaf(core::ModelLeaf::Custom(c)) =
                    core_input_mut(&mut wire.input, &path.0)
                else {
                    unreachable!()
                };
                c.transform = token;
            }
            match leaf {
                ModelLeaf::State(s) => {
                    if let Some(StateEncoding::Custom(c)) = &s.encoding {
                        all_encodings.push(c.clone())
                    }
                }
                ModelLeaf::Concat(c) => {
                    for p in &c.parts {
                        if let ConcatPart::State(State {
                            encoding: Some(StateEncoding::Custom(c)),
                            ..
                        }) = p
                        {
                            all_encodings.push(c.clone())
                        }
                    }
                }
                _ => {}
            }
        }
        for a in &spec.output().components {
            if let Some(ActionEncoding::Custom(c)) = &a.encoding {
                all_encodings.push(c.clone())
            }
        }
        if options.check_inverse {
            check_inverse(&all_encodings)?
        }
        let plan = core::resolve(
            tags,
            &core::SpaceView::from(observation.wire()),
            &core::SpaceView::from(action.wire()),
            &wire,
            true,
        )
        .map_err(resolution_error)?;
        let mut encodings = Encodings::default();
        for (path, leaf) in leaves {
            let parts: Vec<Option<&CustomEncoding>> = match leaf {
                ModelLeaf::State(s) => vec![match &s.encoding {
                    Some(StateEncoding::Custom(c)) => Some(c),
                    _ => None,
                }],
                ModelLeaf::Concat(c) => c
                    .parts
                    .iter()
                    .map(|p| match p {
                        ConcatPart::State(State {
                            encoding: Some(StateEncoding::Custom(c)),
                            ..
                        }) => Some(c),
                        _ => None,
                    })
                    .collect(),
                _ => continue,
            };
            if !parts.iter().any(Option::is_some) {
                continue;
            }
            let state = plan
                .obs_plans
                .iter()
                .find_map(|p| match p {
                    ObsPlan::State(s) if s.placement == path => Some(s),
                    _ => None,
                })
                .ok_or_else(|| Error::adapter_resolution("custom encoding state plan missing"))?;
            if state.pad_to.is_some()
                || state.reshape.is_some()
                || state.container != core::StateContainer::Array
            {
                return Err(Error::adapter_resolution(
                    "a state with CustomEncoding must be an unpadded, unreshaped array",
                ));
            }
            let total = state.native_width.ok_or_else(|| {
                Error::adapter_resolution("custom encoding needs statically known state widths")
            })? as usize;
            let mut offset = 0;
            for (p, encoding) in state.pieces.iter().zip(parts) {
                let width = p.width.ok_or_else(|| {
                    Error::adapter_resolution("custom encoding needs known part widths")
                })? as usize;
                if let Some(c) = encoding {
                    if width != c.width() as usize {
                        return Err(Error::adapter_resolution(
                            "custom encoding part must keep its base width",
                        ));
                    }
                    encodings.obs.push(ObsShim {
                        path: path.clone(),
                        offset,
                        width,
                        total,
                        dtype: crate::DType::parse(&state.dtype)
                            .ok_or_else(|| Error::adapter_resolution("unknown state dtype"))?,
                        transform: arm(&c.from_base, "from_base")?,
                    });
                }
                offset += width;
            }
        }
        let mut offset = 0;
        for a in &spec.output().components {
            if let Some(ActionEncoding::Custom(c)) = &a.encoding {
                if a.binary {
                    return Err(Error::adapter_resolution(
                        "an actuator with CustomEncoding cannot be binary",
                    ));
                }
                encodings.action.push(ActShim {
                    offset,
                    width: a.dim as usize,
                    transform: arm(&c.to_base, "to_base")?,
                });
            }
            offset += a.dim as usize;
        }
        Ok(ResolvedParts {
            plan,
            customs,
            encodings,
            input: spec.input().clone(),
        })
    };
    run().map_err(resolution_error)
}
/// Resolve authored env tags and a model spec against single-lane spaces.
pub fn resolve(
    env_tags: &EnvTags,
    observation_space: &SpaceSpec,
    action_space: &SpaceSpec,
    model_spec: &ModelSpec,
    options: ResolveOptions,
) -> Result<Adapter> {
    let env_tags = env_tags.wire().map_err(resolution_error)?;
    core::reject_unknowns_env(&env_tags).map_err(resolution_error)?;
    let parts = resolve_parts(
        &env_tags,
        observation_space,
        action_space,
        model_spec,
        options,
    )?;
    Ok(Adapter::new(parts))
}
/// Resolve a contract through the core's tolerant tag reader.
pub fn resolve_from_contract(
    contract: &crate::EnvContract,
    model_spec: &ModelSpec,
    options: ResolveOptions,
) -> Result<Adapter> {
    let tags = contract_tags(contract)?;
    let parts = resolve_parts(
        &tags,
        contract.observation_space(),
        contract.action_space(),
        model_spec,
        options,
    )?;
    Ok(Adapter::new(parts))
}
pub(crate) fn contract_tags(c: &crate::EnvContract) -> Result<core::EnvTags> {
    let tags = c
        .metadata()
        .and_then(|m| m.get(ENV_METADATA_KEY))
        .ok_or_else(|| Error::adapter_resolution("env contract publishes no adapter tags"))?;
    serde_json::from_str(&super::codec::value_json(tags).map_err(resolution_error)?)
        .map_err(resolution_error)
}
struct AdapterState {
    parts: ResolvedParts,
    buffers: FrameBuffers,
    step: i64,
    current_step: i64,
}
/// A resolved adapter handle. Clones share state and serialize calls.
#[derive(Clone)]
pub struct Adapter(Arc<Mutex<AdapterState>>);
impl Adapter {
    fn new(parts: ResolvedParts) -> Self {
        Self(Arc::new(Mutex::new(AdapterState {
            parts,
            buffers: FrameBuffers::new(),
            step: 0,
            current_step: 0,
        })))
    }
    /// Non-fatal notes produced while resolving this adapter.
    pub fn advisories(&self) -> Vec<Advisory> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .parts
            .plan
            .advisories()
            .into_iter()
            .map(crate::adapters::Advisory::from_core)
            .collect()
    }
    /// Canonical placements requiring per-episode history.
    pub fn history_keys(&self) -> Vec<String> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .parts
            .plan
            .history_keys()
    }
    /// Placement, span and frame byte cost of each frame window.
    pub fn history_windows(&self) -> Vec<(String, u64, u64)> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .parts
            .plan
            .history_windows()
            .into_iter()
            .map(|w| (w.key, u64::from(w.span), w.frame_bytes))
            .collect()
    }
}
fn raw_map(raw: Value) -> Result<BTreeMap<String, core::Value>> {
    match to_adapter(&raw)? {
        core::Value::Map(m) => Ok(m),
        v => Ok(BTreeMap::from([("<obs>".into(), v)])),
    }
}
impl AdapterBase for Adapter {
    /// Python’s `AdapterBase.transform_obs` operation.
    fn transform_obs(&mut self, raw: Value) -> Result<Value> {
        let mut s = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let AdapterState {
            parts,
            buffers,
            step,
            current_step,
        } = &mut *s;
        let tick = buffers.next_step("local");
        buffers
            .advance_step("local", tick)
            .map_err(Error::from_apply)?;
        *current_step = tick;
        *step = tick + 1;
        let out = core::assemble_obs(
            &parts.plan,
            &raw_map(raw)?,
            "local",
            tick,
            buffers,
            &parts.customs,
            &parts.encodings,
        )
        .map_err(Error::from_apply)?;
        Ok(restore_shape(from_adapter(out)?, &parts.input))
    }
    /// Python’s `AdapterBase.transform_action` operation.
    fn transform_action(&mut self, raw: Value) -> Result<Value> {
        let mut s = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let AdapterState {
            parts,
            buffers,
            current_step,
            ..
        } = &mut *s;
        let mut action = to_adapter(&raw)?;
        parts
            .encodings
            .repack_action(&mut action)
            .map_err(Error::from_apply)?;
        let tensor = parts
            .plan
            .transform_action(&action)
            .map_err(Error::from_apply)?;
        core::record_action(&parts.plan, &action, buffers, "local", *current_step)
            .map_err(Error::from_apply)?;
        crate::binding::tensor_from_core(tensor).map(Value::Tensor)
    }
    /// Python’s `AdapterBase.observe` operation.
    fn observe(&mut self, raw: Value) -> Result<()> {
        let mut s = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let AdapterState {
            parts,
            buffers,
            step,
            current_step,
        } = &mut *s;
        let tick = buffers.next_step("local");
        buffers
            .advance_step("local", tick)
            .map_err(Error::from_apply)?;
        *current_step = tick;
        *step = tick + 1;
        core::observe_obs(&parts.plan, &raw_map(raw)?, "local", buffers).map_err(Error::from_apply)
    }
    /// Python’s `AdapterBase.reset` operation.
    fn reset(&mut self) -> Result<()> {
        let mut s = self.0.lock().unwrap_or_else(|e| e.into_inner());
        s.buffers.clear();
        s.step = 0;
        s.current_step = 0;
        Ok(())
    }
    /// Python’s `AdapterBase.explain` operation.
    fn explain(&self) -> Result<String> {
        Ok(self
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .parts
            .plan
            .describe())
    }
}
pub(crate) fn restore_shape(value: Value, node: &InputNode) -> Value {
    match (node, value) {
        (InputNode::Tuple(nodes), Value::List(values)) => Value::Tuple(
            values
                .into_iter()
                .zip(nodes)
                .map(|(v, n)| restore_shape(v, n))
                .collect(),
        ),
        (InputNode::Dict(nodes), Value::Dict(mut map)) => {
            for (k, n) in nodes {
                if let Some(v) = map.get(k).cloned() {
                    map.insert(k, restore_shape(v, n));
                }
            }
            Value::Dict(map)
        }
        (_, v) => v,
    }
}
/// An authored adapter protocol; defaults match Python's `AdapterBase`.
pub trait AdapterBase {
    /// Python’s `AdapterBase.transform_obs` operation.
    fn transform_obs(&mut self, raw_obs: Value) -> Result<Value>;
    /// Python’s `AdapterBase.transform_action` operation.
    fn transform_action(&mut self, raw_action: Value) -> Result<Value>;
    /// Python’s `AdapterBase.observe` operation.
    fn observe(&mut self, raw_obs: Value) -> Result<()> {
        let _ = raw_obs;
        Ok(())
    }
    /// Python’s `AdapterBase.reset` operation.
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
    /// Python’s `AdapterBase.explain` operation.
    fn explain(&self) -> Result<String> {
        let name = std::any::type_name::<Self>()
            .split('<')
            .next()
            .unwrap_or("AdapterBase")
            .rsplit("::")
            .next()
            .unwrap_or("AdapterBase");
        Ok(format!("{name} (custom adapter)"))
    }
    /// Python’s `AdapterBase.wrap_predict` operation.
    fn wrap_predict<F>(self, predict_fn: F) -> WrappedPredict<Self, F>
    where
        Self: Sized,
        F: FnMut(Value) -> Result<Value> + Send + 'static,
    {
        WrappedPredict {
            adapter: self,
            predict_fn,
        }
    }
}
/// A predict callable composed with an adapter, usable as a model policy.
pub struct WrappedPredict<A, F> {
    adapter: A,
    predict_fn: F,
}
impl<A: AdapterBase, F: FnMut(Value) -> Result<Value>> WrappedPredict<A, F> {
    /// Invoke this callable.
    pub fn call(&mut self, observation: Value) -> Result<Value> {
        let input = self.adapter.transform_obs(observation)?;
        self.adapter.transform_action((self.predict_fn)(input)?)
    }
    /// Borrow the wrapped adapter.
    pub fn adapter(&self) -> &A {
        &self.adapter
    }
    /// Mutably borrow the wrapped adapter.
    pub fn adapter_mut(&mut self) -> &mut A {
        &mut self.adapter
    }
}
impl<A: AdapterBase + Send + 'static, F: FnMut(Value) -> Result<Value> + Send + 'static>
    crate::Policy for WrappedPredict<A, F>
{
    fn predict(&mut self, observation: Value) -> Result<Value> {
        self.call(observation)
    }
}

pub(crate) fn inject_instruction(
    value: Value,
    node: &InputNode,
    instruction: Option<&str>,
) -> Value {
    let Some(text) = instruction else {
        return value;
    };
    match (node, value) {
        (InputNode::Leaf(ModelLeaf::Text(t)), _) => {
            if t.container == TextContainer::List {
                Value::List(vec![Value::Text(text.into())])
            } else {
                Value::Text(text.into())
            }
        }
        (InputNode::Dict(nodes), Value::Dict(mut map)) => {
            for (k, n) in nodes {
                let previous = map.get(k).cloned().unwrap_or(Value::None);
                let v = inject_instruction(previous, n, Some(text));
                if !v.is_none() {
                    map.insert(k, v);
                }
            }
            Value::Dict(map)
        }
        (InputNode::Tuple(nodes), Value::Tuple(values)) => Value::Tuple(
            values
                .into_iter()
                .zip(nodes)
                .map(|(v, n)| inject_instruction(v, n, Some(text)))
                .collect(),
        ),
        (_, v) => v,
    }
}
