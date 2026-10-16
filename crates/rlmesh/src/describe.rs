//! Describe: an env factory's or a model's metadata envelope, as Python's
//! `rlmesh.describe` / `describe_json` build it (COR-10 to COR-14, COR-25).
//!
//! Natives have no module, qualname or signature to reflect, so the four
//! reflected pieces have one native spelling each: `target.qualname` is
//! `rust:<type name>`, `params` lists a model's `LOAD_KEYS` or a factory's
//! `tag_params`, `runtime` is this host's `PeerInfo` plus the edition offer,
//! and an `env_spec` badge's `error_type` is the R-21 class name.

use std::fmt;

use rlmesh_adapters::v1::{build_describe_envelope, space_spec_to_json};
use serde_json::{Value as Json, json};

use crate::env::ServedEnv;
use crate::env::served::ServedRead;
use crate::model::{Corners, PolicyDecl};
use crate::specs::{json_to_value, value_to_json};
use crate::types::SpecArg;
use crate::{
    EnvFactory, EnvFactoryExt, Error, Kwargs, MadeEnv, Metadata, Model, Policy, Result, TagParam,
    Value,
};

/// What `describe` treats its target as: Python's `kind="env"` / `"model"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DescribeKind {
    /// An env factory.
    Env,
    /// A model.
    Model,
}

impl DescribeKind {
    /// Python's spelling (`"env"`, `"model"`).
    pub fn as_str(self) -> &'static str {
        match self {
            DescribeKind::Env => "env",
            DescribeKind::Model => "model",
        }
    }
}

impl fmt::Display for DescribeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The keywords of Python's `describe(obj, *, kind=None, generated_at=None)`.
#[derive(Clone, Debug, Default)]
pub struct DescribeOptions {
    /// A cross-check of the target's kind; a contradiction is the ValueError
    /// class.
    pub kind: Option<DescribeKind>,
    /// An RFC 3339 timestamp stamped on the envelope; omit it for a
    /// content-addressable artifact.
    pub generated_at: Option<String>,
}

/// The keywords of Python's `describe_json(obj, *, kind=None,
/// generated_at=None, served_env=None, workflow_edition=None)`.
#[derive(Debug, Default)]
pub struct DescribeJsonOptions<'a> {
    /// See [`DescribeOptions::kind`].
    pub kind: Option<DescribeKind>,
    /// See [`DescribeOptions::generated_at`].
    pub generated_at: Option<String>,
    /// The env a server hosts: its spaces replace the default branch's build.
    pub served_env: Option<ServedEnv<'a>>,
    /// The edition a server resolved, used as given (`""`: it resolved
    /// none); `None` resolves the type's declaration here, recording a
    /// refusal instead of failing.
    pub workflow_edition: Option<String>,
}

/// What can be described: a model or env factory type, or an instance.
pub struct Describable<'a>(Target<'a>);

#[allow(clippy::large_enum_variant)] // Built once per describe call.
enum Target<'a> {
    Model {
        decl: PolicyDecl,
        spec: Option<SpecArg>,
    },
    Factory(Box<dyn FactoryForm + 'a>),
}

impl fmt::Debug for Describable<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Target::Model { decl, .. } => write!(f, "Describable::model({})", decl.type_name),
            Target::Factory(form) => write!(f, "Describable::env_factory({})", form.type_name()),
        }
    }
}

impl Describable<'static> {
    /// The model type form: `P`'s declarations only, no user code run.
    pub fn model<P: Policy>() -> Describable<'static> {
        let decl = PolicyDecl::of::<P>();
        let spec = decl.spec.clone();
        Describable(Target::Model { decl, spec })
    }
}

impl<'a> Describable<'a> {
    /// The env factory type form: each build default-constructs a fresh
    /// factory, as Python's `cls()` does.
    pub fn env_factory<F: EnvFactory + Default + 'a>() -> Describable<'a> {
        Describable(Target::Factory(Box::new(TypeForm::<F>(
            std::marker::PhantomData,
        ))))
    }
}

/// A built model: its policy's type-level items plus the model's own spec.
impl<'a> From<&'a Model> for Describable<'a> {
    fn from(model: &'a Model) -> Describable<'a> {
        let decl = &model.shared().decl;
        Describable(Target::Model {
            decl: PolicyDecl {
                corners: decl.corners,
                context: decl.context,
                native_chunk: decl.static_native_chunk,
                load_keys: decl.load_keys.clone(),
                spec: None,
                workflow_edition: decl.workflow_edition.clone(),
                authored: decl.authored,
                type_name: decl.type_name.clone(),
            },
            spec: model.spec().cloned(),
        })
    }
}

/// A factory instance: its builds run `prepare()` and `make()` on it.
impl<'a, F: EnvFactory> From<&'a mut F> for Describable<'a> {
    fn from(factory: &'a mut F) -> Describable<'a> {
        Describable(Target::Factory(Box::new(InstanceForm(factory))))
    }
}

/// An env factory behind the describe seam: its type-level items and its
/// builds.
trait FactoryForm {
    fn type_name(&self) -> &'static str;
    fn tags(&self) -> Option<Json>;
    fn tag_params(&self) -> Vec<TagParam>;
    fn branch_table(&self) -> Result<Vec<Kwargs>>;
    fn tags_for(&self, branch: &Kwargs) -> Json;
    fn catalog(&self) -> Option<Result<Json>>;
    fn workflow_edition(&self) -> Option<String>;
    /// One representative build's `env_spec` (or a badge).
    fn env_spec(&mut self, branch: &Kwargs) -> Json;
}

struct TypeForm<F>(std::marker::PhantomData<fn() -> F>);
struct InstanceForm<'a, F>(&'a mut F);

fn tags_json<F: EnvFactory>() -> Option<Json> {
    F::tags().map(|tags| tags.to_json())
}

fn branch_tags_json<F: EnvFactory>(branch: &Kwargs) -> Json {
    match F::tags_for(branch) {
        Ok(Some(tags)) => tags.to_json(),
        Ok(None) => Json::Null,
        Err(err) => json!({ "error": err.to_string() }),
    }
}

fn branch_table<F: EnvFactory>() -> Result<Vec<Kwargs>> {
    crate::env::factory::check_branch_table::<F>()?;
    Ok(crate::env::factory::branches::<F>())
}

fn catalog_json<F: EnvFactory>() -> Option<Result<Json>> {
    match F::enumerate_variants() {
        Ok(None) => None,
        Ok(Some(variants)) => Some(crate::env::factory::catalog_entries(&variants)),
        Err(err) => Some(Err(err)),
    }
}

/// The badge a failed build or space read becomes (COR-25).
fn badge(err: &Error) -> Json {
    json!({ "error": err.to_string(), "error_type": err.class().name() })
}

fn space_json(space: &crate::spaces::Space) -> Result<Json> {
    space_spec_to_json(space.spec().wire()).map_err(|err| Error::invalid_argument(err.to_string()))
}

/// The `env_spec` of one env (scalar or vector).
fn made_spec(made: &MadeEnv) -> Result<Json> {
    match made {
        MadeEnv::Env(env) => Ok(json!({
            "observation_space": space_json(&env.observation_space())?,
            "action_space": space_json(&env.action_space())?,
        })),
        MadeEnv::Vector(env) => Ok(json!({
            "observation_space": space_json(&env.single_observation_space())?,
            "action_space": space_json(&env.single_action_space())?,
            "num_envs": env.num_envs(),
        })),
    }
}

fn served_spec(served: &ServedRead<'_>) -> Result<Json> {
    match served {
        ServedRead::Env(env) => Ok(json!({
            "observation_space": space_json(&env.observation_space())?,
            "action_space": space_json(&env.action_space())?,
        })),
        ServedRead::Vector(env) => Ok(json!({
            "observation_space": space_json(&env.single_observation_space())?,
            "action_space": space_json(&env.single_action_space())?,
            "num_envs": env.num_envs(),
        })),
    }
}

/// Build one env on `factory` for `branch`, read its spaces and close it.
fn build_spec<F: EnvFactory>(factory: &mut F, branch: &Kwargs) -> Json {
    let built = factory.prepare().and_then(|()| factory.make_env(branch));
    let mut made = match built {
        Ok(made) => made,
        Err(err) => return badge(&err),
    };
    let spec = made_spec(&made).unwrap_or_else(|err| badge(&err));
    let _ = match &mut made {
        MadeEnv::Env(env) => env.close(),
        MadeEnv::Vector(env) => env.close(),
    };
    spec
}

impl<F: EnvFactory + Default> FactoryForm for TypeForm<F> {
    fn type_name(&self) -> &'static str {
        std::any::type_name::<F>()
    }
    fn tags(&self) -> Option<Json> {
        tags_json::<F>()
    }
    fn tag_params(&self) -> Vec<TagParam> {
        F::tag_params()
    }
    fn branch_table(&self) -> Result<Vec<Kwargs>> {
        branch_table::<F>()
    }
    fn tags_for(&self, branch: &Kwargs) -> Json {
        branch_tags_json::<F>(branch)
    }
    fn catalog(&self) -> Option<Result<Json>> {
        catalog_json::<F>()
    }
    fn workflow_edition(&self) -> Option<String> {
        F::workflow_edition()
    }
    fn env_spec(&mut self, branch: &Kwargs) -> Json {
        build_spec(&mut F::default(), branch)
    }
}

impl<F: EnvFactory> FactoryForm for InstanceForm<'_, F> {
    fn type_name(&self) -> &'static str {
        std::any::type_name::<F>()
    }
    fn tags(&self) -> Option<Json> {
        tags_json::<F>()
    }
    fn tag_params(&self) -> Vec<TagParam> {
        F::tag_params()
    }
    fn branch_table(&self) -> Result<Vec<Kwargs>> {
        branch_table::<F>()
    }
    fn tags_for(&self, branch: &Kwargs) -> Json {
        branch_tags_json::<F>(branch)
    }
    fn catalog(&self) -> Option<Result<Json>> {
        catalog_json::<F>()
    }
    fn workflow_edition(&self) -> Option<String> {
        F::workflow_edition()
    }
    fn env_spec(&mut self, branch: &Kwargs) -> Json {
        build_spec(self.0, branch)
    }
}

/// Who is describing, for the envelope's `runtime` (COR-25).
pub(crate) struct Host {
    pub(crate) component: &'static str,
    pub(crate) language: &'static str,
    pub(crate) extra: serde_json::Map<String, Json>,
}

impl Host {
    pub(crate) fn rust() -> Host {
        Host {
            component: "rlmesh-rust",
            language: "rust",
            extra: serde_json::Map::new(),
        }
    }
}

/// The envelope's `runtime`: the host's `PeerInfo` plus the edition offer.
/// `preferred` is the declared edition (`None`: this build's cohort).
pub(crate) fn runtime_json(host: &Host, preferred: Option<&str>, error: Option<String>) -> Json {
    let preferred = preferred
        .map(str::trim)
        .filter(|edition| !edition.is_empty())
        .unwrap_or(rlmesh_proto::CURRENT_WORKFLOW_EDITION);
    let mut runtime = json!({
        "component": host.component,
        "language": host.language,
        "package_version": crate::VERSION,
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "protocol_generation": rlmesh_proto::PROTOCOL_GENERATION,
        "supported_workflow_editions": rlmesh_proto::supported_workflow_editions(),
        "preferred_workflow_edition": preferred,
    });
    if let Some(object) = runtime.as_object_mut() {
        if let Some(error) = error {
            object.insert("workflow_edition_error".to_string(), Json::String(error));
        }
        for (key, value) in &host.extra {
            object.insert(key.clone(), value.clone());
        }
    }
    runtime
}

/// COR-12: a given edition is used as given (blank: none); `None` runs the
/// environment variable and the declaration, never warning, and records a
/// refusal instead of failing.
fn offer(declared: Option<&str>, given: Option<&str>) -> (Option<String>, Option<String>) {
    if let Some(given) = given {
        let given = given.trim();
        return ((!given.is_empty()).then(|| given.to_string()), None);
    }
    match crate::editions::resolve_rungs(crate::editions::EditionSources {
        call: None,
        option: None,
        declared,
    }) {
        Ok(resolution) => (resolution.into_declared(), None),
        Err(err) => (None, Some(err.to_string())),
    }
}

fn tag_param_type(value: &Value) -> &'static str {
    match value {
        Value::Bool(_) => "bool",
        Value::Int(_) => "int",
        Value::Float(_) => "float",
        _ => "str",
    }
}

fn kind_error(asked: DescribeKind, actual: DescribeKind) -> Error {
    let missing = match asked {
        DescribeKind::Env => "make",
        DescribeKind::Model => "predict",
    };
    Error::invalid_argument(format!(
        "kind='{asked}' contradicts the described {}, which has no {missing}()",
        match actual {
            DescribeKind::Env => "env factory",
            DescribeKind::Model => "model",
        }
    ))
}

fn model_pieces(
    decl: &PolicyDecl,
    spec: Option<&SpecArg>,
    edition: Option<&str>,
    host: &Host,
) -> Result<Json> {
    // A static declaration construction refuses is refused here too (COR-13).
    if let Some(native) = decl.native_chunk {
        if !decl.corners.has_chunk() {
            return Err(Error::Type(
                "native_chunk declares how long ONE action chunk is, but this model defines \
                 no predict_chunk() / predict_chunk_batch(). Add a chunk corner, or drop \
                 native_chunk."
                    .to_string(),
            ));
        }
        if native < 1 {
            return Err(Error::invalid_argument(format!(
                "native_chunk must be >= 1 (or None when undeclared), got {native}"
            )));
        }
    }
    let (preferred, error) = offer(decl.workflow_edition.as_deref(), edition);
    let signature_tier: Vec<Json> = decl
        .load_keys
        .iter()
        .flatten()
        .map(|(name, required)| {
            json!({ "name": name, "type": "str", "default": null, "required": required })
        })
        .collect();
    let mut pieces = json!({
        "target": { "entrypoint": null, "qualname": format!("rust:{}", decl.type_name) },
        "params": { "param_spec": null, "signature_tier": signature_tier },
        "runtime": runtime_json(host, preferred.as_deref(), error),
        "model_spec": match spec {
            Some(SpecArg::ModelSpec(spec)) => spec.to_json(),
            _ => Json::Null,
        },
    });
    let corners = decl.corners.names();
    if decl.corners != Corners::NONE {
        pieces["corners"] = json!(corners);
    }
    if let Some(native) = decl.native_chunk {
        pieces["native_chunk"] = json!(native);
    }
    Ok(pieces)
}

fn factory_pieces(
    form: &mut dyn FactoryForm,
    served: Option<&ServedEnv<'_>>,
    edition: Option<&str>,
    host: &Host,
) -> Result<Json> {
    let served = served.map(ServedEnv::read).transpose()?;
    let declared = form.workflow_edition();
    let (preferred, error) = offer(declared.as_deref(), edition);
    let tag_params = form.tag_params();
    let signature_tier = tag_params
        .iter()
        .map(|param| {
            Ok(json!({
                "name": param.name,
                "type": tag_param_type(&param.default_),
                "default": value_to_json(&param.default_)?,
                "required": false,
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    let env_tags = form.tags().unwrap_or(Json::Null);
    let env_spec = match &served {
        Some(served) => served_spec(served).unwrap_or_else(|err| badge(&err)),
        None => form.env_spec(&Kwargs::new()),
    };
    let mut pieces = json!({
        "target": { "entrypoint": null, "qualname": format!("rust:{}", form.type_name()) },
        "params": { "param_spec": null, "signature_tier": signature_tier },
        "runtime": runtime_json(host, preferred.as_deref(), error),
        "env_tags": env_tags,
        "env_spec": env_spec,
    });
    match form.catalog() {
        None => {}
        Some(Ok(catalog)) => pieces["variants"] = json!({ "catalog": catalog }),
        Some(Err(err)) => pieces["variants"] = json!({ "catalog_error": err.to_string() }),
    }
    if !tag_params.is_empty() {
        pieces["env_contracts"] = match form.branch_table() {
            Err(err) => json!({ "error": err.to_string() }),
            Ok(branches) => {
                let mut out = Vec::with_capacity(branches.len());
                for (index, binding) in branches.iter().enumerate() {
                    let default = index == 0;
                    let (branch_tags, branch_spec) = if default {
                        (pieces["env_tags"].clone(), pieces["env_spec"].clone())
                    } else {
                        (form.tags_for(binding), form.env_spec(binding))
                    };
                    out.push(json!({
                        "params": value_to_json(&Value::Dict(binding.clone()))?,
                        "default": default,
                        "env_tags": branch_tags,
                        "env_spec": branch_spec,
                    }));
                }
                json!({
                    "discriminants": tag_params.iter().map(|param| &param.name).collect::<Vec<_>>(),
                    "branches": out,
                })
            }
        };
    }
    Ok(pieces)
}

/// The canonical envelope string for `obj`, described by `host`.
pub(crate) fn envelope(
    obj: Describable<'_>,
    options: DescribeJsonOptions<'_>,
    host: &Host,
) -> Result<String> {
    let DescribeJsonOptions {
        kind,
        generated_at,
        served_env,
        workflow_edition,
    } = options;
    let mut target = obj.0;
    let actual = match &target {
        Target::Model { .. } => DescribeKind::Model,
        Target::Factory(_) => DescribeKind::Env,
    };
    if let Some(kind) = kind
        && kind != actual
    {
        return Err(kind_error(kind, actual));
    }
    if let Some(served) = &served_env {
        served.read()?;
    }
    let pieces = match &mut target {
        Target::Model { decl, spec } => {
            model_pieces(decl, spec.as_ref(), workflow_edition.as_deref(), host)?
        }
        Target::Factory(form) => factory_pieces(
            form.as_mut(),
            served_env.as_ref(),
            workflow_edition.as_deref(),
            host,
        )?,
    };
    build_describe_envelope(
        actual.as_str(),
        &pieces.to_string(),
        generated_at.as_deref(),
    )
    .map_err(|err| Error::invalid_argument(format!("invalid describe envelope: {err}")))
}

/// `obj`'s metadata envelope: Python's `rlmesh.describe`.
pub fn describe<'a>(obj: impl Into<Describable<'a>>, options: DescribeOptions) -> Result<Metadata> {
    let text = describe_json(
        obj,
        DescribeJsonOptions {
            kind: options.kind,
            generated_at: options.generated_at,
            ..Default::default()
        },
    )?;
    let json: Json = serde_json::from_str(&text)
        .map_err(|err| Error::Internal(format!("describe envelope is not JSON: {err}")))?;
    match json_to_value(json) {
        Value::Dict(map) => Ok(map),
        _ => Err(Error::Internal(
            "describe envelope is not an object".to_string(),
        )),
    }
}

/// `obj`'s canonical envelope string: Python's `rlmesh.describe_json`.
pub fn describe_json<'a>(
    obj: impl Into<Describable<'a>>,
    options: DescribeJsonOptions<'a>,
) -> Result<String> {
    envelope(obj.into(), options, &Host::rust())
}

impl Model {
    /// `P`'s envelope: Python's `Model.describe()` classmethod. Reads `P`'s
    /// declarations only; nothing is constructed or loaded.
    pub fn describe<P: Policy>() -> Result<Metadata> {
        describe(
            Describable::model::<P>(),
            DescribeOptions {
                kind: Some(DescribeKind::Model),
                ..Default::default()
            },
        )
    }
}

#[cfg(test)]
mod tests;
