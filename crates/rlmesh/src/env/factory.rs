//! [`EnvFactory`]: Python's `EnvFactory` base, with its class attributes as
//! associated functions, and [`Variant`].

use std::fmt;
use std::time::Instant;

use rlmesh_adapters::v1::ENV_BRANCH_METADATA_KEY;

use super::server::{EnvServer, EnvServerOptions, Publish, Tagged};
use super::{Env, VectorEnv, VectorSeed};
use crate::adapters::EnvTags;
use crate::editions::EditionSources;
use crate::spaces::Space;
use crate::specs::value_to_json;
use crate::{
    ENV_RESET_OPTIONS_KEY, Error, InfoDict, Kwargs, Metadata, Result, ServeOptions, Value,
};

/// The most contract branches a factory may declare (Python's
/// `_MAX_TAG_BRANCHES`).
const MAX_TAG_BRANCHES: usize = 8;

/// One contract discriminant of a branched factory: Python's `tag_params`
/// entry with what Python reads from `make`'s signature default and the
/// declared `Param`'s choices (ENV-15).
#[derive(Clone, Debug, PartialEq)]
pub struct TagParam {
    /// The `make` keyword.
    pub name: String,
    /// The value `make` gets when the keyword is omitted; one of `choices`.
    pub default_: Value,
    /// Every value the discriminant may take.
    pub choices: Vec<Value>,
}

/// What `make` returns: one env or one vector env.
pub enum MadeEnv {
    /// A scalar env.
    Env(Box<dyn Env + Send>),
    /// A vector env.
    Vector(Box<dyn VectorEnv + Send>),
}

impl MadeEnv {
    /// Wrap a scalar env.
    pub fn env<E: Env + Send + 'static>(env: E) -> MadeEnv {
        MadeEnv::Env(Box::new(env))
    }

    /// Wrap a vector env.
    pub fn vector<V: VectorEnv + Send + 'static>(env: V) -> MadeEnv {
        MadeEnv::Vector(Box::new(env))
    }
}

impl fmt::Debug for MadeEnv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MadeEnv::Env(_) => f.write_str("MadeEnv::Env(..)"),
            MadeEnv::Vector(_) => f.write_str("MadeEnv::Vector(..)"),
        }
    }
}

/// Builds envs: Python's `EnvFactory`.
///
/// Python's class attributes and classmethods are associated functions
/// (ENV-14): override them on the type. Implement [`make`](Self::make); the
/// provided [`EnvFactoryExt::make_env`] stamps what it returns (tags, branch,
/// reset options), and every library path uses it.
pub trait EnvFactory: Sized {
    /// The contract's tags: Python's `tags` (default `None`).
    fn tags() -> Option<EnvTags> {
        None
    }

    /// The contract discriminants: Python's `tag_params` (default none).
    fn tag_params() -> Vec<TagParam> {
        Vec::new()
    }

    /// The reserved reset-option keys this env wants on `reset`: Python's
    /// `reset_options` (default none).
    fn reset_options() -> Vec<String> {
        Vec::new()
    }

    /// The workflow edition this env was authored against: Python's
    /// `workflow_edition` (default `None`, floating).
    fn workflow_edition() -> Option<String> {
        None
    }

    /// The tags of one discriminant binding: Python's `tags_for(**params)`
    /// (default [`tags`](Self::tags)). Override it only alongside
    /// [`tag_params`](Self::tag_params).
    fn tags_for(_params: &Kwargs) -> Result<Option<EnvTags>> {
        Ok(Self::tags())
    }

    /// The factory's named catalog: Python's optional
    /// `enumerate_variants()`. `Ok(None)` (the default) is "not defined".
    fn enumerate_variants() -> Result<Option<Vec<Variant>>> {
        Ok(None)
    }

    /// One-time setup before `make` (default nothing).
    fn prepare(&mut self) -> Result<()> {
        Ok(())
    }

    /// Construct an env. `kwargs` holds every discriminant, its default
    /// applied. The returned env is unstamped; library paths call
    /// [`EnvFactoryExt::make_env`].
    fn make(&mut self, kwargs: &Kwargs) -> Result<MadeEnv>;

    /// Release resources (default nothing). The library never closes a
    /// factory for you.
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

mod sealed {
    pub trait Sealed {}
    impl<F: super::EnvFactory> Sealed for F {}
}

/// The keywords of Python's `EnvFactory.serve(address, *, num_envs=1,
/// workflow_edition=None, **make_kwargs)`.
#[derive(Clone, Debug)]
pub struct EnvFactoryServeOptions {
    /// Lanes to make and serve (default 1; `<= 1` serves one env).
    pub num_envs: i64,
    /// The edition this endpoint declares, above the type's own and below
    /// `RLMESH_WORKFLOW_EDITION`.
    pub workflow_edition: Option<String>,
    /// Keywords passed to `make`.
    pub make_kwargs: Kwargs,
    /// Stops the serve as SIGINT does (R-18): drain, close the envs, then
    /// the cancellation class. Already fired, it fails the call at once.
    pub cancel: Option<crate::ext::CancelToken>,
}

impl Default for EnvFactoryServeOptions {
    fn default() -> Self {
        EnvFactoryServeOptions {
            num_envs: 1,
            workflow_edition: None,
            make_kwargs: Kwargs::new(),
            cancel: None,
        }
    }
}

/// What every [`EnvFactory`] provides: Python's stamping `make`, `serve`.
pub trait EnvFactoryExt: EnvFactory + sealed::Sealed {
    /// `make` with the kwargs bound (each discriminant's default applied;
    /// a value outside its choices is the ValueError class before `make`
    /// runs), then stamped: the branch's tags, the branch under
    /// `ENV_BRANCH_METADATA_KEY`, and `reset_options` under
    /// `ENV_RESET_OPTIONS_KEY`.
    fn make_env(&mut self, kwargs: &Kwargs) -> Result<MadeEnv> {
        check_branch_table::<Self>()?;
        let (bound, branch) = bind_discriminants::<Self>(kwargs)?;
        let made = self.make(&bound)?;
        let tags = match &branch {
            Some(branch) => Self::tags_for(branch)?,
            None => Self::tags(),
        };
        let mut fragment = Metadata::new();
        if let Some(branch) = branch {
            fragment.insert(ENV_BRANCH_METADATA_KEY, Value::Dict(branch));
        }
        let reset_options = Self::reset_options();
        if !reset_options.is_empty() {
            fragment.insert(
                ENV_RESET_OPTIONS_KEY,
                Value::List(reset_options.into_iter().map(Value::Text).collect()),
            );
        }
        stamp(made, tags, fragment)
    }

    /// This factory type's describe envelope: Python's `EnvFactory.describe()`
    /// classmethod. Best-effort: each build default-constructs a factory,
    /// and a failed build or space read is an `env_spec` badge.
    fn describe() -> Result<Metadata>
    where
        Self: Default,
    {
        crate::describe::describe(
            crate::describe::Describable::env_factory::<Self>(),
            crate::describe::DescribeOptions {
                kind: Some(crate::describe::DescribeKind::Env),
                ..Default::default()
            },
        )
    }

    /// Host this factory's env on `address`, blocking: Python's `serve`.
    /// Runs the branch check and `prepare()`, makes one env (or `num_envs`
    /// lanes), resolves the edition, publishes the describe envelope on the
    /// handshake, and serves in the foreground. The factory is not closed.
    fn serve(&mut self, address: &str, options: EnvFactoryServeOptions) -> Result<()> {
        if tokio::task::try_id().is_some() {
            return Err(Error::Runtime(
                "EnvFactoryExt::serve() blocks; call serve_async() from async code".to_string(),
            ));
        }
        let cancel = options.cancel.clone();
        let server = prepare_serve(self, address, &options)?;
        let watcher = cancel.map(|cancel| CancelWatcher::start(cancel, &server));
        let result = server.serve();
        let cancelled = watcher.is_some_and(|watcher| watcher.stop());
        match result {
            Ok(()) if cancelled => Err(Error::Cancelled),
            other => other,
        }
    }

    /// The async twin of [`serve`](Self::serve): the pre-serve hooks run on
    /// the polling thread inside the first poll, then every lane serves from
    /// library threads; dropping the future stops the server and closes the
    /// envs it made (ENV-29, ENV-32).
    fn serve_async(
        &mut self,
        address: &str,
        options: EnvFactoryServeOptions,
    ) -> impl std::future::Future<Output = Result<()>> {
        async move {
            let cancel = options.cancel.clone();
            let server = prepare_serve(self, address, &options)?;
            let watcher = cancel.map(|cancel| CancelWatcher::start(cancel, &server));
            let result = server.serve_async().await;
            let cancelled = watcher.is_some_and(|watcher| watcher.stop());
            match result {
                Ok(()) if cancelled => Err(Error::Cancelled),
                other => other,
            }
        }
    }
}

impl<F: EnvFactory> EnvFactoryExt for F {}

/// Steps 0 to 5 of a factory serve: the built, bound server.
fn prepare_serve<F: EnvFactory>(
    factory: &mut F,
    address: &str,
    options: &EnvFactoryServeOptions,
) -> Result<EnvServer> {
    let started = Instant::now();
    if options
        .cancel
        .as_ref()
        .is_some_and(crate::ext::CancelToken::take)
    {
        return Err(Error::Cancelled);
    }
    check_branch_table::<F>()?;
    factory.prepare()?;
    let num_envs = options.num_envs.max(1);
    // Step 3: the edition, resolved once here (env var, the option, the
    // type) and handed to the server as the caller's answer.
    let edition = crate::editions::resolve_native(
        EditionSources {
            call: None,
            option: options.workflow_edition.as_deref(),
            declared: F::workflow_edition().as_deref(),
        },
        true,
    )?;
    let server_options = EnvServerOptions {
        address: Some(address.to_string()),
        options: Some(ServeOptions {
            workflow_edition: edition.clone(),
            ..ServeOptions::default()
        }),
        // ENV-18: a branched factory's stamped tags stand.
        tags: if F::tag_params().is_empty() {
            F::tags()
        } else {
            None
        },
        ..EnvServerOptions::default()
    };
    // Step 5: the full describe envelope, read off the served env (its
    // default branch) and the resolved edition; best-effort.
    let describe_with = |factory: &mut F, served: crate::env::ServedEnv<'_>| {
        crate::describe::describe_json(
            factory,
            crate::describe::DescribeJsonOptions {
                served_env: Some(served),
                workflow_edition: Some(edition.clone().unwrap_or_default()),
                ..Default::default()
            },
        )
    };
    let server = if num_envs == 1 {
        let made = factory.make_env(&options.make_kwargs)?;
        let publish = Publish {
            envelope: describe_with(factory, crate::env::ServedEnv::from(&made)),
        };
        match made {
            MadeEnv::Env(env) => EnvServer::lanes_resolved(vec![env], server_options, publish)?,
            MadeEnv::Vector(env) => EnvServer::vector_resolved(env, server_options, publish)?,
        }
    } else {
        let mut lanes: Vec<Box<dyn Env + Send>> = Vec::with_capacity(num_envs as usize);
        for _ in 0..num_envs {
            match factory.make_env(&options.make_kwargs)? {
                MadeEnv::Env(env) => lanes.push(env),
                MadeEnv::Vector(_) => {
                    return Err(Error::Type(
                        "lanes must be scalar environments; serve a vectorized env on its own \
                         instead of inside a list"
                            .to_string(),
                    ));
                }
            }
        }
        let served = crate::env::ServedEnv::Lanes(
            lanes.iter().map(|lane| lane.as_ref() as &dyn Env).collect(),
        );
        let publish = Publish {
            envelope: describe_with(factory, served),
        };
        EnvServer::lanes_resolved(lanes, server_options, publish)?
    };
    print_serving(server.address(), started);
    Ok(server)
}

/// A variant catalog as describe's `variants.catalog`: `{id, params,
/// metadata}` entries; a missing or repeated id is the ValueError class
/// (describe records it as `catalog_error`).
pub(crate) fn catalog_entries(variants: &[Variant]) -> Result<serde_json::Value> {
    let mut seen = std::collections::HashSet::new();
    let mut entries = Vec::with_capacity(variants.len());
    for variant in variants {
        if variant.id.is_empty() {
            return Err(Error::invalid_argument(
                "variant id must be a non-empty str; got ''",
            ));
        }
        if !seen.insert(variant.id.as_str()) {
            return Err(Error::invalid_argument(format!(
                "duplicate variant id '{}'",
                variant.id
            )));
        }
        entries.push(variant_json(variant)?);
    }
    Ok(serde_json::Value::Array(entries))
}

#[allow(clippy::print_stdout)]
fn print_serving(address: &str, started: Instant) {
    println!(
        "RLMesh serving env on {address} (startup: listen {:.1}s)",
        started.elapsed().as_secs_f64()
    );
}

fn variant_json(variant: &Variant) -> Result<serde_json::Value> {
    Ok(serde_json::json!({
        "id": variant.id,
        "params": value_to_json(&Value::Dict(variant.params.clone()))?,
        "metadata": value_to_json(&Value::Dict(variant.metadata.clone()))?,
    }))
}

/// Polls a factory serve's cancel token and stops the server when it fires.
struct CancelWatcher {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    fired: std::sync::Arc<std::sync::atomic::AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl CancelWatcher {
    fn start(cancel: crate::ext::CancelToken, server: &EnvServer) -> CancelWatcher {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        let stop = Arc::new(AtomicBool::new(false));
        let fired = Arc::new(AtomicBool::new(false));
        let trigger = server.shutdown_handle();
        let (stop_flag, fired_flag) = (Arc::clone(&stop), Arc::clone(&fired));
        let thread = std::thread::Builder::new()
            .name("rlmesh-cancel".to_string())
            .spawn(move || {
                while !stop_flag.load(Ordering::SeqCst) {
                    if cancel.take() {
                        fired_flag.store(true, Ordering::SeqCst);
                        trigger.trigger("cancelled");
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
            })
            .ok();
        CancelWatcher {
            stop,
            fired,
            thread,
        }
    }

    /// Stop watching; whether the token fired.
    fn stop(mut self) -> bool {
        use std::sync::atomic::Ordering;
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        self.fired.load(Ordering::SeqCst)
    }
}

/// Bind `kwargs` with each discriminant's default; the branch binding when
/// the factory declares discriminants.
fn bind_discriminants<F: EnvFactory>(kwargs: &Kwargs) -> Result<(Kwargs, Option<Kwargs>)> {
    let params = F::tag_params();
    if params.is_empty() {
        return Ok((kwargs.clone(), None));
    }
    let mut bound = kwargs.clone();
    let mut branch = Kwargs::new();
    for param in &params {
        let value = match bound.get(&param.name) {
            Some(value) => value.clone(),
            None => {
                bound.insert(param.name.clone(), param.default_.clone());
                param.default_.clone()
            }
        };
        if !param.choices.contains(&value) {
            return Err(Error::invalid_argument(format!(
                "{}.make() got {}={}, which is not one of the contract discriminant's choices \
                 {}",
                short_type_name::<F>(),
                param.name,
                py_repr(&value),
                py_repr(&Value::List(param.choices.clone()))
            )));
        }
        branch.insert(param.name.clone(), value);
    }
    Ok((bound, Some(branch)))
}

fn short_type_name<F>() -> &'static str {
    let name = std::any::type_name::<F>();
    name.rsplit("::").next().unwrap_or(name)
}

/// Every branch the discriminants declare, the default branch first.
pub(crate) fn branches<F: EnvFactory>() -> Vec<Kwargs> {
    let params = F::tag_params();
    let mut branches = vec![Kwargs::new()];
    for param in &params {
        branches = branches
            .into_iter()
            .flat_map(|branch| {
                param.choices.iter().map(move |choice| {
                    let mut next = branch.clone();
                    next.insert(param.name.clone(), choice.clone());
                    next
                })
            })
            .collect();
    }
    let is_default = |branch: &Kwargs| {
        params
            .iter()
            .all(|param| branch.get(&param.name) == Some(&param.default_))
    };
    // A stable sort keeps the product order after the default branch.
    branches.sort_by_key(|branch| !is_default(branch));
    branches
}

/// The branch table's checks (Python's `_check_tag_params`), each a
/// TypeError. Rust cannot see whether `tags_for` was overridden (ENV-16).
pub(crate) fn check_branch_table<F: EnvFactory>() -> Result<()> {
    let params = F::tag_params();
    if params.is_empty() {
        return Ok(());
    }
    let name = short_type_name::<F>();
    for param in &params {
        if param.choices.is_empty() {
            return Err(Error::Type(format!(
                "{name}.tag_params names {:?}, whose Param declares no choices; a contract \
                 discriminant must be enumerable so the branch table is finite",
                param.name
            )));
        }
        if !param.choices.contains(&param.default_) {
            return Err(Error::Type(format!(
                "{name}.make() must give the contract discriminant {:?} a signature default \
                 drawn from its choices {}; the default branch is the contract a reader with \
                 no branch information sees",
                param.name,
                py_repr(&Value::List(param.choices.clone()))
            )));
        }
    }
    let count: usize = params.iter().map(|param| param.choices.len()).product();
    if count > MAX_TAG_BRANCHES {
        return Err(Error::Type(format!(
            "{name}.tag_params declares {count} contract branches, over the limit of \
             {MAX_TAG_BRANCHES}; discriminants are contract axes, not shape dials -- move the \
             free knobs back to plain params"
        )));
    }
    let branches = branches::<F>();
    let tags = branches
        .iter()
        .map(F::tags_for)
        .collect::<Result<Vec<_>>>()?;
    if tags.iter().any(Option::is_none) && tags.iter().any(Option::is_some) {
        return Err(Error::Type(format!(
            "{name}.tags_for() returns EnvTags on some branches and None on others; a factory \
             is adapted or generic, not both -- a branch-blind reader cannot tell which one it \
             got"
        )));
    }
    if tags[0] != F::tags() {
        return Err(Error::Type(format!(
            "{name}.tags disagrees with tags_for({}), its default branch; they are the same \
             contract read two ways, so set tags to what the default branch returns",
            kwargs_repr(&branches[0])
        )));
    }
    Ok(())
}

fn kwargs_repr(binding: &Kwargs) -> String {
    binding
        .iter()
        .map(|(name, value)| format!("{name}={}", py_repr(value)))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Stamp a made env: tags (unvalidated, as Python's `validate=False`) and a
/// metadata fragment.
fn stamp(made: MadeEnv, tags: Option<EnvTags>, fragment: Metadata) -> Result<MadeEnv> {
    if tags.is_none() && fragment.is_empty() {
        return Ok(made);
    }
    let merge = |metadata: Option<Metadata>| {
        let mut metadata = metadata.unwrap_or_default();
        if let Some(tags) = &tags {
            tags.stamp(&mut metadata)?;
        }
        for (key, value) in &fragment {
            metadata.insert(key.to_string(), value.clone());
        }
        Ok::<_, Error>(metadata)
    };
    Ok(match made {
        MadeEnv::Env(env) => {
            let metadata = merge(env.metadata())?;
            MadeEnv::Env(Box::new(Tagged::with_metadata(env, metadata)))
        }
        MadeEnv::Vector(env) => {
            let metadata = merge(env.metadata())?;
            MadeEnv::Vector(Box::new(StampedVector { env, metadata }))
        }
    })
}

/// A vector env serving stamped metadata.
struct StampedVector {
    env: Box<dyn VectorEnv + Send>,
    metadata: Metadata,
}

impl VectorEnv for StampedVector {
    fn num_envs(&self) -> u64 {
        self.env.num_envs()
    }
    fn single_observation_space(&self) -> Space {
        self.env.single_observation_space()
    }
    fn single_action_space(&self) -> Space {
        self.env.single_action_space()
    }
    fn reset(
        &mut self,
        seed: Option<VectorSeed>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        self.env.reset(seed, options)
    }
    fn step(
        &mut self,
        actions: Value,
    ) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
        self.env.step(actions)
    }
    fn render(&mut self) -> Result<Option<Value>> {
        self.env.render()
    }
    fn close(&mut self) -> Result<()> {
        self.env.close()
    }
    fn render_mode(&self) -> Option<String> {
        self.env.render_mode()
    }
    fn metadata(&self) -> Option<Metadata> {
        Some(self.metadata.clone())
    }
    fn id(&self) -> Option<String> {
        self.env.id()
    }
}

/// Python's `**metadata` of `Variant(id, params, /, **metadata)`.
#[derive(Clone, Debug, Default)]
pub struct VariantOptions {
    /// Open display bag; `name` is the title a dashboard renders.
    pub metadata: Kwargs,
}

/// One concrete sub-environment in a factory's catalog: Python's `Variant`.
#[derive(Clone, PartialEq)]
pub struct Variant {
    /// The factory-unique handle.
    pub id: String,
    /// The `make` keywords that define this sub-environment.
    pub params: Kwargs,
    /// Display metadata.
    pub metadata: Kwargs,
}

impl Variant {
    /// Python's `Variant(id, params, **metadata)`.
    pub fn new(id: impl Into<String>, params: Kwargs, options: VariantOptions) -> Variant {
        Variant {
            id: id.into(),
            params,
            metadata: options.metadata,
        }
    }
}

/// Python's `repr`: `Variant(id='a', params={'task': 1}, metadata={})`.
impl fmt::Display for Variant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Variant(id={}, params={}, metadata={})",
            py_repr(&Value::Text(self.id.clone())),
            py_repr(&Value::Dict(self.params.clone())),
            py_repr(&Value::Dict(self.metadata.clone()))
        )
    }
}

impl fmt::Debug for Variant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// A value as Python's `repr` prints it.
pub(crate) fn py_repr(value: &Value) -> String {
    match value {
        Value::None => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Int(int) => int.to_string(),
        Value::Float(float) => py_float(*float),
        Value::Text(text) => py_str(text),
        Value::Bytes(bytes) => {
            let mut out = String::from("b'");
            for &byte in bytes {
                match byte {
                    b'\\' => out.push_str("\\\\"),
                    b'\'' => out.push_str("\\'"),
                    b'\n' => out.push_str("\\n"),
                    b'\r' => out.push_str("\\r"),
                    b'\t' => out.push_str("\\t"),
                    0x20..=0x7e => out.push(byte as char),
                    other => out.push_str(&format!("\\x{other:02x}")),
                }
            }
            out.push('\'');
            out
        }
        Value::Tensor(tensor) => tensor.to_string(),
        Value::List(items) => format!(
            "[{}]",
            items.iter().map(py_repr).collect::<Vec<_>>().join(", ")
        ),
        Value::Tuple(items) if items.len() == 1 => format!("({},)", py_repr(&items[0])),
        Value::Tuple(items) => format!(
            "({})",
            items.iter().map(py_repr).collect::<Vec<_>>().join(", ")
        ),
        Value::Dict(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(key, value)| format!("{}: {}", py_str(key), py_repr(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

pub(crate) fn py_float(float: f64) -> String {
    if float.is_nan() {
        "nan".to_string()
    } else if float.is_infinite() {
        if float > 0.0 { "inf" } else { "-inf" }.to_string()
    } else if float.fract() == 0.0 && float.abs() < 1e16 {
        format!("{float:.1}")
    } else {
        let text = format!("{float:?}");
        match text.split_once('e') {
            Some((mantissa, exponent)) if !exponent.starts_with('-') => {
                format!("{mantissa}e+{exponent:0>2}")
            }
            Some((mantissa, exponent)) => {
                format!("{mantissa}e-{:0>2}", exponent.trim_start_matches('-'))
            }
            None => text,
        }
    }
}

pub(crate) fn py_str(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::new();
    out.push(quote);
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch == quote => {
                out.push('\\');
                out.push(ch);
            }
            ch if (ch as u32) < 0x20 || ch as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", ch as u32));
            }
            ch => out.push(ch),
        }
    }
    out.push(quote);
    out
}
