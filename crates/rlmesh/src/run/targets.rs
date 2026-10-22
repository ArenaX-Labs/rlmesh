//! The model and env a run or binding takes.

use std::fmt;

use crate::env::client::Lendable;
use crate::{
    Env, EnvFactory, EnvFactoryExt, EnvServer, Kwargs, MadeEnv, Model, RemoteEnv, RemoteVectorEnv,
    Result, VectorEnv,
};

/// The random baseline: a model that samples the env's action space.
/// Accepted by [`run`](crate::run) and `bind`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct RandomSample;

/// The random action-sampling baseline.
pub const RANDOM_SAMPLE: RandomSample = RandomSample;

impl fmt::Display for RandomSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RANDOM_SAMPLE")
    }
}

/// The model argument of [`run`](crate::run) and `bind`:
/// a borrowed [`Model`], a borrowed [`RemoteModel`](crate::RemoteModel) or
/// [`RANDOM_SAMPLE`].
pub struct ModelTarget<'a>(pub(crate) ModelKind<'a>);

pub(crate) enum ModelKind<'a> {
    Model(&'a Model),
    Remote(&'a crate::RemoteModel),
    Random,
}

impl<'a> From<&'a Model> for ModelTarget<'a> {
    fn from(model: &'a Model) -> Self {
        ModelTarget(ModelKind::Model(model))
    }
}

impl<'a> From<&'a crate::RemoteModel> for ModelTarget<'a> {
    fn from(model: &'a crate::RemoteModel) -> Self {
        ModelTarget(ModelKind::Remote(model))
    }
}

impl From<RandomSample> for ModelTarget<'_> {
    fn from(_: RandomSample) -> Self {
        ModelTarget(ModelKind::Random)
    }
}

impl fmt::Debug for ModelTarget<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            ModelKind::Model(model) => f.debug_tuple("ModelTarget").field(model).finish(),
            ModelKind::Remote(model) => f.debug_tuple("ModelTarget").field(model).finish(),
            ModelKind::Random => f.write_str("ModelTarget(RANDOM_SAMPLE)"),
        }
    }
}

/// A factory erased at construction: `EnvFactory` is not dyn-compatible, so
/// the target keeps only what a run calls.
pub(crate) trait ErasedFactory {
    /// Call `prepare()`, then the stamping `make_env()` with no arguments.
    fn make(&mut self) -> Result<MadeEnv>;
}

struct Factory<'a, F>(&'a mut F);

impl<F: EnvFactory> ErasedFactory for Factory<'_, F> {
    fn make(&mut self) -> Result<MadeEnv> {
        self.0.prepare()?;
        self.0.make_env(&Kwargs::new())
    }
}

/// The env argument of a run or binding: an address, a served [`EnvServer`],
/// a borrowed [`RemoteEnv`] / [`RemoteVectorEnv`], or a borrowed local env,
/// vector env or factory.
///
/// A single env converts with `From` (`model.run(&mut env, ..)`); a vector
/// env and a factory name their kind ([`EnvTarget::vector`],
/// [`EnvTarget::factory`]).
pub struct EnvTarget<'a>(pub(crate) EnvKind<'a>);

pub(crate) enum EnvKind<'a> {
    Address(String),
    Server(&'a EnvServer),
    /// A caller's remote client: a run dials a v1 one's address and refuses
    /// a session one, which holds its env's lease; a binding refuses both.
    Remote(&'a mut (dyn Lendable + Send + 'a)),
    Env(&'a mut (dyn Env + Send + 'a)),
    Vector(&'a mut (dyn VectorEnv + Send + 'a)),
    Factory(Box<dyn ErasedFactory + Send + 'a>),
}

impl<'a> EnvTarget<'a> {
    /// A borrowed vector env.
    pub fn vector(env: &'a mut (impl VectorEnv + Send + 'a)) -> EnvTarget<'a> {
        EnvTarget(EnvKind::Vector(env))
    }

    /// A borrowed factory: the run builds one env from it (only when it
    /// needs one) and closes that env; it never closes the factory.
    pub fn factory<F: EnvFactory + Send + 'a>(factory: &'a mut F) -> EnvTarget<'a> {
        EnvTarget(EnvKind::Factory(Box::new(Factory(factory))))
    }
}

impl From<&str> for EnvTarget<'_> {
    fn from(address: &str) -> Self {
        EnvTarget(EnvKind::Address(address.to_string()))
    }
}

impl From<String> for EnvTarget<'_> {
    fn from(address: String) -> Self {
        EnvTarget(EnvKind::Address(address))
    }
}

impl<'a> From<&'a EnvServer> for EnvTarget<'a> {
    fn from(server: &'a EnvServer) -> Self {
        EnvTarget(EnvKind::Server(server))
    }
}

impl<'a> From<&'a mut RemoteEnv> for EnvTarget<'a> {
    fn from(env: &'a mut RemoteEnv) -> Self {
        EnvTarget(EnvKind::Remote(env))
    }
}

impl<'a> From<&'a mut RemoteVectorEnv> for EnvTarget<'a> {
    fn from(env: &'a mut RemoteVectorEnv) -> Self {
        EnvTarget(EnvKind::Remote(env))
    }
}

impl<'a, E: Env + Send + 'a> From<&'a mut E> for EnvTarget<'a> {
    fn from(env: &'a mut E) -> Self {
        EnvTarget(EnvKind::Env(env))
    }
}

impl fmt::Debug for EnvTarget<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            EnvKind::Address(address) => write!(f, "EnvTarget({address:?})"),
            EnvKind::Server(server) => write!(f, "EnvTarget({server})"),
            EnvKind::Remote(env) => write!(f, "EnvTarget(<remote {}>)", env.remote_address()),
            EnvKind::Env(_) => f.write_str("EnvTarget(<env>)"),
            EnvKind::Vector(_) => f.write_str("EnvTarget(<vector env>)"),
            EnvKind::Factory(_) => f.write_str("EnvTarget(<factory>)"),
        }
    }
}
