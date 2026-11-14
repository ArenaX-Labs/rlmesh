//! Building an [`EnvServer`]: its constructors for one env, lanes or a
//! vector env, and the checks they run on the options and metadata.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use crate::lifecycle::ShutdownTrigger;
use rlmesh_adapters::v1::ENV_BRANCH_METADATA_KEY;

use super::wrappers::Tagged;
use super::{
    DEFAULT_SHUTDOWN_GRACE, EnvServer, EnvServerOptions, Listener, NEXT_SERVER, Publish, Resources,
    Served, Shared, State, lock,
};
use crate::adapters::EnvTags;
use crate::editions::EditionSources;
use crate::env::envelope::{DescribingHost, env_envelope};
use crate::env::single_lane::SingleLaneEnv;
use crate::env::sync_wire::{env_contract_of, vector_contract_of};
use crate::env::{Env, VectorEnv};
use crate::specs::{EnvContract, value_to_json};
use crate::{Error, Result, ServeOptions};

impl EnvServer {
    /// Serve one env.
    pub fn new<E: Env + Send + 'static>(env: E, options: EnvServerOptions) -> Result<EnvServer> {
        Self::build(Served::Lanes(vec![Box::new(env)]), options, None, true)
    }

    /// Serve `envs` as the lanes of one `num_envs = N` endpoint, each on its
    /// own thread. Use `Box<dyn Env + Send>` to mix env types.
    pub fn lanes<E: Env + Send + 'static>(
        envs: Vec<E>,
        options: EnvServerOptions,
    ) -> Result<EnvServer> {
        Self::build(
            Served::Lanes(
                envs.into_iter()
                    .map(|env| Box::new(env) as Box<dyn Env + Send>)
                    .collect(),
            ),
            options,
            None,
            true,
        )
    }

    /// Serve a vector env. A vector env with `num_envs == 1` is served as a
    /// single scalar lane.
    pub fn vector<V: VectorEnv + Send + 'static>(
        env: V,
        options: EnvServerOptions,
    ) -> Result<EnvServer> {
        Self::build(Served::Vector(Box::new(env)), options, None, true)
    }

    /// A factory serve's server: the edition was resolved by the caller.
    pub(crate) fn lanes_resolved(
        envs: Vec<Box<dyn Env + Send>>,
        options: EnvServerOptions,
        publish: Publish,
    ) -> Result<EnvServer> {
        Self::build(Served::Lanes(envs), options, Some(publish), true)
    }

    pub(crate) fn vector_resolved(
        env: Box<dyn VectorEnv + Send>,
        options: EnvServerOptions,
        publish: Publish,
    ) -> Result<EnvServer> {
        Self::build(Served::Vector(env), options, Some(publish), true)
    }

    /// Override publication identity for an in-repo non-Rust binding, before
    /// starting the server. The facade still builds and validates the envelope.
    pub(crate) fn binding_host(
        &self,
        component: &'static str,
        language: &'static str,
        extra: serde_json::Map<String, serde_json::Value>,
    ) -> Result<()> {
        let mut state = lock(&self.shared.state);
        let State::Ready(resources) = &mut *state else {
            return Err(Error::Runtime("server has already started".into()));
        };
        let envelope = env_envelope(
            &self.shared.contract,
            None,
            None,
            &DescribingHost {
                component,
                language,
                extra,
            },
            resources.options.workflow_edition.as_deref(),
            matches!(resources.served, Served::Vector(_)),
        )?;
        *lock(&self.shared.binding_description) = Some(envelope.clone());
        Ok(())
    }

    /// Print the serving line once the env answers, timed from `started`.
    pub(crate) fn print_when_ready(&self, started: std::time::Instant) {
        if let State::Ready(resources) = &mut *lock(&self.shared.state)
            && let Some(announce) = resources.announce.as_mut()
        {
            announce.banner = Some(started);
        }
    }

    pub(crate) fn binding_description(&self) -> Result<String> {
        lock(&self.shared.binding_description)
            .clone()
            .ok_or_else(|| Error::Runtime("server has no binding description".into()))
    }

    /// The loopback server a run or session stands up in front of a local
    /// env: in a session of its own, publishing nothing, never closing the
    /// env (its caller does) and taking the caller's resolved edition. The
    /// run's driver, in this process, calls the env directly either way.
    /// Driving the served model at `beside`, the session is on that model's
    /// host, which the model reaches in its own process; else on this
    /// process's shared loopback host.
    pub(crate) fn loopback(
        env: crate::run::LoopbackEnv,
        edition: Option<String>,
        beside: Option<&str>,
    ) -> Result<EnvServer> {
        let options = EnvServerOptions {
            address: Some(format!(
                "{}/env",
                match beside {
                    // The model's address, already checked: only resolving
                    // its host name can fail, which a retry may get past.
                    Some(model) => crate::lockstep::run_session_beside(model)
                        .map_err(|err| Error::Connection(err.to_string()))?,
                    None => crate::lockstep::local_run_session()
                        .map_err(|err| Error::Internal(err.to_string()))?,
                }
            )),
            options: Some(ServeOptions {
                workflow_edition: edition,
                // A session server has nothing to drain.
                ..Default::default()
            }),
            close_env_on_shutdown: false,
            ..Default::default()
        };
        let served = match env {
            crate::run::LoopbackEnv::Env(env) => Served::Lanes(vec![env]),
            crate::run::LoopbackEnv::Vector(env) => Served::Vector(env),
        };
        Self::build(served, options, None, false)
    }

    /// `factory`: a factory serve's publication, whose edition the caller
    /// resolved. `announce`: whether the env is served for others to reach
    /// (not a run's own loopback), so it says when it answers.
    fn build(
        served: Served,
        options: EnvServerOptions,
        factory: Option<Publish>,
        announce: bool,
    ) -> Result<EnvServer> {
        // Step 1: at least one lane (the traits guarantee the spaces).
        let mut served = match served {
            Served::Lanes(lanes) if lanes.is_empty() => {
                return Err(Error::invalid_argument(
                    "EnvServer needs at least one environment",
                ));
            }
            // Step 2: a one-lane vector env is a scalar lane; 0 is refused.
            Served::Vector(env) if env.num_envs() == 1 => {
                Served::Lanes(vec![Box::new(SingleLaneEnv::new(env)?)])
            }
            Served::Vector(env) if env.num_envs() < 2 => {
                return Err(Error::invalid_argument(format!(
                    "a vector env must have num_envs >= 1, got {}",
                    env.num_envs()
                )));
            }
            other => other,
        };

        // Step 3: tags, merged into each scalar env's metadata and validated
        // against its spaces; without tags, published tags are re-validated.
        // Steps 4 and 7: the expected branch and each lane's contract.
        let contract = match &mut served {
            Served::Lanes(lanes) => {
                if let Some(tags) = &options.tags {
                    *lanes = std::mem::take(lanes)
                        .into_iter()
                        .map(|lane| {
                            Tagged::new(lane, tags.clone())
                                .map(|tagged| Box::new(tagged) as Box<dyn Env + Send>)
                        })
                        .collect::<Result<_>>()?;
                }
                let mut contracts = Vec::with_capacity(lanes.len());
                for lane in lanes.iter() {
                    let contract = env_contract_of(lane.as_ref())?;
                    if options.tags.is_none()
                        && let Some(metadata) = contract.metadata()
                        && let Some(published) = EnvTags::from_metadata(metadata)?
                    {
                        published.validate(&lane.observation_space(), &lane.action_space())?;
                    }
                    check_expected_branch(contract.metadata())?;
                    contracts.push(contract);
                }
                let first = contracts[0].clone();
                for (index, contract) in contracts.iter().enumerate().skip(1) {
                    if !contract.same_lane_contract(&first) {
                        return Err(Error::invalid_argument(format!(
                            "lane {index} disagrees with lane 0 on the env contract (spaces, \
                             metadata, tags): every lane of one endpoint must be the same make()"
                        )));
                    }
                }
                first.with_num_envs(lanes.len() as u64)
            }
            Served::Vector(env) => {
                let contract = match &options.tags {
                    // Deferred validation: tags describe one lane, the vector's
                    // spaces are its single spaces.
                    Some(tags) => {
                        let mut metadata = env.metadata().unwrap_or_default();
                        tags.stamp(&mut metadata)?;
                        let base = vector_contract_of(env.as_ref())?;
                        EnvContract::new(
                            base.id().to_string(),
                            base.observation_space().clone(),
                            base.action_space().clone(),
                            base.render_mode().map(str::to_string),
                            base.num_envs(),
                            Some(metadata),
                            base.autoreset_mode(),
                        )
                    }
                    None => vector_contract_of(env.as_ref())?,
                };
                check_expected_branch(contract.metadata())?;
                contract
            }
        };

        // Step 5: the address.
        let address = normalize_bind_address(&options)?;
        let announce = announce.then(|| crate::binding::Announce::from_env(&address));
        let serve = options.options.clone().unwrap_or_default();
        serve.validate()?;

        // Step 6: the edition. An explicit options value is the call's own.
        let edition = crate::editions::resolve_native(EditionSources {
            call: serve.workflow_edition.as_deref(),
            option: None,
            declared: None,
        })?;

        // Retain the full factory envelope for binding server introspection.
        // Rebuilding it from the contract would lose branches and variants.
        let binding_description = match factory.as_ref() {
            Some(publish) => publish.envelope.as_ref().ok().cloned(),
            None if options.ext.as_ref().is_some_and(|ext| ext.publish_describe) => {
                Some(env_envelope(
                    &contract,
                    None,
                    None,
                    &DescribingHost::rust(),
                    edition.as_deref(),
                    matches!(served, Served::Vector(_)),
                )?)
            }
            None => None,
        };

        // Step 9: bind.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("rlmesh-env-server")
            .build()
            .map_err(|err| Error::Internal(format!("failed to create tokio runtime: {err}")))?;
        let (listener, address) = Listener::bind(&address, &runtime, &contract)?;

        let close_timeout = serve.close_timeout.unwrap_or(DEFAULT_SHUTDOWN_GRACE);
        let close_env_on_shutdown = options.close_env_on_shutdown;
        let options = ServeOptions {
            workflow_edition: edition,
            ..serve
        };
        let (done, _) = tokio::sync::watch::channel(false);
        Ok(EnvServer {
            shared: Arc::new(Shared {
                id: NEXT_SERVER.fetch_add(1, Ordering::Relaxed),
                address,
                contract,
                binding_description: Mutex::new(binding_description),
                shutdown: ShutdownTrigger::new(),
                background_cancel: Mutex::new(std::sync::Weak::new()),
                state: Mutex::new(State::Ready(Box::new(Resources {
                    served,
                    runtime,
                    listener,
                    options,
                    close_timeout,
                    close_env_on_shutdown,
                    announce,
                }))),
                detached: Mutex::new(None),
                abandoned: Arc::new(Mutex::new(Vec::new())),
                drain: futures::lock::Mutex::new(()),
                done,
            }),
        })
    }
}

/// Step 4: `RLMESH_EXPECTED_ENV_BRANCH`, when set and non-empty, must equal the
/// env's published branch.
fn check_expected_branch(metadata: Option<&crate::Metadata>) -> Result<()> {
    let Ok(expected) = std::env::var("RLMESH_EXPECTED_ENV_BRANCH") else {
        return Ok(());
    };
    if expected.trim().is_empty() {
        return Ok(());
    }
    let expected_json: serde_json::Value = serde_json::from_str(&expected).map_err(|err| {
        Error::invalid_argument(format!(
            "RLMESH_EXPECTED_ENV_BRANCH is not valid JSON: {err}"
        ))
    })?;
    let actual = match metadata.and_then(|metadata| metadata.get(ENV_BRANCH_METADATA_KEY)) {
        Some(value) => value_to_json(value)?,
        None => serde_json::Value::Null,
    };
    if actual != expected_json {
        return Err(Error::invalid_argument(format!(
            "this image was published for env branch {expected_json} but the served env is \
             branch {actual}; build and push the image for the branch it serves"
        )));
    }
    Ok(())
}

fn normalize_bind_address(options: &EnvServerOptions) -> Result<String> {
    let address = options.address.as_deref().unwrap_or(":0/env");
    crate::ConnectAddress::parse(address)?;
    Ok(address.to_string())
}
