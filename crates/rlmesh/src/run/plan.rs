//! Evaluation plans: an evaluation request (the file `rlmesh eval submit`
//! sends) compiled into the workloads one machine runs, under the managed
//! platform's rules, so one request draws the same episodes on both.
//!
//! The reference is the platform's run builder (`runs/builder.go`) and its
//! controller's `routeSeedKey`: models × tasks, a task's episodes split over
//! its `workers` (one per episode by default, scaled back to 300 per run),
//! seed 0 by default, and each workload's seed key, slot offset and trial
//! base. The env binding follows `bindEnvironment`: the scenario's make
//! kwargs, the model's pairing for the env refined by the scenario, and the
//! episode cap. Its data is each image's `dev.rlmesh.package` label and the
//! env's describe envelope, which are what the platform hub reads too, so the
//! catalog changes a binding by rebuilding its images, with no rlmesh
//! release. `conformance/eval_plans.v1.json` pins the result against the
//! platform's builder.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::run::runtime::{TRIAL_INDEX_OPTION, episode_seed};
use crate::{Error, Result};

/// The env var `rlmesh serve` binds to `make(**kwargs)`; a request pins an
/// env's or a model's make kwargs through it.
pub const MAKE_KWARGS_VAR: &str = "RLMESH_MAKE_KWARGS";

/// The seed a task without `seed` or `seeds` runs under (`defaultTaskSeed`).
const DEFAULT_SEED: i64 = 0;

/// The routes the default worker count stays under (`maxWorkersPerModel`).
const MAX_DEFAULT_WORKERS: i64 = 300;

/// Pairing keys the runner consumes as the episode cap instead of passing
/// them to the env (`ReservedPairingKeys`).
const EPISODE_CAP_KEYS: [&str; 2] = ["max_episode_steps", "max_steps"];

/// Metadata keys `workers` stamps on each shard.
const SHARD_KEYS: [&str; 3] = [
    "episodeShardIndex",
    "episodeShardCount",
    "episodeShardOffset",
];

const REQUEST_FIELDS: &[&str] = &[
    "models",
    "tasks",
    "maxActionChunk",
    "asyncPrefetchLead",
    "metadata",
    "tags",
];
const MODEL_FIELDS: &[&str] = &["model", "checkpoint", "envVars"];
const TASK_FIELDS: &[&str] = &[
    "taskId", "env", "episodes", "models", "seed", "seeds", "envVars", "metadata", "workers",
    "pairing",
];

const BATCHER: &str = "a local model container batches with its own defaults (rlmesh.BatchOptions)";
const RECORD: &str = "local runs do not record media yet";

/// Request fields only the platform can honour, with why.
const PLATFORM_ONLY: &[(&str, &str)] = &[
    ("projectId", "projects scope evaluations on the platform"),
    (
        "managed",
        "cluster placement and scaling (clusterId, maxParallelWorkloads, modelReplicas, \
         accelerator, requireVerified) are the platform's",
    ),
    (
        "resources",
        "container sizing is the platform's; a local container takes what the host gives it",
    ),
    (
        "residentKeepAliveSeconds",
        "resident models are a platform feature",
    ),
    ("timeoutSeconds", "the run deadline is the platform's"),
    ("maxAttempts", "attempt retries are the platform's"),
    ("maxBatchSize", BATCHER),
    ("batchWindowMs", BATCHER),
    ("maxInFlightBatches", BATCHER),
    (
        "clientsPerServer",
        "model server admission is the platform's",
    ),
    ("record", RECORD),
];
const MODEL_PLATFORM_ONLY: &[(&str, &str)] = &[
    (
        "variant",
        "compute variants are chosen by the platform's placement",
    ),
    ("resources", "container sizing is the platform's"),
];
const TASK_PLATFORM_ONLY: &[(&str, &str)] = &[
    ("record", RECORD),
    ("perturbation", "perturbations run on the platform"),
    (
        "variant",
        "environment profiles are chosen by the platform's placement",
    ),
];

/// One `models[]` entry.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelRef {
    /// The model: an image, or a catalog name (`prebuilt/<model>/<env>`).
    pub model: String,
    /// A checkpoint the model's package label declares; absent, its default.
    #[serde(default)]
    pub checkpoint: Option<String>,
    /// Only `RLMESH_MAKE_KWARGS` (the model's make kwargs).
    #[serde(default)]
    pub env_vars: BTreeMap<String, String>,
}

/// One `tasks[]` entry.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskRef {
    /// The task, or for an env with scenarios the scenario it runs
    /// (`<group>/<scenario>`).
    pub task_id: String,
    /// The env: an image, or a catalog name (`prebuilt/<env>`).
    pub env: String,
    /// Episodes the task runs.
    pub episodes: i64,
    /// The request models it runs on (`model` or `model@checkpoint`); all
    /// when empty.
    #[serde(default)]
    pub models: Vec<String>,
    /// The seed base; 0 when neither it nor `seeds` is given.
    #[serde(default)]
    pub seed: Option<i64>,
    /// One explicit seed per episode.
    #[serde(default)]
    pub seeds: Vec<i64>,
    /// Only `RLMESH_MAKE_KWARGS` (the env's make kwargs).
    #[serde(default)]
    pub env_vars: BTreeMap<String, String>,
    /// Labels; `environmentId` names the task's seed key.
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    /// Env workers its episodes split over.
    #[serde(default)]
    pub workers: Option<i64>,
    /// `native` (the default) only.
    #[serde(default)]
    pub pairing: Option<String>,
}

/// An evaluation request a local run can honour.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvalRequest {
    /// The models.
    pub models: Vec<ModelRef>,
    /// The tasks, run on every model their filter keeps.
    pub tasks: Vec<TaskRef>,
    /// Actions executed per model inference.
    #[serde(default)]
    pub max_action_chunk: Option<u32>,
    /// Async-inference lead (needs `maxActionChunk` > 1).
    #[serde(default)]
    pub async_prefetch_lead: Option<u32>,
    /// Labels, kept in the run index.
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    /// Labels, kept in the run index.
    #[serde(default)]
    pub tags: BTreeMap<String, String>,
}

impl EvalRequest {
    /// The request in `document`, checked as the platform checks it. Every
    /// field a local run cannot honour is refused by name, all at once,
    /// rather than ignored.
    pub fn parse(document: &Value) -> Result<Self> {
        let mut refused = Vec::new();
        refuse_fields(document, "", REQUEST_FIELDS, PLATFORM_ONLY, &mut refused);
        let entries = |key: &str| document[key].as_array().cloned().unwrap_or_default();
        for (i, model) in entries("models").iter().enumerate() {
            let at = format!("models[{i}].");
            refuse_fields(model, &at, MODEL_FIELDS, MODEL_PLATFORM_ONLY, &mut refused);
            refuse_env_vars(model, &at, "model", &mut refused);
            if model["checkpoint"]
                .as_str()
                .is_some_and(|c| c.contains("://"))
            {
                // TODO: fetch checkpoint URIs locally once eval run downloads artifacts.
                refused.push(format!(
                    "{at}checkpoint: a checkpoint URI is fetched by the platform; name one the \
                     model's package label declares"
                ));
            }
        }
        for (i, task) in entries("tasks").iter().enumerate() {
            let at = format!("tasks[{i}].");
            refuse_fields(task, &at, TASK_FIELDS, TASK_PLATFORM_ONLY, &mut refused);
            refuse_env_vars(task, &at, "env", &mut refused);
            match task["pairing"].as_str() {
                None | Some("" | "native") => {}
                Some("transfer") => refused.push(format!(
                    "{at}pairing: transfer runs are not in local runs yet"
                )),
                Some(other) => {
                    refused.push(format!("{at}pairing: {other:?} is not native or transfer"))
                }
            }
        }
        if !refused.is_empty() {
            return Err(Error::invalid_argument(format!(
                "this request cannot run locally:\n  {}",
                refused.join("\n  ")
            )));
        }
        let request: Self = serde_json::from_value(document.clone())
            .map_err(|err| Error::invalid_argument(format!("evaluation request: {err}")))?;
        request.validate()?;
        Ok(request)
    }

    /// The platform's request checks (`validateEvaluationRequest`).
    fn validate(&self) -> Result<()> {
        let invalid = |message: String| Err(Error::invalid_argument(message));
        if self.models.is_empty() {
            return invalid("at least one model is required".into());
        }
        if self.tasks.is_empty() {
            return invalid("at least one task is required".into());
        }
        let mut seen = BTreeSet::new();
        for model in &self.models {
            if !seen.insert((&model.model, &model.checkpoint)) {
                return invalid(format!(
                    "model {} appears twice with the same checkpoint",
                    model.model
                ));
            }
        }
        for task in &self.tasks {
            let id = &task.task_id;
            if id.trim().is_empty() {
                return invalid("every task requires a taskId".into());
            }
            if task.episodes <= 0 {
                return invalid(format!("task {id}: episodes must be greater than zero"));
            }
            if !task.seeds.is_empty() {
                if task.seed.is_some() {
                    return invalid(format!("task {id}: seed and seeds are mutually exclusive"));
                }
                if task.seeds.len() as i64 != task.episodes {
                    return invalid(format!(
                        "task {id}: seeds has {} entries, want {} (one per episode)",
                        task.seeds.len(),
                        task.episodes
                    ));
                }
            }
            if let Some(workers) = task.workers {
                if workers < 1 {
                    return invalid(format!("task {id}: workers must be at least 1"));
                }
                if workers > task.episodes {
                    return invalid(format!(
                        "task {id}: workers ({workers}) exceeds episodes ({})",
                        task.episodes
                    ));
                }
                if let Some(key) = SHARD_KEYS.iter().find(|k| task.metadata.contains_key(**k)) {
                    return invalid(format!(
                        "task {id}: metadata key {key} is set by the plan when workers is given"
                    ));
                }
            }
            for name in &task.models {
                let model = name.split_once('@').map_or(name.as_str(), |(m, _)| m);
                if !self.models.iter().any(|m| m.model == model) {
                    return invalid(format!(
                        "task {id}: models filter names {name:?}, which is not in request.models"
                    ));
                }
            }
        }
        Ok(())
    }
}

fn refuse_fields(
    object: &Value,
    at: &str,
    accepted: &[&str],
    platform_only: &[(&str, &str)],
    refused: &mut Vec<String>,
) {
    for key in object.as_object().into_iter().flat_map(Map::keys) {
        if accepted.contains(&key.as_str()) {
            continue;
        }
        match platform_only.iter().find(|(name, _)| name == key) {
            Some((_, why)) => refused.push(format!("{at}{key}: {why}")),
            None => refused.push(format!("{at}{key}: not an evaluation request field")),
        }
    }
}

fn refuse_env_vars(object: &Value, at: &str, side: &str, refused: &mut Vec<String>) {
    for key in object["envVars"]
        .as_object()
        .into_iter()
        .flat_map(Map::keys)
    {
        if key != MAKE_KWARGS_VAR {
            refused.push(format!(
                "{at}envVars.{key}: a local {side} container gets no env vars besides \
                 {MAKE_KWARGS_VAR}"
            ));
        }
    }
}

/// What a plan reads about one image.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ImageEntry {
    /// The image reference to run.
    pub image: String,
    /// Its `dev.rlmesh.package` label, parsed; `null` when it has none.
    #[serde(default)]
    pub package: Value,
    /// An env's describe envelope; `null` for a model.
    #[serde(default)]
    pub describe: Value,
}

/// The images a request names, keyed by the request's own `model` and `env`
/// strings.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Images {
    /// Per `models[].model`.
    pub models: BTreeMap<String, ImageEntry>,
    /// Per `tasks[].env`.
    pub envs: BTreeMap<String, ImageEntry>,
}

/// One model server: a model ref at one checkpoint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanModel {
    /// The result label: the request's model, `@checkpoint` when the request
    /// runs it at several.
    pub label: String,
    /// The request's model.
    pub model: String,
    /// The checkpoint it runs, when its package declares any.
    pub checkpoint: Option<String>,
    /// The image.
    pub image: String,
    /// Its make kwargs: the package's, the request's over them, the
    /// checkpoint's params over both.
    pub params: Map<String, Value>,
    /// Its package declares `resourceClass: gpu`.
    pub gpu: bool,
}

/// One shard of a task split over workers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shard {
    /// 0-based.
    pub index: u64,
    /// The task's workers.
    pub count: u64,
}

/// One workload: one route, one env container, one result bundle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Workload {
    /// Stable within the request; names the workload's bundle.
    pub id: String,
    /// Index into [`Plan::models`].
    pub model: usize,
    /// The request's `taskId`.
    pub task: String,
    /// The shard, when the task splits over workers.
    pub shard: Option<Shard>,
    /// The request's env.
    pub env: String,
    /// The env image.
    pub env_image: String,
    /// The env's make kwargs.
    pub env_params: Map<String, Value>,
    /// The env's package declares `resourceClass: gpu`.
    pub env_gpu: bool,
    /// The scenario (`<group>/<scenario>`), on an env that has scenarios.
    pub scenario: Option<String>,
    /// The scenario's group, the workload's `config["suite"]`.
    pub suite: Option<String>,
    /// Episodes the workload runs.
    pub episodes: u64,
    /// The seed base, absent when `seeds` is given.
    pub seed: Option<i64>,
    /// Explicit seeds, one per episode.
    pub seeds: Vec<i64>,
    /// What the seeds key on: the env and its scenario, never the model.
    pub seed_key: String,
    /// The global slot of the workload's first episode within its seed key.
    pub seed_slot_offset: u64,
    /// The trial ordinal of its first episode, for an env that takes one
    /// (`resetOptions` names `trial_index`).
    pub trial_index_base: Option<u64>,
    /// The episode step cap the model's pairing (or the task) binds.
    pub max_episode_steps: Option<u64>,
}

impl Workload {
    /// Each episode's reset seed: the explicit seeds, else derived from the
    /// base, the seed key and the episode's slot.
    pub fn episode_seeds(&self) -> Vec<i64> {
        if !self.seeds.is_empty() {
            return self.seeds.clone();
        }
        let base = self.seed.unwrap_or(DEFAULT_SEED);
        (0..self.episodes)
            .map(|i| episode_seed(base, &self.seed_key, self.seed_slot_offset + i))
            .collect()
    }
}

/// A compiled request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Plan {
    /// One per model ref, in request order.
    pub models: Vec<PlanModel>,
    /// Models × tasks × shards, in that order.
    pub workloads: Vec<Workload>,
    /// The request's `maxActionChunk`.
    pub execution_horizon: Option<u32>,
    /// The request's `asyncPrefetchLead`.
    pub prefetch_lead: Option<u32>,
    /// What the plan could not carry over, for the caller to print.
    pub warnings: Vec<String>,
}

/// A task after `workers` split it.
struct TaskShard<'a> {
    task: &'a TaskRef,
    /// The shard's own id, `<taskId>--s<n>`.
    id: String,
    episodes: i64,
    seed: Option<i64>,
    seeds: Vec<i64>,
    metadata: BTreeMap<String, String>,
}

/// Compile `request` against the images it names.
pub fn compile(request: &EvalRequest, images: &Images) -> Result<Plan> {
    let shards = expand_workers(&request.tasks);
    let split = checkpoint_split(&request.models);
    let mut plan = Plan {
        models: Vec::new(),
        workloads: Vec::new(),
        execution_horizon: request.max_action_chunk,
        prefetch_lead: request.async_prefetch_lead,
        warnings: Vec::new(),
    };
    let mut matched = BTreeSet::new();
    let mut ids = BTreeSet::new();
    for model_ref in &request.models {
        let entry = images.models.get(&model_ref.model).ok_or_else(|| {
            Error::invalid_argument(format!("no image for model {}", model_ref.model))
        })?;
        let package = &entry.package;
        let checkpoints = list(&package["checkpoints"]);
        let checkpoint = resolve_checkpoint(model_ref, &checkpoints)?;
        let label = match (&checkpoint, split.contains(&model_ref.model)) {
            (Some(checkpoint), true) => format!("{}@{checkpoint}", model_ref.model),
            _ => model_ref.model.clone(),
        };
        for key in package["envVars"]
            .as_object()
            .into_iter()
            .flat_map(Map::keys)
        {
            if key != MAKE_KWARGS_VAR {
                plan.warnings.push(format!(
                    "{}: its package sets {key}, which a local container does not get",
                    model_ref.model
                ));
            }
        }
        let model_index = plan.models.len();
        plan.models.push(PlanModel {
            label: label.clone(),
            model: model_ref.model.clone(),
            checkpoint: checkpoint.clone(),
            image: entry.image.clone(),
            params: model_params(model_ref, package, &checkpoints, checkpoint.as_deref())?,
            gpu: package["resourceClass"] == "gpu",
        });
        for shard in &shards {
            let task = shard.task;
            let env = images.envs.get(&task.env).ok_or_else(|| {
                Error::invalid_argument(format!(
                    "task {}: no image for env {}",
                    task.task_id, task.env
                ))
            })?;
            let groups = scenario_groups(env);
            let scenario = resolve_scenario(&groups, task, &shard.id)?;
            if !task.models.is_empty() {
                if !task_models_match(&task.models, &model_ref.model, checkpoint.as_deref()) {
                    continue;
                }
                matched.insert(task.task_id.clone());
            }
            refuse_unscoped_checkpoint(model_ref, task, &checkpoints, &checkpoint, &scenario)?;
            let binding = bind_env(package, env, &groups, &scenario, task, &shard.id, model_ref)?;

            let mut seed_env = shard
                .metadata
                .get("environmentId")
                .filter(|id| !id.is_empty())
                .cloned()
                .unwrap_or_else(|| match &scenario {
                    Some(s) => format!("{}:{}", task.env, s.key),
                    None => task.env.clone(),
                });
            let delivers_trial = list(&env.package["resetOptions"])
                .iter()
                .any(|option| option == TRIAL_INDEX_OPTION);
            let (offset, trial) = match shard.metadata.get("episodeShardIndex") {
                None => (0, Some(0)),
                Some(index) => match shard
                    .metadata
                    .get("episodeShardOffset")
                    .and_then(|offset| offset.parse::<u64>().ok())
                {
                    Some(offset) => (offset, Some(offset)),
                    // A shard with no offset keys on its index (routeSeedKey's
                    // legacy branch) and has no global ordinal to offer.
                    None => {
                        seed_env = format!("{seed_env}#{index}");
                        (0, None)
                    }
                },
            };
            let id = format!("{}__{}", file_safe(&label), file_safe(&shard.id));
            if !ids.insert(id.clone()) {
                return Err(Error::invalid_argument(format!(
                    "two workloads of the request are both named {id}"
                )));
            }
            let shard_of = |key: &str| shard.metadata.get(key).and_then(|v| v.parse::<u64>().ok());
            plan.workloads.push(Workload {
                id,
                model: model_index,
                task: task.task_id.clone(),
                shard: shard_of(SHARD_KEYS[0])
                    .zip(shard_of(SHARD_KEYS[1]))
                    .map(|(index, count)| Shard { index, count }),
                env: task.env.clone(),
                env_image: env.image.clone(),
                env_params: binding.params,
                env_gpu: env.package["resourceClass"] == "gpu",
                suite: scenario
                    .as_ref()
                    .map(|s| s.key.split('/').next().unwrap_or_default().to_string()),
                scenario: scenario.map(|s| s.key),
                episodes: shard.episodes as u64,
                seed: shard.seed,
                seeds: shard.seeds.clone(),
                seed_key: seed_env,
                seed_slot_offset: offset,
                trial_index_base: trial.filter(|_| delivers_trial),
                max_episode_steps: binding.cap,
            });
        }
    }
    for task in &request.tasks {
        if !task.models.is_empty() && !matched.contains(&task.task_id) {
            return Err(Error::invalid_argument(format!(
                "task {}: models filter {:?} matches no request model@checkpoint",
                task.task_id, task.models
            )));
        }
    }
    Ok(plan)
}

/// `withDefaultSeeds`, `defaultWorkers` and `expandWorkers`: every task with
/// workers > 1 becomes that many shards of one episode space, the remainder
/// on the first, at consecutive offsets.
fn expand_workers(tasks: &[TaskRef]) -> Vec<TaskShard<'_>> {
    let total: i64 = tasks.iter().map(|t| t.episodes).sum();
    let per_worker = ((total + MAX_DEFAULT_WORKERS - 1) / MAX_DEFAULT_WORKERS).max(1);
    let mut shards = Vec::new();
    for task in tasks {
        let seed = match (task.seed, task.seeds.is_empty()) {
            (None, true) => Some(DEFAULT_SEED),
            (seed, _) => seed,
        };
        let workers = match task.workers {
            Some(workers) => workers,
            None if task.episodes > per_worker => (task.episodes + per_worker - 1) / per_worker,
            None => 1,
        };
        if workers <= 1 || task.episodes < workers {
            shards.push(TaskShard {
                task,
                id: task.task_id.clone(),
                episodes: task.episodes,
                seed,
                seeds: task.seeds.clone(),
                metadata: task.metadata.clone(),
            });
            continue;
        }
        let (each, remainder) = (task.episodes / workers, task.episodes % workers);
        let mut offset = 0;
        for i in 0..workers {
            let episodes = each + i64::from(i < remainder);
            let mut metadata = task.metadata.clone();
            for (key, value) in SHARD_KEYS.iter().zip([i, workers, offset]) {
                metadata.insert((*key).to_string(), value.to_string());
            }
            shards.push(TaskShard {
                task,
                id: format!("{}--s{}", task.task_id, i + 1),
                episodes,
                seed,
                seeds: task
                    .seeds
                    .get(offset as usize..(offset + episodes) as usize)
                    .map(<[i64]>::to_vec)
                    .unwrap_or_default(),
                metadata,
            });
            offset += episodes;
        }
    }
    shards
}

/// The models a request lists at two or more checkpoints.
fn checkpoint_split(models: &[ModelRef]) -> BTreeSet<String> {
    let mut checkpoints: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for model in models {
        checkpoints
            .entry(&model.model)
            .or_default()
            .insert(model.checkpoint.as_deref().unwrap_or_default());
    }
    checkpoints
        .into_iter()
        .filter(|(_, set)| set.len() > 1)
        .map(|(model, _)| model.to_string())
        .collect()
}

/// `resolveCheckpointName`: the named checkpoint (which must be declared),
/// else the default, else the first, else none.
fn resolve_checkpoint(model: &ModelRef, checkpoints: &[Value]) -> Result<Option<String>> {
    for checkpoint in checkpoints {
        let name = checkpoint["name"].as_str().unwrap_or_default();
        if !list(&checkpoint["scenarios"]).is_empty() || checkpoint.get("entrypoint").is_some() {
            // TODO: port checkpointForScenario's per-scenario pick and the
            // entrypoint override once a catalog image declares either.
            return Err(Error::invalid_argument(format!(
                "model {}: checkpoint {name} declares scenarios or an entrypoint, which local \
                 runs do not pick by yet",
                model.model
            )));
        }
    }
    let names: Vec<&str> = checkpoints
        .iter()
        .filter_map(|c| c["name"].as_str())
        .collect();
    match &model.checkpoint {
        Some(requested) if names.contains(&requested.as_str()) => Ok(Some(requested.clone())),
        Some(requested) => Err(Error::invalid_argument(format!(
            "model {} declares no checkpoint {requested:?}",
            model.model
        ))),
        None => Ok(checkpoints
            .iter()
            .find(|c| c["default"] == true)
            .and_then(|c| c["name"].as_str())
            .or(names.first().copied())
            .map(str::to_string)),
    }
}

/// `unscopedCheckpointReason`: a ref naming no checkpoint of a model that
/// declares several would run its default on every suite unchecked.
fn refuse_unscoped_checkpoint(
    model: &ModelRef,
    task: &TaskRef,
    checkpoints: &[Value],
    chosen: &Option<String>,
    scenario: &Option<Scenario>,
) -> Result<()> {
    let names_one = task.models.iter().any(|entry| {
        entry.contains('@')
            && task_models_match(std::slice::from_ref(entry), &model.model, chosen.as_deref())
    });
    if model.checkpoint.is_some() || scenario.is_none() || checkpoints.len() < 2 || names_one {
        return Ok(());
    }
    let names: Vec<&str> = checkpoints
        .iter()
        .filter_map(|c| c["name"].as_str())
        .collect();
    Err(Error::invalid_argument(format!(
        "task {}: model {} declares checkpoints {} but not which scenarios they fit, so its \
         default {} would run them unchecked; set models[].checkpoint to the one trained for them",
        task.task_id,
        model.model,
        names.join(", "),
        chosen.as_deref().unwrap_or_default()
    )))
}

/// A model's make kwargs: the package's `RLMESH_MAKE_KWARGS` (the request's
/// replaces it whole), then the checkpoint's params over it.
fn model_params(
    model: &ModelRef,
    package: &Value,
    checkpoints: &[Value],
    checkpoint: Option<&str>,
) -> Result<Map<String, Value>> {
    let raw = model
        .env_vars
        .get(MAKE_KWARGS_VAR)
        .map(String::as_str)
        .or(package["envVars"][MAKE_KWARGS_VAR].as_str());
    let mut params = match raw {
        Some(raw) => kwargs(raw, &format!("model {}", model.model))?,
        None => Map::new(),
    };
    if let Some(found) = checkpoints
        .iter()
        .find(|c| c["name"].as_str() == checkpoint)
    {
        for (key, value) in found["params"].as_object().into_iter().flatten() {
            params.insert(key.clone(), value.clone());
        }
    }
    Ok(params)
}

/// `taskModelsMatch` (variants are refused before this).
fn task_models_match(filter: &[String], model: &str, checkpoint: Option<&str>) -> bool {
    filter.iter().any(|entry| match entry.split_once('@') {
        Some((name, at)) => name == model && Some(at) == checkpoint,
        None => entry == model,
    })
}

/// One scenario of an env.
#[derive(Clone, Debug)]
struct Scenario {
    /// `<group>/<scenario>`.
    key: String,
    /// The make kwargs that pin it.
    params: Option<Map<String, Value>>,
}

/// An env's scenario groups: its package's `scenarioGroups`, else the
/// variants its describe envelope declares (`ScenarioGroupsFromDescribe`).
fn scenario_groups(env: &ImageEntry) -> Vec<(String, Vec<(String, Scenario)>)> {
    let mut groups: Vec<(String, Vec<(String, Scenario)>)> = Vec::new();
    let packaged = list(&env.package["scenarioGroups"]);
    if !packaged.is_empty() {
        for group in packaged {
            let id = text(&group["id"]);
            let scenarios = list(&group["scenarios"])
                .iter()
                .map(|s| {
                    let params = s["envVars"][MAKE_KWARGS_VAR]
                        .as_str()
                        .and_then(|raw| serde_json::from_str(raw).ok());
                    let key = format!("{id}/{}", text(&s["id"]));
                    (text(&s["id"]), Scenario { key, params })
                })
                .collect();
            groups.push((id, scenarios));
        }
        return groups;
    }
    for variant in list(&env.describe["variants"]["catalog"]) {
        if variant.get("error").is_some() {
            continue;
        }
        let id = match &variant["id"] {
            Value::Null => continue,
            Value::String(id) => id.clone(),
            other => other.to_string(),
        };
        let (group, scenario) = match id.rfind('/') {
            Some(0) => ("tasks".to_string(), id[1..].to_string()),
            Some(cut) => (id[..cut].to_string(), id[cut + 1..].to_string()),
            None => ("tasks".to_string(), id.clone()),
        };
        if scenario.is_empty() {
            continue;
        }
        let params = variant["params"]
            .as_object()
            .filter(|p| !p.is_empty())
            .cloned();
        let entry = (
            scenario.clone(),
            Scenario {
                key: format!("{group}/{scenario}"),
                params,
            },
        );
        match groups.iter_mut().find(|(g, _)| *g == group) {
            Some((_, scenarios)) => scenarios.push(entry),
            None => groups.push((group, vec![entry])),
        }
    }
    groups
}

/// `resolveScenario`: the scenario a task names on an env with scenarios,
/// by `<group>/<scenario>` (a `--` suffix ignored), a bare unique scenario
/// id, or `metadata.environmentId` (`<env>:<group>/<scenario>`).
fn resolve_scenario(
    groups: &[(String, Vec<(String, Scenario)>)],
    task: &TaskRef,
    shard_id: &str,
) -> Result<Option<Scenario>> {
    if groups.is_empty() {
        return Ok(None);
    }
    let mut key = shard_id.split("--").next().unwrap_or_default();
    if let Some(tag) = task
        .metadata
        .get("environmentId")
        .and_then(|id| id.strip_prefix(&format!("{}:", task.env)))
    {
        key = tag;
    }
    let mut bare = Vec::new();
    for (_, scenarios) in groups {
        for (id, scenario) in scenarios {
            if scenario.key == key {
                return Ok(Some(scenario.clone()));
            }
            if id == key {
                bare.push(scenario);
            }
        }
    }
    if bare.len() == 1 {
        return Ok(Some(bare[0].clone()));
    }
    if task.env_vars.contains_key(MAKE_KWARGS_VAR) {
        return Ok(None);
    }
    if bare.len() > 1 {
        return Err(Error::invalid_argument(format!(
            "task {}: scenario {key:?} is ambiguous on environment {}; name it as \
             <group>/<scenario>",
            task.task_id, task.env
        )));
    }
    let want = key.split('/').next().unwrap_or_default();
    let examples: Vec<&str> = groups
        .iter()
        .find(|(id, scenarios)| id == want && !scenarios.is_empty())
        .map(|(_, scenarios)| scenarios.iter().map(|(_, s)| s.key.as_str()).collect())
        .unwrap_or_else(|| {
            groups
                .iter()
                .filter_map(|(_, s)| s.first().map(|(_, s)| s.key.as_str()))
                .collect()
        });
    Err(Error::invalid_argument(format!(
        "task {}: environment {} has no scenario {key:?}; taskId names one as \
         <group>/<scenario> (e.g. {}), or envVars pins {MAKE_KWARGS_VAR}",
        task.task_id,
        task.env,
        examples.first().copied().unwrap_or_default()
    )))
}

struct Binding {
    params: Map<String, Value>,
    cap: Option<u64>,
}

/// `bindEnvironment` without the camera dial (the sandbox binds the model's
/// render size on the env itself) or the expected-branch assertion: the
/// scenario's (or the task's) make kwargs, the model's pairing for the env
/// filled in under them key by key, and the episode cap lifted out of both.
fn bind_env(
    model_package: &Value,
    env: &ImageEntry,
    groups: &[(String, Vec<(String, Scenario)>)],
    scenario: &Option<Scenario>,
    task: &TaskRef,
    task_id: &str,
    model: &ModelRef,
) -> Result<Binding> {
    let invalid = |message: String| Error::invalid_argument(format!("task {task_id}: {message}"));
    let present = match task.env_vars.get(MAKE_KWARGS_VAR) {
        Some(raw) => Some(kwargs(raw, &format!("task {task_id}"))?),
        None => scenario.as_ref().and_then(|s| s.params.clone()),
    };
    let pairing = pairing_for(model_package, &task.env, &env.package).map_err(invalid)?;
    let mut scenario_key = scenario.as_ref().map(|s| s.key.clone());
    if scenario_key.is_none() && pairing.is_some_and(|p| !obj(&p["scenarios"]).is_empty()) {
        scenario_key = present.as_ref().and_then(|want| {
            groups
                .iter()
                .flat_map(|(_, s)| s)
                .find(|(_, s)| s.params.as_ref().is_some_and(|p| same_kwargs(want, p)))
                .map(|(_, s)| s.key.clone())
        });
        if scenario_key.is_none() {
            return Err(invalid(format!(
                "pins {MAKE_KWARGS_VAR} that matches no scenario of {}, but {} declares \
                 per-scenario pairing params for it; name the scenario as <group>/<scenario>",
                task.env, model.model
            )));
        }
    }
    let mut bound = Map::new();
    if let Some(pairing) = pairing {
        bound.extend(obj(&pairing["params"]));
        if let Some(key) = &scenario_key {
            let group = key.split('/').next().unwrap_or_default();
            for layer in [group, key.as_str()] {
                if !layer.is_empty() {
                    bound.extend(obj(&pairing["scenarios"][layer]));
                }
            }
        }
    }
    let mut cap = take_cap(&mut bound, &model.model).map_err(invalid)?;
    let mut present = present.unwrap_or_default();
    if let Some(pinned) = take_cap(&mut present, &format!("task {task_id}")).map_err(invalid)? {
        cap = Some(pinned);
    }
    if cap.is_none() && env.package["requiresEpisodeCap"] == true {
        return Err(invalid(format!(
            "{} has no episode horizon of its own (requiresEpisodeCap), and {} pairs no \
             max_episode_steps for it under any of its names ({}); bind one in the model's \
             compatibility.envParams",
            task.env,
            model.model,
            env_names(&task.env, &env.package).concat().join(", ")
        )));
    }
    if let Some(cap) = cap
        && env_declares_param(&env.describe, EPISODE_CAP_KEYS[0])
    {
        bound.insert(EPISODE_CAP_KEYS[0].to_string(), cap.into());
    }
    let discriminants = discriminants(&env.describe);
    let mut keys: Vec<&String> = bound.keys().collect();
    keys.sort();
    for key in keys {
        let value = &bound[key];
        match present.get(key) {
            None => {
                present.insert(key.clone(), value.clone());
            }
            Some(existing) if same_param(existing, value) => {}
            Some(existing) if discriminants.contains(key) => {
                return Err(invalid(format!(
                    "pins {key}={existing}, but {} pairs {key} with {key}={value}; a \
                     discriminant decides which contract the environment serves, so the two \
                     cannot both hold",
                    model.model
                )));
            }
            // The caller is more specific than the catalog.
            Some(_) => {}
        }
    }
    Ok(Binding {
        params: present,
        cap,
    })
}

/// `envCompatNames`: the tiers a pairing table is searched by, most specific
/// first: the env's id, then its name, flattened id and leaf, then its gym id.
fn env_names(env: &str, package: &Value) -> Vec<Vec<String>> {
    let id = repository(env);
    let leaf = id.rsplit('/').next().unwrap_or_default().to_string();
    let name = package["name"].as_str().unwrap_or(id).to_string();
    let gym = package["gymId"]
        .as_str()
        .map_or(leaf.clone(), str::to_string);
    vec![
        vec![id.to_string()],
        vec![name, id.replace('/', "-"), leaf],
        vec![gym],
    ]
}

/// `PairingFor`: the model's pairing for the env, under its most specific
/// name; two names of one tier is an ambiguity.
fn pairing_for<'a>(
    model_package: &'a Value,
    env: &str,
    env_package: &Value,
) -> std::result::Result<Option<&'a Value>, String> {
    let table = &model_package["compatibility"]["envParams"];
    for tier in env_names(env, env_package) {
        let mut hits: Vec<&String> = Vec::new();
        for name in &tier {
            if !name.is_empty() && table.get(name).is_some() && !hits.contains(&name) {
                hits.push(name);
            }
        }
        match hits.as_slice() {
            [] => {}
            [one] => return Ok(Some(&table[one.as_str()])),
            many => {
                let mut many: Vec<&str> = many.iter().map(|s| s.as_str()).collect();
                many.sort_unstable();
                return Err(format!(
                    "the model pairs environment {env} under {}; keep one",
                    many.join(" and ")
                ));
            }
        }
    }
    Ok(None)
}

/// `takeEpisodeCap`: lift the reserved cap keys out of `bound`.
fn take_cap(
    bound: &mut Map<String, Value>,
    owner: &str,
) -> std::result::Result<Option<u64>, String> {
    let mut cap = None;
    for key in EPISODE_CAP_KEYS {
        let Some(value) = bound.remove(key) else {
            continue;
        };
        let steps = positive_int32(&value).ok_or_else(|| {
            format!("{owner} pairs {key}={value}, which is not a positive step count")
        })?;
        if cap.is_some_and(|c| c != steps) {
            return Err(format!(
                "{owner} pairs both max_episode_steps and max_steps with different values; keep one"
            ));
        }
        cap = Some(steps);
    }
    Ok(cap)
}

/// `ParamSpecNames`: whether the env's describe declares `name` as a make
/// param (a passthrough spec declares nothing).
fn env_declares_param(describe: &Value, name: &str) -> bool {
    let params = &describe["params"];
    let spec = &params["param_spec"];
    if !spec.is_object() || spec["extra"] == "passthrough" {
        return false;
    }
    list(&spec["params"])
        .iter()
        .chain(&list(&params["signature_tier"]))
        .any(|entry| entry["name"] == name)
}

fn discriminants(describe: &Value) -> Vec<String> {
    let contracts = &describe["env_contracts"];
    if contracts.get("error").is_some() {
        return Vec::new();
    }
    list(&contracts["discriminants"])
        .iter()
        .filter_map(|d| d.as_str().map(str::to_string))
        .collect()
}

/// `PositiveInt32`.
fn positive_int32(value: &Value) -> Option<u64> {
    let n = match value {
        Value::Number(n) => n.as_i64().or_else(|| {
            n.as_f64()
                .filter(|f| f.fract() == 0.0 && f.abs() < 1e18)
                .map(|f| f as i64)
        })?,
        Value::String(s) => s.parse::<i32>().ok()?.into(),
        _ => return None,
    };
    (1..=i64::from(i32::MAX)).contains(&n).then_some(n as u64)
}

/// `SameParam`: equal once integral floats read as integers.
fn same_param(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| same_param(a, b))
        }
        _ => a == b,
    }
}

fn same_kwargs(a: &Map<String, Value>, b: &Map<String, Value>) -> bool {
    !a.is_empty()
        && a.len() == b.len()
        && a.iter()
            .all(|(k, v)| b.get(k).is_some_and(|o| same_param(v, o)))
}

/// A make-kwargs blob.
fn kwargs(raw: &str, owner: &str) -> Result<Map<String, Value>> {
    serde_json::from_str(raw).map_err(|err| {
        Error::invalid_argument(format!(
            "{owner}: {MAKE_KWARGS_VAR} must be a JSON object: {err}"
        ))
    })
}

/// An image reference without its tag or digest, as the platform's catalog
/// id is a repository path.
fn repository(image: &str) -> &str {
    let image = image.split('@').next().unwrap_or(image);
    match image.rfind(':') {
        Some(colon) if !image[colon..].contains('/') => &image[..colon],
        _ => image,
    }
}

fn file_safe(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '.' | '-' | '_' => c,
            _ => '_',
        })
        .collect()
}

fn list(value: &Value) -> Vec<Value> {
    value.as_array().cloned().unwrap_or_default()
}

fn obj(value: &Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap_or_default()
}

fn text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Equal JSON, with numbers compared by value (the platform's encoder
    /// writes `0.0` as `0`).
    fn same_json(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Object(x), Value::Object(y)) => {
                x.len() == y.len()
                    && x.iter()
                        .all(|(k, v)| y.get(k).is_some_and(|o| same_json(v, o)))
            }
            (Value::Array(x), Value::Array(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(a, b)| same_json(a, b))
            }
            _ => same_param(a, b),
        }
    }

    #[test]
    fn eval_plans_match_the_cross_language_fixture() {
        let fixture: Value =
            serde_json::from_str(include_str!("../../conformance/eval_plans.v1.json")).unwrap();
        let images: Images = serde_json::from_value(fixture["images"].clone()).unwrap();
        let cases = fixture["cases"].as_array().unwrap();
        assert!(!cases.is_empty());
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let request = EvalRequest::parse(&case["request"]).unwrap();
            let outcome = compile(&request, &images);
            if let Some(error) = case["error"].as_str() {
                let message = outcome.err().map(|e| e.to_string()).unwrap_or_default();
                assert!(
                    message.contains(error),
                    "{name}: {message:?} lacks {error:?}"
                );
                continue;
            }
            let plan = outcome.unwrap_or_else(|err| panic!("{name}: {err}"));
            let got: Vec<Value> = plan
                .workloads
                .iter()
                .map(|w| {
                    serde_json::json!({
                        "model": plan.models[w.model].model,
                        "task": w.task,
                        "shard": w.shard.map(|s| s.index),
                        "episodes": w.episodes,
                        "seed": w.seed,
                        "seedKey": w.seed_key,
                        "seedSlotOffset": w.seed_slot_offset,
                        "trialIndexBase": w.trial_index_base,
                        "maxEpisodeSteps": w.max_episode_steps,
                        "envParams": w.env_params,
                        "episodeSeeds": w.episode_seeds(),
                    })
                })
                .collect();
            let want = case["workloads"].as_array().unwrap();
            assert_eq!(got.len(), want.len(), "{name}: workload count");
            for (got, want) in got.iter().zip(want) {
                assert!(same_json(got, want), "{name}:\n got {got}\nwant {want}");
            }
        }
    }

    #[test]
    fn legacy_shard_metadata_without_an_offset_keys_on_the_shard_index() {
        let fixture: Value =
            serde_json::from_str(include_str!("../../conformance/eval_plans.v1.json")).unwrap();
        let images: Images = serde_json::from_value(fixture["images"].clone()).unwrap();
        let request = EvalRequest::parse(&serde_json::json!({
            "models": [{"model": "prebuilt/noop/base"}],
            "tasks": [{
                "taskId": "libero_spatial/0",
                "env": "prebuilt/libero",
                "episodes": 1,
                "metadata": {"episodeShardIndex": "1", "episodeShardCount": "2"},
            }],
        }))
        .unwrap();
        let plan = compile(&request, &images).unwrap();
        let workload = &plan.workloads[0];
        assert!(workload.seed_key.ends_with("#1"), "{}", workload.seed_key);
        assert_eq!(workload.seed_slot_offset, 0);
        assert_eq!(workload.trial_index_base, None);
    }

    #[test]
    fn fields_a_local_run_cannot_honour_are_refused_by_name() {
        let base = serde_json::json!({
            "models": [{"model": "m"}],
            "tasks": [{"taskId": "t", "env": "e", "episodes": 2}],
        });
        for (edit, named) in [
            ("/managed", "managed:"),
            ("/maxBatchSize", "maxBatchSize:"),
            ("/projectId", "projectId:"),
            ("/name", "name: not an evaluation request field"),
            ("/models/0/variant", "models[0].variant:"),
            ("/models/0/envVars", "models[0].envVars.HF_TOKEN:"),
            ("/tasks/0/perturbation", "tasks[0].perturbation:"),
            ("/tasks/0/pairing", "tasks[0].pairing: transfer"),
        ] {
            let mut request = base.clone();
            let (parent, key) = edit.rsplit_once('/').unwrap();
            let value = match key {
                "envVars" => serde_json::json!({"HF_TOKEN": "x"}),
                "pairing" => "transfer".into(),
                _ => serde_json::json!({"clusterId": "c"}),
            };
            request.pointer_mut(parent).unwrap()[key] = value;
            let error = EvalRequest::parse(&request).unwrap_err().to_string();
            assert!(error.contains(named), "{edit}: {error}");
        }
        assert!(EvalRequest::parse(&base).is_ok());
    }
}
