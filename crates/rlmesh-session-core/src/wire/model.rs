//! Served models: one request, one reply, on `m/{model}`.
//!
//! ```text
//! driver ──ModelRequest::Resolve{route, contract}──▶ model   (once per env)
//! driver ──ModelRequest::Predict{groups}──────────▶ model   (per decision)
//! driver ◀──ModelReply::Predicted{outcomes}──────── model
//! ```
//!
//! A route is one env as the model sees it: its contract, resolved once.
//! A predict carries any number of routes' rows (groups), so one caller's
//! envs share one forward pass when the model fuses them; each group gets
//! its own outcome. Values are leaf slabs (`rlmesh-spaces`' leaf codec:
//! one buffer per leaf of the route's space, rows row-major).

use bytes::Bytes;

/// Leaf slabs of some number of rows (the count travels beside them).
pub type Leaves = Vec<Bytes>;

/// One row's episode.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelEpisode {
    /// The episode id.
    pub id: String,
    /// The episode's seed, when it has one.
    pub seed: Option<i64>,
}

/// Resolves a route: the model adapts to the env's contract.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelResolve {
    /// The route (the env as the driver names it).
    pub route: String,
    /// The env contract as JSON (spaces, metadata and tags).
    pub contract_json: String,
    /// Actions of each chunk the driver executes before re-planning.
    pub execution_horizon: u32,
    /// Whether the driver sends the steps it replayed as history.
    pub delivers_history: bool,
    /// The workflow edition the driver pinned (empty: unpinned).
    pub workflow_edition: String,
}

/// What a resolved route needs from its driver.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelResolved {
    /// The model's native chunk length, when declared.
    pub native_chunk: Option<u32>,
    /// The keys of the frame windows the route keeps, when it keeps any
    /// (the driver then sends every replayed step as history), and whether
    /// rows may be pruned to them.
    pub history: Option<(Vec<String>, bool)>,
    /// Whether the model adds options to the resets that start its
    /// episodes (the driver then asks before each).
    pub reset_options: bool,
}

/// A step the driver executed from a chunk without predicting.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelHistoryRow {
    /// Its observation, one row per episode.
    pub observation: Leaves,
    /// Its rows' episodes.
    pub episodes: Vec<ModelEpisode>,
    /// The step.
    pub step: i64,
}

/// One route's rows in a predict.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelGroup {
    /// The route.
    pub route: String,
    /// One per row.
    pub episodes: Vec<ModelEpisode>,
    /// The rows' observation.
    pub observation: Leaves,
    /// Replayed steps since the last predict, oldest first.
    pub history: Vec<ModelHistoryRow>,
    /// The step this predict decides.
    pub step: Option<i64>,
}

/// A refusal or failure.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelFault {
    /// The error code (`INTERNAL`, `INVALID_REQUEST`, `UNSUPPORTED`).
    pub code: String,
    /// What went wrong.
    pub message: String,
    /// Whether the episode may go on.
    pub recoverable: bool,
}

/// One group's outcome: frame 0 (this step) and the chunk's replay
/// frames, each one row per episode, or its fault.
pub type ModelOutcome = Result<Vec<Leaves>, ModelFault>;

/// A request to a served model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelRequest {
    /// Resolves a route.
    Resolve(ModelResolve),
    /// Predicts for each group.
    Predict(Vec<ModelGroup>),
    /// Ends a route's episodes (empty: all of them).
    Reset {
        /// The route.
        route: String,
        /// The episodes.
        episode_ids: Vec<String>,
    },
    /// Drops a route.
    Release {
        /// The route.
        route: String,
    },
    /// Asks for the model's describe envelope.
    Describe,
    /// Asks for the options the model adds to the reset that starts
    /// `episode`, as trial `trial_index`.
    ResetOptions {
        /// The route.
        route: String,
        /// The episode the reset starts.
        episode: ModelEpisode,
        /// Its trial.
        trial_index: Option<u64>,
    },
}

/// A served model's reply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelReply {
    /// A route resolved.
    Resolved(ModelResolved),
    /// One outcome per group, in order.
    Predicted(Vec<ModelOutcome>),
    /// A reset or release done.
    Done,
    /// The describe envelope (JSON), empty when the model has none.
    Described(String),
    /// The reset options (a JSON object), empty when the model adds none.
    Options(String),
    /// The request failed as a whole.
    Fault(ModelFault),
}
