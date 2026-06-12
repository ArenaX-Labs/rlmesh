use crate::spaces;

/// One sub-environment's position within a routed predict request.
///
/// A vectorized predict carries one slot per active sub-environment.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelRouteSlot {
    /// Identifier of the episode this slot belongs to.
    pub episode_id: String,
    /// Index of the sub-environment.
    pub env_index: i32,
    /// Step number within the episode.
    pub step: i64,
    /// Whether this slot is the first step of a new episode.
    pub reset: bool,
}

/// Routing metadata attached to a [`ModelObservation`].
///
/// Identifies the session/route the request belongs to and the per-slot episode
/// state. The `primary_*` accessors read the first slot, which is the natural
/// choice for single-env handlers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelRouteContext {
    /// Identifier of the session this request belongs to.
    pub session_id: String,
    /// Identifier of the route within the session.
    pub route_id: String,
    /// Identifier of this individual request.
    pub request_id: String,
    /// One slot per active sub-environment.
    pub slots: Vec<ModelRouteSlot>,
}

impl ModelRouteContext {
    /// The primary (first) slot's episode id, or `""` if there are no slots.
    pub fn primary_episode_id(&self) -> &str {
        self.primary_slot()
            .map(|slot| slot.episode_id.as_str())
            .unwrap_or("")
    }

    /// The primary slot's env index, or `0` if there are no slots.
    pub fn primary_env_index(&self) -> i32 {
        self.primary_slot().map_or(0, |slot| slot.env_index)
    }

    /// The primary slot's step number, or `0` if there are no slots.
    pub fn primary_step(&self) -> i64 {
        self.primary_slot().map_or(0, |slot| slot.step)
    }

    /// Whether the primary slot marks the start of a new episode.
    pub fn primary_reset(&self) -> bool {
        self.primary_slot().is_some_and(|slot| slot.reset)
    }

    /// The episode id of every slot, in slot order.
    pub fn episode_ids(&self) -> Vec<String> {
        self.slots
            .iter()
            .map(|slot| slot.episode_id.clone())
            .collect()
    }

    /// The primary (first) slot, if any.
    pub fn primary_slot(&self) -> Option<&ModelRouteSlot> {
        self.slots.first()
    }
}

/// An observation delivered to [`ModelHandler::predict`](crate::ModelHandler::predict).
///
/// Carries the encoded observation payload plus the routing context and the
/// env contract / batch size needed to decode it. The convenience accessors
/// forward to the primary slot of [`route`](ModelObservation::route).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelObservation {
    /// The encoded observation payload, if present.
    pub observation: Option<spaces::BinaryPayload>,
    /// Routing/episode metadata for this request.
    pub route: ModelRouteContext,
    /// Whether this request starts a new episode.
    pub reset: bool,
    /// Number of sub-environments in this route's batch.
    pub num_envs: usize,
    /// The env contract (spaces/metadata) for decoding the observation.
    pub env_contract: Option<spaces::EnvContract>,
}

impl ModelObservation {
    /// The primary slot's episode id (see [`ModelRouteContext::primary_episode_id`]).
    pub fn episode_id(&self) -> &str {
        self.route.primary_episode_id()
    }

    /// The episode id of every slot (see [`ModelRouteContext::episode_ids`]).
    pub fn episode_ids(&self) -> Vec<String> {
        self.route.episode_ids()
    }

    /// The primary slot's step number.
    pub fn step(&self) -> i64 {
        self.route.primary_step()
    }

    /// The primary slot's env index.
    pub fn env_index(&self) -> i32 {
        self.route.primary_env_index()
    }
}

/// Notification that an episode has ended, passed to
/// [`ModelHandler::on_episode_end`](crate::ModelHandler::on_episode_end).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelEpisodeEnd {
    /// Identifier of the episode that ended.
    pub episode_id: String,
    /// Index of the sub-environment that produced the episode.
    pub env_index: i32,
}
