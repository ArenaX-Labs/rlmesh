use crate::spaces;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelRouteSlot {
    pub episode_id: String,
    pub env_index: i32,
    pub step: i64,
    pub reset: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelRouteContext {
    pub session_id: String,
    pub route_id: String,
    pub request_id: String,
    pub slots: Vec<ModelRouteSlot>,
}

impl ModelRouteContext {
    pub fn primary_episode_id(&self) -> &str {
        self.primary_slot()
            .map(|slot| slot.episode_id.as_str())
            .unwrap_or("")
    }

    pub fn primary_env_index(&self) -> i32 {
        self.primary_slot().map_or(0, |slot| slot.env_index)
    }

    pub fn primary_step(&self) -> i64 {
        self.primary_slot().map_or(0, |slot| slot.step)
    }

    pub fn primary_reset(&self) -> bool {
        self.primary_slot().is_some_and(|slot| slot.reset)
    }

    pub fn episode_ids(&self) -> Vec<String> {
        self.slots
            .iter()
            .map(|slot| slot.episode_id.clone())
            .collect()
    }

    pub fn primary_slot(&self) -> Option<&ModelRouteSlot> {
        self.slots.first()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelObservation {
    pub observation: Option<spaces::BinaryPayload>,
    pub route: ModelRouteContext,
    pub reset: bool,
    pub num_envs: usize,
    pub env_contract: Option<spaces::EnvContract>,
}

impl ModelObservation {
    pub fn episode_id(&self) -> &str {
        self.route.primary_episode_id()
    }

    pub fn episode_ids(&self) -> Vec<String> {
        self.route.episode_ids()
    }

    pub fn step(&self) -> i64 {
        self.route.primary_step()
    }

    pub fn env_index(&self) -> i32 {
        self.route.primary_env_index()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelEpisodeEnd {
    pub episode_id: String,
    pub env_index: i32,
}
