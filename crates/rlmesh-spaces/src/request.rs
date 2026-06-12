use crate::MetaMap;
use crate::spaces::SpaceValue;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ResetRequest {
    pub seed: Option<i64>,
    pub options: Option<MetaMap>,
    pub timeout_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct StepRequest {
    pub action: Option<SpaceValue>,
    pub timeout_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CloseRequest {
    pub wait_for_episodes: bool,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ResetResult {
    pub observation: Option<SpaceValue>,
    pub info: Option<MetaMap>,
    pub episode_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct StepResult {
    pub observation: Option<SpaceValue>,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    pub info: Option<MetaMap>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CloseResult;
