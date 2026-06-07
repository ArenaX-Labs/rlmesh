use rlmesh_proto::common::v1::MessageBytes;
use rlmesh_proto::model::v1::{CloseRouteRequest, PredictContext, PredictRequest};
use rlmesh_proto::spaces::v1::SpaceValue;

use crate::state::RouteState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RequestPhase {
    ResetObservation,
    StepObservation,
}

impl RequestPhase {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::ResetObservation => "reset_observation",
            Self::StepObservation => "step_observation",
        }
    }
}

fn bytes_value(value: MessageBytes) -> SpaceValue {
    SpaceValue { bytes: Some(value) }
}

impl RouteState {
    pub(crate) fn predict_request(
        &mut self,
        observation: Option<MessageBytes>,
        phase: RequestPhase,
    ) -> PredictRequest {
        PredictRequest {
            context: Some(PredictContext {
                session_id: self.session_id().to_string(),
                route_id: self.route_id().to_string(),
                request_id: self.next_request_id(phase.as_str()),
                slots: self.slots(),
            }),
            observation: observation.map(bytes_value),
        }
    }

    pub(crate) fn close_route_request(&mut self, reason: impl Into<String>) -> CloseRouteRequest {
        CloseRouteRequest {
            context: Some(PredictContext {
                session_id: self.session_id().to_string(),
                route_id: self.route_id().to_string(),
                request_id: self.next_request_id("close_route"),
                slots: self.slots(),
            }),
            reason: reason.into(),
        }
    }
}
