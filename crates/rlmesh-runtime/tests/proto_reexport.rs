//! Checks that runtime protocol types are reachable through `rlmesh_proto`.

use rlmesh_runtime::rlmesh_proto::env::v1::{ResetRequest, ResetResponse, StepRequest};
use rlmesh_runtime::rlmesh_proto::model::v1::{PredictRequest, PredictResponse};
use rlmesh_runtime::rlmesh_proto::spaces::v1::{SpaceSpec, SpaceValue};

#[tokio::test]
async fn proto_types_are_reachable_through_runtime_reexport() {
    let reset_request = ResetRequest::default();
    let _reset_response = ResetResponse::default();
    let step_request = StepRequest::default();
    let _predict_request = PredictRequest::default();
    let _predict_response = PredictResponse::default();
    let _space_spec = SpaceSpec::default();
    let _space_value = SpaceValue::default();

    use rlmesh_runtime::RuntimeEnv;
    let mut env = NoopEnv;
    env.reset(reset_request).await.unwrap();
    env.step(step_request).await.unwrap();
}

struct NoopEnv;

#[async_trait::async_trait]
impl rlmesh_runtime::RuntimeEnv for NoopEnv {
    async fn reset(
        &mut self,
        _request: rlmesh_runtime::rlmesh_proto::env::v1::ResetRequest,
    ) -> Result<rlmesh_runtime::RuntimeEnvReset, rlmesh_runtime::RuntimeError> {
        Ok(rlmesh_runtime::RuntimeEnvReset {
            response: Default::default(),
            endpoint_total_ns: None,
        })
    }

    async fn step(
        &mut self,
        _request: rlmesh_runtime::rlmesh_proto::env::v1::StepRequest,
    ) -> Result<rlmesh_runtime::RuntimeEnvStep, rlmesh_runtime::RuntimeError> {
        Ok(rlmesh_runtime::RuntimeEnvStep {
            response: Default::default(),
            endpoint_total_ns: None,
        })
    }
}
