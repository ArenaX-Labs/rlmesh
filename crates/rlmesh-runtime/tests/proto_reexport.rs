//! Verifies that the protocol types appearing in rlmesh-runtime's public API
//! are nameable through the crate's own `rlmesh_proto` re-export, so a
//! downstream `RuntimeEnv`/`RuntimeModel`/`RuntimeHooks` implementor does not
//! need an independent `rlmesh-proto` dependency. Review findings #49 / #78.

// Reference the types only via the re-export path. These aliases must compile
// and must be the same types the public trait/event signatures use.
use rlmesh_runtime::rlmesh_proto::common::v1::MessageBytes;
use rlmesh_runtime::rlmesh_proto::env::v1::{ResetRequest, ResetResponse, StepRequest};
use rlmesh_runtime::rlmesh_proto::model::v1::{PredictRequest, PredictResponse};
use rlmesh_runtime::rlmesh_proto::spaces::v1::SpaceSpec;

#[tokio::test]
async fn proto_types_are_reachable_through_runtime_reexport() {
    // Construct the re-exported types to prove the path resolves to real,
    // usable items (not just a module alias).
    let reset_request = ResetRequest::default();
    let _reset_response = ResetResponse::default();
    let step_request = StepRequest::default();
    let _predict_request = PredictRequest::default();
    let _predict_response = PredictResponse::default();
    let _space_spec = SpaceSpec::default();
    let _message_bytes = MessageBytes::default();

    // Drive a real RuntimeEnv whose signatures are spelled entirely through the
    // re-export, feeding re-export-typed requests in: this only compiles and
    // runs if the re-exported types are identical to the ones the trait uses.
    use rlmesh_runtime::RuntimeEnv;
    let mut env = NoopEnv;
    env.reset(reset_request).await.unwrap();
    env.step(step_request).await.unwrap();
}

// Compile-time proof that the re-exported types are the same ones the public
// `RuntimeEnv` trait names: implementing the trait while spelling the argument
// types only through the re-export only compiles if they are identical. If the
// re-export pointed at a different `rlmesh-proto`, this would fail to build.
struct NoopEnv;

#[async_trait::async_trait]
impl rlmesh_runtime::RuntimeEnv for NoopEnv {
    async fn reset(
        &mut self,
        _request: rlmesh_runtime::rlmesh_proto::env::v1::ResetRequest,
    ) -> Result<rlmesh_runtime::RuntimeEnvReset, rlmesh_runtime::RuntimeError> {
        Ok(rlmesh_runtime::RuntimeEnvReset {
            response: Default::default(),
            telemetry: None,
        })
    }

    async fn step(
        &mut self,
        _request: rlmesh_runtime::rlmesh_proto::env::v1::StepRequest,
    ) -> Result<rlmesh_runtime::RuntimeEnvStep, rlmesh_runtime::RuntimeError> {
        Ok(rlmesh_runtime::RuntimeEnvStep {
            response: Default::default(),
            telemetry: None,
        })
    }
}
