//! Standard gRPC health service (`grpc.health.v1`) wiring.
//!
//! RLMesh servers bind their listener up front (see the bind-first
//! `BoundEnvServer` / `BoundModelServer` paths), so by the time the health
//! service is registered the socket is already accepting connections. We
//! therefore mark the overall server health as `SERVING` immediately and hand
//! back the ready-to-add tonic service.
//!
//! The signal is the standard `grpc.health.v1.Health` service — no
//! rlmesh-specific protocol surface is added. Supervisors and probes can issue
//! `Check`/`Watch` against the empty (overall) service name and observe
//! `SERVING` once the listener is up.

use tonic_health::ServingStatus;
use tonic_health::pb::health_server::HealthServer;
use tonic_health::server::{HealthReporter, health_reporter};

/// Build a `grpc.health.v1` health service with the overall server health
/// already marked `SERVING`, ready to add to a tonic `Server::builder()`.
///
/// The returned [`HealthReporter`] is handed back so callers may flip the
/// status to `NotServing` during drain/shutdown if they wish; RLMesh keeps the
/// service always-on and does not currently downgrade it.
pub async fn serving_health_service() -> (
    HealthReporter,
    HealthServer<impl tonic_health::pb::health_server::Health>,
) {
    let (reporter, service) = health_reporter();
    // The empty service name is the gRPC convention for overall server health.
    reporter
        .set_service_status("", ServingStatus::Serving)
        .await;
    (reporter, service)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn overall_health_is_serving() {
        let (_reporter, _service) = serving_health_service().await;
        // Construction succeeding with the overall status set is the contract
        // exercised end-to-end by the rlmesh-crate health client tests; here we
        // just assert the helper builds without panicking.
    }
}
