//! Route edition reconciliation (the session floor) for the layer that binds a
//! route. It takes the env and model offers learned at their handshakes, adds this
//! runtime build's own supported window, and returns the floor all three speak.
//!
//! Lives here — not in the `RemoteModel` facade — so the production runtime, which
//! drives the env and model clients directly, computes and sets the same pin
//! instead of leaving the workflow edition unset.

use rlmesh_proto::{RuntimeCap, SessionFloor, SessionOffer, negotiate_session_floor};

use crate::error::{Error as GrpcError, ProtocolError};

/// Reconcile the route's workflow edition across the env, the model, and this
/// runtime build — see [`rlmesh_proto::negotiate_session_floor`]. `env_offer` and
/// `model_offer` are the editions each peer declared at its handshake (from
/// [`EnvHandshake::session_offer`](crate::EnvHandshake::session_offer) and
/// [`ModelClient::model_session_offer`](crate::ModelClient::model_session_offer)).
/// `declared` is the runtime tier's own WANT — the explicit pin the caller was
/// given (`run(workflow_edition=...)`) — or `None` to declare nothing, which
/// wants this build's current edition and so caps no peer.
///
/// Warns when this runtime is the tier holding the edition back (the session still
/// runs, safely, at the floor). Errs when the three share no edition — the caller
/// must fail before opening any Join stream. The returned
/// [`SessionFloor::selected_workflow_edition`] is what the runtime pins onto the
/// model's `ResolveAdapter` and, through
/// [`EnvClient::pin_workflow_edition`](crate::EnvClient::pin_workflow_edition),
/// onto the env's `ConfigureEnv`.
pub fn env_floor(
    env_offer: &SessionOffer,
    model_offer: &SessionOffer,
    declared: Option<&str>,
) -> Result<SessionFloor, GrpcError> {
    let runtime_offer = SessionOffer::this_build(declared);
    let floor =
        negotiate_session_floor(env_offer, model_offer, &runtime_offer).map_err(|refusal| {
            GrpcError::from(ProtocolError::HandshakeFailed(format!(
                "no mutual workflow edition across env, model, and runtime: {refusal}; the \
             runtime re-frames env<->model traffic, so a session can only run at an edition all \
             three support"
            )))
        })?;
    match floor.runtime_cap {
        // A CAN-set limit: the runtime genuinely cannot do what its peers can.
        Some(RuntimeCap::Capability) => tracing::warn!(
            selected_workflow_edition = %floor.selected_workflow_edition,
            desired_workflow_edition = %floor.desired_workflow_edition,
            "runtime is the limiting tier: env and model support workflow edition {} but this \
             runtime caps the session to {} — upgrade the runtime to run at the newer edition",
            floor.desired_workflow_edition,
            floor.selected_workflow_edition,
        ),
        // An explicit declaration: sticky editions working as intended, not a defect.
        Some(RuntimeCap::Declaration) => tracing::info!(
            selected_workflow_edition = %floor.selected_workflow_edition,
            desired_workflow_edition = %floor.desired_workflow_edition,
            "runtime declares workflow edition {}, so the session runs there even though env and \
             model both support {}",
            floor.selected_workflow_edition,
            floor.desired_workflow_edition,
        ),
        None => {}
    }
    Ok(floor)
}
