//! Lockstep and virtual clock modes of a real-time RLMesh session
//! (protocol v2), and the Gymnasium compatibility path: existing Gym envs and
//! v1 model servers run unchanged on v2.
//!
//! - `exchange`: one `LockstepStep`, one `LockstepReply`; the env-side
//!   `Gate` (epoch, lease token and e-stop checked locally per step) and the
//!   links: a direct call in one process, a Zenoh query on `l/{robot}`
//!   otherwise.
//! - `map`: a Gym env as a robot, leaf slabs to part slabs and back,
//!   bit-exact.
//! - `served`, `client`: a served env behind the exchange, and its driver.
//! - `session`: attach, declare, arm and one lease grant.
//! - `model`: a model served on a session (`m/{model}`) and the driver's
//!   link to it.
//! - `view`, `virtual_mode`, `record`: the Gym view over a v2-native
//!   robot, the virtual clock's run loop, and the tee into the recorder.
//! - [`adapters`]: the v1 adapter engine a v1 `ModelSpec` resolves through.

#![warn(missing_docs)]

pub mod client;
pub mod error;
pub mod escape;
pub mod exchange;
#[cfg(test)]
pub(crate) mod linked_host;
pub mod map;
pub mod meta_json;
pub mod model;
pub mod record;
pub mod served;
pub mod session;
pub mod view;
pub mod virtual_mode;

/// The v1 adapter engine the compatibility path resolves `ModelSpec`s
/// with: the `rlmesh-adapters` resolver and its `join` lowering,
pub mod adapters {
    pub use rlmesh_adapters::v1::{
        AdapterResolutionError, ENV_METADATA_KEY, EnvTags, JoinError, ModelSpec, ResolvedAdapter,
        join, resolve,
    };
}

pub use client::{Keep, LeafStep, LockstepClient};
pub use error::GymError;
pub use exchange::{DirectLink, Gate, Limited, LockstepHandler, LockstepLink, Request, ZenohLink};
pub use map::{GymEnvInfo, GymMap};
pub use model::{ModelBinding, ModelConnection, ModelLink, ModelService, ServedModel};
pub use rlmesh_session::address::is_v2_address;
pub use served::{GymEnv, Reset, Served, Step};
pub use session::{
    ConnectOptions, Declared, Driven, LockstepAddress, LockstepBinding, ServeOptions, bind, check,
    connect, connect_driver, connect_view, local_run_session, lockstep_robot_map, run_session,
    run_session_beside,
};
pub use view::{Frame, GymView};
