//! Lockstep and virtual clock modes of a real-time RLMesh session
//! (protocol v2, the v0.2 spec §11.2-§11.3 and §12), and the Gymnasium
//! compatibility path: existing Gym envs and v1 model servers run unchanged
//! on v2, at v1 throughput.
//!
//! - `exchange`: the exchange (one `LockstepStep`, one `LockstepReply`),
//!   the env-side `Gate` (epoch, lease token and e-stop
//!   checked locally per step) and the links: a direct call in one process,
//!   a Zenoh query on `l/{robot}` otherwise.
//! - `map`: a Gym env as a robot (§12.5): spaces and v1 tags to a robot
//!   map, leaf slabs to part slabs and back, bit-exact.
//! - `served`: a served env behind the lockstep exchange (`GymEnv` over
//!   a `Served` env, on native values).
//! - `client`: the driver of a lockstep Gym env (`LockstepClient`), on
//!   native values.
//! - `session`: attach, declare, arm and one lease grant; addresses
//!   (participant addresses).
//! - `model`: a model served on a session (`m/{model}`) and the
//!   driver's link to it.
//! - `view`: the Gym view over a v2-native robot (§12.6).
//! - `virtual_mode`: the virtual clock's run loop (§11.3).
//! - `record`: the lockstep tee into the recorder (§13).
//! - [`adapters`]: the v1 adapter engine (resolver and `join` lowering),
//!   which a v1 `ModelSpec` resolves through on this path, unchanged.

#![warn(missing_docs)]

pub mod client;
pub mod error;
pub mod escape;
pub mod exchange;
pub mod map;
pub mod meta_json;
pub mod model;
pub mod record;
pub mod served;
pub mod session;
pub mod view;
pub mod virtual_mode;

/// The v1 adapter engine the compatibility path resolves `ModelSpec`s
/// with (§12.4): the `rlmesh-adapters` resolver and its `join` lowering,
/// reused as they are while the catalog is ported to path trees.
pub mod adapters {
    pub use rlmesh_adapters::v1::{
        AdapterResolutionError, ENV_METADATA_KEY, EnvTags, JoinError, ModelSpec, ResolvedAdapter,
        join, resolve,
    };
}

pub use client::{Keep, LeafStep, LockstepClient};
pub use error::GymError;
pub use exchange::{DirectLink, Gate, LockstepHandler, LockstepLink, Request, ZenohLink};
pub use map::{GymEnvInfo, GymMap};
pub use model::{ModelBinding, ModelConnection, ModelLink, ModelService, ServedModel};
pub use rlmesh_session::address::is_v2_address;
pub use served::{GymEnv, Reset, Served, Step};
pub use session::{
    ConnectOptions, Declared, Driven, LockstepAddress, LockstepBinding, ServeOptions, bind, check,
    connect, connect_driver, connect_view, local_run_session, lockstep_robot_map,
};
pub use view::{Frame, GymView};
