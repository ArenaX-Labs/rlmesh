//! Lockstep and virtual clock modes of a real-time RLMesh session
//! (protocol v2, the v0.2 spec §11.2-§11.3 and §12), and the Gymnasium
//! compatibility path: existing Gym envs and v1 model servers run unchanged
//! on v2, at v1 throughput.
//!
//! - [`exchange`]: the exchange (one `LockstepStep`, one `LockstepReply`),
//!   the env-side [`Gate`] (epoch, lease token and e-stop
//!   checked locally per step) and the links: a direct call in one process,
//!   a Zenoh query on `l/{robot}` otherwise.
//! - [`map`]: a Gym env as a robot (§12.5): spaces and v1 tags to a robot
//!   map, leaf slabs to part slabs and back, bit-exact.
//! - [`compat`]: the v1 env behind the lockstep exchange
//!   ([`GymEnv`], on the v1 env service's own core), and a
//!   lockstep env as a v1 [`RuntimeEnv`](rlmesh_runtime::RuntimeEnv)
//!   ([`LockstepEnv`]), so the v1 runtime's chunking,
//!   prefetch, batching, autoreset and episode ids run unchanged.
//! - [`session`]: attach, declare, arm and one lease grant; addresses
//!   (`lockstep://...`).
//! - [`view`]: the Gym view over a v2-native robot (§12.6).
//! - [`virtual_mode`]: the virtual clock's run loop (§11.3).
//! - [`record`]: the lockstep tee into the recorder (§13).
//! - [`adapters`]: the v1 adapter engine (resolver and `join` lowering),
//!   which a v1 `ModelSpec` resolves through on this path, unchanged.

#![warn(missing_docs)]

pub mod compat;
pub mod error;
pub mod escape;
pub mod exchange;
pub mod map;
pub mod meta_json;
pub mod record;
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

pub use compat::{GymEnv, Keep, LockstepEnv};
pub use error::GymError;
pub use exchange::{DirectLink, Gate, LockstepHandler, LockstepLink, Request, ZenohLink};
pub use map::{GymEnvInfo, GymMap};
pub use session::{
    ConnectOptions, Declared, LockstepAddress, LockstepBinding, ServeOptions, bind, check, connect,
    connect_view, connect_view_in, is_lockstep_address, lockstep_robot_map,
};
pub use view::{Frame, GymView};
