//! Behavioral fingerprint for workflow edition 2026.10: these lifecycle
//! assertions are the edition contract (episode accounting, per-lane autoreset,
//! request/response ordering). Changing observable behavior here changes the
//! edition; see docs/editions/2026.10.md. The fakes they drive live in
//! `common/mod.rs`, shared with the byte-exact golden records in
//! `runtime_edition_2026_10.rs`.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::FutureExt;
use futures::channel::oneshot;
use prost::bytes::Bytes;
use rlmesh::binding::runtime::{
    ActionReceivedEvent, EndpointPhases, EnvResetRequest, EnvStepRequest, HookError, RuntimeEnv,
    RuntimeEnvReset, RuntimeEnvStep, RuntimeError, RuntimeHooks, RuntimeSessionSpec,
};
use rlmesh_proto::spaces::v1::{MetaValue, SpaceSpec, meta_value};
use rlmesh_spaces::SpaceValue;

#[path = "../runtime_common/mod.rs"]
mod common;

use common::*;

mod autoreset;
mod chunking;
mod lane_sessions;
mod ledger;
mod lifecycle;
mod shutdown;
mod telemetry;
mod transforms;
mod trials;
