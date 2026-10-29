//! Delivery contracts, action grids and publication plans (§8.2, §9.2).
//!
//! A controller's binding states one [`StreamContract`] per bound input
//! stream and one [`ActionGrid`] per commanded part. The host admits the
//! contracts, derives one publication [`Plan`] per stream from every
//! admitted contract on it, and publishes it as a [`Demand`]; the publisher
//! installs the plan and acknowledges its revision (`DemandAck`). A lease is
//! granted only once every stream of the requester's binding has
//! acknowledged a revision that includes its contract (`NOT_READY`
//! otherwise). Nothing renegotiates a contract later.

#[allow(unused_imports)]
use crate::prelude::*;

use core::time::Duration;

use crate::wire::ids::ChannelId;

/// How a subscriber is delivered one stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContractMode {
    /// The newest sample, at most `delivery_hz` times per second (an input
    /// without `history`).
    Latest,
    /// Every sample, as soon as it is published; the subscriber's cache
    /// keeps the window (an input with `history`).
    History,
    /// Every sample, as soon as it is published (`delivery_hz` at or above
    /// the source rate, or a full-rate consumer).
    Every,
}

/// One subscriber's contract on one stream (§8.2).
#[derive(Clone, Debug, PartialEq)]
pub struct StreamContract {
    /// The stream: a part's, or a commandable part's `_applied`.
    pub channel: ChannelId,
    /// How it is delivered.
    pub mode: ContractMode,
    /// The delivery rate D.
    pub delivery_hz: f64,
    /// The deepest window on the stream (1: none).
    pub history: u32,
    /// The effective freshness bound after precedence (§8.7); `None` for
    /// events, which are never stale.
    pub max_age: Option<Duration>,
    /// Whether the bound was declared (policy or robot) rather than
    /// derived. An explicit bound the contract cannot meet is refused.
    pub max_age_explicit: bool,
}

/// The action grid a binding declared for one commanded part: a chunk with
/// `dt_ns = 0` takes this `dt` (§9.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ActionGrid {
    /// The commandable part.
    pub channel: ChannelId,
    /// `1 / A`, the action period.
    pub dt: Duration,
}

/// How a publisher sends one stream, derived from every admitted contract on
/// it. A stream with no admitted contract has no demand and publishes every
/// sample within its `send` ceiling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Plan {
    /// Some contract is `every` or `history`: publish each sample
    /// immediately.
    Every,
    /// Only `latest` contracts: publish the newest sample at this rate,
    /// coalescing older ones.
    Latest {
        /// `max D`, within the `send` ceiling.
        hz: f64,
    },
}

/// The publication plan of one stream, as the host published it.
#[derive(Clone, Debug, PartialEq)]
pub struct Demand {
    /// The stream.
    pub channel: ChannelId,
    /// Bumps whenever the plan changes; from 1.
    pub revision: u64,
    /// The plan.
    pub plan: Plan,
    /// How long the publisher's ring must retain samples (window plus
    /// recovery delay).
    pub retain: Duration,
    /// The newest revision the publisher acknowledged (0: none).
    pub acked: u64,
}

impl Demand {
    /// Whether the publisher installed revision `revision` or a later one.
    pub fn installed(&self, revision: u64) -> bool {
        self.acked >= revision
    }
}
