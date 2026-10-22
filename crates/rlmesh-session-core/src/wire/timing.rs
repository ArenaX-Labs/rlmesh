//! What an endpoint measured answering one request: the v1 endpoint phases,
//! carried on lockstep and served-model replies.

/// An endpoint's timing of one request, in nanoseconds. An absent field was
/// not measured, so a reader can tell silence from zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EndpointTiming {
    /// Arrival to reply.
    pub total_ns: Option<u64>,
    /// Decoding the request.
    pub decode_ns: Option<u64>,
    /// The env's or model's own work, user code included.
    pub user_ns: Option<u64>,
    /// Encoding the reply.
    pub encode_ns: Option<u64>,
    /// Waiting before the handler ran (a lock, the authority view).
    pub queue_ns: Option<u64>,
    /// Model: requests in flight when this one was dispatched.
    pub in_flight: Option<u32>,
    /// Model: adapter work inside `user_ns` (a sub-span, not additive).
    pub adapter_ns: Option<u64>,
    /// Model: episodes with held frame windows.
    pub held_episodes: Option<u32>,
    /// Model: the bytes those windows hold.
    pub held_state_bytes: Option<u64>,
    /// Env: straggler skew across a vector env's lanes.
    pub lane_skew_ns: Option<u64>,
}

/// `elapsed` in nanoseconds, saturating; zero (unmeasured) as `None`.
pub fn ns(elapsed: std::time::Duration) -> Option<u64> {
    let ns = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
    (ns != 0).then_some(ns)
}
