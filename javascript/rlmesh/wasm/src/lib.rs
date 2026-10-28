//! The browser binding's wasm core: [`rlmesh_session_core::wire`] and
//! [`rlmesh_session_core`] behind wasm-bindgen, instantiated once per class
//! Worker of the `rlmesh` npm package (`javascript/rlmesh`).
//!
//! - [`ControlSession`] (the control Worker): this participant's view of the
//!   authority (decoded snapshots), heartbeats and the clock offset estimate
//!   they yield, `ctl` requests, one [`LeaseClient`] per leased part, and
//!   fenced action chunks.
//! - [`Streams`] (the state and bulk Workers): decodes samples, applies the session's receive rules (epochs, seq gaps as
//!   losses, layout checks, freshness) and measures delivery.
//!
//! Every message is a real v2 one, encoded and decoded by `rlmesh_session_core::wire`;
//! the gateway forwards the bytes ([`frame`]).
//!
//! # Boundary conventions
//!
//! - Times cross as `f64` **milliseconds since the UNIX epoch**
//!   (`performance.timeOrigin + performance.now()`) and are `i64` ns
//!   inside. A `u64`/`i64` would cross as a `BigInt`; `f64` ms keep ~0.1 µs.
//!   Lease tokens and seqs cross as `f64` (exact below 2^53).
//! - Inbound frames are taken as `Vec<u8>` (one JS → wasm copy) and decoded
//!   as [`Bytes`], so payloads slice that buffer rather than being copied
//!   again.
//! - Outbound values are copied into caller-owned TypedArrays (`*_into`):
//!   views into wasm memory detach when it grows, so none is handed out.
//! - Malformed input is a `JsError`, never a panic: a panic aborts the
//!   instance (`panic = "abort"`), and the class Worker is respawned.
//! - Names carry no wire prefixes: modes, priorities and kinds are
//!   lowercase strings.

pub mod decl;
pub mod frame;
pub mod stats;

use std::collections::{BTreeMap, VecDeque};

use bytes::Bytes;
use js_sys::{Array, Float64Array, Object, Reflect, Uint8Array};
use rlmesh_session_core::clock::{OffsetEstimate, OffsetEstimator, OffsetEstimatorConfig};
use rlmesh_session_core::lease::{LeaseChange, LeaseClient, LeaseState};
use rlmesh_session_core::stream::{SampleStats, check_sample, is_stale};
use rlmesh_session_core::wire::codec;
use rlmesh_session_core::wire::{
    ActionChunk, AuthoritySnapshot, ChannelId, Command, CtlReply, Epoch, Health, Heartbeat,
    LeaseToken, Mode, ParticipantId, ReasonCode, Request, Sample, StreamSpec, TimingSample,
};
use wasm_bindgen::prelude::*;

use crate::decl::Kind;
use crate::stats::Delivery;

/// Installs the panic hook: without it a Rust panic surfaces in JS only as
/// `RuntimeError: unreachable`.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

fn ms_to_ns(ms: f64) -> i64 {
    (ms * 1e6) as i64
}

fn ns_to_ms(ns: i64) -> f64 {
    ns as f64 / 1e6
}

fn set(o: &Object, k: &str, v: impl Into<JsValue>) {
    let _ = Reflect::set(o, &JsValue::from_str(k), &v.into());
}

fn err(message: impl std::fmt::Display) -> JsError {
    JsError::new(&message.to_string())
}

/// The runtime gotcha, kept on purpose: `std::time::Instant::now()` compiles
/// for wasm32-unknown-unknown and panics at run time ("time not implemented
/// on this platform"). The JS tests and `session.debug.trap()` call it to
/// raise a real trap.
#[wasm_bindgen]
pub fn std_instant_now_panics() -> f64 {
    #[allow(clippy::disallowed_methods)]
    let t = std::time::Instant::now();
    t.elapsed().as_secs_f64()
}

/// The fix: `web-time` reads `performance.now()` on the web.
#[wasm_bindgen]
pub fn web_time_now_ms() -> f64 {
    web_time::SystemTime::now()
        .duration_since(web_time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
        * 1e3
}

/// What was built.
#[wasm_bindgen]
pub fn build_info() -> JsValue {
    let o = Object::new();
    set(&o, "crate", "rlmesh-wasm");
    set(&o, "version", env!("CARGO_PKG_VERSION"));
    set(&o, "wasm_bindgen", "0.2.129");
    set(&o, "protocol", "rlmesh/v2");
    o.into()
}

fn fields_js(fields: &[decl::FieldDecl]) -> Array {
    let out = Array::new();
    for f in fields {
        let o = Object::new();
        set(&o, "name", f.name.as_str());
        set(&o, "dtype", f.dtype);
        let shape = Array::new();
        for d in &f.shape {
            shape.push(&JsValue::from(*d));
        }
        set(&o, "shape", shape);
        set(&o, "unit", f.unit.as_str());
        set(&o, "offset", f.offset as f64);
        set(&o, "len", f.len as f64);
        out.push(&o);
    }
    out
}

fn stream_js(d: &decl::StreamDecl) -> Object {
    let o = Object::new();
    set(&o, "path", d.path.as_str());
    set(&o, "robot", d.robot.as_str());
    set(&o, "part", d.part.as_str());
    set(&o, "applied", d.applied);
    set(&o, "channel", d.channel.0);
    set(&o, "class", d.class);
    set(&o, "kind", d.kind.name());
    set(&o, "rate_hz", d.hz);
    set(&o, "max_age_ms", d.max_age_ms);
    set(&o, "fields", fields_js(&d.fields));
    set(&o, "width", d.width);
    set(&o, "height", d.height);
    o
}

fn control_js(d: &decl::ControlDecl) -> Object {
    let o = Object::new();
    set(&o, "path", d.path.as_str());
    set(&o, "robot", d.robot.as_str());
    set(&o, "part", d.part.as_str());
    set(&o, "channel", d.channel.0);
    set(&o, "class", "control");
    set(&o, "kind", d.kind.as_str());
    set(&o, "mode", d.mode.as_str());
    match &d.group {
        Some(g) => set(&o, "group", g.as_str()),
        None => set(&o, "group", JsValue::NULL),
    }
    set(&o, "fields", fields_js(&d.fields));
    set(&o, "rate_hz", d.hz);
    set(&o, "hold_max_ms", d.hold_max_ms);
    set(&o, "bound", d.bound);
    o
}

/// `session = participant + θ` of `id` in `snapshot`, with its error.
fn mapping(snapshot: Option<&AuthoritySnapshot>, id: &ParticipantId) -> Option<(i64, u64)> {
    snapshot?.clock(id).map(|c| (c.offset_ns, c.error_ns))
}

// ------------------------------------------------------------------ streams

struct Stream {
    path: String,
    spec: StreamSpec,
    kind: Kind,
    stats: SampleStats,
    delivery: Delivery,
    late_ns: i64,
    latest_capture_local_ns: Option<i64>,
}

/// One decoded, accepted sample and its times in the subscriber's clock.
struct Current {
    stream: usize,
    sample: Sample,
    capture_local_ns: i64,
    publish_local_ns: i64,
    receive_ns: i64,
}

/// The decoder and per-stream rules and measurements of one data Worker.
/// After [`Streams::push`], the getters read the current sample;
/// [`Streams::advance`] steps past it.
#[wasm_bindgen]
pub struct Streams {
    state: Option<AuthoritySnapshot>,
    me: Option<ParticipantId>,
    /// This participant's mapping, from the control Worker.
    self_offset_ns: Option<(i64, u64)>,
    streams: Vec<Stream>,
    current: VecDeque<Current>,
    scratch: Vec<f64>,
}

impl Default for Streams {
    fn default() -> Self {
        Self::new()
    }
}

impl Streams {
    /// Publisher clock → this participant's clock, through the host's
    /// mappings: `θ(publisher) − θ(me)`, or 0 while either is unknown.
    fn shift_ns(&self, publisher: &ParticipantId) -> i64 {
        match (mapping(self.state.as_ref(), publisher), self.self_offset_ns) {
            (Some((theirs, _)), Some((mine, _))) => theirs.saturating_sub(mine),
            _ => 0,
        }
    }

    fn current(&self) -> Result<&Current, JsError> {
        self.current
            .front()
            .ok_or_else(|| err("no sample decoded yet"))
    }

    fn accept(&mut self, index: usize, sample: Sample, receive_ns: i64) -> bool {
        let view_epoch = self.state.as_ref().map_or(Epoch(0), |s| s.epoch);
        let shift = self.shift_ns(&self.streams[index].spec.owner);
        let stream = &mut self.streams[index];
        if check_sample(&stream.spec, &sample).is_err() {
            stream.stats.malformed += 1;
            return false;
        }
        if !stream
            .stats
            .accept(sample.seq, sample.epoch, view_epoch, false)
        {
            return false;
        }
        let capture_local_ns = sample.capture_ns.saturating_add(shift);
        let publish_local_ns = sample.publish_ns.saturating_add(shift);
        stream.delivery.record(
            receive_ns,
            receive_ns.saturating_sub(publish_local_ns),
            receive_ns.saturating_sub(capture_local_ns),
        );
        stream.latest_capture_local_ns = Some(capture_local_ns);
        self.current.push_back(Current {
            stream: index,
            sample,
            capture_local_ns,
            publish_local_ns,
            receive_ns,
        });
        true
    }
}

#[wasm_bindgen]
impl Streams {
    /// An empty decoder.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Streams {
        Streams {
            state: None,
            me: None,
            self_offset_ns: None,
            streams: Vec::new(),
            current: VecDeque::new(),
            scratch: Vec::new(),
        }
    }

    /// The authority snapshot (the declarations, the epoch and the clock
    /// mappings), as the control socket delivered it. Re-resolves every
    /// stream; one whose part was re-declared starts its counters over.
    pub fn set_state(&mut self, bytes: &[u8]) -> Result<(), JsError> {
        let snapshot = codec::decode_state(bytes).map_err(err)?;
        if let Some(old) = &self.state
            && !snapshot.is_newer_than(old)
        {
            return Ok(());
        }
        for stream in &mut self.streams {
            if let Some((spec, _)) = decl::stream(&snapshot, &stream.path)
                && spec.channel != stream.spec.channel
            {
                stream.kind = decl::kind_of(&spec);
                stream.spec = spec;
                stream.stats = SampleStats::default();
                stream.delivery = Delivery::default();
            }
        }
        self.state = Some(snapshot);
        Ok(())
    }

    /// This participant's id and mapping (`session = local + offset`, within
    /// `error`), from the control Worker.
    pub fn set_clock(&mut self, participant: &str, offset_ms: f64, error_ms: f64) {
        self.me = ParticipantId::new(participant).ok();
        self.self_offset_ns = offset_ms.is_finite().then(|| {
            (
                ms_to_ns(offset_ms),
                ms_to_ns(error_ms.max(0.0)).unsigned_abs(),
            )
        });
    }

    /// Adds the stream at `path` (`{robot}/{part}[/_applied]`); its index is
    /// the subscription id the gateway tags its samples with. Samples later
    /// than `late_ms` (publish → receipt) count as late. Returns the
    /// stream's declaration.
    pub fn add_stream(&mut self, path: &str, late_ms: f64) -> Result<JsValue, JsError> {
        let snapshot = self
            .state
            .as_ref()
            .ok_or_else(|| err("no authority snapshot yet"))?;
        let (spec, decl) = decl::stream(snapshot, path)
            .ok_or_else(|| err(format!("{path} is not a declared stream")))?;
        let o = stream_js(&decl);
        set(&o, "index", self.streams.len() as f64);
        self.streams.push(Stream {
            path: path.to_string(),
            kind: decl.kind,
            spec,
            stats: SampleStats::default(),
            delivery: Delivery::default(),
            late_ns: ms_to_ns(late_ms),
            latest_capture_local_ns: None,
        });
        Ok(o.into())
    }

    /// Decodes one `SAMPLE` frame received at `now_ms` and applies the
    /// receive rules. Returns the number of samples accepted (0 or 1;
    /// [`Self::advance`] steps past it). An unknown subscription id returns
    /// 0; garbage is an error.
    pub fn push(&mut self, bytes: Vec<u8>, now_ms: f64) -> Result<u32, JsError> {
        self.current.clear();
        if bytes.first() != Some(&frame::SAMPLE) || bytes.len() < frame::SAMPLE_HEADER {
            return Err(err("not a sample frame"));
        }
        let id = frame::read_id(&bytes).ok_or_else(|| err("short frame"))? as usize;
        let body = Bytes::from(bytes).slice(frame::SAMPLE_HEADER..);
        if id >= self.streams.len() {
            return Ok(0);
        }
        let receive_ns = ms_to_ns(now_ms);
        let sample = codec::decode_sample(body).map_err(err)?;
        Ok(u32::from(self.accept(id, sample, receive_ns)))
    }

    /// Moves to the next sample of the last frame; false when none is left.
    pub fn advance(&mut self) -> bool {
        self.current.pop_front();
        !self.current.is_empty()
    }

    /// The current sample's stream index (-1: none).
    pub fn index(&self) -> i32 {
        self.current.front().map_or(-1, |c| c.stream as i32)
    }

    /// Its seq (-1: none).
    pub fn seq(&self) -> f64 {
        self.current.front().map_or(-1.0, |c| c.sample.seq as f64)
    }

    /// Its epoch.
    pub fn epoch(&self) -> f64 {
        self.current
            .front()
            .map_or(0.0, |c| c.sample.epoch.0 as f64)
    }

    /// How its value reaches JS: `f64`, `jpeg`, `rgb8` or `text`.
    pub fn kind(&self) -> String {
        self.current
            .front()
            .map_or("", |c| self.streams[c.stream].kind.name())
            .to_string()
    }

    /// Capture time in this participant's clock (ms since the epoch).
    pub fn capture_ms(&self) -> f64 {
        self.current
            .front()
            .map_or(f64::NAN, |c| ns_to_ms(c.capture_local_ns))
    }

    /// Publish time in this participant's clock.
    pub fn publish_ms(&self) -> f64 {
        self.current
            .front()
            .map_or(f64::NAN, |c| ns_to_ms(c.publish_local_ns))
    }

    /// Capture → receipt.
    pub fn age_ms(&self) -> f64 {
        self.current
            .front()
            .map_or(f64::NAN, |c| ns_to_ms(c.receive_ns - c.capture_local_ns))
    }

    /// Publish → receipt.
    pub fn latency_ms(&self) -> f64 {
        self.current
            .front()
            .map_or(f64::NAN, |c| ns_to_ms(c.receive_ns - c.publish_local_ns))
    }

    /// The payload's byte length.
    pub fn data_len(&self) -> u32 {
        self.current
            .front()
            .map_or(0, |c| c.sample.payload.len() as u32)
    }

    /// Image width and height (frames; 0 otherwise).
    pub fn width(&self) -> u32 {
        self.current
            .front()
            .map_or(0, |c| match self.streams[c.stream].spec.fields.as_slice() {
                [f] => f.rgb8_dims().map_or(0, |(_, w)| w),
                _ => 0,
            })
    }

    /// See [`Self::width`].
    pub fn height(&self) -> u32 {
        self.current
            .front()
            .map_or(0, |c| match self.streams[c.stream].spec.fields.as_slice() {
                [f] => f.rgb8_dims().map_or(0, |(h, _)| h),
                _ => 0,
            })
    }

    /// On an applied stream: the edge mode (`""` otherwise).
    pub fn applied_mode(&self) -> String {
        self.current
            .front()
            .and_then(|c| c.sample.applied)
            .map_or("", |a| decl::edge_mode_name(a.mode))
            .to_string()
    }

    /// On an applied stream: whether a value was dispatched.
    pub fn applied_present(&self) -> bool {
        self.current
            .front()
            .and_then(|c| c.sample.applied)
            .is_some_and(|a| a.present)
    }

    /// On an applied stream: the dispatched action's lease token (0: none).
    pub fn applied_token(&self) -> f64 {
        self.current
            .front()
            .and_then(|c| c.sample.applied)
            .map_or(0.0, |a| a.lease_token.0 as f64)
    }

    fn values(&mut self) -> Result<(), JsError> {
        let current = self
            .current
            .front()
            .ok_or_else(|| err("no sample decoded yet"))?;
        let stream = &self.streams[current.stream];
        if stream.kind != Kind::F64 {
            return Err(err(format!(
                "{} is {}, not numbers",
                stream.path,
                stream.kind.name()
            )));
        }
        decl::unpack_f64(
            &stream.spec.fields,
            &stream.spec.layout,
            &current.sample.payload,
            &mut self.scratch,
        );
        Ok(())
    }

    /// The numeric value as a fresh `Float64Array`, field by field (an
    /// absent applied value is empty).
    pub fn data_f64(&mut self) -> Result<Float64Array, JsError> {
        self.values()?;
        Ok(Float64Array::from(&self.scratch[..]))
    }

    /// Copies the numeric value into `out` (caller-owned, or a
    /// SharedArrayBuffer view); returns the count written.
    pub fn data_f64_into(&mut self, out: &Float64Array) -> Result<u32, JsError> {
        self.values()?;
        let n = self.scratch.len().min(out.length() as usize);
        out.subarray(0, n as u32).copy_from(&self.scratch[..n]);
        Ok(n as u32)
    }

    /// Copies the payload bytes (a JPEG, raw RGB, text or a packed value)
    /// into `out` (a pooled buffer); throws when it does not fit.
    pub fn data_into(&self, out: &Uint8Array) -> Result<u32, JsError> {
        let payload = &self.current()?.sample.payload;
        if payload.len() > out.length() as usize {
            return Err(err(format!(
                "buffer too small: {} < {}",
                out.length(),
                payload.len()
            )));
        }
        out.subarray(0, payload.len() as u32).copy_from(payload);
        Ok(payload.len() as u32)
    }

    /// A text sample's value.
    pub fn text(&self) -> Result<String, JsError> {
        let payload = &self.current()?.sample.payload;
        std::str::from_utf8(payload)
            .map(str::to_string)
            .map_err(err)
    }

    /// Whether stream `index`'s newest value is stale at `now_ms` against
    /// its `max_age` (never, without one; always, before any value).
    pub fn is_stale(&self, index: u32, now_ms: f64) -> bool {
        let Some(stream) = self.streams.get(index as usize) else {
            return true;
        };
        match stream.latest_capture_local_ns {
            Some(capture) => is_stale(ms_to_ns(now_ms) - capture, stream.spec.max_age),
            None => true,
        }
    }

    /// Stream `index`'s counters: the receive rules' (`received`, `lost`,
    /// `out_of_order`, `stale_epoch`, `malformed`) and delivery (rate,
    /// latency and age percentiles, late share).
    pub fn stats(&self, index: u32) -> JsValue {
        let Some(s) = self.streams.get(index as usize) else {
            return JsValue::NULL;
        };
        let o = Object::new();
        set(&o, "path", s.path.as_str());
        set(&o, "received", s.stats.received as f64);
        set(&o, "lost", s.stats.lost as f64);
        set(&o, "out_of_order", s.stats.out_of_order as f64);
        set(&o, "stale_epoch", s.stats.stale_epoch as f64);
        set(&o, "malformed", s.stats.malformed as f64);
        set(&o, "last_seq", s.stats.last_seq as f64);
        set(&o, "rate_hz", s.delivery.rate_hz());
        let l = &s.delivery.latency;
        set(&o, "latency_p50_ms", l.quantile_ms(0.5));
        set(&o, "latency_p99_ms", l.quantile_ms(0.99));
        set(&o, "latency_p999_ms", l.quantile_ms(0.999));
        set(&o, "latency_max_ms", l.max_ms());
        set(&o, "late_pct", l.above_pct(s.late_ns));
        let a = &s.delivery.age;
        set(&o, "age_p50_ms", a.quantile_ms(0.5));
        set(&o, "age_p99_ms", a.quantile_ms(0.99));
        set(&o, "count", l.count() as f64);
        o.into()
    }

    /// Starts every stream's counters over.
    pub fn reset_stats(&mut self) {
        for s in &mut self.streams {
            s.stats = SampleStats {
                last_seq: s.stats.last_seq,
                ..SampleStats::default()
            };
            s.delivery = Delivery::default();
        }
    }
}

// ------------------------------------------------------------------ control

/// What a request id waits for.
enum Pending {
    Grant(String),
    Release(String),
    Verb(&'static str),
}

/// Heartbeats in flight, for matching echoes.
const SENT: usize = 64;

/// This participant's control state in the control Worker: see the crate
/// docs.
#[wasm_bindgen]
pub struct ControlSession {
    me: Option<ParticipantId>,
    state: Option<AuthoritySnapshot>,
    estimator: OffsetEstimator,
    offset_ns: i64,
    hb_seq: u64,
    sent: VecDeque<(u64, i64)>,
    completed: Option<TimingSample>,
    leases: BTreeMap<String, LeaseClient>,
    pending: BTreeMap<u32, Pending>,
    next_request: u32,
    /// Fall-through counter for debug chunks with a crafted token.
    debug_seq: u64,
}

impl Default for ControlSession {
    fn default() -> Self {
        Self::new()
    }
}

impl ControlSession {
    fn me(&self) -> Result<&ParticipantId, JsError> {
        self.me
            .as_ref()
            .ok_or_else(|| err("not attached yet (no hello from the gateway)"))
    }

    /// The best estimate at local time `now_ns`.
    fn estimate(&self, now_ns: i64) -> Option<OffsetEstimate> {
        self.estimator
            .estimate(now_ns.saturating_add(self.offset_ns))
    }

    fn request(&mut self, command: Command, pending: Pending) -> Result<Vec<u8>, JsError> {
        let from = self.me()?.clone();
        let epoch = self.state.as_ref().map(|s| s.epoch);
        let id = self.next_request;
        self.next_request = self.next_request.wrapping_add(1);
        let body = codec::encode_request(&Request {
            from,
            epoch,
            command,
        });
        self.pending.insert(id, pending);
        Ok(frame::with_id(frame::REQUEST, id, &body))
    }

    /// Takes a newer snapshot; returns each held lease's change.
    fn take_state(&mut self, snapshot: AuthoritySnapshot, now_ns: i64) -> (bool, Array) {
        let changes = Array::new();
        if self
            .state
            .as_ref()
            .is_some_and(|old| !snapshot.is_newer_than(old))
        {
            return (false, changes);
        }
        let clock = self.estimate(now_ns);
        for (path, client) in &mut self.leases {
            let token = client.token();
            if let Some(change) = client.on_snapshot(&snapshot, clock.as_ref(), now_ns) {
                let o = Object::new();
                set(&o, "path", path.as_str());
                set(&o, "token", token.0 as f64);
                match change {
                    LeaseChange::Renewed => set(&o, "change", "renewed"),
                    LeaseChange::Revoked(reason) => {
                        set(&o, "change", "revoked");
                        set(&o, "reason", reason);
                    }
                }
                changes.push(&o);
            }
        }
        let structural = self
            .state
            .as_ref()
            .is_none_or(|old| old.revision != snapshot.revision);
        self.state = Some(snapshot);
        (structural, changes)
    }

    fn state_event(&self, structural: bool, changes: Array) -> Object {
        let ev = Object::new();
        set(&ev, "type", "state");
        if let Some(s) = &self.state {
            set(&ev, "revision", s.revision as f64);
            set(&ev, "structural", structural);
            set(&ev, "epoch", s.epoch.0 as f64);
            set(&ev, "mode", decl::mode_name(s.mode));
            set(&ev, "mode_reason", decl::reason_name(s.mode_reason.code));
            set(&ev, "estop_latched", s.estop_latched);
            set(&ev, "session", s.session.to_string());
        }
        set(&ev, "changes", changes);
        ev
    }

    fn lease_event(&mut self, path: &str, event: &str, now_ns: i64) -> Object {
        let ev = Object::new();
        set(&ev, "type", "lease");
        set(&ev, "event", event);
        set(&ev, "path", path);
        set(&ev, "ours", true);
        if let Some(client) = self.leases.get_mut(path) {
            let state = client.state(now_ns);
            set(&ev, "state", state.name());
            set(&ev, "token", client.token().0 as f64);
            if let Some(lease) = client.lease() {
                set(&ev, "holder", lease.holder.as_str());
                set(&ev, "priority", decl::priority_name(lease.priority));
                set(&ev, "ttl_ms", lease.ttl_ns as f64 / 1e6);
            }
            if let LeaseState::Refused { error } = state {
                set(&ev, "reason", error.to_string());
                set(&ev, "status", format!("{:?}", error.status));
            }
        }
        ev
    }
}

#[wasm_bindgen]
impl ControlSession {
    /// A session before the gateway's hello.
    #[wasm_bindgen(constructor)]
    pub fn new() -> ControlSession {
        ControlSession {
            me: None,
            state: None,
            estimator: OffsetEstimator::new(OffsetEstimatorConfig::default()),
            offset_ns: 0,
            hb_seq: 0,
            sent: VecDeque::with_capacity(SENT),
            completed: None,
            leases: BTreeMap::new(),
            pending: BTreeMap::new(),
            next_request: 1,
            debug_seq: u64::from(u32::MAX),
        }
    }

    /// The participant id the gateway attached this connection as.
    pub fn set_participant(&mut self, id: &str) -> Result<(), JsError> {
        self.me = Some(ParticipantId::new(id).map_err(err)?);
        Ok(())
    }

    /// This participant's id (`""` before the hello).
    pub fn participant(&self) -> String {
        self.me
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default()
    }

    /// Handles one binary frame from the control socket at `now_ms`:
    /// returns an event `{type: state | clock | lease | reply, ...}`.
    pub fn on_frame(&mut self, bytes: Vec<u8>, now_ms: f64) -> Result<JsValue, JsError> {
        let now_ns = ms_to_ns(now_ms);
        let (&tag, body) = bytes.split_first().ok_or_else(|| err("empty frame"))?;
        match tag {
            frame::STATE => {
                let snapshot = codec::decode_state(body).map_err(err)?;
                let (structural, changes) = self.take_state(snapshot, now_ns);
                Ok(self.state_event(structural, changes).into())
            }
            frame::ECHO => {
                let echo = codec::decode_echo(body).map_err(err)?;
                let ev = Object::new();
                set(&ev, "type", "clock");
                // Only our own heartbeats: the gateway's participant may
                // have sent one of its own before handing over.
                let ours = self
                    .sent
                    .iter()
                    .any(|(seq, t1)| *seq == echo.seq && *t1 == echo.t1_ns);
                if ours {
                    let sample = echo.complete(now_ns);
                    if self.estimator.add(sample) {
                        if let Some(e) = self.estimator.estimate(echo.t3_ns) {
                            self.offset_ns = e.offset_ns;
                        }
                        self.completed = Some(sample);
                    }
                }
                set(&ev, "ours", ours);
                set(&ev, "samples", self.estimator.len() as f64);
                Ok(ev.into())
            }
            frame::REPLY => {
                let id = frame::read_id(&bytes).ok_or_else(|| err("short reply"))?;
                let reply: CtlReply = codec::decode_reply(&bytes[5..]).map_err(err)?;
                let pending = self.pending.remove(&id);
                let clock = self.estimate(now_ns);
                let ev = match pending {
                    Some(Pending::Grant(path)) => {
                        let held = self
                            .leases
                            .get_mut(&path)
                            .is_some_and(|c| c.on_reply(&reply, clock.as_ref(), now_ns));
                        let ev =
                            self.lease_event(&path, if held { "grant" } else { "refused" }, now_ns);
                        if let Err(e) = &reply.result {
                            set(&ev, "reason", e.to_string());
                        }
                        ev
                    }
                    Some(Pending::Release(path)) => {
                        let ev = Object::new();
                        set(&ev, "type", "reply");
                        set(&ev, "op", "release");
                        set(&ev, "path", path);
                        ev
                    }
                    Some(Pending::Verb(op)) => {
                        let ev = Object::new();
                        set(&ev, "type", "reply");
                        set(&ev, "op", op);
                        ev
                    }
                    None => {
                        let ev = Object::new();
                        set(&ev, "type", "reply");
                        set(&ev, "op", "unknown");
                        ev
                    }
                };
                set(&ev, "rid", id);
                set(&ev, "ok", reply.result.is_ok());
                if let Err(e) = &reply.result {
                    set(&ev, "error", e.to_string());
                }
                let (structural, changes) = self.take_state(reply.state, now_ns);
                set(&ev, "state", self.state_event(structural, changes));
                Ok(ev.into())
            }
            other => Err(err(format!("unexpected control frame 0x{other:02x}"))),
        }
    }

    /// The next heartbeat frame (with the last completed exchange), or
    /// `undefined` before the hello.
    pub fn heartbeat(&mut self, now_ms: f64) -> Option<Vec<u8>> {
        let me = self.me.clone()?;
        self.hb_seq += 1;
        let t1_ns = ms_to_ns(now_ms);
        if self.sent.len() == SENT {
            self.sent.pop_front();
        }
        self.sent.push_back((self.hb_seq, t1_ns));
        let (epoch, mode) = self
            .state
            .as_ref()
            .map_or((Epoch(0), Mode::Disarmed), |s| (s.epoch, s.mode));
        let hb = Heartbeat {
            participant: me,
            epoch,
            mode,
            seq: self.hb_seq,
            t1_ns,
            previous: self.completed.take(),
            last_seq_seen: Vec::new(),
            last_chunk_accepted: 0,
            last_chunk_applied: 0,
            queue_depth: 0,
            health: Health::Ok,
        };
        Some(frame::tagged(
            frame::HEARTBEAT,
            &codec::encode_heartbeat(&hb),
        ))
    }

    /// The clock estimate at `now_ms`: `{offset_ms, rtt_ms, error_ms,
    /// samples}` (session = local + offset), plus the host's published
    /// mapping of this participant when there is one.
    pub fn clock(&self, now_ms: f64) -> JsValue {
        let o = Object::new();
        let now_ns = ms_to_ns(now_ms);
        if let Some(e) = self.estimate(now_ns) {
            set(&o, "offset_ms", ns_to_ms(e.offset_ns));
            set(&o, "rtt_ms", e.rtt_ns as f64 / 1e6);
            set(&o, "error_ms", e.error_ns as f64 / 1e6);
        } else {
            set(&o, "offset_ms", f64::NAN);
            set(&o, "rtt_ms", f64::NAN);
            set(&o, "error_ms", f64::NAN);
        }
        set(&o, "samples", self.estimator.len() as f64);
        if let (Some(me), Some(state)) = (&self.me, &self.state)
            && let Some(m) = state.clock(me)
        {
            set(&o, "host_offset_ms", ns_to_ms(m.offset_ns));
            set(&o, "host_error_ms", m.error_ns as f64 / 1e6);
        }
        o.into()
    }

    /// Session time at local `now_ms` (`NaN` before the first estimate).
    pub fn session_now_ms(&self, now_ms: f64) -> f64 {
        match self.estimate(ms_to_ns(now_ms)) {
            Some(e) => now_ms + ns_to_ms(e.offset_ns),
            None => f64::NAN,
        }
    }

    /// Everything declared: `{session, epoch, mode, streams, controls}`.
    pub fn declarations(&self) -> JsValue {
        let o = Object::new();
        let streams = Array::new();
        let controls = Array::new();
        if let Some(s) = &self.state {
            set(&o, "session", s.session.to_string());
            set(&o, "epoch", s.epoch.0 as f64);
            set(&o, "mode", decl::mode_name(s.mode));
            for d in decl::streams(s) {
                streams.push(&stream_js(&d));
            }
            for d in decl::controls(s, self.me.as_ref()) {
                controls.push(&control_js(&d));
            }
            let robots = Array::new();
            for (robot, fp) in &s.contract {
                let r = Object::new();
                set(&r, "id", robot.as_str());
                set(&r, "fingerprint", fp.to_string());
                robots.push(&r);
            }
            set(&o, "robots", robots);
        }
        set(&o, "streams", streams);
        set(&o, "controls", controls);
        o.into()
    }

    /// The host's view of the commandable part at `path`: `{path, mode,
    /// mode_reason, epoch, holder, priority, token, ours, bound}`.
    pub fn status(&self, path: &str) -> Result<JsValue, JsError> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| err("no authority snapshot yet"))?;
        let part =
            decl::control(state, path).ok_or_else(|| err(format!("no commandable part {path}")))?;
        let o = Object::new();
        set(&o, "path", path);
        set(&o, "mode", decl::mode_name(state.mode));
        set(&o, "mode_reason", decl::reason_name(state.mode_reason.code));
        set(&o, "epoch", state.epoch.0 as f64);
        set(&o, "estop_latched", state.estop_latched);
        match state.lease(part.channel) {
            Some(lease) => {
                set(&o, "holder", lease.holder.as_str());
                set(&o, "priority", decl::priority_name(lease.priority));
                set(&o, "token", lease.token.0 as f64);
                set(&o, "ours", self.me.as_ref() == Some(&lease.holder));
            }
            None => {
                set(&o, "holder", "");
                set(&o, "priority", "");
                set(&o, "token", 0.0);
                set(&o, "ours", false);
            }
        }
        let bound = self
            .me
            .as_ref()
            .and_then(|me| state.binding(me))
            .is_some_and(|b| b.channels.contains(&part.channel));
        set(&o, "bound", bound);
        Ok(o.into())
    }

    /// Requests the lease on the commandable part at `path` (`ttl_ms`:
    /// `NaN` takes the session's lease grace). Returns the `REQUEST` frame;
    /// the reply arrives as a `lease` event.
    pub fn request_lease(
        &mut self,
        path: &str,
        ttl_ms: f64,
        now_ms: f64,
    ) -> Result<Vec<u8>, JsError> {
        let me = self.me()?.clone();
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| err("no authority snapshot yet"))?;
        let part = decl::control(state, path).ok_or_else(|| {
            err(format!(
                "no commandable part {path}; declared: {}",
                decl::controls(state, Some(&me))
                    .iter()
                    .map(|c| c.path.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;
        let channel: ChannelId = part.channel;
        let binding = state
            .binding(&me)
            .filter(|b| b.channels.contains(&channel))
            .map(|b| b.fingerprint)
            .ok_or_else(|| err(format!("{path} is not in this participant's binding yet")))?;
        let ttl_ns = ttl_ms
            .is_finite()
            .then(|| ms_to_ns(ttl_ms.max(1.0)).unsigned_abs());
        let mut client = LeaseClient::new(me, channel, 100_000_000);
        let command = client.request(binding, ttl_ns, ms_to_ns(now_ms));
        self.leases.insert(path.to_string(), client);
        self.request(command, Pending::Grant(path.to_string()))
    }

    /// Releases `path`: the `REQUEST` frame, or `undefined` when nothing was
    /// held. The local client is dropped either way.
    pub fn release(&mut self, path: &str) -> Result<Option<Vec<u8>>, JsError> {
        let Some(mut client) = self.leases.remove(path) else {
            return Ok(None);
        };
        match client.release() {
            Some(command) => self
                .request(command, Pending::Release(path.to_string()))
                .map(Some),
            None => Ok(None),
        }
    }

    /// Drops the local client for `path` without telling the host (the
    /// connection that held it is gone): its token is never used again.
    /// Returns the token.
    pub fn forget(&mut self, path: &str) -> f64 {
        self.leases
            .remove(path)
            .map_or(0.0, |mut c| c.forget().0 as f64)
    }

    /// A session verb: `arm`, `disarm` or `estop` (with `reason`).
    pub fn verb(&mut self, name: &str, reason: &str) -> Result<Vec<u8>, JsError> {
        let (command, op) = match name {
            "arm" => (Command::Arm, "arm"),
            "disarm" => (Command::Disarm, "disarm"),
            "estop" => (
                Command::Estop {
                    reason: reason.to_string(),
                    code: ReasonCode::EstopRemote,
                },
                "estop",
            ),
            other => {
                return Err(err(format!(
                    "unknown verb {other:?} (arm | disarm | estop)"
                )));
            }
        };
        self.request(command, Pending::Verb(op))
    }

    /// One command for `path` under its lease: `values` are the command
    /// fields' elements in order, held for `period_ms` (the drive period).
    /// The chunk carries two actions `period_ms` apart from now (session
    /// time), so one late chunk does not starve the edge, and is void after
    /// both. Throws when the lease is not held, or before the first clock
    /// estimate.
    pub fn command(
        &mut self,
        path: &str,
        values: &[f64],
        period_ms: f64,
        now_ms: f64,
    ) -> Result<Vec<u8>, JsError> {
        let now_ns = ms_to_ns(now_ms);
        let estimate = self
            .estimate(now_ns)
            .ok_or_else(|| err("no clock estimate yet"))?;
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| err("no authority snapshot yet"))?;
        let part =
            decl::control(state, path).ok_or_else(|| err(format!("no commandable part {path}")))?;
        let command = part
            .command
            .as_ref()
            .ok_or_else(|| err("not commandable"))?;
        let action = decl::pack_command(command, values).map_err(err)?;
        let period_ns = ms_to_ns(period_ms.max(0.1));
        let t0_ns = now_ns.saturating_add(estimate.offset_ns);
        let mut actions = action.clone();
        actions.extend_from_slice(&action);
        let client = self
            .leases
            .get_mut(path)
            .ok_or_else(|| err(format!("no lease on {path}")))?;
        let chunk = client
            .chunk(
                Bytes::from(actions),
                2,
                t0_ns,
                period_ns.unsigned_abs(),
                t0_ns.saturating_add(period_ns.saturating_mul(2)),
                Vec::new(),
                now_ns,
            )
            .map_err(err)?;
        encode_chunk(&chunk)
    }

    /// Debug only: a chunk for `path` stamped with an explicit `token`,
    /// bypassing the local lease check, to show the edge's fencing.
    /// Returns the frame and the fence verdict of the current snapshot
    /// (`""`: it would pass).
    pub fn command_with_token(
        &mut self,
        path: &str,
        values: &[f64],
        token: f64,
        now_ms: f64,
    ) -> Result<Array, JsError> {
        let now_ns = ms_to_ns(now_ms);
        let me = self.me()?.clone();
        let offset = self.estimate(now_ns).map_or(0, |e| e.offset_ns);
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| err("no authority snapshot yet"))?;
        let part =
            decl::control(state, path).ok_or_else(|| err(format!("no commandable part {path}")))?;
        let command = part
            .command
            .as_ref()
            .ok_or_else(|| err("not commandable"))?;
        let actions = decl::pack_command(command, values).map_err(err)?;
        let token = LeaseToken(token as u64);
        let verdict = match state.check_fence(part.channel, token, state.epoch) {
            Ok(_) => String::new(),
            Err(e) => e.to_string(),
        };
        self.debug_seq += 1;
        let t0_ns = now_ns.saturating_add(offset);
        let chunk = ActionChunk {
            channel: part.channel,
            epoch: state.epoch,
            lease_token: token,
            seq: self.debug_seq,
            source: me,
            binding: state
                .binding(self.me()?)
                .map(|b| b.fingerprint)
                .unwrap_or_default(),
            based_on: Vec::new(),
            t0_ns,
            dt_ns: 0,
            replace_from_ns: t0_ns,
            valid_until_ns: t0_ns.saturating_add(20_000_000),
            count: 1,
            actions: Bytes::from(actions),
            publish_ns: now_ns,
            tick: 0,
        };
        let out = Array::new();
        out.push(&Uint8Array::from(&encode_chunk(&chunk)?[..]));
        out.push(&JsValue::from_str(&verdict));
        Ok(out)
    }

    /// Paths with a local lease client (any state).
    pub fn lease_paths(&self) -> Vec<String> {
        self.leases.keys().cloned().collect()
    }

    /// The local lease state of `path` at `now_ms` (`idle` without one).
    pub fn lease_state(&mut self, path: &str, now_ms: f64) -> String {
        self.leases
            .get_mut(path)
            .map_or("idle", |c| c.state(ms_to_ns(now_ms)).name())
            .to_string()
    }

    /// The token held (or last held) on `path` (0: none).
    pub fn lease_token(&self, path: &str) -> f64 {
        self.leases.get(path).map_or(0.0, |c| c.token().0 as f64)
    }

    /// Time left before the local deadline (negative: not held).
    pub fn lease_remaining_ms(&mut self, path: &str, now_ms: f64) -> f64 {
        let now_ns = ms_to_ns(now_ms);
        match self.leases.get_mut(path) {
            Some(c) => {
                c.tick(now_ns);
                c.deadline_ns().map_or(-1.0, |d| ns_to_ms(d - now_ns))
            }
            None => -1.0,
        }
    }
}

fn encode_chunk(chunk: &ActionChunk) -> Result<Vec<u8>, JsError> {
    let encoded = codec::encode_chunk(chunk).map_err(err)?;
    let mut out = Vec::with_capacity(1 + encoded.segments.iter().map(Bytes::len).sum::<usize>());
    out.push(frame::CHUNK);
    for segment in &encoded.segments {
        out.extend_from_slice(segment);
    }
    Ok(out)
}
