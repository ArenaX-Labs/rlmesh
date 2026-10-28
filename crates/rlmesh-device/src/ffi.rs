//! The C ABI (`include/rlmesh_device.h`).

#![allow(unsafe_code)] // C callbacks and out-pointers.

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::ffi::{CStr, c_char, c_int};

use rlmesh_session_core::auth::Credential;
use rlmesh_session_core::crypto::{Rng, RngError};
use rlmesh_session_core::wire::ids::{ParticipantId, SessionId, TenantId};
use rlmesh_session_core::wire::{CtlStatus, Mode};

use crate::robot::{Action, Config, Fault, Phase, Robot, Sub};
use crate::timing::{self, TimingError};

/// Success.
pub const RLMESH_DEVICE_OK: c_int = 0;
/// A null argument.
pub const RLMESH_DEVICE_ERR_ARG: c_int = -1;
/// The platform's randomness failed.
pub const RLMESH_DEVICE_ERR_RNG: c_int = -2;
/// The crypto failed where it can't (a bug, or memory corrupted).
pub const RLMESH_DEVICE_ERR_CRYPTO: c_int = -3;
/// A config field is invalid: a tenant or name that isn't one, a zero
/// heartbeat period.
pub const RLMESH_DEVICE_ERR_CONFIG: c_int = -4;

/// The platform's microsecond clock.
pub type NowUs = unsafe extern "C" fn() -> u64;
/// The platform's cryptographic randomness: fills `len` bytes, 0 on
/// success.
pub type Random = unsafe extern "C" fn(buf: *mut u8, len: usize) -> c_int;

/// The crate's version: major, minor and patch, a byte each.
#[unsafe(no_mangle)]
pub extern "C" fn rlmesh_device_version() -> u32 {
    let n = |s: &str| s.parse::<u32>().unwrap_or(0);
    (n(env!("CARGO_PKG_VERSION_MAJOR")) << 16)
        | (n(env!("CARGO_PKG_VERSION_MINOR")) << 8)
        | n(env!("CARGO_PKG_VERSION_PATCH"))
}

/// `rlmesh_device_timing_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RlmeshDeviceTiming {
    pub handshake_start_us: u32,
    pub handshake_finish_us: u32,
    pub state_verify_us: u32,
    pub chunk_open_sign_us: u32,
    pub chunk_open_seal_us: u32,
    pub sample_seal_sign_us: u32,
}

struct CRng(Random);

impl Rng for CRng {
    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), RngError> {
        // SAFETY: the caller's callback, writing `dest.len()` bytes.
        match unsafe { (self.0)(dest.as_mut_ptr(), dest.len()) } {
            0 => Ok(()),
            _ => Err(RngError),
        }
    }
}

/// Runs [`timing::run`] on the platform's clock and randomness.
///
/// # Safety
/// `out` is null or valid for a write; the callbacks are sound to call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_timing(
    now_us: Option<NowUs>,
    random: Option<Random>,
    out: *mut RlmeshDeviceTiming,
) -> c_int {
    let (Some(now_us), Some(random)) = (now_us, random) else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    if out.is_null() {
        return RLMESH_DEVICE_ERR_ARG;
    }
    // SAFETY: the caller's clock.
    let mut clock = || unsafe { now_us() };
    match timing::run(&mut clock, &mut CRng(random)) {
        Ok(t) => {
            // SAFETY: checked non-null; valid per the contract.
            unsafe {
                out.write(RlmeshDeviceTiming {
                    handshake_start_us: t.handshake_start_us,
                    handshake_finish_us: t.handshake_finish_us,
                    state_verify_us: t.state_verify_us,
                    chunk_open_sign_us: t.chunk_open_sign_us,
                    chunk_open_seal_us: t.chunk_open_seal_us,
                    sample_seal_sign_us: t.sample_seal_sign_us,
                });
            }
            RLMESH_DEVICE_OK
        }
        Err(TimingError::Rng) => RLMESH_DEVICE_ERR_RNG,
        Err(TimingError::Crypto) => RLMESH_DEVICE_ERR_CRYPTO,
    }
}

/// `rlmesh_device_robot_config_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RlmeshDeviceRobotConfig {
    /// The session's tenant (`local` for an embedded host).
    pub tenant: *const c_char,
    /// The session's name (`default`) or its id, a hyphenated UUID.
    pub session: *const c_char,
    /// This robot's name.
    pub name: *const c_char,
    /// The address's local secret, 32 bytes.
    pub secret: *const u8,
    /// The address's host key, 32 bytes.
    pub host_key: *const u8,
    /// The robot's map, as authored JSON.
    pub robot_json: *const c_char,
    /// The heartbeat period; 0 is refused.
    pub heartbeat_ms: u32,
    /// Cryptographic randomness, for the handshake.
    pub random: Option<Random>,
}

/// `rlmesh_device_action_t`'s kinds.
pub const RLMESH_DEVICE_SUBSCRIBE: u8 = 1;
/// Declare a liveliness token.
pub const RLMESH_DEVICE_TOKEN: u8 = 2;
/// Query a key.
pub const RLMESH_DEVICE_QUERY: u8 = 3;
/// Publish.
pub const RLMESH_DEVICE_PUBLISH: u8 = 4;

/// `rlmesh_device_action_t`: borrowed from the robot until the next call
/// on it.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RlmeshDeviceAction {
    /// `RLMESH_DEVICE_SUBSCRIBE`, `_TOKEN`, `_QUERY` or `_PUBLISH`.
    pub kind: u8,
    /// A subscription's id, for `rlmesh_device_robot_sample`.
    pub sub: u8,
    /// Whether a query carries `payload`.
    pub has_payload: u8,
    /// A query's id, for `_reply` and `_query_done`.
    pub query_id: u32,
    /// The key expression, NUL-terminated.
    pub key: *const c_char,
    /// A query's or publication's payload.
    pub payload: *const u8,
    /// Its length.
    pub payload_len: usize,
}

/// `rlmesh_device_robot_status_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RlmeshDeviceRobotStatus {
    /// 0 reading the host, 1 handshake, 2 attach, 3 declaring its parts, 4
    /// running, 5 waiting to retry.
    pub phase: u8,
    /// Why it last started over: 0 never, 1 no reply, 2 the host's state
    /// unsigned, 3 the handshake failed (another host's key), 4 refused
    /// (`refused_status`, the proto's `CtlStatus` number), 5 the host lost, 6 internal.
    pub fault: u8,
    /// The host's status code for a refusal.
    pub refused_status: u8,
    /// The session mode: 0 none yet, 1 disarmed, 2 armed, 3 active, 4
    /// safe, 5 e-stop.
    pub mode: u8,
    /// This attach's generation.
    pub generation: u32,
    /// The authority epoch.
    pub epoch: u64,
    /// Heartbeat echoes received.
    pub echoes: u64,
}

/// `rlmesh_device_robot_t`: the robot, its randomness and the action the
/// caller holds.
pub struct DeviceRobot {
    robot: Robot,
    rng: CRng,
    queue: VecDeque<Action>,
    /// The action last handed out: its key with a NUL, and its payload.
    key: Vec<u8>,
    payload: Vec<u8>,
}

/// # Safety
/// `p` is null or a NUL-terminated string.
unsafe fn text<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    // SAFETY: per the contract.
    unsafe { CStr::from_ptr(p) }.to_str().ok()
}

/// # Safety
/// `p` is null or valid for 32 bytes.
unsafe fn key32(p: *const u8) -> Option<[u8; 32]> {
    if p.is_null() {
        return None;
    }
    let mut out = [0; 32];
    // SAFETY: per the contract.
    out.copy_from_slice(unsafe { core::slice::from_raw_parts(p, 32) });
    Some(out)
}

/// A new robot, joining on the first `rlmesh_device_robot_next`.
///
/// # Safety
/// `config` and `out` are null or valid; the config's strings are null or
/// NUL-terminated, its keys null or 32 bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_robot_new(
    config: *const RlmeshDeviceRobotConfig,
    out: *mut *mut DeviceRobot,
) -> c_int {
    if config.is_null() || out.is_null() {
        return RLMESH_DEVICE_ERR_ARG;
    }
    // SAFETY: checked non-null; valid per the contract.
    let c = unsafe { &*config };
    // SAFETY: per the contract.
    let fields = unsafe {
        (
            text(c.tenant),
            text(c.session),
            text(c.name),
            key32(c.secret),
            key32(c.host_key),
            text(c.robot_json),
            c.random,
        )
    };
    let (
        Some(tenant),
        Some(session),
        Some(name),
        Some(secret),
        Some(host_key),
        Some(json),
        Some(random),
    ) = fields
    else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    // A session's name (`default`) or its UUID, as in an address.
    let session = session
        .parse::<SessionId>()
        .unwrap_or_else(|_| SessionId::named(session));
    let (Ok(tenant), Ok(name)) = (TenantId::new(tenant), ParticipantId::new(name)) else {
        return RLMESH_DEVICE_ERR_CONFIG;
    };
    if c.heartbeat_ms == 0 {
        return RLMESH_DEVICE_ERR_CONFIG;
    }
    let robot = Robot::new(Config {
        tenant,
        session,
        name,
        credential: Credential::local(&secret, host_key),
        robot_json: json.into(),
        heartbeat_ms: c.heartbeat_ms,
    });
    let robot = Box::new(DeviceRobot {
        robot,
        rng: CRng(random),
        queue: VecDeque::new(),
        key: Vec::new(),
        payload: Vec::new(),
    });
    // SAFETY: checked non-null; valid for a write per the contract.
    unsafe { out.write(Box::into_raw(robot)) };
    RLMESH_DEVICE_OK
}

/// Frees a robot from `rlmesh_device_robot_new`; null is a no-op.
///
/// # Safety
/// `robot` is null or from `rlmesh_device_robot_new`, and not used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_robot_free(robot: *mut DeviceRobot) {
    if !robot.is_null() {
        // SAFETY: from `Box::into_raw`, per the contract.
        drop(unsafe { Box::from_raw(robot) });
    }
}

/// # Safety
/// `robot` is null or from `rlmesh_device_robot_new`.
unsafe fn robot_mut<'a>(robot: *mut DeviceRobot) -> Option<&'a mut DeviceRobot> {
    // SAFETY: per the contract.
    unsafe { robot.as_mut() }
}

/// Runs the robot's timers and hands out its next action: 1 with `*out`
/// written (valid until the next call on `robot`), 0 when there is none
/// for now. Call it until it returns 0, every loop.
///
/// # Safety
/// `robot` is null or from `rlmesh_device_robot_new`; `out` null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_robot_next(
    robot: *mut DeviceRobot,
    now_us: u64,
    out: *mut RlmeshDeviceAction,
) -> c_int {
    // SAFETY: per the contract.
    let Some(r) = (unsafe { robot_mut(robot) }) else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    if out.is_null() {
        return RLMESH_DEVICE_ERR_ARG;
    }
    if r.queue.is_empty() {
        r.queue.extend(r.robot.poll(now_us));
    }
    let Some(action) = r.queue.pop_front() else {
        return 0;
    };
    let (kind, sub, query_id, key, payload) = match action {
        Action::Subscribe { sub, key } => (RLMESH_DEVICE_SUBSCRIBE, sub as u8, 0, key, None),
        Action::Token { key } => (RLMESH_DEVICE_TOKEN, 0, 0, key, None),
        Action::Query { id, key, payload } => (RLMESH_DEVICE_QUERY, 0, id, key, payload),
        Action::Publish { key, payload } => (RLMESH_DEVICE_PUBLISH, 0, 0, key, Some(payload)),
    };
    r.key = key.into_bytes();
    r.key.push(0);
    let has_payload = payload.is_some();
    r.payload = payload.unwrap_or_default();
    // SAFETY: checked non-null; valid for a write per the contract.
    unsafe {
        out.write(RlmeshDeviceAction {
            kind,
            sub,
            has_payload: u8::from(has_payload),
            query_id,
            key: r.key.as_ptr().cast(),
            payload: r.payload.as_ptr(),
            payload_len: r.payload.len(),
        });
    }
    1
}

/// # Safety
/// `data` is null with `len` 0, or valid for `len` bytes.
unsafe fn bytes<'a>(data: *const u8, len: usize) -> Option<&'a [u8]> {
    if data.is_null() {
        return (len == 0).then_some(&[]);
    }
    // SAFETY: per the contract.
    Some(unsafe { core::slice::from_raw_parts(data, len) })
}

/// One reply to query `query_id`.
///
/// # Safety
/// `robot` is null or from `rlmesh_device_robot_new`; `data` valid for
/// `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_robot_reply(
    robot: *mut DeviceRobot,
    query_id: u32,
    data: *const u8,
    len: usize,
    now_us: u64,
) -> c_int {
    // SAFETY: per the contract.
    let (Some(r), Some(data)) = (unsafe { robot_mut(robot) }, unsafe { bytes(data, len) }) else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    r.robot.reply(query_id, data, now_us, &mut r.rng);
    RLMESH_DEVICE_OK
}

/// Query `query_id` is over: every reply has arrived, or it timed out.
///
/// # Safety
/// `robot` is null or from `rlmesh_device_robot_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_robot_query_done(
    robot: *mut DeviceRobot,
    query_id: u32,
    now_us: u64,
) -> c_int {
    // SAFETY: per the contract.
    let Some(r) = (unsafe { robot_mut(robot) }) else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    r.robot.query_done(query_id, now_us);
    RLMESH_DEVICE_OK
}

/// A sample on subscription `sub`.
///
/// # Safety
/// `robot` is null or from `rlmesh_device_robot_new`; `data` valid for
/// `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_robot_sample(
    robot: *mut DeviceRobot,
    sub: u8,
    data: *const u8,
    len: usize,
    now_us: u64,
) -> c_int {
    // SAFETY: per the contract.
    let (Some(r), Some(data), Some(sub)) = (
        unsafe { robot_mut(robot) },
        unsafe { bytes(data, len) },
        Sub::from_u8(sub),
    ) else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    // Opened in place: the transport's buffer is its own.
    let mut copy = data.to_vec();
    r.robot.sample(sub, &mut copy, now_us);
    RLMESH_DEVICE_OK
}

/// The proto's `CtlStatus` number.
fn status_code(status: CtlStatus) -> u8 {
    match status {
        CtlStatus::InvalidRequest => 2,
        CtlStatus::NotAttached => 3,
        CtlStatus::Forbidden => 4,
        CtlStatus::InvalidTransition => 5,
        CtlStatus::EstopLatched => 6,
        CtlStatus::StaleEpoch => 7,
        CtlStatus::StaleToken => 8,
        CtlStatus::LeaseBusy => 9,
        CtlStatus::UnknownPath => 11,
        CtlStatus::Unsupported => 12,
        CtlStatus::Incompatible => 13,
        CtlStatus::GroupBusy => 14,
        CtlStatus::NotBound => 15,
        CtlStatus::NotReady => 16,
        CtlStatus::Unauthenticated => 17,
        CtlStatus::NotAuthorized => 18,
        CtlStatus::Expired => 19,
    }
}

/// Where the robot is.
///
/// # Safety
/// `robot` is null or from `rlmesh_device_robot_new`; `out` null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_robot_status(
    robot: *const DeviceRobot,
    out: *mut RlmeshDeviceRobotStatus,
) -> c_int {
    // SAFETY: per the contract.
    let Some(r) = (unsafe { robot.as_ref() }) else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    if out.is_null() {
        return RLMESH_DEVICE_ERR_ARG;
    }
    let robot = &r.robot;
    let phase = match robot.phase() {
        Phase::Host => 0,
        Phase::Handshake => 1,
        Phase::Attach => 2,
        Phase::Declare => 3,
        Phase::Running => 4,
        Phase::Retry => 5,
    };
    let (fault, refused_status) = match robot.fault() {
        None => (0, 0),
        Some(Fault::NoReply) => (1, 0),
        Some(Fault::Unsigned) => (2, 0),
        Some(Fault::Handshake) => (3, 0),
        Some(Fault::Refused(status, _)) => (4, status_code(*status)),
        Some(Fault::HostLost) => (5, 0),
        Some(Fault::Internal) => (6, 0),
    };
    let (mode, epoch) = robot.snapshot().map_or((0, 0), |s| {
        let mode = match s.mode {
            Mode::Disarmed => 1,
            Mode::Armed => 2,
            Mode::Active => 3,
            Mode::Safe => 4,
            Mode::Estop => 5,
        };
        (mode, s.epoch.0)
    });
    // SAFETY: checked non-null; valid for a write per the contract.
    unsafe {
        out.write(RlmeshDeviceRobotStatus {
            phase,
            fault,
            refused_status,
            mode,
            generation: robot.generation(),
            epoch,
            echoes: robot.echoes(),
        });
    }
    RLMESH_DEVICE_OK
}

#[cfg(test)]
mod tests {
    use core::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static CLOCK: AtomicU64 = AtomicU64::new(0);

    unsafe extern "C" fn now_us() -> u64 {
        CLOCK.fetch_add(10, Ordering::Relaxed)
    }

    static DRAWN: AtomicU64 = AtomicU64::new(0);

    /// A different byte stream on every call: no two keys alike.
    unsafe extern "C" fn random(buf: *mut u8, len: usize) -> c_int {
        for i in 0..len {
            let n = DRAWN
                .fetch_add(1, Ordering::Relaxed)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15);
            unsafe { buf.add(i).write((n >> 56) as u8) };
        }
        0
    }

    unsafe extern "C" fn broken(_: *mut u8, _: usize) -> c_int {
        -1
    }

    #[test]
    fn timing_through_the_c_abi() {
        let mut out = RlmeshDeviceTiming::default();
        let rc = unsafe { rlmesh_device_timing(Some(now_us), Some(random), &mut out) };
        assert_eq!(rc, RLMESH_DEVICE_OK);
        assert_eq!(out.chunk_open_seal_us, 10);
        let rc = unsafe { rlmesh_device_timing(Some(now_us), Some(broken), &mut out) };
        assert_eq!(rc, RLMESH_DEVICE_ERR_RNG);
        let rc = unsafe { rlmesh_device_timing(None, Some(random), &mut out) };
        assert_eq!(rc, RLMESH_DEVICE_ERR_ARG);
        let rc = unsafe { rlmesh_device_timing(Some(now_us), Some(random), core::ptr::null_mut()) };
        assert_eq!(rc, RLMESH_DEVICE_ERR_ARG);
    }

    fn config(tenant: &core::ffi::CStr, heartbeat_ms: u32) -> RlmeshDeviceRobotConfig {
        static KEY: [u8; 32] = [7; 32];
        RlmeshDeviceRobotConfig {
            tenant: tenant.as_ptr(),
            session: c"default".as_ptr(),
            name: c"eyes".as_ptr(),
            secret: KEY.as_ptr(),
            host_key: KEY.as_ptr(),
            robot_json: c"{}".as_ptr(),
            heartbeat_ms,
            random: Some(random),
        }
    }

    const TENANT: &core::ffi::CStr = c"local";

    #[test]
    fn a_robot_through_the_c_abi_asks_for_its_subscriptions_then_the_host() {
        let mut robot = core::ptr::null_mut();
        let rc = unsafe { rlmesh_device_robot_new(&config(TENANT, 50), &mut robot) };
        assert_eq!(rc, RLMESH_DEVICE_OK);
        let mut kinds = Vec::new();
        let mut keys = Vec::new();
        let mut action = RlmeshDeviceAction {
            kind: 0,
            sub: 0,
            has_payload: 0,
            query_id: 0,
            key: core::ptr::null(),
            payload: core::ptr::null(),
            payload_len: 0,
        };
        while unsafe { rlmesh_device_robot_next(robot, 0, &mut action) } == 1 {
            kinds.push((action.kind, action.sub, action.has_payload));
            let key = unsafe { core::ffi::CStr::from_ptr(action.key) };
            keys.push(key.to_str().unwrap().to_owned());
        }
        assert_eq!(
            kinds,
            [
                (RLMESH_DEVICE_SUBSCRIBE, 0, 0),
                (RLMESH_DEVICE_SUBSCRIBE, 1, 0),
                (RLMESH_DEVICE_SUBSCRIBE, 2, 0),
                (RLMESH_DEVICE_TOKEN, 0, 0),
                (RLMESH_DEVICE_QUERY, 0, 0),
            ]
        );
        assert!(keys[0].ends_with("/host/state"), "{keys:?}");
        assert!(keys[4].ends_with("/host/state"), "{keys:?}");
        assert_eq!(action.query_id, 1);

        let mut status = RlmeshDeviceRobotStatus::default();
        assert_eq!(
            unsafe { rlmesh_device_robot_status(robot, &mut status) },
            RLMESH_DEVICE_OK
        );
        assert_eq!((status.phase, status.fault, status.mode), (0, 0, 0));
        // An unsigned state fails the attempt, and it retries later.
        let rc = unsafe { rlmesh_device_robot_reply(robot, 1, b"plain".as_ptr(), 5, 10) };
        assert_eq!(rc, RLMESH_DEVICE_OK);
        unsafe { rlmesh_device_robot_status(robot, &mut status) };
        assert_eq!((status.phase, status.fault), (5, 2));
        let rc = unsafe { rlmesh_device_robot_sample(robot, 9, b"x".as_ptr(), 1, 10) };
        assert_eq!(rc, RLMESH_DEVICE_ERR_ARG);
        unsafe { rlmesh_device_robot_free(robot) };
    }

    #[test]
    fn a_robot_config_is_checked() {
        let mut robot = core::ptr::null_mut();
        let rc = unsafe { rlmesh_device_robot_new(&config(c"not/a/tenant", 50), &mut robot) };
        assert_eq!(rc, RLMESH_DEVICE_ERR_CONFIG);
        let rc = unsafe { rlmesh_device_robot_new(&config(TENANT, 0), &mut robot) };
        assert_eq!(rc, RLMESH_DEVICE_ERR_CONFIG);
        let mut missing = config(TENANT, 50);
        missing.secret = core::ptr::null();
        let rc = unsafe { rlmesh_device_robot_new(&missing, &mut robot) };
        assert_eq!(rc, RLMESH_DEVICE_ERR_ARG);
        assert!(robot.is_null());
        unsafe { rlmesh_device_robot_free(core::ptr::null_mut()) };
    }

    #[test]
    fn version_is_the_crate_version() {
        let v = rlmesh_device_version();
        assert_eq!(
            v >> 16,
            env!("CARGO_PKG_VERSION_MAJOR").parse::<u32>().unwrap()
        );
    }
}
