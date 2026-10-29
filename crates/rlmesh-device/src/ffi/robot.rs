//! The robot's C ABI: `rlmesh_device_robot_*`.

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;
use core::ffi::{CStr, c_char, c_int};

use rlmesh_session_core::wire::{
    CtlStatus, EdgeMode, Mode, ParticipantId, ReasonCode, SessionId, TenantId,
};

use super::{
    CRng, RLMESH_DEVICE_ERR_ARG, RLMESH_DEVICE_ERR_CONFIG, RLMESH_DEVICE_ERR_INVALID,
    RLMESH_DEVICE_ERR_NOT_READY, RLMESH_DEVICE_OK, Random,
};
use rlmesh_session_core::auth::Credential;
use rlmesh_session_core::crypto::StaticKeypair;
use zeroize::Zeroizing;

use crate::address;
use crate::provision::Record;
use crate::robot::{Action, Config, Fault, Phase, PublishError, Robot};

/// `rlmesh_device_robot_config_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RlmeshDeviceRobotConfig {
    /// The address the host printed, with its key; null for a provisioned
    /// board.
    pub address: *const c_char,
    /// A provisioned board's device key (32 bytes), with `address` null.
    pub device_key: *const u8,
    /// Its stored provisioning record (tether spec §14).
    pub record: *const u8,
    /// The record's length.
    pub record_len: usize,
    /// This robot's name; null takes the address's.
    pub name: *const c_char,
    /// The robot's map, as authored JSON.
    pub robot_json: *const c_char,
    /// The heartbeat period; 0 is refused.
    pub heartbeat_ms: u32,
    /// Cryptographic randomness, for the handshake.
    pub random: Option<Random>,
}

/// `rlmesh_device_action_t`'s kinds: subscribe.
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
    /// The traffic class: the host's port plus this.
    pub class: u8,
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
    /// (`refused_status`, the proto's `CtlStatus` number), 5 the host lost,
    /// 6 internal.
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
    /// Command chunks the edges accepted.
    pub chunks_accepted: u64,
    /// The e-stop latch: 0 clear, 1 this robot's own stop, 2 a member's
    /// signal.
    pub estop: u8,
}

/// No part: `rlmesh_device_tick_t.part` and the handover's.
pub const RLMESH_DEVICE_NO_PART: u8 = 0xFF;

/// `rlmesh_device_tick_t`: borrowed from the robot until the next call on
/// it.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RlmeshDeviceTick {
    /// The edge mode: 0 idle, 1 driving, 2 hold, 3 safe, 4 e-stop.
    pub mode: u8,
    /// The part it concerns (the active or last-driven member of the
    /// group), or `RLMESH_DEVICE_NO_PART`.
    pub part: u8,
    /// Whether `value` is an action to dispatch to `part`.
    pub has_action: u8,
    /// A handover between members starts: switch the control mode from
    /// `handover_from` to `handover_to`, then call
    /// `rlmesh_device_robot_handover_done`.
    pub handover: u8,
    /// The handover's parts.
    pub handover_from: u8,
    /// The handover's parts.
    pub handover_to: u8,
    /// The action, in the part's dense command layout.
    pub value: *const u8,
    /// Its length.
    pub value_len: usize,
    /// The declared safe behaviour to run now (`"hold"`, `"stop"`, ...),
    /// NUL-terminated, or null.
    pub safe: *const c_char,
}

/// `rlmesh_device_robot_t`: the robot, its randomness, and what the caller
/// borrows from it.
pub struct DeviceRobot {
    robot: Robot,
    rng: CRng,
    queue: VecDeque<Action>,
    /// The host's address and base port, NUL-terminated.
    host: Vec<u8>,
    port: u16,
    /// The action last handed out: its key with a NUL, and its payload.
    key: Vec<u8>,
    payload: Vec<u8>,
    /// The tick last handed out: its action and safe behaviour.
    value: Vec<u8>,
    safe: Vec<u8>,
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

/// Where a robot joins, and as which device.
struct Target {
    host: String,
    port: u16,
    session: String,
    credential: Credential,
    name: Option<String>,
}

/// The target of `address` (device `local`), or of a provisioned board's
/// device key and record (session `default` on the record's first
/// address).
///
/// # Safety
/// The config's key and record are null or valid for their lengths.
unsafe fn target(address: Option<&str>, c: &RlmeshDeviceRobotConfig) -> Result<Target, c_int> {
    if let Some(address) = address {
        let address = address::parse(address).ok_or(RLMESH_DEVICE_ERR_CONFIG)?;
        return Ok(Target {
            host: address.host,
            port: address.port,
            session: address.session,
            credential: address.key.credential(),
            name: address.name,
        });
    }
    if c.device_key.is_null() {
        return Err(RLMESH_DEVICE_ERR_ARG);
    }
    // SAFETY: per the contract: 32 bytes, and `record_len` bytes.
    let (key, record) = unsafe {
        (
            core::slice::from_raw_parts(c.device_key, 32),
            bytes(c.record, c.record_len),
        )
    };
    let record = record
        .and_then(Record::parse)
        .ok_or(RLMESH_DEVICE_ERR_CONFIG)?;
    let (host, port) = record
        .addresses
        .first()
        .and_then(|a| address::host_port(a))
        .ok_or(RLMESH_DEVICE_ERR_CONFIG)?;
    let mut private = Zeroizing::new([0_u8; 32]);
    private.copy_from_slice(key);
    Ok(Target {
        host,
        port,
        session: String::from(address::DEFAULT_SESSION),
        credential: Credential::new(StaticKeypair::from_private(&mut private), record.host_key),
        name: None,
    })
}

/// A new robot, joining on the first `rlmesh_device_robot_next`. The
/// config's strings are copied.
///
/// # Safety
/// `config` and `out` are null or valid; the config's strings are null or
/// NUL-terminated.
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
    let (address, name, json) = unsafe { (text(c.address), text(c.name), text(c.robot_json)) };
    let (Some(json), Some(random)) = (json, c.random) else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    // SAFETY: per the contract.
    let target = match unsafe { target(address, c) } {
        Ok(target) => target,
        Err(rc) => return rc,
    };
    let Some(name) = name.map(String::from).or(target.name) else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    // A session's name (`default`) or its UUID.
    let session = target
        .session
        .parse::<SessionId>()
        .unwrap_or_else(|_| SessionId::named(&target.session));
    let (Ok(tenant), Ok(name)) = (TenantId::new(address::TENANT), ParticipantId::new(name)) else {
        return RLMESH_DEVICE_ERR_CONFIG;
    };
    let Ok(robot) = Robot::new(Config {
        tenant,
        session,
        name,
        credential: target.credential,
        robot_json: json.into(),
        heartbeat_ms: c.heartbeat_ms,
    }) else {
        return RLMESH_DEVICE_ERR_CONFIG;
    };
    let mut host = target.host.into_bytes();
    host.push(0);
    let robot = Box::new(DeviceRobot {
        robot,
        rng: CRng(random),
        queue: VecDeque::new(),
        host,
        port: target.port,
        key: Vec::new(),
        payload: Vec::new(),
        value: Vec::new(),
        safe: Vec::new(),
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

/// The host to connect to, NUL-terminated (owned by the robot), and in
/// `*port` its first port: class `i` listens on `port + i`.
///
/// # Safety
/// `robot` is null or from `rlmesh_device_robot_new`; `port` null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_robot_host(
    robot: *const DeviceRobot,
    port: *mut u16,
) -> *const c_char {
    // SAFETY: per the contract.
    let Some(r) = (unsafe { robot.as_ref() }) else {
        return core::ptr::null();
    };
    if !port.is_null() {
        // SAFETY: checked non-null; valid per the contract.
        unsafe { port.write(r.port) };
    }
    r.host.as_ptr().cast()
}

/// The index of the part at `path` in the robot's map, which every part
/// argument takes, or -1.
///
/// # Safety
/// `robot` is null or from `rlmesh_device_robot_new`; `path` null or
/// NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_robot_part(
    robot: *const DeviceRobot,
    path: *const c_char,
) -> c_int {
    // SAFETY: per the contract.
    let (Some(r), Some(path)) = (unsafe { robot.as_ref() }, unsafe { text(path) }) else {
        return -1;
    };
    r.robot
        .part(path)
        .and_then(|i| c_int::try_from(i).ok())
        .unwrap_or(-1)
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
    let (kind, class, sub, query_id, key, payload) = match action {
        Action::Subscribe { class, sub, key } => {
            (RLMESH_DEVICE_SUBSCRIBE, class, sub, 0, key, None)
        }
        Action::Token { class, key } => (RLMESH_DEVICE_TOKEN, class, 0, 0, key, None),
        Action::Query {
            class,
            id,
            key,
            payload,
        } => (RLMESH_DEVICE_QUERY, class, 0, id, key, payload),
        Action::Publish {
            class,
            key,
            payload,
        } => (RLMESH_DEVICE_PUBLISH, class, 0, 0, key, Some(payload)),
    };
    r.key = key.into_bytes();
    r.key.push(0);
    let has_payload = payload.is_some();
    r.payload = payload.unwrap_or_default();
    // SAFETY: checked non-null; valid for a write per the contract.
    unsafe {
        out.write(RlmeshDeviceAction {
            kind,
            class,
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

/// A sample on subscription `sub`, received on key `key` (`key_len`
/// bytes, UTF-8: an e-stop signal names its sender in it).
///
/// # Safety
/// `robot` is null or from `rlmesh_device_robot_new`; `key` valid for
/// `key_len` bytes and `data` for `len`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_robot_sample(
    robot: *mut DeviceRobot,
    sub: u8,
    key: *const c_char,
    key_len: usize,
    data: *const u8,
    len: usize,
    now_us: u64,
) -> c_int {
    // SAFETY: per the contract.
    let (Some(r), Some(key), Some(data)) = (
        unsafe { robot_mut(robot) },
        unsafe { bytes(key.cast(), key_len) },
        unsafe { bytes(data, len) },
    ) else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    let Ok(key) = core::str::from_utf8(key) else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    // Opened in place: the transport's buffer is its own.
    let mut copy = data.to_vec();
    r.robot.sample(sub, key, &mut copy, now_us);
    RLMESH_DEVICE_OK
}

/// Latches this robot's e-stop for its own stop (an e-stop pin,
/// `ESTOP_LOCAL`, `reason` NUL-terminated or null): every part stops at
/// once, with or without a host, and the host is told. Only
/// `rlmesh_device_robot_clear_estop` clears it.
///
/// # Safety
/// `robot` is null or from `rlmesh_device_robot_new`; `reason` null or
/// NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_robot_estop(
    robot: *mut DeviceRobot,
    reason: *const c_char,
    now_us: u64,
) -> c_int {
    // SAFETY: per the contract.
    let Some(r) = (unsafe { robot_mut(robot) }) else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    // SAFETY: per the contract.
    let reason = unsafe { text(reason) }.unwrap_or("e-stop pin");
    r.robot.estop(reason, now_us);
    RLMESH_DEVICE_OK
}

/// Clears the e-stop latch, whoever set it: the only way to. Call it only
/// from a local action on the board (a reset button, with the e-stop
/// released). The host ends in ESTOP, unlatched; leaving that still needs
/// disarm and arm.
///
/// # Safety
/// `robot` is null or from `rlmesh_device_robot_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_robot_clear_estop(robot: *mut DeviceRobot) -> c_int {
    // SAFETY: per the contract.
    let Some(r) = (unsafe { robot_mut(robot) }) else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    r.robot.clear_estop();
    RLMESH_DEVICE_OK
}

fn part_u8(part: Option<usize>) -> u8 {
    part.and_then(|p| u8::try_from(p).ok())
        .unwrap_or(RLMESH_DEVICE_NO_PART)
}

/// One control tick of part `part`'s command group: 1 with `*out` written
/// (valid until the next call on `robot`) when there is something to do (an
/// action, a safe behaviour to run, or a handover), 0 when there is
/// nothing, `RLMESH_DEVICE_ERR_NOT_READY` before the robot runs or for a
/// part that takes no commands. Call it at the part's control rate; it
/// publishes the group's applied samples.
///
/// # Safety
/// `robot` is null or from `rlmesh_device_robot_new`; `out` null or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_robot_tick(
    robot: *mut DeviceRobot,
    part: u8,
    now_us: u64,
    out: *mut RlmeshDeviceTick,
) -> c_int {
    // SAFETY: per the contract.
    let Some(r) = (unsafe { robot_mut(robot) }) else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    if out.is_null() {
        return RLMESH_DEVICE_ERR_ARG;
    }
    let Some(tick) = r.robot.tick(usize::from(part), now_us) else {
        return RLMESH_DEVICE_ERR_NOT_READY;
    };
    let edge = &tick.tick;
    if edge.action.is_none() && edge.safe_behavior.is_none() && edge.handover.is_none() {
        return 0;
    }
    r.value = edge
        .action
        .as_ref()
        .map(|a| a.value.to_vec())
        .unwrap_or_default();
    r.safe = edge
        .safe_behavior
        .as_ref()
        .map(|s| {
            let mut name = s.as_bytes().to_vec();
            name.push(0);
            name
        })
        .unwrap_or_default();
    let member = |channel| r.robot.part_of(channel);
    let (handover_from, handover_to) = edge
        .handover
        .as_ref()
        .map_or((RLMESH_DEVICE_NO_PART, RLMESH_DEVICE_NO_PART), |h| {
            (part_u8(member(h.from)), part_u8(member(h.to)))
        });
    let mode = match edge.mode {
        EdgeMode::Idle => 0,
        EdgeMode::Driving => 1,
        EdgeMode::Hold => 2,
        EdgeMode::Safe => 3,
        EdgeMode::Estop => 4,
    };
    // SAFETY: checked non-null; valid for a write per the contract.
    unsafe {
        out.write(RlmeshDeviceTick {
            mode,
            part: part_u8(tick.part),
            has_action: u8::from(edge.action.is_some()),
            handover: u8::from(edge.handover.is_some()),
            handover_from,
            handover_to,
            value: r.value.as_ptr(),
            value_len: r.value.len(),
            safe: if r.safe.is_empty() {
                core::ptr::null()
            } else {
                r.safe.as_ptr().cast()
            },
        });
    }
    1
}

/// The robot finished the handover of `part`'s group (it switched its
/// control mode and neutralised the old member's target): the group may
/// dispatch again.
///
/// # Safety
/// `robot` is null or from `rlmesh_device_robot_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_robot_handover_done(
    robot: *mut DeviceRobot,
    part: u8,
) -> c_int {
    // SAFETY: per the contract.
    let Some(r) = (unsafe { robot_mut(robot) }) else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    r.robot.handover_done(usize::from(part));
    RLMESH_DEVICE_OK
}

/// Publishes a sample of part `part`: `value` in its dense layout,
/// captured at `capture_us` (this board's clock).
/// `RLMESH_DEVICE_ERR_NOT_READY` before the robot runs or its key arrived,
/// `_ERR_INVALID` for a value not in the part's layout, `_ERR_ARG` for no
/// such part.
///
/// # Safety
/// `robot` is null or from `rlmesh_device_robot_new`; `value` valid for
/// `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlmesh_device_robot_publish(
    robot: *mut DeviceRobot,
    part: u8,
    capture_us: u64,
    value: *const u8,
    len: usize,
    now_us: u64,
) -> c_int {
    // SAFETY: per the contract.
    let (Some(r), Some(value)) = (unsafe { robot_mut(robot) }, unsafe { bytes(value, len) }) else {
        return RLMESH_DEVICE_ERR_ARG;
    };
    match r
        .robot
        .publish(usize::from(part), capture_us, value, now_us)
    {
        Ok(()) => RLMESH_DEVICE_OK,
        Err(PublishError::UnknownPart) => RLMESH_DEVICE_ERR_ARG,
        Err(PublishError::NotReady) => RLMESH_DEVICE_ERR_NOT_READY,
        Err(PublishError::Invalid) => RLMESH_DEVICE_ERR_INVALID,
    }
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
        CtlStatus::RobotBusy => 20,
        CtlStatus::NotPresent => 21,
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
            chunks_accepted: robot.chunks_accepted(),
            estop: match robot.estop_reason().map(|r| r.code) {
                None => 0,
                Some(ReasonCode::EstopSignal) => 2,
                Some(_) => 1,
            },
        });
    }
    RLMESH_DEVICE_OK
}

#[cfg(test)]
mod tests {
    use core::sync::atomic::{AtomicU64, Ordering};

    use rlmesh_session_core::auth::address::base32;

    use super::*;

    static DRAWN: AtomicU64 = AtomicU64::new(0);

    unsafe extern "C" fn random(buf: *mut u8, len: usize) -> c_int {
        for i in 0..len {
            let n = DRAWN
                .fetch_add(1, Ordering::Relaxed)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15);
            unsafe { buf.add(i).write((n >> 56) as u8) };
        }
        0
    }

    const MAP: &CStr = c"{\"eyes\": {\"kind\": \"joints\", \"names\": [\"pan\", \"tilt\"], \"fields\": [\"pos\"], \"hz\": 50, \"command\": \"position\", \"safe\": \"hold\"}}";

    fn address(tail: &str) -> Vec<u8> {
        let mut text = alloc::format!(
            "10.0.0.2:7447#{}.{}{tail}",
            base32(&[1; 32]),
            base32(&[2; 32])
        )
        .into_bytes();
        text.push(0);
        text
    }

    fn config(address: &[u8], name: Option<&CStr>, map: &CStr) -> RlmeshDeviceRobotConfig {
        RlmeshDeviceRobotConfig {
            address: address.as_ptr().cast(),
            device_key: core::ptr::null(),
            record: core::ptr::null(),
            record_len: 0,
            name: name.map_or(core::ptr::null(), CStr::as_ptr),
            robot_json: map.as_ptr(),
            heartbeat_ms: 50,
            random: Some(random),
        }
    }

    fn empty_action() -> RlmeshDeviceAction {
        RlmeshDeviceAction {
            kind: 0,
            class: 0,
            sub: 0,
            has_payload: 0,
            query_id: 0,
            key: core::ptr::null(),
            payload: core::ptr::null(),
            payload_len: 0,
        }
    }

    #[test]
    fn a_robot_through_the_c_abi_asks_for_its_subscriptions_then_the_host() {
        let addr = address("");
        let mut robot = core::ptr::null_mut();
        let rc = unsafe { rlmesh_device_robot_new(&config(&addr, Some(c"eyes"), MAP), &mut robot) };
        assert_eq!(rc, RLMESH_DEVICE_OK);
        let mut port = 0;
        let host = unsafe { CStr::from_ptr(rlmesh_device_robot_host(robot, &mut port)) };
        assert_eq!((host.to_str().unwrap(), port), ("10.0.0.2", 7447));
        assert_eq!(
            unsafe { rlmesh_device_robot_part(robot, c"eyes".as_ptr()) },
            0
        );
        assert_eq!(
            unsafe { rlmesh_device_robot_part(robot, c"nose".as_ptr()) },
            -1
        );

        let mut kinds = Vec::new();
        let mut keys = Vec::new();
        let mut action = empty_action();
        while unsafe { rlmesh_device_robot_next(robot, 0, &mut action) } == 1 {
            kinds.push((action.kind, action.class, action.sub, action.has_payload));
            let key = unsafe { CStr::from_ptr(action.key) };
            keys.push(String::from(key.to_str().unwrap()));
        }
        assert_eq!(
            kinds,
            [
                (RLMESH_DEVICE_SUBSCRIBE, 0, 0, 0),
                (RLMESH_DEVICE_SUBSCRIBE, 0, 1, 0),
                (RLMESH_DEVICE_SUBSCRIBE, 0, 2, 0),
                (RLMESH_DEVICE_SUBSCRIBE, 0, 3, 0),
                (RLMESH_DEVICE_SUBSCRIBE, 0, 4, 0),
                (RLMESH_DEVICE_TOKEN, 0, 0, 0),
                (RLMESH_DEVICE_QUERY, 0, 0, 0),
            ]
        );
        assert!(keys[0].ends_with("/host/state"), "{keys:?}");
        assert!(keys[3].ends_with("/p/eyes/keys"), "{keys:?}");
        assert!(keys[4].ends_with("/estop/*"), "{keys:?}");
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
        // Not running: nothing to tick or publish.
        let mut tick = core::mem::MaybeUninit::<RlmeshDeviceTick>::uninit();
        let rc = unsafe { rlmesh_device_robot_tick(robot, 0, 10, tick.as_mut_ptr()) };
        assert_eq!(rc, RLMESH_DEVICE_ERR_NOT_READY);
        let value = [0_u8; 8];
        let rc = unsafe { rlmesh_device_robot_publish(robot, 0, 10, value.as_ptr(), 8, 10) };
        assert_eq!(rc, RLMESH_DEVICE_ERR_NOT_READY);
        let rc = unsafe { rlmesh_device_robot_publish(robot, 9, 10, value.as_ptr(), 8, 10) };
        assert_eq!(rc, RLMESH_DEVICE_ERR_ARG);
        // The pin's latch holds before any session, and only a clear
        // clears it.
        assert_eq!(
            unsafe { rlmesh_device_robot_estop(robot, c"pin".as_ptr(), 20) },
            RLMESH_DEVICE_OK
        );
        unsafe { rlmesh_device_robot_status(robot, &mut status) };
        assert_eq!(status.estop, 1);
        assert_eq!(
            unsafe { rlmesh_device_robot_clear_estop(robot) },
            RLMESH_DEVICE_OK
        );
        unsafe { rlmesh_device_robot_status(robot, &mut status) };
        assert_eq!(status.estop, 0);
        unsafe { rlmesh_device_robot_free(robot) };
    }

    #[test]
    fn a_provisioned_board_joins_its_records_first_address() {
        use rlmesh_tether::provision::{self, Addresses, Prov, Provision};
        let p = Prov::Provision(Provision {
            flags: 0,
            host_key: [9; 32],
            host_id: b"rlm-3f9a2c",
            addresses: Addresses::new(&[b"[fe80::1]:7447", b"10.0.0.9:7447"]).unwrap(),
        });
        let mut record = [0; 252];
        let n = provision::payload(&p, &mut record).unwrap();
        let key = [5_u8; 32];
        let mut config = config(b"\0", Some(c"eyes"), MAP);
        config.address = core::ptr::null();
        config.device_key = key.as_ptr();
        config.record = record.as_ptr();
        config.record_len = n;
        let mut robot = core::ptr::null_mut();
        assert_eq!(
            unsafe { rlmesh_device_robot_new(&config, &mut robot) },
            RLMESH_DEVICE_OK
        );
        let mut port = 0;
        let host = unsafe { CStr::from_ptr(rlmesh_device_robot_host(robot, &mut port)) };
        assert_eq!((host.to_str().unwrap(), port), ("[fe80::1]", 7447));
        unsafe { rlmesh_device_robot_free(robot) };
        // A record that isn't one, and no key at all.
        config.record_len = 3;
        assert_eq!(
            unsafe { rlmesh_device_robot_new(&config, &mut robot) },
            RLMESH_DEVICE_ERR_CONFIG
        );
        config.device_key = core::ptr::null();
        assert_eq!(
            unsafe { rlmesh_device_robot_new(&config, &mut robot) },
            RLMESH_DEVICE_ERR_ARG
        );
    }

    #[test]
    fn the_name_can_come_from_the_address() {
        let addr = address("/eyes");
        let mut robot = core::ptr::null_mut();
        let rc = unsafe { rlmesh_device_robot_new(&config(&addr, None, MAP), &mut robot) };
        assert_eq!(rc, RLMESH_DEVICE_OK);
        unsafe { rlmesh_device_robot_free(robot) };
        let addr = address("");
        let rc = unsafe { rlmesh_device_robot_new(&config(&addr, None, MAP), &mut robot) };
        assert_eq!(rc, RLMESH_DEVICE_ERR_ARG, "no name anywhere");
    }

    #[test]
    fn a_robot_config_is_checked() {
        let mut robot = core::ptr::null_mut();
        let addr = address("");
        let rc = unsafe {
            rlmesh_device_robot_new(&config(b"10.0.0.2:7447\0", Some(c"eyes"), MAP), &mut robot)
        };
        assert_eq!(rc, RLMESH_DEVICE_ERR_CONFIG, "no key");
        let rc = unsafe {
            rlmesh_device_robot_new(&config(&addr, Some(c"not/a/name"), MAP), &mut robot)
        };
        assert_eq!(rc, RLMESH_DEVICE_ERR_CONFIG);
        let rc = unsafe {
            rlmesh_device_robot_new(&config(&addr, Some(c"eyes"), c"{\"eyes\": {}}"), &mut robot)
        };
        assert_eq!(rc, RLMESH_DEVICE_ERR_CONFIG, "a map the host refuses");
        let mut no_beat = config(&addr, Some(c"eyes"), MAP);
        no_beat.heartbeat_ms = 0;
        let rc = unsafe { rlmesh_device_robot_new(&no_beat, &mut robot) };
        assert_eq!(rc, RLMESH_DEVICE_ERR_CONFIG);
        let mut no_rng = config(&addr, Some(c"eyes"), MAP);
        no_rng.random = None;
        let rc = unsafe { rlmesh_device_robot_new(&no_rng, &mut robot) };
        assert_eq!(rc, RLMESH_DEVICE_ERR_ARG);
        assert!(robot.is_null());
        unsafe { rlmesh_device_robot_free(core::ptr::null_mut()) };
    }
}
