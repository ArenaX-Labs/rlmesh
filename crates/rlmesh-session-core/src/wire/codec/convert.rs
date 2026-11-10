//! Conversions every message shares: enums, fingerprints, channel lists
//! and fixed-size byte fields.

use super::pb;
use crate::prelude::*;

use crate::wire::authority::{ClockMode, Mode, PriorityClass, Role};
use crate::wire::error::WireError;
use crate::wire::ids::{ChannelId, Fingerprint};

pub(super) fn collect<T, U, E>(
    items: Vec<T>,
    f: impl FnMut(T) -> Result<U, E>,
) -> Result<Vec<U>, WireError>
where
    WireError: From<E>,
{
    Ok(items.into_iter().map(f).collect::<Result<Vec<U>, E>>()?)
}

pub(super) fn bad_enum(what: &'static str, raw: i32) -> WireError {
    WireError::malformed(what, format!("unknown value {raw}"))
}

pub(super) fn fingerprint_from_pb(
    what: &'static str,
    bytes: &[u8],
) -> Result<Fingerprint, WireError> {
    Fingerprint::from_slice(bytes)
        .ok_or_else(|| WireError::malformed(what, format!("a {}-byte fingerprint", bytes.len())))
}

pub(super) fn bytes32(
    message: &'static str,
    field: &str,
    raw: &[u8],
) -> Result<[u8; 32], WireError> {
    raw.try_into()
        .map_err(|_| WireError::malformed(message, format!("{field} is not 32 bytes")))
}

pub(super) fn channels(ids: &[ChannelId]) -> Vec<u32> {
    ids.iter().map(|c| c.0).collect()
}

pub(super) fn role_to_pb(role: Role) -> pb::Role {
    match role {
        Role::Embodiment => pb::Role::Embodiment,
        Role::Controller => pb::Role::Controller,
        Role::Observer => pb::Role::Observer,
    }
}

pub(super) fn role_from_pb(raw: i32) -> Result<Role, WireError> {
    match pb::Role::try_from(raw) {
        Ok(pb::Role::Embodiment) => Ok(Role::Embodiment),
        Ok(pb::Role::Controller) => Ok(Role::Controller),
        Ok(pb::Role::Observer) => Ok(Role::Observer),
        _ => Err(bad_enum("role", raw)),
    }
}

pub(super) fn mode_to_pb(mode: Mode) -> pb::Mode {
    match mode {
        Mode::Disarmed => pb::Mode::Disarmed,
        Mode::Armed => pb::Mode::Armed,
        Mode::Active => pb::Mode::Active,
        Mode::Safe => pb::Mode::Safe,
        Mode::Estop => pb::Mode::Estop,
    }
}

pub(super) fn mode_from_pb(raw: i32) -> Result<Mode, WireError> {
    match pb::Mode::try_from(raw) {
        Ok(pb::Mode::Disarmed) => Ok(Mode::Disarmed),
        Ok(pb::Mode::Armed) => Ok(Mode::Armed),
        Ok(pb::Mode::Active) => Ok(Mode::Active),
        Ok(pb::Mode::Safe) => Ok(Mode::Safe),
        Ok(pb::Mode::Estop) => Ok(Mode::Estop),
        _ => Err(bad_enum("mode", raw)),
    }
}

pub(super) fn clock_mode_to_pb(mode: ClockMode) -> pb::ClockMode {
    match mode {
        ClockMode::Wall => pb::ClockMode::Wall,
        ClockMode::Virtual => pb::ClockMode::Virtual,
    }
}

pub(super) fn clock_mode_from_pb(raw: i32) -> Result<ClockMode, WireError> {
    match pb::ClockMode::try_from(raw) {
        Ok(pb::ClockMode::Wall | pb::ClockMode::Unspecified) => Ok(ClockMode::Wall),
        Ok(pb::ClockMode::Virtual) => Ok(ClockMode::Virtual),
        Err(_) => Err(bad_enum("clock mode", raw)),
    }
}

pub(super) fn priority_to_pb(priority: PriorityClass) -> pb::PriorityClass {
    match priority {
        PriorityClass::Scripted => pb::PriorityClass::Scripted,
        PriorityClass::Model => pb::PriorityClass::Model,
        PriorityClass::Operator => pb::PriorityClass::Operator,
        PriorityClass::Safety => pb::PriorityClass::Safety,
    }
}

pub(super) fn priority_from_pb(raw: i32) -> Result<PriorityClass, WireError> {
    match pb::PriorityClass::try_from(raw) {
        Ok(pb::PriorityClass::Scripted) => Ok(PriorityClass::Scripted),
        Ok(pb::PriorityClass::Model) => Ok(PriorityClass::Model),
        Ok(pb::PriorityClass::Operator) => Ok(PriorityClass::Operator),
        Ok(pb::PriorityClass::Safety) => Ok(PriorityClass::Safety),
        _ => Err(bad_enum("priority class", raw)),
    }
}
