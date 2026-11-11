//! The session handshake: request, reply and accept.

use super::convert::{priority_from_pb, priority_to_pb};
use super::ctl::{status_from_pb, status_to_pb};
use super::pb;
use crate::prelude::*;

use bytes::Bytes;
use prost::Message as _;

use crate::wire::authority::{CtlError, CtlStatus, HandshakeAccept, HandshakeRequest};
use crate::wire::datagram::DatagramOffer;
use crate::wire::error::WireError;
use crate::wire::ids::ParticipantId;

/// Encodes msg 1 of the handshake.
pub fn encode_handshake_request(request: &HandshakeRequest) -> Vec<u8> {
    pb::HandshakeRequest {
        participant: request.participant.to_string(),
        noise: Bytes::copy_from_slice(&request.noise),
    }
    .encode_to_vec()
}

/// Decodes msg 1 of the handshake.
pub fn decode_handshake_request(buf: &[u8]) -> Result<HandshakeRequest, WireError> {
    let request = pb::HandshakeRequest::decode(buf)?;
    Ok(HandshakeRequest {
        participant: ParticipantId::new(request.participant)?,
        noise: request.noise.to_vec(),
    })
}

/// Encodes the host's answer to a handshake: msg 2, or a refusal.
pub fn encode_handshake_reply(reply: &Result<Vec<u8>, CtlError>) -> Vec<u8> {
    use pb::handshake_reply::Result as Answer;
    let result = match reply {
        Ok(noise) => Answer::Noise(Bytes::copy_from_slice(noise)),
        Err(err) => Answer::Error(pb::CtlError {
            status: status_to_pb(err.status) as i32,
            message: err.message.clone(),
        }),
    };
    pb::HandshakeReply {
        result: Some(result),
    }
    .encode_to_vec()
}

/// Decodes the host's answer to a handshake: `Ok(msg 2)`, or its refusal
/// (`Err(Ok(refusal))`, an unauthenticated hint).
pub fn decode_handshake_reply(buf: &[u8]) -> Result<Result<Vec<u8>, CtlError>, WireError> {
    use pb::handshake_reply::Result as Answer;
    match pb::HandshakeReply::decode(buf)?.result {
        Some(Answer::Noise(noise)) => Ok(Ok(noise.to_vec())),
        Some(Answer::Error(err)) => Ok(Err(status_from_pb(err.status, err.message)
            .err()
            .unwrap_or_else(|| CtlError::new(CtlStatus::InvalidRequest, "a refusal with OK")))),
        None => Err(WireError::malformed("handshake reply", "no result")),
    }
}

/// Encodes msg 2's payload.
pub fn encode_handshake_accept(accept: &HandshakeAccept) -> Vec<u8> {
    pb::HandshakeAccept {
        generation: accept.generation,
        device: accept.device.clone(),
        // Unspecified: observer only.
        max_priority: accept.max_priority.map_or(0, |p| priority_to_pb(p) as i32),
        host_sign_key: Bytes::copy_from_slice(&accept.host_sign_key),
        host_instance: accept.host_instance,
        up_key_id: accept.up_key_id,
        down_key_id: accept.down_key_id,
        datagram: accept.datagram.map(|d| pb::DatagramOffer {
            port: u32::from(d.port),
            version: u32::from(d.version),
        }),
    }
    .encode_to_vec()
}

/// Decodes msg 2's payload.
pub fn decode_handshake_accept(buf: &[u8]) -> Result<HandshakeAccept, WireError> {
    let accept = pb::HandshakeAccept::decode(buf)?;
    Ok(HandshakeAccept {
        generation: accept.generation,
        device: accept.device,
        max_priority: match accept.max_priority {
            0 => None,
            raw => Some(priority_from_pb(raw)?),
        },
        host_sign_key: accept.host_sign_key[..].try_into().map_err(|_| {
            WireError::malformed("handshake accept", "host_sign_key is not 32 bytes")
        })?,
        host_instance: accept.host_instance,
        up_key_id: accept.up_key_id,
        down_key_id: accept.down_key_id,
        // An offer this side can't read is no offer: the session carries
        // everything, as from a host that offers none.
        datagram: accept.datagram.and_then(|d| {
            Some(DatagramOffer {
                port: u16::try_from(d.port).ok().filter(|p| *p != 0)?,
                version: u8::try_from(d.version).ok()?,
            })
        }),
    })
}
