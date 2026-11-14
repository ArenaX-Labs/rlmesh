//! Pairing messages.

use super::convert::bytes32;
use super::ctl::{error_from_pb, error_to_pb};
use super::pb;
use crate::prelude::*;

use bytes::Bytes;
use prost::Message as _;

use crate::wire::authority::CtlError;
use crate::wire::error::WireError;
use crate::wire::pair::{PairInfo, PairNonce, PairReply, PairRequest, PairStep};

/// Encodes a pairing packet.
pub fn encode_pair_request(request: &PairRequest) -> Vec<u8> {
    use pb::pair_request::Step;
    let step = match &request.step {
        PairStep::Xx1(b) => Step::Xx1(Bytes::copy_from_slice(b)),
        PairStep::Xx3(b) => Step::Xx3(Bytes::copy_from_slice(b)),
        PairStep::Sealed(b) => Step::Sealed(Bytes::copy_from_slice(b)),
    };
    pb::PairRequest {
        exchange: Bytes::copy_from_slice(&request.exchange),
        step: Some(step),
    }
    .encode_to_vec()
}

/// Decodes a pairing packet.
pub fn decode_pair_request(buf: &[u8]) -> Result<PairRequest, WireError> {
    use pb::pair_request::Step;
    let request = pb::PairRequest::decode(buf)?;
    let exchange = request.exchange[..]
        .try_into()
        .map_err(|_| WireError::malformed("pair request", "exchange is not 16 bytes"))?;
    let step = match request.step {
        Some(Step::Xx1(b)) => PairStep::Xx1(b.to_vec()),
        Some(Step::Xx3(b)) => PairStep::Xx3(b.to_vec()),
        Some(Step::Sealed(b)) => PairStep::Sealed(b.to_vec()),
        None => return Err(WireError::malformed("pair request", "no step")),
    };
    Ok(PairRequest { exchange, step })
}

/// Encodes rlmeshd's answer to a pairing packet.
pub fn encode_pair_reply(reply: &PairReply) -> Vec<u8> {
    use pb::pair_reply::Step;
    let step = match reply {
        PairReply::Xx2(b) => Step::Xx2(Bytes::copy_from_slice(b)),
        PairReply::Sealed(b) => Step::Sealed(Bytes::copy_from_slice(b)),
        PairReply::Error(err) => Step::Error(error_to_pb(err)),
    };
    pb::PairReply { step: Some(step) }.encode_to_vec()
}

/// Decodes rlmeshd's answer to a pairing packet.
pub fn decode_pair_reply(buf: &[u8]) -> Result<PairReply, WireError> {
    use pb::pair_reply::Step;
    match pb::PairReply::decode(buf)?.step {
        Some(Step::Xx2(b)) => Ok(PairReply::Xx2(b.to_vec())),
        Some(Step::Sealed(b)) => Ok(PairReply::Sealed(b.to_vec())),
        Some(Step::Error(err)) => Ok(PairReply::Error(error_from_pb(err))),
        None => Err(WireError::malformed("pair reply", "no step")),
    }
}

/// Encodes `PairCommit { c }`.
pub fn encode_pair_commit(c: &[u8; 32]) -> Vec<u8> {
    pb::PairCommit {
        c: Bytes::copy_from_slice(c),
    }
    .encode_to_vec()
}

/// Decodes `PairCommit`: `C`.
pub fn decode_pair_commit(buf: &[u8]) -> Result<[u8; 32], WireError> {
    bytes32("pair commit", "c", &pb::PairCommit::decode(buf)?.c)
}

/// Encodes `PairNonce`.
pub fn encode_pair_nonce(nonce: &PairNonce) -> Vec<u8> {
    pb::PairNonce {
        ni: Bytes::copy_from_slice(&nonce.ni),
        name: nonce.name.clone(),
        kind: nonce.kind.clone(),
    }
    .encode_to_vec()
}

/// Decodes `PairNonce`.
pub fn decode_pair_nonce(buf: &[u8]) -> Result<PairNonce, WireError> {
    let nonce = pb::PairNonce::decode(buf)?;
    Ok(PairNonce {
        ni: bytes32("pair nonce", "ni", &nonce.ni)?,
        name: nonce.name,
        kind: nonce.kind,
    })
}

/// Encodes `PairReveal { nr }`.
pub fn encode_pair_reveal(nr: &[u8; 32]) -> Vec<u8> {
    pb::PairReveal {
        nr: Bytes::copy_from_slice(nr),
    }
    .encode_to_vec()
}

/// Decodes `PairReveal`: `Nr`.
pub fn decode_pair_reveal(buf: &[u8]) -> Result<[u8; 32], WireError> {
    bytes32("pair reveal", "nr", &pb::PairReveal::decode(buf)?.nr)
}

/// Encodes `PairWait {}`.
pub fn encode_pair_wait() -> Vec<u8> {
    pb::PairWait {}.encode_to_vec()
}

/// Encodes `PairResult`: the approved machine's details, or the refusal.
pub fn encode_pair_result(result: &Result<PairInfo, CtlError>) -> Vec<u8> {
    use pb::pair_result::R;
    let r = match result {
        Ok(info) => R::Ok(pb::PairInfo {
            device: info.device.clone(),
            host_key: Bytes::copy_from_slice(&info.host_key),
            host_id: info.host_id.clone(),
            addresses: info.addresses.clone(),
        }),
        Err(err) => R::Error(error_to_pb(err)),
    };
    pb::PairResult { r: Some(r) }.encode_to_vec()
}

/// Decodes `PairResult`.
pub fn decode_pair_result(buf: &[u8]) -> Result<Result<PairInfo, CtlError>, WireError> {
    use pb::pair_result::R;
    match pb::PairResult::decode(buf)?.r {
        Some(R::Ok(info)) => Ok(Ok(PairInfo {
            device: info.device,
            host_key: bytes32("pair result", "host_key", &info.host_key)?,
            host_id: info.host_id,
            addresses: info.addresses,
        })),
        Some(R::Error(err)) => Ok(Err(error_from_pb(err))),
        None => Err(WireError::malformed("pair result", "no result")),
    }
}
