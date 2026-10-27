//! Length-prefixed bytes and the keyed hash (§8.1).

use blake2::digest::{KeyInit, Mac};

use crate::prelude::*;

/// The longest input `lp` takes: its length must fit a `u16`.
pub const LP_MAX: usize = u16::MAX as usize;

/// An input to `lp` was over 65,535 bytes. `lp` never truncates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("a length-prefixed field is over 65535 bytes")]
pub struct LpTooLong;

/// The two-byte little-endian length prefix of `x`.
///
/// # Errors
/// [`LpTooLong`] if `x` is over 65,535 bytes.
pub fn lp_prefix(x: &[u8]) -> Result<[u8; 2], LpTooLong> {
    u16::try_from(x.len())
        .map(u16::to_le_bytes)
        .map_err(|_| LpTooLong)
}

/// `lp(x)` = `u16` little-endian length ‖ `x`.
///
/// # Errors
/// [`LpTooLong`] if `x` is over 65,535 bytes.
pub fn lp(x: &[u8]) -> Result<Vec<u8>, LpTooLong> {
    let mut out = Vec::with_capacity(2 + x.len());
    push_lp(&mut out, x)?;
    Ok(out)
}

/// Append `lp(x)` to `out`. `out` is unchanged on error.
///
/// # Errors
/// [`LpTooLong`] if `x` is over 65,535 bytes.
pub fn push_lp(out: &mut Vec<u8>, x: &[u8]) -> Result<(), LpTooLong> {
    out.extend_from_slice(&lp_prefix(x)?);
    out.extend_from_slice(x);
    Ok(())
}

/// `kdf(key, label, fields…)` = BLAKE2s-256 keyed with `key`, over
/// `lp(label) ‖ fields`.
///
/// # Errors
/// [`LpTooLong`] if `label` is over 65,535 bytes.
pub fn kdf(key: &[u8; 32], label: &[u8], fields: &[&[u8]]) -> Result<[u8; 32], LpTooLong> {
    let prefix = lp_prefix(label)?;
    let mut mac = <blake2::Blake2sMac256 as KeyInit>::new(key.into());
    Mac::update(&mut mac, &prefix);
    Mac::update(&mut mac, label);
    for field in fields {
        Mac::update(&mut mac, field);
    }
    Ok(mac.finalize().into_bytes().into())
}

/// BLAKE2s-256 (unkeyed) over `lp(label) ‖ fields`: the pairing commitment
/// and SAS (§3).
pub(crate) fn hash_labelled(label: &[u8], fields: &[&[u8]]) -> Result<[u8; 32], LpTooLong> {
    use blake2::Digest;
    let prefix = lp_prefix(label)?;
    let mut hash = blake2::Blake2s256::new();
    hash.update(prefix);
    hash.update(label);
    for field in fields {
        hash.update(field);
    }
    Ok(hash.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lp_prefixes_the_little_endian_length() {
        assert_eq!(lp(b"abc").unwrap(), [3, 0, b'a', b'b', b'c']);
        assert_eq!(lp(b"").unwrap(), [0, 0]);
        let long = vec![7u8; 0x0102];
        assert_eq!(lp(&long).unwrap()[..2], [0x02, 0x01]);
    }

    #[test]
    fn lp_takes_65535_bytes_and_rejects_more() {
        let max = vec![0u8; LP_MAX];
        assert_eq!(lp(&max).unwrap()[..2], [0xff, 0xff]);
        let over = vec![0u8; LP_MAX + 1];
        assert_eq!(lp(&over), Err(LpTooLong));
        let mut out = vec![1];
        assert_eq!(push_lp(&mut out, &over), Err(LpTooLong));
        assert_eq!(out, [1], "never truncates or writes a partial field");
        assert_eq!(kdf(&[0; 32], &over, &[]), Err(LpTooLong));
    }

    #[test]
    fn kdf_is_keyed_blake2s_over_the_labelled_fields() {
        let key = [9u8; 32];
        let a = kdf(&key, b"label", &[b"x", b"y"]).unwrap();
        // Fields are concatenated: the split doesn't matter, the key does.
        assert_eq!(a, kdf(&key, b"label", &[b"xy"]).unwrap());
        assert_ne!(a, kdf(&[8u8; 32], b"label", &[b"xy"]).unwrap());
        assert_ne!(a, kdf(&key, b"labelx", &[b"y"]).unwrap());
    }
}
