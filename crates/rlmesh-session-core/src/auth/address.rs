//! The key an embedded host's address carries (`…/{session}#{secret}.{host}`,
//! design §12 decision 2): what a script, a sandbox or a board parses out of
//! the address it was given, to attach as device `local`.

use core::fmt;
use core::str::FromStr;

use zeroize::Zeroizing;

use super::{Credential, HostIdentity};
use crate::prelude::*;

/// The key an embedded host's address carries (`#{secret}.{host}`): its
/// local secret, from which a holder derives the `local` device's key, and
/// its static public key. A secret: never logged.
#[derive(Clone, PartialEq, Eq)]
pub struct AddressKey {
    secret: Zeroizing<[u8; 32]>,
    host: [u8; 32],
}

impl fmt::Debug for AddressKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AddressKey(..)")
    }
}

/// RFC 4648 base32, lowercase, unpadded: 52 characters for 32 bytes.
const BASE32: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";

/// `bytes` as unpadded lowercase base32 (52 characters).
pub fn base32(bytes: &[u8; 32]) -> String {
    let mut out = String::with_capacity(52);
    let (mut acc, mut bits) = (0_u32, 0_u32);
    for &b in bytes {
        acc = (acc << 8) | u32::from(b);
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            out.push(char::from(BASE32[((acc >> bits) & 31) as usize]));
        }
    }
    if bits > 0 {
        out.push(char::from(BASE32[((acc << (5 - bits)) & 31) as usize]));
    }
    out
}

/// The 32 bytes of unpadded base32 `text` (either case), if it is exactly
/// that.
pub fn unbase32(text: &str) -> Option<[u8; 32]> {
    if text.len() != 52 {
        return None;
    }
    let mut out = [0_u8; 32];
    let (mut acc, mut bits, mut at) = (0_u32, 0_u32, 0);
    for c in text.bytes() {
        let value = BASE32.iter().position(|&b| b == c.to_ascii_lowercase())?;
        acc = (acc << 5) | u32::try_from(value).ok()?;
        bits += 5;
        if bits >= 8 {
            bits -= 8;
            *out.get_mut(at)? = (acc >> bits) as u8;
            at += 1;
        }
    }
    // The 4 padding bits of the last character are zero.
    (at == 32 && acc & ((1 << bits) - 1) == 0).then_some(out)
}

/// An address key that isn't `{secret}.{host key}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("not {{secret}}.{{host key}}")]
pub struct AddressKeyError;

impl AddressKey {
    /// The key of `identity`'s address.
    pub fn of(identity: &HostIdentity) -> Self {
        Self {
            secret: Zeroizing::new(*identity.local_secret()),
            host: *identity.static_public(),
        }
    }

    /// The credential of its holder (device `local`).
    pub fn credential(&self) -> Credential {
        Credential::local(&self.secret, self.host)
    }
}

impl FromStr for AddressKey {
    type Err = AddressKeyError;

    /// `{secret}.{host key}`, as an address carries it after `#`.
    fn from_str(text: &str) -> Result<Self, AddressKeyError> {
        let (secret, host) = text.trim().split_once('.').ok_or(AddressKeyError)?;
        Ok(Self {
            secret: Zeroizing::new(unbase32(secret).ok_or(AddressKeyError)?),
            host: unbase32(host).ok_or(AddressKeyError)?,
        })
    }
}

impl fmt::Display for AddressKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", base32(&self.secret), base32(&self.host))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::test_rng::TestRng;

    #[test]
    fn an_address_key_round_trips_and_derives_the_local_device() {
        let identity = HostIdentity::generate(&mut TestRng::new(4)).expect("identity");
        let key = AddressKey::of(&identity);
        assert_eq!(key.to_string().parse::<AddressKey>(), Ok(key.clone()));
        assert_eq!(
            key.credential().device_public(),
            &identity.local_public(),
            "a holder derives the local device's key"
        );
        assert!(format!("{key:?}").len() < 20, "never logged");
        assert_eq!("nope".parse::<AddressKey>(), Err(AddressKeyError));
    }
}
