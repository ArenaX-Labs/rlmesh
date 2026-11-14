//! CRC-16/CCITT-FALSE for frames: polynomial 0x1021, initial value
//! 0xFFFF, no reflection, no final XOR. Bitwise, so an AVR board carries
//! no table. Also CRC-32/ISO-HDLC for the map.

/// The CRC's initial value.
pub const INIT: u16 = 0xFFFF;

/// Folds `bytes` into a running CRC (start from [`INIT`]).
pub fn update(mut crc: u16, bytes: &[u8]) -> u16 {
    for &byte in bytes {
        crc ^= u16::from(byte) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 == 0 {
                crc << 1
            } else {
                (crc << 1) ^ 0x1021
            };
        }
    }
    crc
}

/// The CRC of `bytes`.
pub fn crc16(bytes: &[u8]) -> u16 {
    update(INIT, bytes)
}

/// CRC-32/ISO-HDLC (the zlib CRC): reflected polynomial 0xEDB88320,
/// initial value and final XOR 0xFFFFFFFF. `HELLO.map_crc` uses it.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 0 {
                crc >> 1
            } else {
                (crc >> 1) ^ 0xEDB8_8320
            };
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_catalogue_check_value() {
        // The CRC catalogue's check: "123456789" → 0x29B1.
        assert_eq!(crc16(b"123456789"), 0x29B1);
        assert_eq!(crc16(b""), 0xFFFF);
    }

    #[test]
    fn crc32_matches_the_catalogue_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn running_updates_equal_one_pass() {
        let whole = crc16(b"hello, tether");
        let split = update(update(INIT, b"hello, "), b"tether");
        assert_eq!(whole, split);
    }
}
