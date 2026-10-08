//! Byte quantities: `16Gi`, `24G`, `1.5Gi`, `80000000000`.

/// The quantity suffixes, largest first: binary ones are powers of 1024,
/// decimal ones powers of 1000.
const UNITS: [(&str, u64); 8] = [
    ("Ti", 1 << 40),
    ("T", 1_000_000_000_000),
    ("Gi", 1 << 30),
    ("G", 1_000_000_000),
    ("Mi", 1 << 20),
    ("M", 1_000_000),
    ("Ki", 1 << 10),
    ("K", 1_000),
];

/// Why a string is not a byte quantity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantityError {
    /// It does not match `^[0-9]+(\.[0-9]+)?(Ki|Mi|Gi|Ti|K|M|G|T)?$`.
    Syntax,
    /// It is zero.
    Zero,
    /// It is a fraction of a byte.
    Fractional,
    /// It is more than a signed 64-bit integer holds.
    TooLarge,
}

/// Split a quantity into its whole digits, its fraction digits, and its
/// unit's multiplier (1 without a suffix).
fn split_quantity(raw: &str) -> Option<(&str, &str, u64)> {
    let (number, multiplier) = UNITS
        .iter()
        .find_map(|(suffix, multiplier)| Some((raw.strip_suffix(suffix)?, *multiplier)))
        .unwrap_or((raw, 1));
    let (whole, fraction) = number.split_once('.').unwrap_or((number, ""));
    let digits = |part: &str| part.bytes().all(|b| b.is_ascii_digit());
    let fraction_ok = !number.contains('.') || !fraction.is_empty();
    (!whole.is_empty() && digits(whole) && digits(fraction) && fraction_ok)
        .then_some((whole, fraction, multiplier))
}

/// The exact byte count of `whole.fraction × multiplier`, in the platform's
/// signed 64-bit range.
fn quantity_bytes(whole: &str, fraction: &str, multiplier: u64) -> Result<u64, QuantityError> {
    let fraction = fraction.trim_end_matches('0');
    let scale = u32::try_from(fraction.len())
        .ok()
        .and_then(|len| 10_u128.checked_pow(len))
        .ok_or(QuantityError::TooLarge)?;
    let digits = format!("{whole}{fraction}");
    let digits = digits.trim_start_matches('0');
    let numerator = if digits.is_empty() {
        0
    } else {
        digits
            .parse::<u128>()
            .ok()
            .and_then(|n| n.checked_mul(u128::from(multiplier)))
            .ok_or(QuantityError::TooLarge)?
    };
    if numerator == 0 {
        return Err(QuantityError::Zero);
    }
    if !numerator.is_multiple_of(scale) {
        return Err(QuantityError::Fractional);
    }
    u64::try_from(numerator / scale)
        .ok()
        .filter(|bytes| i64::try_from(*bytes).is_ok())
        .ok_or(QuantityError::TooLarge)
}

/// Parse a byte quantity: digits with an optional fraction and an optional
/// `Ki`, `Mi`, `Gi`, `Ti` (powers of 1024) or `K`, `M`, `G`, `T` (powers of
/// 1000) suffix, no suffix meaning bytes. It must be a positive whole number
/// of bytes within the signed 64-bit range.
pub fn parse_quantity(raw: &str) -> Result<u64, QuantityError> {
    let (whole, fraction, multiplier) = split_quantity(raw).ok_or(QuantityError::Syntax)?;
    quantity_bytes(whole, fraction, multiplier)
}

/// Render a byte count with the first unit, largest first (`Ti`, `T`, `Gi`,
/// `G`, `Mi`, `M`, `Ki`, `K`), that divides it exactly, else as plain bytes:
/// 17179869184 is `16Gi`, 16000000000 is `16G`.
pub fn format_quantity(bytes: u64) -> String {
    UNITS
        .iter()
        .find(|(_, multiplier)| bytes > 0 && bytes.is_multiple_of(*multiplier))
        .map_or_else(
            || bytes.to_string(),
            |(suffix, multiplier)| format!("{}{suffix}", bytes / multiplier),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantity_grammar_table() {
        for (raw, bytes) in [
            ("16Gi", 16 << 30),
            ("24G", 24_000_000_000),
            ("1.5Gi", 3 << 29),
            ("512Mi", 512 << 20),
            ("80000000000", 80_000_000_000),
            ("1K", 1_000),
            ("2Ti", 2 << 40),
            ("0.5Ki", 512),
            ("016Gi", 16 << 30),
            ("1.50Gi", 3 << 29),
            ("9223372036854775807", i64::MAX as u64),
        ] {
            assert_eq!(parse_quantity(raw), Ok(bytes), "{raw}");
        }
        for (raw, error) in [
            ("16 Gi", QuantityError::Syntax),
            ("16gi", QuantityError::Syntax),
            ("16GB", QuantityError::Syntax),
            ("-1Gi", QuantityError::Syntax),
            ("1e9", QuantityError::Syntax),
            (">=16Gi", QuantityError::Syntax),
            ("", QuantityError::Syntax),
            ("Gi", QuantityError::Syntax),
            ("1.Gi", QuantityError::Syntax),
            (".5Gi", QuantityError::Syntax),
            ("100m", QuantityError::Syntax),
            (" 16Gi", QuantityError::Syntax),
            ("0", QuantityError::Zero),
            ("0.0Gi", QuantityError::Zero),
            ("1.5", QuantityError::Fractional),
            ("0.0001Ki", QuantityError::Fractional),
            ("9999999999Ti", QuantityError::TooLarge),
            ("9223372036854775808", QuantityError::TooLarge),
            (
                "99999999999999999999999999999999999999999",
                QuantityError::TooLarge,
            ),
        ] {
            assert_eq!(parse_quantity(raw), Err(error), "{raw}");
        }
    }

    #[test]
    fn quantities_render_with_the_largest_exact_unit() {
        for (bytes, rendered) in [
            (16_000_000_000, "16G"),
            (17_179_869_184, "16Gi"),
            (24_000_000_000, "24G"),
            (123, "123"),
            (1_000, "1K"),
            (1_024, "1Ki"),
            (1 << 40, "1Ti"),
            (2_000_000_000_000, "2T"),
            (3 << 29, "1536Mi"),
            (i64::MAX as u64, "9223372036854775807"),
        ] {
            assert_eq!(format_quantity(bytes), rendered, "{bytes}");
            assert_eq!(parse_quantity(rendered), Ok(bytes), "{rendered}");
        }
    }
}
