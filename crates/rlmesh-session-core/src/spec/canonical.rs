//! Canonical JSON (RFC 8785, the JSON Canonicalization Scheme) and the
//! fingerprints taken over it (§3.3, §6.5).
//!
//! A fingerprint is the first 8 bytes of BLAKE3 over the canonical bytes.
//! Only Rust computes fingerprints, so every language agrees on them.
//!
//! RFC 8785 sorts object members by their UTF-16 code units and writes
//! numbers the way ECMAScript's `Number.prototype.toString` does. Order that
//! matters (a part's fields, which are its payload order) is therefore never
//! carried in an object: the normalised forms use arrays for it.

#[allow(unused_imports)]
use crate::prelude::*;

use crate::wire::Fingerprint;
use serde_json::Value;

/// `value` as RFC 8785 canonical JSON.
pub fn to_string(value: &Value) -> String {
    let mut out = String::new();
    write_value(value, &mut out);
    out
}

/// The first 8 bytes of BLAKE3 over `value`'s canonical JSON.
pub fn fingerprint(value: &Value) -> Fingerprint {
    let hash = blake3::hash(to_string(value).as_bytes());
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&hash.as_bytes()[..8]);
    Fingerprint(bytes)
}

fn write_value(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            // JSON numbers are IEEE doubles to RFC 8785.
            out.push_str(&number(n.as_f64().unwrap_or(0.0)));
        }
        Value::String(s) => write_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut members: Vec<(&String, &Value)> = map.iter().collect();
            members.sort_by(|a, b| a.0.encode_utf16().cmp(b.0.encode_utf16()));
            out.push('{');
            for (i, (key, item)) in members.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(key, out);
                out.push(':');
                write_value(item, out);
            }
            out.push('}');
        }
    }
}

fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\u{c}' => out.push_str("\\f"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A finite double the way ECMAScript's `Number.prototype.toString` writes
/// it: the shortest digits that round-trip, in fixed notation for decimal
/// exponents in (-7, 21], exponential otherwise. Non-finite values cannot
/// occur in JSON; they write as `null`.
pub fn number(x: f64) -> String {
    if !x.is_finite() {
        return "null".into();
    }
    if x == 0.0 {
        return "0".into();
    }
    if x < 0.0 {
        return format!("-{}", number(-x));
    }
    // Rust's `{:e}` writes the shortest round-trip digits: `d.ddde±x`.
    let sci = format!("{x:e}");
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((&sci, "0"));
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let k = digits.len() as i32;
    // x = 0.digits × 10^n.
    let n = exponent + 1;
    if k <= n && n <= 21 {
        format!("{digits}{}", "0".repeat((n - k) as usize))
    } else if 0 < n && n <= 21 {
        let (int, frac) = digits.split_at(n as usize);
        format!("{int}.{frac}")
    } else if -6 < n && n <= 0 {
        format!("0.{}{digits}", "0".repeat((-n) as usize))
    } else {
        let sign = if n - 1 < 0 { '-' } else { '+' };
        let (first, rest) = digits.split_at(1);
        if rest.is_empty() {
            format!("{first}e{sign}{}", (n - 1).abs())
        } else {
            format!("{first}.{rest}e{sign}{}", (n - 1).abs())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn numbers_follow_ecmascript() {
        for (x, want) in [
            (0.0, "0"),
            (-0.0, "0"),
            (1.0, "1"),
            (500.0, "500"),
            (0.085, "0.085"),
            (-6.25, "-6.25"),
            (1e21, "1e+21"),
            (1e20, "100000000000000000000"),
            (1e-7, "1e-7"),
            (1.5e-7, "1.5e-7"),
            (0.000_001, "0.000001"),
            (123_456.789, "123456.789"),
            (4.5, "4.5"),
            (2e-308, "2e-308"),
            (1.797_693_134_862_315_7e308, "1.7976931348623157e+308"),
            (0.1 + 0.2, "0.30000000000000004"),
        ] {
            assert_eq!(number(x), want, "{x:e}");
        }
    }

    #[test]
    fn members_sort_by_utf16_and_strings_escape_minimally() {
        let value = json!({"b": [3, {"z": null, "a": true}], "a": "é\n\"\u{1}", "\u{fb33}": 1, "\u{1d11e}": 2});
        // U+1D11E is a surrogate pair (D834 DD1E): by UTF-16 code units it
        // sorts before U+FB33, though its code point is higher.
        assert_eq!(
            to_string(&value),
            "{\"a\":\"é\\n\\\"\\u0001\",\"b\":[3,{\"a\":true,\"z\":null}],\"\u{1d11e}\":2,\"\u{fb33}\":1}"
        );
    }

    #[test]
    fn fingerprints_ignore_member_order_but_not_array_order() {
        let a = fingerprint(&json!({"x": 1, "y": [1, 2]}));
        assert_eq!(a, fingerprint(&json!({"y": [1, 2], "x": 1.0})));
        assert_ne!(a, fingerprint(&json!({"x": 1, "y": [2, 1]})));
    }
}
