//! How numbers read in binding tables and refusals: three significant
//! digits, never exponential, trailing zeros trimmed (`66.7`, `1000`,
//! `0.25`).

use std::time::Duration;

/// `x` to three significant digits (integers above 100 kept whole).
pub fn num(x: f64) -> String {
    if !x.is_finite() {
        return x.to_string();
    }
    let a = x.abs();
    let decimals = if a >= 100.0 || a == 0.0 {
        0
    } else if a >= 10.0 {
        1
    } else if a >= 1.0 {
        2
    } else {
        3
    };
    let text = format!("{x:.decimals$}");
    let text = if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    };
    if text == "-0" { "0".into() } else { text }
}

/// A rate in Hz, as a number.
pub fn hz(x: f64) -> String {
    num(x)
}

/// A duration in milliseconds, as a number.
pub fn ms(d: Duration) -> String {
    num(d.as_secs_f64() * 1e3)
}

/// A duration in seconds, as a number.
pub fn seconds(d: Duration) -> String {
    num(d.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_read_at_three_significant_digits() {
        for (x, want) in [
            (66.666_666, "66.7"),
            (1000.0, "1000"),
            (15.0, "15"),
            (0.25, "0.25"),
            (2.0 / 3.0, "0.667"),
            (12.5, "12.5"),
            (0.0, "0"),
            (-0.000_1, "0"),
            (-1.5, "-1.5"),
        ] {
            assert_eq!(num(x), want, "{x}");
        }
        assert_eq!(ms(Duration::from_nanos(66_666_667)), "66.7");
        assert_eq!(ms(Duration::from_millis(667)), "667");
    }
}
