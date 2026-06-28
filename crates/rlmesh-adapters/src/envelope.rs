//! The describe-envelope format: the one Rust-owned, versioned JSON artifact
//! that fully describes an `EnvFactory` or `Model`.
//!
//! Producers (Python today; a future C++/TS SDK) *gather* the language-specific
//! pieces -- signature reflection, the author's variant enumeration, the env's
//! obs/action spaces, the local runtime versions -- and hand them here as a
//! single JSON object. [`build_describe_envelope`] stamps the schema version,
//! validates the wrapper, and re-serializes the whole tree through one serde_json
//! pass. Because the *final* serialization happens here -- and `serde_json::Value`
//! serializes object keys in sorted (`BTreeMap`) order -- the bytes are identical
//! across producer languages given identical logical input. Sub-pieces are
//! carried as opaque [`serde_json::Value`]: each was already validated by its own
//! codec (env_tags/model_spec via [`adapters_spec_normalize`]) or by the
//! producer (params/variants), so the envelope only owns the wrapper + ordering +
//! serialization, never the sub-piece schemas.
//!
//! [`adapters_spec_normalize`]: (the PyO3 spec gate)

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::keys::DESCRIBE_SCHEMA_VERSION;

/// Total-byte ceiling on the gathered pieces blob. Mirrors the spec codec's
/// `MAX_SPEC_BYTES`: the pieces can originate from a possibly-untrusted producer
/// and are retained + re-emitted, so an unbounded blob is a DoS amplifier. 4 MiB
/// is absurdly generous for any real envelope.
const MAX_PIECES_BYTES: usize = 4 << 20;

/// What an envelope describes. A closed enum: an unknown kind is rejected at the
/// boundary, and the canonical wire form is `"env"` / `"model"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Env,
    Model,
}

impl Kind {
    fn parse(raw: &str) -> Result<Self, EnvelopeError> {
        match raw {
            "env" => Ok(Self::Env),
            "model" => Ok(Self::Model),
            other => Err(EnvelopeError::UnknownKind(other.to_owned())),
        }
    }
}

/// The producer-gathered sub-pieces. `deny_unknown_fields` makes the envelope's
/// key set part of the contract: a producer that invents a top-level field fails
/// here rather than silently shipping a non-standard envelope. Every field is
/// optional so the env/model shapes share one struct; the env-vs-model invariant
/// is enforced in [`Pieces::check_kind`].
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Pieces {
    #[serde(default)]
    target: Option<Value>,
    #[serde(default)]
    env_spec: Option<Value>,
    #[serde(default)]
    env_tags: Option<Value>,
    #[serde(default)]
    model_spec: Option<Value>,
    #[serde(default)]
    params: Option<Value>,
    #[serde(default)]
    variants: Option<Value>,
    #[serde(default)]
    runtime: Option<Value>,
}

impl Pieces {
    /// Enforce that env-only and model-only pieces don't cross kinds, so the
    /// emitted envelope can never carry a model_spec under `kind:"env"` (or env
    /// spaces/tags under `kind:"model"`).
    fn check_kind(&self, kind: Kind) -> Result<(), EnvelopeError> {
        let offender = match kind {
            Kind::Env if self.model_spec.is_some() => Some("model_spec"),
            Kind::Model if self.env_spec.is_some() => Some("env_spec"),
            Kind::Model if self.env_tags.is_some() => Some("env_tags"),
            _ => None,
        };
        match offender {
            Some(field) => Err(EnvelopeError::KindMismatch { kind, field }),
            None => Ok(()),
        }
    }
}

/// The serialized env form. The kind's own fields (`env_spec`, `env_tags`) are
/// always present -- `env_tags` may be `null`, but a `model_spec` never appears on
/// an env envelope. Field order here is the top-level byte order (wrapper first);
/// nested object keys sort via `BTreeMap`.
#[derive(Debug, Serialize)]
struct EnvEnvelope {
    schema_version: u32,
    kind: Kind,
    #[serde(skip_serializing_if = "Option::is_none")]
    generated_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target: Option<Value>,
    env_spec: Value,
    env_tags: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    variants: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime: Option<Value>,
}

/// The serialized model form. `model_spec` is always present (may be `null`); env
/// spaces/tags never appear on a model envelope.
#[derive(Debug, Serialize)]
struct ModelEnvelope {
    schema_version: u32,
    kind: Kind,
    #[serde(skip_serializing_if = "Option::is_none")]
    generated_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target: Option<Value>,
    model_spec: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    variants: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime: Option<Value>,
}

/// Errors a producer can hit building an envelope. All map to a Python
/// `ValueError` at the PyO3 boundary.
#[derive(Debug, thiserror::Error)]
pub enum EnvelopeError {
    #[error("pieces blob is {0} bytes, over the {MAX_PIECES_BYTES}-byte limit")]
    TooLarge(usize),
    #[error("unknown kind {0:?}; expected \"env\" or \"model\"")]
    UnknownKind(String),
    #[error("{field} is not allowed on a {kind:?} envelope")]
    KindMismatch { kind: Kind, field: &'static str },
    #[error("generated_at must be an RFC-3339 timestamp, got {0:?}")]
    BadTimestamp(String),
    #[error("could not parse pieces: {0}")]
    Parse(serde_json::Error),
    #[error("could not serialize envelope: {0}")]
    Serialize(serde_json::Error),
}

/// Build the canonical describe-envelope JSON string.
///
/// `kind` is `"env"` or `"model"`. `pieces_json` is the producer-gathered
/// sub-pieces as one JSON object. `generated_at`, if given, must be RFC-3339 (the
/// caller supplies it -- a build pipeline pins a reproducible timestamp; omit for
/// content-addressable artifacts). The schema version is stamped here, never by
/// the caller. This is the function a future native producer calls directly; the
/// PyO3 entry is a thin shim over it.
pub fn build_describe_envelope(
    kind: &str,
    pieces_json: &str,
    generated_at: Option<&str>,
) -> Result<String, EnvelopeError> {
    if pieces_json.len() > MAX_PIECES_BYTES {
        return Err(EnvelopeError::TooLarge(pieces_json.len()));
    }
    let kind = Kind::parse(kind)?;
    if let Some(ts) = generated_at
        && !is_rfc3339(ts)
    {
        return Err(EnvelopeError::BadTimestamp(ts.to_owned()));
    }
    let pieces: Pieces = serde_json::from_str(pieces_json).map_err(EnvelopeError::Parse)?;
    pieces.check_kind(kind)?;

    let generated_at = generated_at.map(str::to_owned);
    match kind {
        Kind::Env => serde_json::to_string(&EnvEnvelope {
            schema_version: DESCRIBE_SCHEMA_VERSION,
            kind,
            generated_at,
            target: pieces.target,
            env_spec: pieces.env_spec.unwrap_or(Value::Null),
            env_tags: pieces.env_tags.unwrap_or(Value::Null),
            params: pieces.params,
            variants: pieces.variants,
            runtime: pieces.runtime,
        }),
        Kind::Model => serde_json::to_string(&ModelEnvelope {
            schema_version: DESCRIBE_SCHEMA_VERSION,
            kind,
            generated_at,
            target: pieces.target,
            model_spec: pieces.model_spec.unwrap_or(Value::Null),
            params: pieces.params,
            variants: pieces.variants,
            runtime: pieces.runtime,
        }),
    }
    .map_err(EnvelopeError::Serialize)
}

/// Lightweight RFC-3339 shape check: `YYYY-MM-DDThh:mm:ss` then a zone
/// (`Z`/`+hh:mm`/`-hh:mm`) with an optional fractional second.
///
// ponytail: structural check, not a calendar validator (no date dep). It rejects
// obvious garbage and pins the cross-language format; if true calendar validation
// is ever needed, swap in `time`/`jiff` here -- the one call site.
fn is_rfc3339(s: &str) -> bool {
    let b = s.as_bytes();
    // Minimum "1970-01-01T00:00:00Z" is 20 bytes.
    if b.len() < 20 || b.len() > 64 {
        return false;
    }
    let digit = |i: usize| b[i].is_ascii_digit();
    let at = |i: usize, c: u8| b[i] == c;
    // date: YYYY-MM-DD
    if !(digit(0)
        && digit(1)
        && digit(2)
        && digit(3)
        && at(4, b'-')
        && digit(5)
        && digit(6)
        && at(7, b'-')
        && digit(8)
        && digit(9))
    {
        return false;
    }
    // 'T' (RFC-3339 allows a space, but we pin 'T' for one canonical form).
    if !at(10, b'T') {
        return false;
    }
    // time: hh:mm:ss
    if !(digit(11)
        && digit(12)
        && at(13, b':')
        && digit(14)
        && digit(15)
        && at(16, b':')
        && digit(17)
        && digit(18))
    {
        return false;
    }
    // optional .fraction, then a timezone.
    let mut i = 19;
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == start {
            return false; // a dot with no digits
        }
    }
    match b.get(i) {
        Some(b'Z') => i + 1 == b.len(),
        Some(b'+') | Some(b'-') => {
            // +hh:mm
            b.len() - i == 6
                && digit(i + 1)
                && digit(i + 2)
                && at(i + 3, b':')
                && digit(i + 4)
                && digit(i + 5)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Value {
        serde_json::from_str(s).expect("valid json")
    }

    #[test]
    fn stamps_version_and_orders_wrapper_first() {
        let out = build_describe_envelope(
            "env",
            r#"{"env_tags": {"b": 1, "a": 2}, "params": {"signature_tier": []}}"#,
            None,
        )
        .expect("builds");
        // schema_version + kind lead; nested keys sort (a before b).
        assert!(out.starts_with(r#"{"schema_version":1,"kind":"env","#));
        assert!(out.contains(r#""env_tags":{"a":2,"b":1}"#));
        // no generated_at when omitted; no model_spec on an env.
        assert!(!out.contains("generated_at"));
        assert!(!out.contains("model_spec"));
        // round-trips to a value with the stamped version.
        let v = parse(&out);
        assert_eq!(v["schema_version"], 1);
        assert_eq!(v["kind"], "env");
    }

    #[test]
    fn byte_identical_regardless_of_input_key_order() {
        let a = build_describe_envelope("model", r#"{"model_spec":{"x":1,"y":2}}"#, None).unwrap();
        let b = build_describe_envelope("model", r#"{"model_spec":{"y":2,"x":1}}"#, None).unwrap();
        assert_eq!(
            a, b,
            "sorted re-serialization must erase producer key order"
        );
    }

    #[test]
    fn caller_supplied_timestamp_passes_through() {
        let out =
            build_describe_envelope("env", "{}", Some("2026-06-28T19:30:00Z")).expect("builds");
        assert!(out.contains(r#""generated_at":"2026-06-28T19:30:00Z""#));
        // with fractional seconds + numeric offset
        assert!(build_describe_envelope("env", "{}", Some("2026-06-28T19:30:00.5+01:00")).is_ok());
    }

    #[test]
    fn rejects_bad_timestamp() {
        let err = build_describe_envelope("env", "{}", Some("June 28")).unwrap_err();
        assert!(matches!(err, EnvelopeError::BadTimestamp(_)));
        assert!(!is_rfc3339("2026-06-28 19:30:00Z")); // space, not T
        assert!(!is_rfc3339("2026-06-28T19:30:00")); // no zone
        assert!(!is_rfc3339("2026-06-28T19:30:00.Z")); // empty fraction
    }

    #[test]
    fn rejects_unknown_kind() {
        let err = build_describe_envelope("widget", "{}", None).unwrap_err();
        assert!(matches!(err, EnvelopeError::UnknownKind(k) if k == "widget"));
    }

    #[test]
    fn rejects_unknown_top_level_field() {
        let err = build_describe_envelope("env", r#"{"bogus": 1}"#, None).unwrap_err();
        assert!(matches!(err, EnvelopeError::Parse(_)));
    }

    #[test]
    fn enforces_env_vs_model_invariant() {
        // model_spec under an env envelope is rejected...
        let err = build_describe_envelope("env", r#"{"model_spec":{"x":1}}"#, None).unwrap_err();
        assert!(matches!(
            err,
            EnvelopeError::KindMismatch {
                kind: Kind::Env,
                field: "model_spec"
            }
        ));
        // ...and env spaces/tags under a model envelope.
        assert!(matches!(
            build_describe_envelope("model", r#"{"env_spec":{}}"#, None).unwrap_err(),
            EnvelopeError::KindMismatch {
                kind: Kind::Model,
                field: "env_spec"
            }
        ));
    }

    #[test]
    fn model_spec_is_present_even_when_null() {
        // A model with no resolved spec must still carry model_spec: null (the
        // kind's own field is always present); env spaces never appear.
        let out = build_describe_envelope("model", "{}", None).expect("builds");
        assert!(out.contains(r#""model_spec":null"#));
        assert!(!out.contains("env_spec") && !out.contains("env_tags"));
        // ...while an env carries env_spec/env_tags and never model_spec.
        let env = build_describe_envelope("env", r#"{"env_spec":{"a":1}}"#, None).unwrap();
        assert!(env.contains(r#""env_tags":null"#));
        assert!(!env.contains("model_spec"));
    }

    #[test]
    fn rejects_oversized_pieces() {
        let big = format!(r#"{{"params":"{}"}}"#, "x".repeat(MAX_PIECES_BYTES));
        assert!(matches!(
            build_describe_envelope("env", &big, None).unwrap_err(),
            EnvelopeError::TooLarge(_)
        ));
    }
}
