//! The `requires` and `facets` maps of a declaration, and the key pattern.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde_json::Value;

use super::constraint::{Clause, dotted, parse_constraint};
use super::quantity::{QuantityError, format_quantity, parse_quantity};

/// The `requires` keys the platform matches hardware against.
pub const REQUIRE_KEYS: [&str; 7] = [
    "accel.vendor",
    "accel.compute",
    "accel.gfx",
    "accel.cuda",
    "accel.driver",
    "accel.vram",
    "accel.vram_bytes",
];
/// The deprecated spelling of `accel.vram`: the minimum as integer bytes.
pub const DEPRECATED_VRAM_KEY: &str = "accel.vram_bytes";
/// The `accel.vendor` values.
pub const VENDORS: [&str; 2] = ["nvidia", "amd"];
/// The facet keys a declaration may use.
pub const FACET_KEYS: [&str; 3] = ["framework", "accel", "render"];
/// The `facets.accel` values: the accelerator stack an image is built for.
pub const ACCEL_FACETS: [&str; 3] = ["cpu", "cuda", "rocm"];
/// The `facets.render` values: the GL backend an environment renders with.
pub const RENDER_FACETS: [&str; 3] = ["osmesa", "egl", "none"];
/// The longest variant, profile, or row key.
pub const MAX_KEY_LEN: usize = 32;

/// `requires` keys only an NVIDIA GPU has.
const NVIDIA_KEYS: [&str; 3] = ["accel.compute", "accel.cuda", "accel.driver"];

/// One parsed `requires` entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Requirement {
    /// `accel.vendor`.
    Vendor(String),
    /// `accel.gfx`: the GPU's target must be one of these.
    Targets(Vec<String>),
    /// `accel.compute`, `accel.cuda`, `accel.driver`: every clause must hold.
    Version(Vec<Clause>),
    /// `accel.vram`, or the deprecated `accel.vram_bytes`: a minimum per GPU.
    MinBytes(u64),
}

impl fmt::Display for Requirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vendor(vendor) => write!(f, "={vendor}"),
            Self::Targets(targets) => write!(f, " in [{}]", targets.join(",")),
            Self::Version(clauses) => {
                let clauses: Vec<String> = clauses
                    .iter()
                    .map(|clause| {
                        format!("{}{}", clause.comparator.symbol(), dotted(&clause.version))
                    })
                    .collect();
                f.write_str(&clauses.join(","))
            }
            Self::MinBytes(bytes) => write!(f, ">={}", format_quantity(*bytes)),
        }
    }
}

/// `requires` entries by key, in key order; `accel.vram_bytes` is recorded
/// as `accel.vram` (see [`requirement_key`]).
pub type Requires = BTreeMap<String, Requirement>;

/// The key a `requires` entry is recorded under: `accel.vram_bytes` is
/// `accel.vram`, so one spelling overrides the other key by key.
pub fn requirement_key(key: &str) -> &str {
    if key == DEPRECATED_VRAM_KEY {
        "accel.vram"
    } else {
        key
    }
}

/// Render `requires` as `accel.cuda>=12.4, accel.vendor=nvidia`.
pub fn format_requires(requires: &Requires) -> String {
    requires
        .iter()
        .map(|(key, requirement)| format!("{key}{requirement}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Render facets as `accel=cuda, framework=torch`.
pub fn format_facets(facets: &BTreeMap<String, String>) -> String {
    facets
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn is_gfx_target(value: &str) -> bool {
    value.strip_prefix("gfx").is_some_and(|hex| {
        (3..=4).contains(&hex.len()) && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    })
}

/// Parse one `requires` entry, in the platform's words when it is refused.
pub fn parse_requirement(key: &str, value: &Value) -> Result<Requirement, String> {
    match key {
        "accel.vendor" => match value.as_str() {
            Some(vendor) if VENDORS.contains(&vendor) => Ok(Requirement::Vendor(vendor.to_owned())),
            _ => Err(format!("accel.vendor {value} is not \"nvidia\" or \"amd\"")),
        },
        "accel.compute" | "accel.cuda" | "accel.driver" => value
            .as_str()
            .and_then(parse_constraint)
            .map(Requirement::Version)
            .ok_or_else(|| format!("{key} {value} is not a version constraint like \">=12.1\"")),
        "accel.gfx" => {
            let targets: Option<Vec<String>> = value
                .as_array()
                .filter(|list| !list.is_empty())
                .map(|list| {
                    list.iter()
                        .filter_map(|target| {
                            target
                                .as_str()
                                .filter(|t| is_gfx_target(t))
                                .map(str::to_owned)
                        })
                        .collect()
                });
            match (targets, value.as_array()) {
                (Some(targets), Some(list)) if targets.len() == list.len() => {
                    Ok(Requirement::Targets(targets))
                }
                (Some(_), Some(list)) => Err(format!(
                    "accel.gfx entry {} is not a gfx target like \"gfx942\"",
                    list.iter()
                        .find(|target| !target.as_str().is_some_and(is_gfx_target))
                        .unwrap_or(&Value::Null)
                )),
                _ => Err(format!(
                    "accel.gfx {value} is not a non-empty list of gfx targets"
                )),
            }
        }
        "accel.vram" => parse_vram(value),
        // The platform stores the minimum as an int64.
        "accel.vram_bytes" => value
            .as_i64()
            .and_then(|bytes| u64::try_from(bytes).ok())
            .filter(|bytes| *bytes > 0)
            .map(Requirement::MinBytes)
            .ok_or_else(|| {
                let hint = value
                    .as_str()
                    .and_then(|raw| parse_quantity(raw.trim().trim_start_matches(">=").trim()).ok())
                    .map(|bytes| format!(" (write accel.vram: \"{}\")", format_quantity(bytes)))
                    .unwrap_or_default();
                format!("accel.vram_bytes {value} is not a positive integer byte count{hint}")
            }),
        _ => Err(format!(
            "unknown key {key:?} (allowed: {})",
            REQUIRE_KEYS.join(", ")
        )),
    }
}

/// `accel.vram`: a quantity string like `"16Gi"`, a positive whole number of
/// bytes within the signed 64-bit range.
fn parse_vram(value: &Value) -> Result<Requirement, String> {
    let Some(raw) = value.as_str() else {
        let hint = if value.is_number() {
            "; write it as a quantity string, e.g. \"16Gi\""
        } else {
            ""
        };
        return Err(format!("accel.vram {value} is not a quantity string{hint}"));
    };
    parse_quantity(raw)
        .map(Requirement::MinBytes)
        .map_err(|error| match error {
            QuantityError::Syntax => format!(
                "accel.vram {value} is not a quantity like \"16Gi\" (digits with an optional \
                 Ki, Mi, Gi, Ti, K, M, G or T suffix)"
            ),
            QuantityError::Zero => format!("accel.vram {value} is not a positive quantity"),
            QuantityError::Fractional => {
                format!("accel.vram {value} is not a whole number of bytes")
            }
            QuantityError::TooLarge => {
                format!("accel.vram {value} is more than {} bytes", i64::MAX)
            }
        })
}

/// One `requires` map may spell the VRAM minimum one way only.
fn vram_spelling_problems(keys: &BTreeSet<&str>) -> Vec<String> {
    if keys.contains("accel.vram") && keys.contains(DEPRECATED_VRAM_KEY) {
        vec!["set accel.vram or the deprecated accel.vram_bytes, not both".to_owned()]
    } else {
        Vec::new()
    }
}

/// The deprecated `accel.vram_bytes`, with the quantity to write instead.
pub(super) fn requires_warnings(raw: &Value) -> Vec<String> {
    match raw
        .get(DEPRECATED_VRAM_KEY)
        .map(|value| parse_requirement(DEPRECATED_VRAM_KEY, value))
    {
        Some(Ok(Requirement::MinBytes(bytes))) => vec![format!(
            "accel.vram_bytes is deprecated; write accel.vram: \"{}\"",
            format_quantity(bytes)
        )],
        _ => Vec::new(),
    }
}

/// The vendor-coherence rules on one `requires` map: NVIDIA keys under `amd`,
/// `accel.gfx` under `nvidia`, and any hardware key without a vendor.
pub(super) fn vendor_problems(keys: &BTreeSet<&str>, vendor: Option<&str>) -> Vec<String> {
    let mut problems = Vec::new();
    match vendor {
        Some("amd") => {
            for key in NVIDIA_KEYS.iter().filter(|key| keys.contains(*key)) {
                problems.push(format!(
                    "{key} is an NVIDIA requirement but accel.vendor is amd"
                ));
            }
        }
        Some("nvidia") => {
            if keys.contains("accel.gfx") {
                problems
                    .push("accel.gfx is an AMD requirement but accel.vendor is nvidia".to_owned());
            }
        }
        Some(_) => {}
        None => {
            if let Some(key) = NVIDIA_KEYS
                .iter()
                .chain(&["accel.gfx", "accel.vram", DEPRECATED_VRAM_KEY])
                .find(|key| keys.contains(*key))
            {
                problems.push(format!("{key} needs accel.vendor"));
            }
        }
    }
    problems
}

/// Parse a `requires` object as written: every entry, the VRAM spelling, and
/// the vendor rules.
pub(super) fn parse_requires(raw: &Value) -> (Requires, Vec<String>) {
    let Value::Object(entries) = raw else {
        return (
            Requires::new(),
            vec!["requires is not an object".to_owned()],
        );
    };
    let mut requires = Requires::new();
    let mut problems = Vec::new();
    for (key, value) in entries {
        match parse_requirement(key, value) {
            Ok(requirement) => {
                requires.insert(requirement_key(key).to_owned(), requirement);
            }
            Err(problem) => problems.push(problem),
        }
    }
    let keys: BTreeSet<&str> = entries.keys().map(String::as_str).collect();
    problems.extend(vram_spelling_problems(&keys));
    problems.extend(vendor_problems(
        &keys,
        entries.get("accel.vendor").and_then(Value::as_str),
    ));
    (requires, problems)
}

pub(super) fn requires_vendor(requires: &Requires) -> Option<&str> {
    match requires.get("accel.vendor") {
        Some(Requirement::Vendor(vendor)) => Some(vendor),
        _ => None,
    }
}

/// Parse a `facets` object as written.
pub(super) fn parse_facets(raw: &Value) -> (BTreeMap<String, String>, Vec<String>) {
    let Value::Object(entries) = raw else {
        return (BTreeMap::new(), vec!["facets is not an object".to_owned()]);
    };
    let mut facets = BTreeMap::new();
    let mut problems = Vec::new();
    for (key, value) in entries {
        let text = value.as_str().unwrap_or_default();
        let ok = match key.as_str() {
            "framework" => is_token(text),
            "accel" => ACCEL_FACETS.contains(&text),
            "render" => RENDER_FACETS.contains(&text),
            _ => {
                problems.push(format!(
                    "unknown facet {key:?} (allowed: {})",
                    FACET_KEYS.join(", ")
                ));
                continue;
            }
        };
        if ok && value.is_string() {
            facets.insert(key.clone(), text.to_owned());
        } else {
            problems.push(match key.as_str() {
                "framework" => format!("facets.framework {value} is not a lowercase token"),
                "accel" => format!("facets.accel {value} is not cpu, cuda or rocm"),
                _ => format!("facets.render {value} is not osmesa, egl or none"),
            });
        }
    }
    (facets, problems)
}

/// A lowercase token: `^[a-z0-9][a-z0-9_.-]{0,31}$`.
fn is_token(value: &str) -> bool {
    let mut bytes = value.bytes();
    value.len() <= MAX_KEY_LEN
        && bytes
            .next()
            .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        && bytes.all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'_' | b'.' | b'-')
        })
}

/// A variant, profile, or row key: `^[a-z0-9][a-z0-9-]{0,31}$`.
pub fn valid_key(key: &str) -> bool {
    let mut bytes = key.bytes();
    key.len() <= MAX_KEY_LEN
        && bytes
            .next()
            .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        && bytes.all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// The vendor a `facets.accel` value implies.
pub(super) fn accel_vendor(accel: &str) -> Option<&'static str> {
    match accel {
        "cuda" => Some("nvidia"),
        "rocm" => Some("amd"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn requirement_values_table() {
        let ok = |key: &str, value: Value| {
            parse_requirement(key, &value).unwrap_or_else(|e| panic!("{key} {value}: {e}"))
        };
        assert_eq!(
            ok("accel.vendor", json!("amd")),
            Requirement::Vendor("amd".to_owned())
        );
        assert_eq!(
            ok("accel.gfx", json!(["gfx942", "gfx90a"])),
            Requirement::Targets(vec!["gfx942".to_owned(), "gfx90a".to_owned()])
        );
        assert_eq!(
            ok("accel.vram_bytes", json!(24000000000_u64)),
            Requirement::MinBytes(24000000000)
        );
        assert_eq!(
            ok("accel.vram_bytes", json!(i64::MAX)),
            Requirement::MinBytes(i64::MAX as u64)
        );
        for (raw, bytes) in [
            ("16Gi", 17_179_869_184),
            ("24G", 24_000_000_000),
            ("1.5Gi", 1_610_612_736),
            ("512Mi", 536_870_912),
            ("80000000000", 80_000_000_000),
        ] {
            assert_eq!(ok("accel.vram", json!(raw)), Requirement::MinBytes(bytes));
        }
        // Both spellings render as a quantity.
        assert_eq!(
            ok("accel.vram_bytes", json!(17179869184_u64)).to_string(),
            ">=16Gi"
        );
        assert_eq!(ok("accel.vram", json!("24000M")).to_string(), ">=24G");
        assert_eq!(
            ok("accel.driver", json!("535.104.05")).to_string(),
            ">=535.104.5"
        );
        for (key, value, needle) in [
            (
                "accel.vendor",
                json!("intel"),
                "is not \"nvidia\" or \"amd\"",
            ),
            (
                "accel.vendor",
                json!(["nvidia"]),
                "is not \"nvidia\" or \"amd\"",
            ),
            ("accel.cuda", json!(12.4), "is not a version constraint"),
            ("accel.cuda", json!(["12.4"]), "is not a version constraint"),
            ("accel.compute", json!("~8"), "is not a version constraint"),
            ("accel.gfx", json!("gfx942"), "not a non-empty list"),
            ("accel.gfx", json!([]), "not a non-empty list"),
            (
                "accel.gfx",
                json!(["gfx942", "mi300"]),
                "entry \"mi300\" is not a gfx target",
            ),
            (
                "accel.vram_bytes",
                json!(">=24000000000"),
                "is not a positive integer byte count (write accel.vram: \"24G\")",
            ),
            (
                "accel.vram_bytes",
                json!("lots"),
                "\"lots\" is not a positive integer byte count",
            ),
            (
                "accel.vram",
                json!(16),
                "16 is not a quantity string; write it as a quantity string, e.g. \"16Gi\"",
            ),
            (
                "accel.vram",
                json!(["16Gi"]),
                "[\"16Gi\"] is not a quantity string",
            ),
            (
                "accel.vram",
                json!("16 Gi"),
                "is not a quantity like \"16Gi\"",
            ),
            ("accel.vram", json!("16gi"), "is not a quantity like"),
            ("accel.vram", json!("16GB"), "is not a quantity like"),
            ("accel.vram", json!("-1Gi"), "is not a quantity like"),
            ("accel.vram", json!("1e9"), "is not a quantity like"),
            ("accel.vram", json!(">=16Gi"), "is not a quantity like"),
            ("accel.vram", json!(""), "is not a quantity like"),
            ("accel.vram", json!("0"), "\"0\" is not a positive quantity"),
            (
                "accel.vram",
                json!("1.5"),
                "\"1.5\" is not a whole number of bytes",
            ),
            (
                "accel.vram",
                json!("9999999999Ti"),
                "\"9999999999Ti\" is more than 9223372036854775807 bytes",
            ),
            ("accel.vram_bytes", json!(0), "not a positive integer"),
            ("accel.vram_bytes", json!(-1), "not a positive integer"),
            ("accel.vram_bytes", json!(1.5), "not a positive integer"),
            // The platform stores an int64.
            (
                "accel.vram_bytes",
                json!(i64::MAX as u64 + 1),
                "not a positive integer",
            ),
            (
                "accel.vram_bytes",
                json!(u64::MAX),
                "not a positive integer",
            ),
            ("accel.memory", json!(1), "unknown key \"accel.memory\""),
        ] {
            let error = parse_requirement(key, &value).unwrap_err();
            assert!(error.contains(needle), "{key} {value}: {error}");
        }
    }

    #[test]
    fn hardware_keys_need_a_consistent_vendor() {
        let problems = |raw: Value| parse_requires(&raw).1;
        assert_eq!(
            problems(json!({"accel.cuda": ">=12"})),
            ["accel.cuda needs accel.vendor"]
        );
        assert_eq!(
            problems(json!({"accel.vram_bytes": 1})),
            ["accel.vram_bytes needs accel.vendor"]
        );
        assert_eq!(
            problems(json!({"accel.vram": "16Gi"})),
            ["accel.vram needs accel.vendor"]
        );
        assert_eq!(
            problems(json!({"accel.vendor": "amd", "accel.cuda": ">=12", "accel.driver": "550"})),
            [
                "accel.cuda is an NVIDIA requirement but accel.vendor is amd",
                "accel.driver is an NVIDIA requirement but accel.vendor is amd"
            ]
        );
        assert_eq!(
            problems(json!({"accel.vendor": "nvidia", "accel.gfx": ["gfx942"]})),
            ["accel.gfx is an AMD requirement but accel.vendor is nvidia"]
        );
        assert!(
            problems(
                json!({"accel.vendor": "amd", "accel.gfx": ["gfx942"], "accel.vram_bytes": 1})
            )
            .is_empty()
        );
        assert_eq!(problems(json!([])), ["requires is not an object"]);
    }

    #[test]
    fn vram_is_spelled_one_way() {
        let (requires, problems) = parse_requires(
            &json!({"accel.vendor": "nvidia", "accel.vram": "16Gi", "accel.vram_bytes": 1}),
        );
        assert_eq!(
            problems,
            ["set accel.vram or the deprecated accel.vram_bytes, not both"]
        );
        assert!(!requires.contains_key("accel.vram_bytes"), "{requires:?}");
        // The deprecated key is recorded, and rendered, as accel.vram.
        let (requires, problems) =
            parse_requires(&json!({"accel.vendor": "nvidia", "accel.vram_bytes": 16000000000_u64}));
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(
            format_requires(&requires),
            "accel.vendor=nvidia, accel.vram>=16G"
        );
    }

    #[test]
    fn the_deprecated_vram_key_warns_with_its_quantity() {
        for (bytes, quantity) in [
            (json!(16000000000_u64), "16G"),
            (json!(17179869184_u64), "16Gi"),
            (json!(24000000000_u64), "24G"),
            (json!(123), "123"),
        ] {
            assert_eq!(
                requires_warnings(&json!({"accel.vendor": "nvidia", "accel.vram_bytes": bytes})),
                [format!(
                    "accel.vram_bytes is deprecated; write accel.vram: \"{quantity}\""
                )]
            );
        }
        // A value the platform refuses is a failure, not a deprecation.
        for raw in [
            json!({"accel.vendor": "nvidia", "accel.vram": "16Gi"}),
            json!({"accel.vram_bytes": 0}),
            json!({"accel.vram_bytes": "16Gi"}),
            json!([]),
        ] {
            assert!(requires_warnings(&raw).is_empty(), "{raw}");
        }
    }
}
