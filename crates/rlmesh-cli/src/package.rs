//! The compute-variant blocks of the `dev.rlmesh.package` label.
//!
//! One model or environment version can ship several images (a PyTorch CUDA
//! build, a ROCm build, a JAX build) and several runtime profiles of one image
//! (osmesa or EGL rendering). The managed platform reads them structurally: an
//! OCI image index is a version, its non-attestation children are variants,
//! and each child declares itself in its `dev.rlmesh.package` label:
//!
//! ```json
//! {
//!   "schemaVersion": 1,
//!   "variant": {
//!     "key": "torch-cuda12",
//!     "facets": {"framework": "torch", "accel": "nvidia"},
//!     "requires": {"accel.vendor": "nvidia", "accel.cuda": ">=12.2"},
//!     "priority": 10
//!   },
//!   "profiles": [
//!     {"key": "osmesa", "default": true, "envVars": {"MUJOCO_GL": "osmesa"}},
//!     {"key": "egl", "envVars": {"MUJOCO_GL": "egl"}, "gpu": {"count": 1},
//!      "requires": {"accel.vendor": "nvidia"}}
//!   ]
//! }
//! ```
//!
//! Every field is optional. `facets` are free-form string tags; `requires`
//! keys are the fixed set [`REQUIRE_KEYS`], each a comparator string (`>=X`,
//! `<X`, `=X`, or a bare `X` meaning `=X`) or an array of bare values (set
//! membership); a higher `priority` wins ties. The platform implements exactly
//! this schema, so like [`crate::image_check`] the parsing rules and the
//! report's message prefixes (`variant:`, `profiles:`) are a contract.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde_json::{Map, Value};

use crate::image_check::{ADDRESS_ENV, CheckReport, ImageConfig};

/// The `requires` keys the platform matches hardware against.
pub const REQUIRE_KEYS: [&str; 6] = [
    "accel.vendor",
    "accel.compute",
    "accel.gfx",
    "accel.cuda",
    "accel.driver",
    "accel.vram_bytes",
];
/// The `accel.vendor` values the platform's hardware classes report.
pub const KNOWN_VENDORS: [&str; 3] = ["nvidia", "amd", "intel"];

const VARIANT_FIELDS: [&str; 4] = ["key", "facets", "requires", "priority"];
const PROFILE_FIELDS: [&str; 6] = ["key", "default", "facets", "envVars", "gpu", "requires"];

/// How a `requires` value compares against the hardware's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Comparator {
    /// `=X` or a bare `X`.
    Eq,
    /// `>=X`.
    Ge,
    /// `<X`.
    Lt,
}

impl Comparator {
    fn symbol(self) -> &'static str {
        match self {
            Self::Eq => "=",
            Self::Ge => ">=",
            Self::Lt => "<",
        }
    }
}

/// One parsed `requires` entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Requirement {
    /// A comparator string; the value is stored without its operator.
    Compare(Comparator, String),
    /// A string array: the hardware's value must be one of these.
    OneOf(Vec<String>),
}

impl fmt::Display for Requirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compare(comparator, value) => write!(f, "{}{value}", comparator.symbol()),
            Self::OneOf(values) => write!(f, " in [{}]", values.join(",")),
        }
    }
}

/// `requires` entries by key, in key order.
pub type Requires = BTreeMap<String, Requirement>;

/// Render `requires` as `accel.vendor=nvidia, accel.cuda>=12.2`.
pub fn format_requires(requires: &Requires) -> String {
    requires
        .iter()
        .map(|(key, requirement)| format!("{key}{requirement}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Render facets as `framework=torch, accel=nvidia`.
pub fn format_facets(facets: &BTreeMap<String, String>) -> String {
    facets
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// What kind of value a `requires` key compares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Domain {
    /// Equality and set membership only (`accel.vendor`, `accel.gfx`).
    Text,
    /// Dotted numeric versions, up to four parts (`8.0`, `12.2`, `535.104.05`).
    Version,
    /// A whole number of bytes.
    Bytes,
}

fn domain(key: &str) -> Option<Domain> {
    match key {
        "accel.vendor" | "accel.gfx" => Some(Domain::Text),
        "accel.compute" | "accel.cuda" | "accel.driver" => Some(Domain::Version),
        "accel.vram_bytes" => Some(Domain::Bytes),
        _ => None,
    }
}

/// A version or byte count as numeric parts; compared with zero padding, so
/// `12` equals `12.0`.
fn ordered(domain: Domain, value: &str) -> Option<Vec<u64>> {
    match domain {
        Domain::Text => None,
        Domain::Version => {
            let parts: Vec<&str> = value.split('.').collect();
            if parts.len() > 4
                || parts
                    .iter()
                    .any(|part| part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()))
            {
                return None;
            }
            parts.iter().map(|part| part.parse().ok()).collect()
        }
        Domain::Bytes => {
            if !value.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            value.parse().ok().map(|bytes| vec![bytes])
        }
    }
}

fn compare(left: &[u64], right: &[u64]) -> Ordering {
    let len = left.len().max(right.len());
    (0..len)
        .map(|index| {
            let a = left.get(index).copied().unwrap_or(0);
            let b = right.get(index).copied().unwrap_or(0);
            a.cmp(&b)
        })
        .find(|ordering| ordering.is_ne())
        .unwrap_or(Ordering::Equal)
}

fn check_value(key: &str, domain: Domain, value: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err(format!("{key} has an empty value"));
    }
    match domain {
        Domain::Text if value.chars().any(char::is_whitespace) => {
            Err(format!("{key} value {value:?} contains whitespace"))
        }
        Domain::Text => Ok(()),
        Domain::Version => ordered(domain, value).map(|_| ()).ok_or_else(|| {
            format!("{key} value {value:?} is not a dotted numeric version (e.g. 12.2)")
        }),
        Domain::Bytes => ordered(domain, value)
            .map(|_| ())
            .ok_or_else(|| format!("{key} value {value:?} is not a whole number of bytes")),
    }
}

/// Parse one `requires` entry. Accepted comparators are exactly `>=`, `<`,
/// and `=` (or none, meaning `=`); `accel.vendor` and `accel.gfx` take only
/// equality or a list. A JSON number is refused rather than read as `=`, since
/// `"accel.vram_bytes": 24000000000` almost always meant `>=`.
pub fn parse_requirement(key: &str, value: &Value) -> Result<Requirement, String> {
    let Some(domain) = domain(key) else {
        return Err(format!(
            "unknown requires key {key:?}; the platform matches only {}",
            REQUIRE_KEYS.join(", ")
        ));
    };
    match value {
        Value::String(raw) => {
            let raw = raw.trim();
            let (comparator, rest) = if let Some(rest) = raw.strip_prefix(">=") {
                (Comparator::Ge, rest)
            } else if raw.starts_with("<=") || raw.starts_with("!=") || raw.starts_with('>') {
                return Err(format!(
                    "{key} comparator in {raw:?} is not supported; use >=, <, or ="
                ));
            } else if let Some(rest) = raw.strip_prefix('<') {
                (Comparator::Lt, rest)
            } else if let Some(rest) = raw.strip_prefix('=') {
                (Comparator::Eq, rest)
            } else {
                (Comparator::Eq, raw)
            };
            let rest = rest.trim();
            if domain == Domain::Text && comparator != Comparator::Eq {
                return Err(format!(
                    "{key} takes a value or a list of values, not a {} comparison",
                    comparator.symbol()
                ));
            }
            check_value(key, domain, rest)?;
            Ok(Requirement::Compare(comparator, rest.to_owned()))
        }
        Value::Array(items) => {
            if items.is_empty() {
                return Err(format!("{key} is an empty list, which no hardware matches"));
            }
            let mut values = Vec::with_capacity(items.len());
            for item in items {
                let Some(raw) = item.as_str() else {
                    return Err(format!("{key} list entries must be strings, got {item}"));
                };
                let raw = raw.trim();
                if raw.starts_with(['<', '>', '=', '!']) {
                    return Err(format!(
                        "{key} list entry {raw:?} carries a comparator; list entries are \
                         plain values"
                    ));
                }
                check_value(key, domain, raw)?;
                values.push(raw.to_owned());
            }
            Ok(Requirement::OneOf(values))
        }
        Value::Number(number) => Err(format!(
            "{key} is the number {number}; write a comparator string such as \">={number}\""
        )),
        other => Err(format!(
            "{key} must be a comparator string or a list of strings, got {other}"
        )),
    }
}

/// Whether some hardware value meets every requirement in `requirements`
/// (all for the same `key`).
pub fn satisfiable(key: &str, requirements: &[&Requirement]) -> bool {
    let Some(domain) = domain(key) else {
        return true;
    };
    let same = |a: &str, b: &str| match domain {
        Domain::Text => a == b,
        _ => match (ordered(domain, a), ordered(domain, b)) {
            (Some(a), Some(b)) => compare(&a, &b).is_eq(),
            _ => a == b,
        },
    };
    let mut allowed: Option<Vec<&str>> = None;
    let mut lower: Option<Vec<u64>> = None;
    let mut upper: Option<Vec<u64>> = None;
    for requirement in requirements {
        let values: Vec<&str> = match requirement {
            Requirement::Compare(Comparator::Eq, value) => vec![value.as_str()],
            Requirement::OneOf(values) => values.iter().map(String::as_str).collect(),
            Requirement::Compare(comparator, value) => {
                let Some(bound) = ordered(domain, value) else {
                    continue;
                };
                let slot = if *comparator == Comparator::Ge {
                    &mut lower
                } else {
                    &mut upper
                };
                let tighter = match slot {
                    None => true,
                    Some(current) if *comparator == Comparator::Ge => {
                        compare(&bound, current).is_gt()
                    }
                    Some(current) => compare(&bound, current).is_lt(),
                };
                if tighter {
                    *slot = Some(bound);
                }
                continue;
            }
        };
        allowed = Some(match allowed {
            None => values,
            Some(current) => current
                .into_iter()
                .filter(|a| values.iter().any(|b| same(a, b)))
                .collect(),
        });
    }
    let in_range = |value: &[u64]| {
        lower.as_ref().is_none_or(|l| compare(value, l).is_ge())
            && upper.as_ref().is_none_or(|u| compare(value, u).is_lt())
    };
    match allowed {
        Some(values) => values.iter().any(|value| {
            domain == Domain::Text || ordered(domain, value).is_some_and(|v| in_range(&v))
        }),
        None => match (&lower, &upper) {
            (Some(l), Some(u)) => compare(l, u).is_lt(),
            _ => true,
        },
    }
}

/// The lowest hardware value a requirement admits, when it has one.
fn lower_bound(key: &str, requirement: &Requirement) -> Option<Vec<u64>> {
    let domain = domain(key)?;
    match requirement {
        Requirement::Compare(Comparator::Ge | Comparator::Eq, value) => ordered(domain, value),
        Requirement::Compare(Comparator::Lt, _) => None,
        Requirement::OneOf(values) => values
            .iter()
            .filter_map(|value| ordered(domain, value))
            .min_by(|a, b| compare(a, b)),
    }
}

/// Whether `requirement` admits `vendor` (`None` when it says nothing).
fn admits_vendor(requires: &Requires, vendor: &str) -> Option<bool> {
    requires.get("accel.vendor").map(|requirement| {
        satisfiable(
            "accel.vendor",
            &[
                requirement,
                &Requirement::Compare(Comparator::Eq, vendor.to_owned()),
            ],
        )
    })
}

/// A parsed `variant` block.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Variant {
    pub key: Option<String>,
    pub facets: BTreeMap<String, String>,
    pub requires: Requires,
    pub priority: Option<i64>,
}

/// A parsed `profiles[]` entry.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Profile {
    pub key: String,
    pub default: bool,
    pub facets: BTreeMap<String, String>,
    pub env_vars: BTreeMap<String, String>,
    pub gpu_count: Option<u64>,
    pub requires: Requires,
}

/// The `variant` and `profiles` blocks of one image's package label.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ComputeBlocks {
    pub variant: Option<Variant>,
    pub profiles: Vec<Profile>,
}

impl ComputeBlocks {
    /// Whether the label declares either block.
    pub fn is_empty(&self) -> bool {
        self.variant.is_none() && self.profiles.is_empty()
    }

    /// The profile marked `default`, when exactly one is.
    pub fn default_profile(&self) -> Option<&Profile> {
        let mut defaults = self.profiles.iter().filter(|profile| profile.default);
        match (defaults.next(), defaults.next()) {
            (Some(profile), None) => Some(profile),
            _ => None,
        }
    }
}

/// Parse the `variant` and `profiles` blocks out of a package label object,
/// reporting structural problems (wrong types, unknown `requires` keys, bad
/// comparators, duplicate or missing profile keys, not exactly one default)
/// into the returned report. Whatever parsed cleanly is kept, so one bad
/// field does not hide the rest.
pub fn parse_compute_blocks(package: &Map<String, Value>) -> (ComputeBlocks, CheckReport) {
    let mut report = CheckReport::default();
    let mut blocks = ComputeBlocks::default();
    match package.get("variant") {
        None | Some(Value::Null) => {}
        Some(Value::Object(raw)) => blocks.variant = Some(parse_variant(raw, &mut report)),
        Some(other) => report.failed.push(format!(
            "variant: must be a JSON object, got {}",
            type_name(other)
        )),
    }
    match package.get("profiles") {
        None | Some(Value::Null) => {}
        Some(Value::Array(entries)) => {
            let mut seen = BTreeSet::new();
            for (index, entry) in entries.iter().enumerate() {
                let Value::Object(raw) = entry else {
                    report.failed.push(format!(
                        "profiles: entry {index} must be a JSON object, got {}",
                        type_name(entry)
                    ));
                    continue;
                };
                let Some(profile) = parse_profile(index, raw, &mut report) else {
                    continue;
                };
                if !seen.insert(profile.key.clone()) {
                    report.failed.push(format!(
                        "profiles: key {:?} appears more than once; profile keys must be unique",
                        profile.key
                    ));
                    continue;
                }
                blocks.profiles.push(profile);
            }
            if !entries.is_empty() {
                let defaults: Vec<&str> = blocks
                    .profiles
                    .iter()
                    .filter(|profile| profile.default)
                    .map(|profile| profile.key.as_str())
                    .collect();
                match defaults.len() {
                    1 => {}
                    0 => report.failed.push(
                        "profiles: no profile is marked \"default\": true; mark exactly one, \
                         the one the platform runs when a request names none"
                            .to_owned(),
                    ),
                    _ => report.failed.push(format!(
                        "profiles: {} are all marked default; mark exactly one",
                        defaults.join(", ")
                    )),
                }
            }
        }
        Some(other) => report.failed.push(format!(
            "profiles: must be a JSON array, got {}",
            type_name(other)
        )),
    }
    (blocks, report)
}

fn parse_variant(raw: &Map<String, Value>, report: &mut CheckReport) -> Variant {
    let mut variant = Variant::default();
    match raw.get("key") {
        None | Some(Value::Null) => report.warnings.push(
            "variant: no key; the platform cannot address the variant by name and \
             `rlmesh registry publish` refuses it"
                .to_owned(),
        ),
        Some(Value::String(key)) if !key.trim().is_empty() => {
            check_key("variant:", key, report);
            variant.key = Some(key.trim().to_owned());
        }
        Some(other) => report.failed.push(format!(
            "variant: key must be a non-empty string, got {other}"
        )),
    }
    variant.facets = parse_facets("variant:", raw.get("facets"), report);
    variant.requires = parse_requires("variant:", raw.get("requires"), report);
    match raw.get("priority") {
        None | Some(Value::Null) => {}
        Some(value) => match value.as_i64() {
            Some(priority) => variant.priority = Some(priority),
            None => report
                .failed
                .push(format!("variant: priority must be an integer, got {value}")),
        },
    }
    warn_unknown_fields("variant:", raw, &VARIANT_FIELDS, report);
    variant
}

fn parse_profile(
    index: usize,
    raw: &Map<String, Value>,
    report: &mut CheckReport,
) -> Option<Profile> {
    let key = match raw.get("key") {
        Some(Value::String(key)) if !key.trim().is_empty() => key.trim().to_owned(),
        None | Some(Value::Null) => {
            report
                .failed
                .push(format!("profiles: entry {index} has no key"));
            return None;
        }
        Some(other) => {
            report.failed.push(format!(
                "profiles: entry {index} key must be a non-empty string, got {other}"
            ));
            return None;
        }
    };
    let scope = format!("profiles: {key}:");
    check_key(&scope, &key, report);
    let mut profile = Profile {
        key,
        ..Profile::default()
    };
    match raw.get("default") {
        None | Some(Value::Null) => {}
        Some(Value::Bool(default)) => profile.default = *default,
        Some(other) => report.failed.push(format!(
            "{scope} default must be true or false, got {other}"
        )),
    }
    profile.facets = parse_facets(&scope, raw.get("facets"), report);
    match raw.get("envVars") {
        None | Some(Value::Null) => {}
        Some(Value::Object(vars)) => {
            for (name, value) in vars {
                if !is_env_name(name) {
                    report.failed.push(format!(
                        "{scope} envVars name {name:?} is not a valid environment variable name"
                    ));
                    continue;
                }
                let Some(value) = value.as_str() else {
                    report.failed.push(format!(
                        "{scope} envVars {name} must be a string, got {value}"
                    ));
                    continue;
                };
                if name == ADDRESS_ENV {
                    report.warnings.push(format!(
                        "{scope} envVars sets {ADDRESS_ENV}; the platform assigns it per pod, \
                         so it has no effect there"
                    ));
                }
                profile.env_vars.insert(name.clone(), value.to_owned());
            }
        }
        Some(other) => report.failed.push(format!(
            "{scope} envVars must be an object of strings, got {}",
            type_name(other)
        )),
    }
    match raw.get("gpu") {
        None | Some(Value::Null) => {}
        Some(Value::Object(gpu)) => {
            match gpu.get("count") {
                None | Some(Value::Null) => {}
                Some(value) => match value.as_u64() {
                    Some(count) => profile.gpu_count = Some(count),
                    None => report.failed.push(format!(
                        "{scope} gpu.count must be a non-negative integer, got {value}"
                    )),
                },
            }
            warn_unknown_fields(&format!("{scope} gpu"), gpu, &["count"], report);
        }
        Some(other) => report.failed.push(format!(
            "{scope} gpu must be an object, got {}",
            type_name(other)
        )),
    }
    profile.requires = parse_requires(&scope, raw.get("requires"), report);
    warn_unknown_fields(&scope, raw, &PROFILE_FIELDS, report);
    Some(profile)
}

fn parse_facets(
    scope: &str,
    raw: Option<&Value>,
    report: &mut CheckReport,
) -> BTreeMap<String, String> {
    let mut facets = BTreeMap::new();
    match raw {
        None | Some(Value::Null) => {}
        Some(Value::Object(entries)) => {
            for (name, value) in entries {
                match value.as_str() {
                    Some(value) if !name.is_empty() && !value.is_empty() => {
                        facets.insert(name.clone(), value.to_owned());
                    }
                    Some(_) => report.failed.push(format!(
                        "{scope} facets {name:?} has an empty name or value"
                    )),
                    None => report.failed.push(format!(
                        "{scope} facets {name:?} must be a string, got {value}"
                    )),
                }
            }
        }
        Some(other) => report.failed.push(format!(
            "{scope} facets must be an object of strings, got {}",
            type_name(other)
        )),
    }
    facets
}

fn parse_requires(scope: &str, raw: Option<&Value>, report: &mut CheckReport) -> Requires {
    let mut requires = Requires::new();
    match raw {
        None | Some(Value::Null) => {}
        Some(Value::Object(entries)) => {
            for (key, value) in entries {
                match parse_requirement(key, value) {
                    Ok(requirement) => {
                        requires.insert(key.clone(), requirement);
                    }
                    Err(message) => report.failed.push(format!("{scope} requires {message}")),
                }
            }
            let vendors: Vec<&str> = match requires.get("accel.vendor") {
                Some(Requirement::Compare(_, vendor)) => vec![vendor.as_str()],
                Some(Requirement::OneOf(vendors)) => vendors.iter().map(String::as_str).collect(),
                None => Vec::new(),
            };
            for vendor in vendors.into_iter().filter(|v| !KNOWN_VENDORS.contains(v)) {
                report.warnings.push(format!(
                    "{scope} requires accel.vendor {vendor:?} is not one the platform's \
                     hardware reports ({})",
                    KNOWN_VENDORS.join(", ")
                ));
            }
        }
        Some(other) => report.failed.push(format!(
            "{scope} requires must be an object, got {}",
            type_name(other)
        )),
    }
    requires
}

/// The accelerator stack an image's environment declares.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImageAccel {
    /// `CUDA_VERSION`, the toolkit the image was built on.
    pub cuda_version: Option<String>,
    /// The `cuda>=X` floor in `NVIDIA_REQUIRE_CUDA`, which the NVIDIA
    /// container runtime enforces against the host driver.
    pub require_cuda: Option<String>,
    /// `ROCM_VERSION`.
    pub rocm_version: Option<String>,
}

impl ImageAccel {
    /// Read the CUDA and ROCm markers off an image's `Env`.
    pub fn from_config(config: &ImageConfig) -> Self {
        let set = |key: &str| {
            config
                .env_value(key)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        };
        Self {
            cuda_version: set("CUDA_VERSION"),
            require_cuda: set("NVIDIA_REQUIRE_CUDA").and_then(|value| {
                // `cuda>=12.4 brand=tesla,driver>=470,driver<471 ...`: only the
                // standalone cuda floor; the comma groups are alternatives.
                value.split_whitespace().find_map(|word| {
                    word.strip_prefix("cuda>=")
                        .filter(|floor| ordered(Domain::Version, floor).is_some())
                        .map(str::to_owned)
                })
            }),
            rocm_version: set("ROCM_VERSION"),
        }
    }

    fn is_cuda(&self) -> bool {
        self.cuda_version.is_some() || self.require_cuda.is_some()
    }

    /// The host CUDA floor and where it came from: the enforced
    /// `NVIDIA_REQUIRE_CUDA` when present, else `CUDA_VERSION`'s major.minor.
    fn cuda_floor(&self) -> Option<(String, String)> {
        if let Some(floor) = &self.require_cuda {
            return Some((floor.clone(), format!("NVIDIA_REQUIRE_CUDA cuda>={floor}")));
        }
        let version = self.cuda_version.as_ref()?;
        let parts: Vec<&str> = version.split('.').take(2).collect();
        let floor = parts.join(".");
        ordered(Domain::Version, &floor).map(|_| (floor, format!("CUDA_VERSION={version}")))
    }
}

/// Check one scope's effective `requires` against the image's accelerator
/// stack. `variant_scope` adds the "declare it" warnings, which would repeat
/// for every profile otherwise.
fn check_against_image(
    scope: &str,
    requires: &Requires,
    accel: &ImageAccel,
    variant_scope: bool,
    report: &mut CheckReport,
) {
    if accel.is_cuda() {
        let marker = accel
            .cuda_version
            .as_ref()
            .map(|v| format!("CUDA_VERSION={v}"))
            .unwrap_or_else(|| "NVIDIA_REQUIRE_CUDA".to_owned());
        match admits_vendor(requires, "nvidia") {
            Some(false) => report.warnings.push(format!(
                "{scope} the image is a CUDA image ({marker}) but requires accel.vendor{}",
                requires["accel.vendor"]
            )),
            None if variant_scope => report.warnings.push(format!(
                "{scope} the image is a CUDA image ({marker}) but requires no accel.vendor; \
                 declare \"accel.vendor\": \"nvidia\" so the platform does not place it on \
                 other hardware"
            )),
            _ => {}
        }
        if requires.contains_key("accel.gfx") {
            report.warnings.push(format!(
                "{scope} requires accel.gfx (an AMD GPU target) on a CUDA image"
            ));
        }
        if let Some((floor, source)) = accel.cuda_floor() {
            let floor_requirement = Requirement::Compare(Comparator::Ge, floor.clone());
            match requires.get("accel.cuda") {
                Some(declared) if !satisfiable("accel.cuda", &[declared, &floor_requirement]) => {
                    report.failed.push(format!(
                        "{scope} requires accel.cuda{declared} but the image needs a host with \
                         CUDA >={floor} ({source}); no host satisfies both"
                    ));
                }
                Some(declared) => {
                    let below = lower_bound("accel.cuda", declared).is_none_or(|bound| {
                        compare(
                            &bound,
                            &ordered(Domain::Version, &floor).unwrap_or_default(),
                        )
                        .is_lt()
                    });
                    if below {
                        report.warnings.push(format!(
                            "{scope} requires accel.cuda{declared}, which admits hosts below \
                             the image's {source}; declare \">={floor}\""
                        ));
                    }
                }
                None if variant_scope => report.warnings.push(format!(
                    "{scope} the image needs a host with CUDA >={floor} ({source}) but \
                     requires no accel.cuda; declare \"accel.cuda\": \">={floor}\""
                )),
                None => {}
            }
        }
    }
    if let Some(rocm) = &accel.rocm_version {
        match admits_vendor(requires, "amd") {
            Some(false) => report.warnings.push(format!(
                "{scope} the image is a ROCm image (ROCM_VERSION={rocm}) but requires \
                 accel.vendor{}",
                requires["accel.vendor"]
            )),
            None if variant_scope => report.warnings.push(format!(
                "{scope} the image is a ROCm image (ROCM_VERSION={rocm}) but requires no \
                 accel.vendor; declare \"accel.vendor\": \"amd\""
            )),
            _ => {}
        }
        for key in ["accel.cuda", "accel.compute"] {
            if requires.contains_key(key) && !accel.is_cuda() {
                report.warnings.push(format!(
                    "{scope} requires {key} (an NVIDIA property) on a ROCm image"
                ));
            }
        }
    }
}

/// Validate a package label's `variant` and `profiles` blocks: the schema,
/// comparators, profile keys and the single default, whether each profile's
/// `requires` can hold together with the variant's, and whether the
/// `requires` agree with the CUDA/ROCm stack the image's `Env` declares. A
/// label with neither block reports nothing.
pub fn check_compute_blocks(package: &Map<String, Value>, config: &ImageConfig) -> CheckReport {
    let (blocks, mut report) = parse_compute_blocks(package);
    if blocks.is_empty() {
        return report;
    }
    let accel = ImageAccel::from_config(config);
    let no_requires = Requires::new();
    let variant_requires = blocks
        .variant
        .as_ref()
        .map_or(&no_requires, |variant| &variant.requires);
    if let Some(variant) = &blocks.variant {
        check_against_image("variant:", &variant.requires, &accel, true, &mut report);
    }
    for profile in &blocks.profiles {
        let scope = format!("profiles: {}:", profile.key);
        for (key, requirement) in &profile.requires {
            if let Some(inherited) = variant_requires.get(key)
                && !satisfiable(key, &[inherited, requirement])
            {
                report.failed.push(format!(
                    "{scope} requires {key}{requirement} but the variant requires \
                     {key}{inherited}; no host satisfies both"
                ));
            }
        }
        // Only what the profile itself declares: inherited entries were
        // judged at the variant scope.
        check_against_image(&scope, &profile.requires, &accel, false, &mut report);
    }
    let failed = |prefix: &str| report.failed.iter().any(|m| m.starts_with(prefix));
    let (variant_failed, profiles_failed) = (failed("variant:"), failed("profiles:"));
    if let Some(variant) = blocks.variant.as_ref().filter(|_| !variant_failed) {
        let mut details = Vec::new();
        if !variant.facets.is_empty() {
            details.push(format!("facets {}", format_facets(&variant.facets)));
        }
        if !variant.requires.is_empty() {
            details.push(format!("requires {}", format_requires(&variant.requires)));
        }
        if let Some(priority) = variant.priority {
            details.push(format!("priority {priority}"));
        }
        report.passed.push(format!(
            "variant: {}{}",
            variant.key.as_deref().unwrap_or("(no key)"),
            if details.is_empty() {
                String::new()
            } else {
                format!(" ({})", details.join("; "))
            }
        ));
    }
    if !blocks.profiles.is_empty() && !profiles_failed {
        report
            .passed
            .push(format!("profiles: {}", profile_summary(&blocks.profiles)));
    }
    report
}

/// `osmesa (default), egl`.
pub fn profile_summary(profiles: &[Profile]) -> String {
    profiles
        .iter()
        .map(|profile| {
            if profile.default {
                format!("{} (default)", profile.key)
            } else {
                profile.key.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Keys name workloads and appear in evaluation requests (`variant: ...`),
/// so they should read as DNS labels, like checkpoint names.
fn check_key(scope: &str, key: &str, report: &mut CheckReport) {
    if !is_dns_label(key) {
        report.warnings.push(format!(
            "{scope} key {key:?} is not a DNS label; keep keys to lowercase letters, digits, \
             and '-' (at most 63) so they can name workloads"
        ));
    }
}

fn is_dns_label(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 63
        && value
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        && !value.starts_with('-')
        && !value.ends_with('-')
}

fn is_env_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes
        .next()
        .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        && bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

fn warn_unknown_fields(
    scope: &str,
    raw: &Map<String, Value>,
    known: &[&str],
    report: &mut CheckReport,
) {
    let scope = scope.trim_end_matches(':');
    for field in raw.keys().filter(|field| !known.contains(&field.as_str())) {
        report.warnings.push(format!(
            "{scope}: unknown field {field:?}; the platform ignores it"
        ));
    }
}

fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn object(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(map) => map,
            other => panic!("not an object: {other}"),
        }
    }

    fn image(env: &[&str]) -> ImageConfig {
        ImageConfig {
            os: "linux".to_owned(),
            architecture: "amd64".to_owned(),
            env: env.iter().map(|e| (*e).to_owned()).collect(),
            ..ImageConfig::default()
        }
    }

    /// Assert each bucket holds exactly one message per needle, in order.
    fn assert_buckets(report: &CheckReport, failed: &[&str], warnings: &[&str]) {
        for (bucket, expected) in [(&report.failed, failed), (&report.warnings, warnings)] {
            assert_eq!(bucket.len(), expected.len(), "{report:#?}");
            for (message, needle) in bucket.iter().zip(expected) {
                assert!(message.contains(needle), "{needle:?} not in {message:?}");
            }
        }
    }

    /// The label from the schema this module documents.
    fn spec_package() -> Map<String, Value> {
        object(json!({
            "schemaVersion": 1,
            "variant": {
                "key": "torch-cuda12",
                "facets": {"framework": "torch", "accel": "nvidia", "render": "egl"},
                "requires": {
                    "accel.vendor": "nvidia",
                    "accel.compute": ">=8.0",
                    "accel.cuda": ">=12.2",
                    "accel.driver": ">=535",
                    "accel.vram_bytes": ">=24000000000"
                },
                "priority": 10
            },
            "profiles": [
                {"key": "osmesa", "default": true, "facets": {"render": "osmesa"},
                 "envVars": {"MUJOCO_GL": "osmesa"}},
                {"key": "egl", "facets": {"render": "egl"}, "envVars": {"MUJOCO_GL": "egl"},
                 "gpu": {"count": 1}, "requires": {"accel.vendor": "nvidia"}}
            ]
        }))
    }

    #[test]
    fn requirement_parsing_table() {
        let ok = |key: &str, value: Value, expected: Requirement| {
            assert_eq!(
                parse_requirement(key, &value),
                Ok(expected),
                "{key} {value}"
            );
        };
        let cmp = |c: Comparator, v: &str| Requirement::Compare(c, v.to_owned());
        ok(
            "accel.vendor",
            json!("nvidia"),
            cmp(Comparator::Eq, "nvidia"),
        );
        ok("accel.vendor", json!("=amd"), cmp(Comparator::Eq, "amd"));
        ok("accel.cuda", json!(">=12.2"), cmp(Comparator::Ge, "12.2"));
        ok(
            "accel.cuda",
            json!(" >= 12.2 "),
            cmp(Comparator::Ge, "12.2"),
        );
        ok("accel.compute", json!("<9.0"), cmp(Comparator::Lt, "9.0"));
        ok(
            "accel.driver",
            json!("535.104.05"),
            cmp(Comparator::Eq, "535.104.05"),
        );
        ok(
            "accel.vram_bytes",
            json!(">=24000000000"),
            cmp(Comparator::Ge, "24000000000"),
        );
        ok(
            "accel.gfx",
            json!(["gfx942", "gfx90a"]),
            Requirement::OneOf(vec!["gfx942".to_owned(), "gfx90a".to_owned()]),
        );
        ok(
            "accel.compute",
            json!(["8.0", "9.0"]),
            Requirement::OneOf(vec!["8.0".to_owned(), "9.0".to_owned()]),
        );
        for (key, value, needle) in [
            ("accel.memory", json!(">=1"), "unknown requires key"),
            ("accel.cuda", json!(">12"), "not supported"),
            ("accel.cuda", json!("<=12"), "not supported"),
            ("accel.cuda", json!("!=12"), "not supported"),
            (
                "accel.cuda",
                json!(">=twelve"),
                "not a dotted numeric version",
            ),
            (
                "accel.cuda",
                json!(">=1.2.3.4.5"),
                "not a dotted numeric version",
            ),
            ("accel.cuda", json!(">="), "empty value"),
            ("accel.vram_bytes", json!(">=24G"), "whole number of bytes"),
            (
                "accel.vram_bytes",
                json!(24_000_000_000_u64),
                "write a comparator string",
            ),
            ("accel.vendor", json!(">=nvidia"), "not a >= comparison"),
            ("accel.vendor", json!("nv idia"), "whitespace"),
            ("accel.gfx", json!([]), "empty list"),
            ("accel.gfx", json!([942]), "must be strings"),
            ("accel.compute", json!([">=8.0"]), "carries a comparator"),
            ("accel.vendor", json!(true), "comparator string or a list"),
        ] {
            let error = parse_requirement(key, &value).unwrap_err();
            assert!(error.contains(needle), "{key} {value}: {error}");
        }
    }

    #[test]
    fn requirements_combine_by_key() {
        let parse = |key: &str, value: Value| parse_requirement(key, &value).unwrap();
        let sat = |key: &str, values: &[Value]| {
            let parsed: Vec<Requirement> = values.iter().map(|v| parse(key, v.clone())).collect();
            satisfiable(key, &parsed.iter().collect::<Vec<_>>())
        };
        assert!(sat("accel.cuda", &[json!(">=12.2"), json!("<13")]));
        assert!(!sat("accel.cuda", &[json!(">=12.4"), json!("<12.4")]));
        assert!(!sat("accel.cuda", &[json!(">=12"), json!("<12")]));
        // Zero padding: 12 is 12.0.
        assert!(sat("accel.cuda", &[json!("=12"), json!(">=12.0")]));
        assert!(!sat("accel.cuda", &[json!("=12.1"), json!(">=12.2")]));
        assert!(sat(
            "accel.compute",
            &[json!(["8.0", "9.0"]), json!(">=8.6")]
        ));
        assert!(!sat(
            "accel.compute",
            &[json!(["7.5", "8.0"]), json!(">=8.6")]
        ));
        assert!(sat(
            "accel.vendor",
            &[json!(["nvidia", "amd"]), json!("amd")]
        ));
        assert!(!sat("accel.vendor", &[json!("nvidia"), json!("amd")]));
        assert!(!sat("accel.gfx", &[json!(["gfx942"]), json!(["gfx90a"])]));
        assert!(sat(
            "accel.vram_bytes",
            &[json!(">=24000000000"), json!("<80000000001")]
        ));
    }

    #[test]
    fn the_documented_label_is_clean() {
        let (blocks, report) = parse_compute_blocks(&spec_package());
        assert_eq!(report, CheckReport::default());
        let variant = blocks.variant.as_ref().unwrap();
        assert_eq!(variant.key.as_deref(), Some("torch-cuda12"));
        assert_eq!(variant.priority, Some(10));
        assert_eq!(variant.requires.len(), 5);
        assert_eq!(blocks.profiles.len(), 2);
        assert_eq!(
            blocks.default_profile().map(|p| p.key.as_str()),
            Some("osmesa")
        );
        let egl = &blocks.profiles[1];
        assert_eq!(egl.gpu_count, Some(1));
        assert_eq!(egl.env_vars["MUJOCO_GL"], "egl");
        // Against a CUDA 12.4 image that declares no stricter floor than 12.2.
        let config = image(&[
            "CUDA_VERSION=12.2.2",
            "NVIDIA_REQUIRE_CUDA=cuda>=12.2 brand=tesla",
        ]);
        let report = check_compute_blocks(&spec_package(), &config);
        assert_buckets(&report, &[], &[]);
        assert_eq!(report.passed.len(), 2, "{report:#?}");
        assert!(report.passed[0].starts_with("variant: torch-cuda12 (facets "));
        assert!(report.passed[0].contains("accel.cuda>=12.2"), "{report:#?}");
        assert_eq!(report.passed[1], "profiles: osmesa (default), egl");
    }

    #[test]
    fn a_label_without_the_blocks_reports_nothing() {
        let package = object(json!({"schemaVersion": 1, "tags": ["gpu"]}));
        let config = image(&["CUDA_VERSION=12.4.1"]);
        assert_eq!(
            check_compute_blocks(&package, &config),
            CheckReport::default()
        );
    }

    #[test]
    fn structure_table() {
        struct Case {
            name: &'static str,
            package: Value,
            failed: &'static [&'static str],
            warnings: &'static [&'static str],
        }
        let cases = [
            Case {
                name: "variant not an object",
                package: json!({"variant": "torch"}),
                failed: &["variant: must be a JSON object, got a string"],
                warnings: &[],
            },
            Case {
                name: "bad field types, unknown field, non-DNS key",
                package: json!({"variant": {"key": "Torch_CUDA", "facets": {"gpu": 1},
                    "priority": "high", "requires": {"accel.cuda": "~12"}, "tier": "a"}}),
                failed: &[
                    "variant: facets \"gpu\" must be a string",
                    "variant: requires accel.cuda value \"~12\" is not a dotted numeric version",
                    "variant: priority must be an integer",
                ],
                warnings: &[
                    "variant: key \"Torch_CUDA\" is not a DNS label",
                    "unknown field \"tier\"",
                ],
            },
            Case {
                name: "variant without a key",
                package: json!({"variant": {"facets": {"framework": "jax"}}}),
                failed: &[],
                warnings: &["variant: no key"],
            },
            Case {
                name: "unknown vendor",
                package: json!({"variant": {"key": "x", "requires": {"accel.vendor": ["nvidia", "qualcomm"]}}}),
                failed: &[],
                warnings: &["accel.vendor \"qualcomm\" is not one"],
            },
            Case {
                name: "profiles not an array",
                package: json!({"profiles": {"key": "egl"}}),
                failed: &["profiles: must be a JSON array, got an object"],
                warnings: &[],
            },
            Case {
                name: "duplicate keys",
                package: json!({"profiles": [{"key": "egl", "default": true}, {"key": "egl"}]}),
                failed: &["profiles: key \"egl\" appears more than once"],
                warnings: &[],
            },
            Case {
                name: "no default",
                package: json!({"profiles": [{"key": "osmesa"}, {"key": "egl"}]}),
                failed: &["no profile is marked \"default\": true"],
                warnings: &[],
            },
            Case {
                name: "two defaults",
                package: json!({"profiles": [{"key": "osmesa", "default": true},
                    {"key": "egl", "default": true}]}),
                failed: &["osmesa, egl are all marked default"],
                warnings: &[],
            },
            Case {
                name: "missing key, non-object entry, bad default",
                package: json!({"profiles": [{"default": true}, "egl",
                    {"key": "osmesa", "default": "yes"}]}),
                failed: &[
                    "profiles: entry 0 has no key",
                    "profiles: entry 1 must be a JSON object",
                    "profiles: osmesa: default must be true or false",
                    "no profile is marked",
                ],
                warnings: &[],
            },
            Case {
                name: "envVars and gpu shapes",
                package: json!({"profiles": [{"key": "egl", "default": true,
                    "envVars": {"MUJOCO_GL": "egl", "1BAD": "x", "N": 3, "RLMESH_ADDRESS": "0.0.0.0:1"},
                    "gpu": {"count": -1, "type": "h100"}}]}),
                failed: &[
                    "profiles: egl: envVars name \"1BAD\" is not a valid",
                    "profiles: egl: envVars N must be a string",
                    "profiles: egl: gpu.count must be a non-negative integer",
                ],
                warnings: &[
                    "profiles: egl: envVars sets RLMESH_ADDRESS",
                    "profiles: egl: gpu: unknown field \"type\"",
                ],
            },
        ];
        for case in cases {
            let (_, report) = parse_compute_blocks(&object(case.package));
            assert_eq!(
                report.failed.len(),
                case.failed.len(),
                "{}: {report:#?}",
                case.name
            );
            assert_eq!(
                report.warnings.len(),
                case.warnings.len(),
                "{}: {report:#?}",
                case.name
            );
            for (bucket, expected) in [
                (&report.failed, case.failed),
                (&report.warnings, case.warnings),
            ] {
                for (message, needle) in bucket.iter().zip(expected) {
                    assert!(
                        message.contains(needle),
                        "{}: {needle:?} not in {message:?}",
                        case.name
                    );
                }
            }
        }
    }

    #[test]
    fn requires_are_checked_against_the_image_env() {
        let variant =
            |requires: Value| object(json!({"variant": {"key": "v", "requires": requires}}));
        let cuda = image(&[
            "CUDA_VERSION=12.4.1",
            "NVIDIA_REQUIRE_CUDA=cuda>=12.4 brand=tesla,driver>=470,driver<471",
        ]);
        // Consistent.
        let report = check_compute_blocks(
            &variant(json!({"accel.vendor": "nvidia", "accel.cuda": ">=12.4"})),
            &cuda,
        );
        assert_buckets(&report, &[], &[]);
        // Admits hosts the NVIDIA runtime refuses.
        let report = check_compute_blocks(
            &variant(json!({"accel.vendor": "nvidia", "accel.cuda": ">=12.2"})),
            &cuda,
        );
        assert_buckets(
            &report,
            &[],
            &["admits hosts below the image's NVIDIA_REQUIRE_CUDA cuda>=12.4"],
        );
        // No host can satisfy both.
        let report = check_compute_blocks(
            &variant(json!({"accel.vendor": "nvidia", "accel.cuda": "<12"})),
            &cuda,
        );
        assert_buckets(
            &report,
            &["accel.cuda<12 but the image needs a host with CUDA >=12.4"],
            &[],
        );
        // Nothing declared on a CUDA image: say what to declare.
        let report = check_compute_blocks(&variant(json!({})), &cuda);
        assert_buckets(
            &report,
            &[],
            &[
                "requires no accel.vendor",
                "declare \"accel.cuda\": \">=12.4\"",
            ],
        );
        // Wrong vendor and an AMD target on a CUDA image.
        let report = check_compute_blocks(
            &variant(
                json!({"accel.vendor": "amd", "accel.gfx": ["gfx942"], "accel.cuda": ">=12.4"}),
            ),
            &cuda,
        );
        assert_buckets(
            &report,
            &[],
            &[
                "CUDA image (CUDA_VERSION=12.4.1) but requires accel.vendor=amd",
                "accel.gfx",
            ],
        );
        // CUDA_VERSION alone: its major.minor is the floor.
        let toolkit = image(&["CUDA_VERSION=12.1.0"]);
        let report = check_compute_blocks(
            &variant(json!({"accel.vendor": "nvidia", "accel.cuda": ">=11.8"})),
            &toolkit,
        );
        assert_buckets(
            &report,
            &[],
            &["below the image's CUDA_VERSION=12.1.0; declare \">=12.1\""],
        );
        // ROCm.
        let rocm = image(&["ROCM_VERSION=6.2.1"]);
        let report = check_compute_blocks(
            &variant(json!({"accel.vendor": "nvidia", "accel.cuda": ">=12"})),
            &rocm,
        );
        assert_buckets(
            &report,
            &[],
            &[
                "ROCm image (ROCM_VERSION=6.2.1) but requires accel.vendor=nvidia",
                "accel.cuda (an NVIDIA property)",
            ],
        );
        let report = check_compute_blocks(
            &variant(json!({"accel.vendor": "amd", "accel.gfx": ["gfx942", "gfx90a"]})),
            &rocm,
        );
        assert_buckets(&report, &[], &[]);
        // A CPU image is not judged for its requires.
        let report = check_compute_blocks(&variant(json!({"accel.cuda": ">=12"})), &image(&[]));
        assert_buckets(&report, &[], &[]);
    }

    #[test]
    fn profiles_must_hold_together_with_the_variant() {
        let package = object(json!({
            "variant": {"key": "torch-rocm6", "requires": {"accel.vendor": "amd"}},
            "profiles": [
                {"key": "osmesa", "default": true},
                {"key": "egl", "requires": {"accel.vendor": "nvidia"}},
            ]
        }));
        let report = check_compute_blocks(&package, &image(&[]));
        assert_buckets(
            &report,
            &[
                "profiles: egl: requires accel.vendor=nvidia but the variant requires accel.vendor=amd",
            ],
            &[],
        );
        // A profile on a CUDA image that pins another vendor is called out per profile.
        let package = object(json!({
            "variant": {"key": "v", "requires": {"accel.vendor": ["nvidia", "amd"], "accel.cuda": ">=12.4"}},
            "profiles": [
                {"key": "rocm", "default": true, "requires": {"accel.vendor": "amd"}},
                {"key": "plain"}
            ]
        }));
        let report = check_compute_blocks(&package, &image(&["NVIDIA_REQUIRE_CUDA=cuda>=12.4"]));
        // Inherited entries are judged once, at the variant; "plain" adds nothing.
        assert_buckets(&report, &[], &["profiles: rocm: the image is a CUDA image"]);
        assert_eq!(
            report.passed.last().unwrap(),
            "profiles: rocm (default), plain"
        );
        // A block that failed gets no summary line.
        let package = object(json!({"variant": {"key": "v", "priority": "x"},
            "profiles": [{"key": "a"}]}));
        let report = check_compute_blocks(&package, &image(&[]));
        assert_eq!(report.failed.len(), 2, "{report:#?}");
        assert!(report.passed.is_empty(), "{report:#?}");
    }

    #[test]
    fn image_accel_reads_the_cuda_floor() {
        let accel = ImageAccel::from_config(&image(&[
            "NVIDIA_REQUIRE_CUDA=brand=tesla,driver>=470 cuda>=12.4",
            "ROCM_VERSION=",
        ]));
        assert_eq!(accel.require_cuda.as_deref(), Some("12.4"));
        assert_eq!(accel.cuda_version, None);
        assert_eq!(accel.rocm_version, None);
    }
}
