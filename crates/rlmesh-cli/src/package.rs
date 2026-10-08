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
//!     "key": "cuda12",
//!     "facets": {"framework": "torch", "accel": "cuda"},
//!     "requires": {"accel.vendor": "nvidia", "accel.cuda": ">=12.4"},
//!     "priority": 10
//!   },
//!   "profiles": [
//!     {"key": "osmesa", "default": true, "envVars": {"MUJOCO_GL": "osmesa"}},
//!     {"key": "egl", "envVars": {"MUJOCO_GL": "egl"}, "gpu": {"count": 1}}
//!   ]
//! }
//! ```
//!
//! Each block is optional; a `variant` block needs a `key`. `facets` are
//! `framework` (a lowercase token), `accel` (`cpu`, `cuda`, `rocm`), and
//! `render` (`osmesa`, `egl`, `none`). `requires` keys are the fixed set
//! [`REQUIRE_KEYS`]: `accel.vendor` is `nvidia` or `amd`; `accel.compute`,
//! `accel.cuda`, and `accel.driver` are comma-joined version clauses
//! (`>=8.0,<10.0`, a bare version meaning a minimum); `accel.gfx` is a list of
//! AMD targets; `accel.vram_bytes` is a minimum in integer bytes; every key but
//! `accel.vendor` needs `accel.vendor`. A profile's `requires` and `facets`
//! override the variant's key by key. An image without a variant block gets
//! its requires inferred from its CUDA/ROCm markers ([`infer`]).
//!
//! The managed platform implements exactly these rules (its
//! `variant_requires_schema`, `variant_requires_vs_env`, and row naming), so
//! like [`crate::image_check`] the parsing rules, the row keys, and the
//! report's message prefixes (`variant:`, `profiles:`, `rows:`) are a
//! contract.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde_json::{Map, Value};

use crate::image_check::{ADDRESS_ENV, CheckReport, DESCRIBE_LABEL, ImageConfig, PACKAGE_LABEL};

/// The `requires` keys the platform matches hardware against.
pub const REQUIRE_KEYS: [&str; 6] = [
    "accel.vendor",
    "accel.compute",
    "accel.gfx",
    "accel.cuda",
    "accel.driver",
    "accel.vram_bytes",
];
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
/// `variant.priority` lies in `[-MAX_PRIORITY, MAX_PRIORITY]`.
pub const MAX_PRIORITY: i64 = 1000;
/// The most GPUs a profile may request.
pub const MAX_PROFILE_GPUS: u64 = 8;
/// The row key of an undeclared image.
pub const DEFAULT_VARIANT_KEY: &str = "default";
/// The package keys an index annotation may set for the whole version
/// (besides `schemaVersion` and `rev`); the annotation wins over a child.
pub const VERSION_PACKAGE_KEYS: [&str; 6] = [
    "name",
    "description",
    "checkpoints",
    "compatibility",
    "capabilities",
    "inputArtifacts",
];

const VARIANT_FIELDS: [&str; 4] = ["key", "facets", "requires", "priority"];
const PROFILE_FIELDS: [&str; 7] = [
    "key",
    "default",
    "facets",
    "envVars",
    "gpu",
    "requires",
    "resources",
];
/// `requires` keys only an NVIDIA GPU has.
const NVIDIA_KEYS: [&str; 3] = ["accel.compute", "accel.cuda", "accel.driver"];

/// One comparator in a version clause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Comparator {
    /// `>=X`, or a bare `X`.
    Ge,
    /// `>X`.
    Gt,
    /// `<=X`.
    Le,
    /// `<X`.
    Lt,
    /// `==X` or `=X`.
    Eq,
}

impl Comparator {
    fn symbol(self) -> &'static str {
        match self {
            Self::Ge => ">=",
            Self::Gt => ">",
            Self::Le => "<=",
            Self::Lt => "<",
            Self::Eq => "==",
        }
    }
}

/// One clause of a version constraint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clause {
    pub comparator: Comparator,
    /// Up to three dotted numeric parts.
    pub version: Vec<u64>,
}

/// One parsed `requires` entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Requirement {
    /// `accel.vendor`.
    Vendor(String),
    /// `accel.gfx`: the GPU's target must be one of these.
    Targets(Vec<String>),
    /// `accel.compute`, `accel.cuda`, `accel.driver`: every clause must hold.
    Version(Vec<Clause>),
    /// `accel.vram_bytes`: a minimum per GPU.
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
            Self::MinBytes(bytes) => write!(f, ">={bytes}"),
        }
    }
}

fn dotted(version: &[u64]) -> String {
    version
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(".")
}

/// `requires` entries by key, in key order.
pub type Requires = BTreeMap<String, Requirement>;

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

/// Parse a dotted numeric version of one to three parts.
fn parse_version(raw: &str) -> Option<Vec<u64>> {
    let parts: Vec<&str> = raw.split('.').collect();
    if parts.len() > 3
        || parts
            .iter()
            .any(|part| part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()))
    {
        return None;
    }
    parts.iter().map(|part| part.parse().ok()).collect()
}

fn compare(left: &[u64], right: &[u64]) -> std::cmp::Ordering {
    let len = left.len().max(right.len());
    (0..len)
        .map(|i| {
            let a = left.get(i).copied().unwrap_or(0);
            let b = right.get(i).copied().unwrap_or(0);
            a.cmp(&b)
        })
        .find(|ordering| ordering.is_ne())
        .unwrap_or(std::cmp::Ordering::Equal)
}

/// Parse a comma-joined version constraint: each clause an optional `>=`,
/// `>`, `<=`, `<`, `==`, or `=` and a version; a bare version is a minimum.
pub fn parse_constraint(raw: &str) -> Option<Vec<Clause>> {
    if raw.trim().is_empty() {
        return None;
    }
    raw.split(',')
        .map(|clause| {
            let clause = clause.trim();
            let (comparator, rest) = [
                (">=", Comparator::Ge),
                ("<=", Comparator::Le),
                ("==", Comparator::Eq),
                ("=", Comparator::Eq),
                (">", Comparator::Gt),
                ("<", Comparator::Lt),
            ]
            .iter()
            .find_map(|(symbol, comparator)| {
                clause.strip_prefix(symbol).map(|rest| (*comparator, rest))
            })
            .unwrap_or((Comparator::Ge, clause));
            Some(Clause {
                comparator,
                version: parse_version(rest.trim_start())?,
            })
        })
        .collect()
}

/// The lowest version a constraint admits, from its clauses that bound from
/// below (bare, `>=`, `>`, `==`); `None` when none does.
fn constraint_floor(clauses: &[Clause]) -> Option<&[u64]> {
    clauses
        .iter()
        .filter(|clause| !matches!(clause.comparator, Comparator::Lt | Comparator::Le))
        .map(|clause| clause.version.as_slice())
        .max_by(|a, b| compare(a, b))
}

/// Whether a constraint's upper bound already excludes `version`.
fn caps_below(clauses: &[Clause], version: &[u64]) -> bool {
    clauses.iter().any(|clause| match clause.comparator {
        Comparator::Lt => compare(&clause.version, version).is_le(),
        Comparator::Le => compare(&clause.version, version).is_lt(),
        _ => false,
    })
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
        "accel.vram_bytes" => value
            .as_u64()
            .filter(|bytes| *bytes > 0)
            .map(Requirement::MinBytes)
            .ok_or_else(|| {
                let hint = value
                    .as_str()
                    .map(|raw| {
                        format!(
                            " (write the minimum as a JSON number: {})",
                            raw.trim().trim_start_matches(">=").trim()
                        )
                    })
                    .unwrap_or_default();
                format!("accel.vram_bytes {value} is not a positive integer byte count{hint}")
            }),
        _ => Err(format!(
            "unknown key {key:?} (allowed: {})",
            REQUIRE_KEYS.join(", ")
        )),
    }
}

/// The vendor-coherence rules on one `requires` map: NVIDIA keys under `amd`,
/// `accel.gfx` under `nvidia`, and any hardware key without a vendor.
fn vendor_problems(keys: &BTreeSet<&str>, vendor: Option<&str>) -> Vec<String> {
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
                .chain(&["accel.gfx", "accel.vram_bytes"])
                .find(|key| keys.contains(*key))
            {
                problems.push(format!("{key} needs accel.vendor"));
            }
        }
    }
    problems
}

/// Parse a `requires` object as written: every entry and the vendor rules.
fn parse_requires(raw: &Value) -> (Requires, Vec<String>) {
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
                requires.insert(key.clone(), requirement);
            }
            Err(problem) => problems.push(problem),
        }
    }
    let keys: BTreeSet<&str> = entries.keys().map(String::as_str).collect();
    problems.extend(vendor_problems(
        &keys,
        entries.get("accel.vendor").and_then(Value::as_str),
    ));
    (requires, problems)
}

fn requires_vendor(requires: &Requires) -> Option<&str> {
    match requires.get("accel.vendor") {
        Some(Requirement::Vendor(vendor)) => Some(vendor),
        _ => None,
    }
}

/// Parse a `facets` object as written.
fn parse_facets(raw: &Value) -> (BTreeMap<String, String>, Vec<String>) {
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
fn accel_vendor(accel: &str) -> Option<&'static str> {
    match accel {
        "cuda" => Some("nvidia"),
        "rocm" => Some("amd"),
        _ => None,
    }
}

/// A parsed `variant` block.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Variant {
    pub key: Option<String>,
    pub facets: BTreeMap<String, String>,
    /// `None` when the block has no `requires`, so the platform infers them.
    pub requires: Option<Requires>,
    /// Defaults to 0.
    pub priority: i64,
}

/// A parsed `profiles[]` entry.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Profile {
    pub key: String,
    pub default: bool,
    pub facets: BTreeMap<String, String>,
    pub env_vars: BTreeMap<String, String>,
    pub gpu_count: Option<u64>,
    /// Overrides the variant's, key by key.
    pub requires: Requires,
    /// Whether the profile sets `resources`, which the platform validates
    /// against its own ceilings.
    pub resources: bool,
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

    /// The index of the default profile: the one marked, else the first.
    pub fn default_profile(&self) -> Option<usize> {
        if self.profiles.is_empty() {
            return None;
        }
        Some(self.profiles.iter().position(|p| p.default).unwrap_or(0))
    }

    /// The rows the platform derives from this image, given its variant key
    /// (declared, or synthesized by [`synth_key`]): one per profile, named
    /// `<variant.key>-<profile.key>` (the bare profile key without a variant
    /// block), else the one key.
    pub fn row_keys(&self, base: &str) -> Vec<String> {
        if self.profiles.is_empty() {
            return vec![base.to_owned()];
        }
        self.profiles
            .iter()
            .map(|profile| match &self.variant {
                Some(_) => format!("{base}-{}", profile.key),
                None => profile.key.clone(),
            })
            .collect()
    }
}

/// Parse the `variant` and `profiles` blocks out of a package label object,
/// reporting what the platform's `variant_requires_schema` fails on into the
/// returned report. Whatever parsed cleanly is kept, so one bad field does not
/// hide the rest.
pub fn parse_compute_blocks(package: &Map<String, Value>) -> (ComputeBlocks, CheckReport) {
    let mut report = CheckReport::default();
    let mut blocks = ComputeBlocks::default();
    match package.get("variant") {
        None => {}
        Some(Value::Object(raw)) => blocks.variant = Some(parse_variant(raw, &mut report)),
        Some(_) => report
            .failed
            .push("variant: variant is not an object".to_owned()),
    }
    match package.get("profiles") {
        None => {}
        Some(Value::Array(entries)) => {
            let mut seen = BTreeSet::new();
            let mut defaults = Vec::new();
            for (index, entry) in entries.iter().enumerate() {
                let Some(profile) = parse_profile(index, entry, &mut report) else {
                    continue;
                };
                if !seen.insert(profile.key.clone()) {
                    report.failed.push(format!(
                        "profiles: key {:?} appears more than once; profile keys must be unique",
                        profile.key
                    ));
                    continue;
                }
                if profile.default {
                    defaults.push(profile.key.clone());
                }
                blocks.profiles.push(profile);
            }
            match defaults.len() {
                0 if !blocks.profiles.is_empty() => report.warnings.push(format!(
                    "profiles: none is marked \"default\": true, so the first, {}, is the default; \
                     mark one to say so",
                    blocks.profiles[0].key
                )),
                0 | 1 => {}
                _ => report.failed.push(format!(
                    "profiles: {} are all marked default; at most one may be",
                    defaults.join(", ")
                )),
            }
        }
        Some(_) => report
            .failed
            .push("profiles: profiles is not a list".to_owned()),
    }
    (blocks, report)
}

fn parse_variant(raw: &Map<String, Value>, report: &mut CheckReport) -> Variant {
    let mut variant = Variant::default();
    match raw.get("key").and_then(Value::as_str) {
        Some(key) if valid_key(key) => variant.key = Some(key.to_owned()),
        Some(key) if !key.is_empty() => report.failed.push(format!(
            "variant: key {key:?} is not 1-32 lowercase letters, digits or dashes"
        )),
        _ => report
            .failed
            .push("variant: variant.key is required".to_owned()),
    }
    if let Some(facets) = raw.get("facets") {
        let (facets, problems) = parse_facets(facets);
        variant.facets = facets;
        report
            .failed
            .extend(problems.into_iter().map(|p| format!("variant: {p}")));
    }
    if let Some(requires) = raw.get("requires") {
        let (requires, problems) = parse_requires(requires);
        report.failed.extend(
            problems
                .into_iter()
                .map(|p| format!("variant: requires {p}")),
        );
        variant.requires = Some(requires);
    }
    if let Some(priority) = raw.get("priority") {
        match priority.as_i64().filter(|p| p.abs() <= MAX_PRIORITY) {
            Some(priority) => variant.priority = priority,
            None => report.failed.push(format!(
                "variant: priority {priority} is not an integer in [-{MAX_PRIORITY}, {MAX_PRIORITY}]"
            )),
        }
    }
    // A declared stack must agree with the declared vendor when both are set.
    if let (Some(accel), Some(requires)) = (variant.facets.get("accel"), raw.get("requires")) {
        let vendor = requires.get("accel.vendor").and_then(Value::as_str);
        if accel_vendor(accel) != vendor {
            report.failed.push(format!(
                "variant: facets.accel {accel} contradicts requires accel.vendor {}",
                vendor.unwrap_or("(none)")
            ));
        }
    }
    for field in raw
        .keys()
        .filter(|field| !VARIANT_FIELDS.contains(&field.as_str()))
    {
        report
            .failed
            .push(format!("variant: unknown key {field:?}"));
    }
    variant
}

fn parse_profile(index: usize, entry: &Value, report: &mut CheckReport) -> Option<Profile> {
    let Value::Object(raw) = entry else {
        report
            .failed
            .push(format!("profiles: profiles[{index}] is not an object"));
        return None;
    };
    let key = match raw.get("key").and_then(Value::as_str) {
        Some(key) if valid_key(key) => key.to_owned(),
        Some(key) if !key.is_empty() => {
            report.failed.push(format!(
                "profiles: profiles[{index}].key {key:?} is not 1-32 lowercase letters, digits or \
                 dashes"
            ));
            return None;
        }
        _ => {
            report
                .failed
                .push(format!("profiles: profiles[{index}].key is required"));
            return None;
        }
    };
    let scope = format!("profiles: {key}:");
    let mut profile = Profile {
        key,
        ..Profile::default()
    };
    match raw.get("default") {
        None => {}
        Some(Value::Bool(default)) => profile.default = *default,
        Some(other) => report
            .failed
            .push(format!("{scope} default {other} is not true or false")),
    }
    if let Some(facets) = raw.get("facets") {
        let (facets, problems) = parse_facets(facets);
        profile.facets = facets;
        report
            .failed
            .extend(problems.into_iter().map(|p| format!("{scope} {p}")));
    }
    match raw.get("envVars") {
        None => {}
        Some(Value::Object(vars)) => {
            for (name, value) in vars {
                let Some(value) = value.as_str() else {
                    report
                        .failed
                        .push(format!("{scope} envVars {name} {value} is not a string"));
                    continue;
                };
                if !is_env_name(name) {
                    report.warnings.push(format!(
                        "{scope} envVars name {name:?} is not a portable environment variable name"
                    ));
                }
                if name == ADDRESS_ENV {
                    report.warnings.push(format!(
                        "{scope} envVars sets {ADDRESS_ENV}; the platform assigns it per pod, so it \
                         has no effect there"
                    ));
                }
                profile.env_vars.insert(name.clone(), value.to_owned());
            }
        }
        Some(_) => report
            .failed
            .push(format!("{scope} envVars is not an object of strings")),
    }
    match raw.get("gpu") {
        None => {}
        Some(Value::Object(gpu)) => match gpu.get("count") {
            Some(count) => match count.as_u64().filter(|n| *n <= MAX_PROFILE_GPUS) {
                Some(count) => profile.gpu_count = Some(count),
                None => report.failed.push(format!(
                    "{scope} gpu.count {count} is not in [0, {MAX_PROFILE_GPUS}]"
                )),
            },
            None => report.failed.push(format!("{scope} gpu has no count")),
        },
        Some(_) => report.failed.push(format!("{scope} gpu is not an object")),
    }
    if let Some(requires) = raw.get("requires") {
        let (requires, problems) = parse_requires(requires);
        profile.requires = requires;
        report.failed.extend(
            problems
                .into_iter()
                .map(|p| format!("{scope} requires {p}")),
        );
    }
    profile.resources = raw.contains_key("resources");
    for field in raw
        .keys()
        .filter(|field| !PROFILE_FIELDS.contains(&field.as_str()))
    {
        report.failed.push(format!("{scope} unknown key {field:?}"));
    }
    Some(profile)
}

fn is_env_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes
        .next()
        .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        && bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

/// What an image says about its accelerator without declaring it, read the
/// way the platform reads an undeclared image.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Inferred {
    /// `accel` (`cuda`, `rocm`, `cpu`; absent when the image carries both
    /// CUDA and ROCm markers) and `framework` (from the describe label).
    pub facets: BTreeMap<String, String>,
    pub requires: Requires,
    /// The CUDA toolkit found, as major.minor.
    pub cuda: Option<String>,
    /// The ROCm version found, as major.minor.
    pub rocm: Option<String>,
    /// What the inference was read from, for the report.
    pub evidence: String,
}

impl Inferred {
    /// The key an undeclared child of an index is named for when its
    /// platform has several: `cuda12`, `rocm6`, `cpu`, else `default`.
    pub fn stack_key(&self) -> String {
        let major = |version: &Option<String>| {
            version
                .as_deref()
                .and_then(|v| v.split('.').next())
                .unwrap_or_default()
                .to_owned()
        };
        match self.facets.get("accel").map(String::as_str) {
            Some("cuda") => format!("cuda{}", major(&self.cuda)),
            Some("rocm") => format!("rocm{}", major(&self.rocm)),
            Some("cpu") => "cpu".to_owned(),
            _ => DEFAULT_VARIANT_KEY.to_owned(),
        }
    }
}

/// `12.4.1` → `12.4`; `None` when a part is not a number.
fn major_minor(version: &str) -> Option<String> {
    let parts: Vec<&str> = version
        .trim()
        .trim_start_matches('v')
        .split('.')
        .take(2)
        .collect();
    parts
        .iter()
        .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
        .then(|| parts.join("."))
}

/// `describe.runtime.framework_versions`, keys lowercased.
fn framework_versions(config: &ImageConfig) -> BTreeMap<String, String> {
    config
        .labels
        .get(DESCRIBE_LABEL)
        .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
        .and_then(|envelope| envelope.pointer("/runtime/framework_versions").cloned())
        .and_then(|versions| match versions {
            Value::Object(versions) => Some(versions),
            _ => None,
        })
        .map(|versions| {
            versions
                .into_iter()
                .filter_map(|(k, v)| v.as_str().map(|v| (k.to_lowercase(), v.to_owned())))
                .collect()
        })
        .unwrap_or_default()
}

/// The `framework_versions` packages that name each framework facet, in
/// precedence order for an image that ships more than one.
const FRAMEWORK_PACKAGES: [(&str, &str); 4] = [
    ("torch", "torch"),
    ("jax", "jax"),
    ("jaxlib", "jax"),
    ("tensorflow", "tensorflow"),
];

/// Infer an image's accelerator from its markers: `CUDA_VERSION` (else the
/// `cuda>=X.Y` in `NVIDIA_REQUIRE_CUDA`) means NVIDIA with `accel.cuda>=X.Y`;
/// `ROCM_VERSION` means AMD; without either, a torch build tag in the
/// describe label (`2.3.0+cu121`, `+rocm6.0`) decides; else a CPU image. An
/// image with both markers gets neither accel nor requires.
pub fn infer(config: &ImageConfig) -> Inferred {
    let env = |key: &str| {
        config
            .env_value(key)
            .map(str::trim)
            .filter(|v| !v.is_empty())
    };
    let mut out = Inferred::default();
    let mut evidence = Vec::new();
    if let Some(version) = env("CUDA_VERSION") {
        out.cuda = major_minor(version);
        evidence.push(format!("CUDA_VERSION={version}"));
    } else if let Some(floor) = env("NVIDIA_REQUIRE_CUDA").and_then(|value| {
        // The first `cuda>=X.Y` anywhere in the value, as the platform reads it.
        value.match_indices("cuda>=").find_map(|(at, _)| {
            let rest = &value[at + "cuda>=".len()..];
            let end = rest
                .find(|c: char| !c.is_ascii_digit() && c != '.')
                .unwrap_or(rest.len());
            let mut parts = rest[..end].split('.');
            match (parts.next(), parts.next()) {
                (Some(major), Some(minor)) if !major.is_empty() && !minor.is_empty() => {
                    let minor: String = minor.chars().take_while(char::is_ascii_digit).collect();
                    Some(format!("{major}.{minor}"))
                }
                _ => None,
            }
        })
    }) {
        evidence.push(format!("NVIDIA_REQUIRE_CUDA cuda>={floor}"));
        out.cuda = Some(floor);
    }
    if let Some(version) = env("ROCM_VERSION") {
        out.rocm = major_minor(version);
        evidence.push(format!("ROCM_VERSION={version}"));
    }
    let versions = framework_versions(config);
    if let Some((_, framework)) = FRAMEWORK_PACKAGES
        .iter()
        .find(|(package, _)| versions.contains_key(*package))
    {
        out.facets
            .insert("framework".to_owned(), (*framework).to_owned());
    }
    if out.cuda.is_none()
        && out.rocm.is_none()
        && let Some(torch) = versions.get("torch")
        && let Some((_, tag)) = torch.rsplit_once('+')
    {
        let (stack, version) = if let Some(v) = tag.strip_prefix("cu") {
            ("cu", v)
        } else if let Some(v) = tag.strip_prefix("rocm") {
            ("rocm", v)
        } else {
            ("", "")
        };
        if !version.is_empty() && version.bytes().all(|b| b.is_ascii_digit() || b == b'.') {
            let version = if stack == "cu" && version.len() >= 3 && !version.contains('.') {
                // cu121 is CUDA 12.1; cu118 is 11.8.
                format!(
                    "{}.{}",
                    &version[..version.len() - 1],
                    &version[version.len() - 1..]
                )
            } else {
                version.to_owned()
            };
            if stack == "cu" {
                out.cuda = major_minor(&version);
            } else if stack == "rocm" {
                out.rocm = major_minor(&version);
            }
            evidence.push(format!("torch {torch}"));
        }
    }
    match (&out.cuda, &out.rocm) {
        (Some(cuda), None) => {
            out.facets.insert("accel".to_owned(), "cuda".to_owned());
            out.requires.insert(
                "accel.vendor".to_owned(),
                Requirement::Vendor("nvidia".to_owned()),
            );
            if let Some(clauses) = parse_constraint(&format!(">={cuda}")) {
                out.requires
                    .insert("accel.cuda".to_owned(), Requirement::Version(clauses));
            }
        }
        (None, Some(_)) => {
            out.facets.insert("accel".to_owned(), "rocm".to_owned());
            out.requires.insert(
                "accel.vendor".to_owned(),
                Requirement::Vendor("amd".to_owned()),
            );
        }
        (None, None) => {
            out.facets.insert("accel".to_owned(), "cpu".to_owned());
            evidence.push("no CUDA or ROCm markers".to_owned());
        }
        // Both markers: a base that layers one toolkit on the other says
        // nothing reliable, so neither accel nor requires is guessed.
        (Some(_), Some(_)) => evidence.push("both CUDA and ROCm markers, so nothing".to_owned()),
    }
    out.evidence = evidence.join(", ");
    out
}

/// The requires and facets the platform records for the image's base row:
/// the declared requires when the variant block has them; else, when the
/// declared `facets.accel` names a stack the markers do not show, only the
/// vendor that stack implies; else the inferred ones.
pub fn effective(
    blocks: &ComputeBlocks,
    inferred: &Inferred,
) -> (Requires, BTreeMap<String, String>) {
    let Some(variant) = &blocks.variant else {
        return (inferred.requires.clone(), inferred.facets.clone());
    };
    let mut facets = inferred.facets.clone();
    facets.extend(variant.facets.clone());
    let requires = match (&variant.requires, variant.facets.get("accel")) {
        (Some(requires), _) => requires.clone(),
        (None, Some(accel)) if inferred.facets.get("accel") != Some(accel) => accel_vendor(accel)
            .map(|vendor| {
                Requires::from([(
                    "accel.vendor".to_owned(),
                    Requirement::Vendor(vendor.to_owned()),
                )])
            })
            .unwrap_or_default(),
        (None, _) => inferred.requires.clone(),
    };
    (requires, facets)
}

/// Whether the platform can select an image of this platform.
pub fn selectable(os: &str, architecture: &str) -> bool {
    os == "linux" && architecture == "amd64"
}

/// The key the platform gives an image with no variant block: `default` when
/// it is the only undeclared image on its platform (always, for a single
/// image), else named for its stack (`cuda12`, `rocm6`, `cpu`); a platform the
/// fleet cannot select is appended (`default-arm64v8`).
pub fn synth_key(
    inferred: &Inferred,
    os: &str,
    architecture: &str,
    platform_variant: &str,
    undeclared_on_platform: usize,
) -> String {
    let mut key = if undeclared_on_platform > 1 {
        inferred.stack_key()
    } else {
        DEFAULT_VARIANT_KEY.to_owned()
    };
    if !selectable(os, architecture) {
        let suffix: String = format!("{architecture}{platform_variant}")
            .to_lowercase()
            .chars()
            .filter(char::is_ascii_alphanumeric)
            .collect();
        if !suffix.is_empty() {
            key = format!("{key}-{suffix}");
        }
    }
    key
}

/// The image's package label, when it is a JSON object (a malformed label is
/// [`crate::image_check::check_labels`]' to report).
pub fn package_object(config: &ImageConfig) -> Option<Map<String, Value>> {
    match serde_json::from_str(config.labels.get(PACKAGE_LABEL)?) {
        Ok(Value::Object(package)) => Some(package),
        _ => None,
    }
}

/// Check an image's compute variant the way the platform will: the `variant`
/// and `profiles` blocks as written (`variant_requires_schema`, failures),
/// the declaration against the image's CUDA/ROCm markers and describe
/// (`variant_requires_vs_env`, warnings), and the row keys the platform
/// derives when the image is pushed alone. An image without a variant block
/// reports what is inferred for it.
pub fn check_variant(config: &ImageConfig) -> CheckReport {
    let package = package_object(config).unwrap_or_default();
    let (blocks, mut report) = parse_compute_blocks(&package);
    let inferred = infer(config);
    let (requires, facets) = effective(&blocks, &inferred);

    if let Some(variant) = &blocks.variant {
        check_against_markers(variant, config, &inferred, &mut report);
    }
    for profile in &blocks.profiles {
        // Profiles override key by key; the merged map must still hold together.
        if profile.requires.is_empty() {
            continue;
        }
        let mut merged = requires.clone();
        merged.extend(profile.requires.clone());
        let keys: BTreeSet<&str> = merged.keys().map(String::as_str).collect();
        for problem in vendor_problems(&keys, requires_vendor(&merged)) {
            report.warnings.push(format!(
                "profiles: {}: with the variant's requires it inherits, {problem}",
                profile.key
            ));
        }
        if let Some(vendor) = profile.requires.get("accel.vendor").and_then(|r| match r {
            Requirement::Vendor(v) => Some(v.as_str()),
            _ => None,
        }) {
            marker_vendor_warning(
                &format!("profiles: {}:", profile.key),
                vendor,
                &inferred,
                &mut report,
            );
        }
        if profile.resources {
            report.not_checked.push(format!(
                "profiles: {}: resources are validated by the platform against its ceilings",
                profile.key
            ));
        }
    }

    let variant_failed = report.failed.iter().any(|m| m.starts_with("variant:"));
    let profiles_failed = report.failed.iter().any(|m| m.starts_with("profiles:"));
    let mut details = Vec::new();
    if !facets.is_empty() {
        details.push(format!("facets {}", format_facets(&facets)));
    }
    details.push(if requires.is_empty() {
        "no requires (a CPU image)".to_owned()
    } else {
        format!("requires {}", format_requires(&requires))
    });
    match &blocks.variant {
        Some(variant) if !variant_failed => {
            if variant.priority != 0 {
                details.push(format!("priority {}", variant.priority));
            }
            let inferred_note = if variant.requires.is_none() && requires == inferred.requires {
                format!("; requires inferred from {}", inferred.evidence)
            } else {
                String::new()
            };
            report.passed.push(format!(
                "variant: {} ({}){inferred_note}",
                variant.key.as_deref().unwrap_or_default(),
                details.join("; ")
            ));
        }
        Some(_) => {}
        None => report.passed.push(format!(
            "variant: none declared; inferred from {}: {}",
            inferred.evidence,
            details.join("; ")
        )),
    }

    if !(variant_failed || profiles_failed) {
        let base = match blocks.variant.as_ref().and_then(|v| v.key.as_deref()) {
            Some(key) => key.to_owned(),
            None => synth_key(&inferred, &config.os, &config.architecture, "", 1),
        };
        let rows = blocks.row_keys(&base);
        let invalid: Vec<&String> = rows.iter().filter(|row| !valid_key(row)).collect();
        if invalid.is_empty() {
            let default = blocks.default_profile().unwrap_or(0);
            let excluded = if selectable(&config.os, &config.architecture) {
                ""
            } else {
                " (excluded: not linux/amd64)"
            };
            report.passed.push(format!(
                "rows: {}{excluded}",
                rows.iter()
                    .enumerate()
                    .map(|(i, row)| if rows.len() > 1 && i == default {
                        format!("{row} (default)")
                    } else {
                        row.clone()
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        } else {
            for row in invalid {
                report.failed.push(format!(
                    "rows: row key {row:?} (variant.key-profile.key) is not 1-32 lowercase letters, \
                     digits or dashes; shorten the keys"
                ));
            }
        }
    }
    report
}

fn marker_vendor_warning(scope: &str, vendor: &str, inferred: &Inferred, report: &mut CheckReport) {
    match (vendor, &inferred.cuda, &inferred.rocm) {
        ("nvidia", None, Some(rocm)) => report.warnings.push(format!(
            "{scope} requires accel.vendor nvidia but the image carries ROCm {rocm}"
        )),
        ("amd", Some(cuda), None) => report.warnings.push(format!(
            "{scope} requires accel.vendor amd but the image carries CUDA {cuda}"
        )),
        _ => {}
    }
}

/// `variant_requires_vs_env`: a declaration the image's own markers
/// contradict. Warnings only: the markers are evidence, and a probe on real
/// hardware settles it.
fn check_against_markers(
    variant: &Variant,
    config: &ImageConfig,
    inferred: &Inferred,
    report: &mut CheckReport,
) {
    let declared = variant.requires.clone().unwrap_or_default();
    if let Some(vendor) = requires_vendor(&declared) {
        marker_vendor_warning("variant:", vendor, inferred, report);
    }
    if let (Some(Requirement::Version(clauses)), Some(cuda)) =
        (declared.get("accel.cuda"), &inferred.cuda)
        && let Some(image) = parse_version(cuda)
    {
        let requirement = &declared["accel.cuda"];
        if constraint_floor(clauses).is_some_and(|floor| compare(floor, &image).is_lt()) {
            report.warnings.push(format!(
                "variant: requires accel.cuda{requirement} admits drivers older than the image's \
                 CUDA {cuda} runtime needs; declare \">={cuda}\""
            ));
        } else if caps_below(clauses, &image) {
            report.warnings.push(format!(
                "variant: requires accel.cuda{requirement} admits no driver that can run the \
                 image's CUDA {cuda} runtime"
            ));
        }
    }
    if let (Some(declared), Some(built)) =
        (variant.facets.get("accel"), inferred.facets.get("accel"))
        && built != "cpu"
        && declared != built
    {
        report.warnings.push(format!(
            "variant: facets.accel {declared} but the image is built on {built}"
        ));
    }
    if let Some(framework) = variant.facets.get("framework") {
        let versions = framework_versions(config);
        let named = FRAMEWORK_PACKAGES
            .iter()
            .any(|(package, facet)| facet == framework && versions.contains_key(*package));
        if !versions.is_empty() && !named && !versions.contains_key(framework) {
            report.warnings.push(format!(
                "variant: facets.framework {framework} is not among describe.runtime.framework_versions"
            ));
        }
    }
}

/// Check an index annotation's `dev.rlmesh.package` object: it may carry only
/// the version-level keys ([`VERSION_PACKAGE_KEYS`], plus `schemaVersion` and
/// `rev`); the platform ignores anything else, which belongs on the child
/// images. Returns the warnings.
pub fn index_annotation_warnings(package: &Map<String, Value>) -> Vec<String> {
    let ignored: Vec<&str> = package
        .keys()
        .map(String::as_str)
        .filter(|key| {
            *key != "schemaVersion" && *key != "rev" && !VERSION_PACKAGE_KEYS.contains(key)
        })
        .collect();
    if ignored.is_empty() {
        return Vec::new();
    }
    vec![format!(
        "index annotation keys ignored (they belong on the child images): {}",
        ignored.join(", ")
    )]
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

    fn image(env: &[&str], package: Option<Value>) -> ImageConfig {
        let mut config = ImageConfig {
            os: "linux".to_owned(),
            architecture: "amd64".to_owned(),
            env: env.iter().map(|e| (*e).to_owned()).collect(),
            ..ImageConfig::default()
        };
        if let Some(package) = package {
            config
                .labels
                .insert(PACKAGE_LABEL.to_owned(), package.to_string());
        }
        config
    }

    fn with_torch(mut config: ImageConfig, versions: Value) -> ImageConfig {
        config.labels.insert(
            DESCRIBE_LABEL.to_owned(),
            json!({"schema_version": 1, "runtime": {"framework_versions": versions}}).to_string(),
        );
        config
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

    /// The label docs/compute-variants.md documents.
    fn documented() -> Value {
        json!({
            "schemaVersion": 1,
            "variant": {
                "key": "cuda12",
                "facets": {"framework": "torch", "accel": "cuda", "render": "egl"},
                "requires": {
                    "accel.vendor": "nvidia",
                    "accel.compute": ">=8.0,<10.0",
                    "accel.cuda": ">=12.4",
                    "accel.driver": "550",
                    "accel.vram_bytes": 24000000000_u64
                },
                "priority": 10
            },
            "profiles": [
                {"key": "osmesa", "default": true, "facets": {"render": "osmesa"},
                 "envVars": {"MUJOCO_GL": "osmesa"}},
                {"key": "egl", "facets": {"render": "egl"}, "envVars": {"MUJOCO_GL": "egl"},
                 "gpu": {"count": 1}}
            ]
        })
    }

    #[test]
    fn constraints_parse_like_the_platform() {
        let parse = |raw: &str| {
            parse_constraint(raw).map(|clauses| Requirement::Version(clauses).to_string())
        };
        assert_eq!(parse(">=8.0,<10.0").as_deref(), Some(">=8.0,<10.0"));
        // A bare version is a minimum.
        assert_eq!(parse("12.4").as_deref(), Some(">=12.4"));
        assert_eq!(parse(" >= 12.2 ").as_deref(), Some(">=12.2"));
        assert_eq!(parse("==12.4").as_deref(), Some("==12.4"));
        assert_eq!(parse("=12").as_deref(), Some("==12"));
        assert_eq!(parse(">12,<=12.6.1").as_deref(), Some(">12,<=12.6.1"));
        for bad in [
            "", " ", "12.4.1.2", "~12", ">=", "12,", "abc", "!=12", "12.x",
        ] {
            assert_eq!(parse(bad), None, "{bad:?}");
        }
        let floor = |raw: &str| {
            let clauses = parse_constraint(raw).unwrap();
            constraint_floor(&clauses).map(dotted)
        };
        assert_eq!(floor(">=12.2,>12.4,<13").as_deref(), Some("12.4"));
        assert_eq!(floor("<13"), None);
        assert_eq!(floor("==12.1").as_deref(), Some("12.1"));
    }

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
                "write the minimum as a JSON number: 24000000000",
            ),
            ("accel.vram_bytes", json!(0), "not a positive integer"),
            ("accel.vram_bytes", json!(1.5), "not a positive integer"),
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
    fn the_documented_label_is_clean() {
        let package = object(documented());
        let (blocks, report) = parse_compute_blocks(&package);
        assert_eq!(report, CheckReport::default());
        let variant = blocks.variant.as_ref().unwrap();
        assert_eq!(variant.priority, 10);
        assert_eq!(variant.requires.as_ref().unwrap().len(), 5);
        assert_eq!(blocks.row_keys("cuda12"), ["cuda12-osmesa", "cuda12-egl"]);
        let config = image(&["CUDA_VERSION=12.4.1"], Some(documented()));
        let report = check_variant(&config);
        assert_buckets(&report, &[], &[]);
        assert_eq!(
            report.passed,
            [
                "variant: cuda12 (facets accel=cuda, framework=torch, render=egl; requires \
                 accel.compute>=8.0,<10.0, accel.cuda>=12.4, accel.driver>=550, \
                 accel.vendor=nvidia, accel.vram_bytes>=24000000000; priority 10)",
                "rows: cuda12-osmesa (default), cuda12-egl",
            ]
        );
    }

    #[test]
    fn structure_table() {
        let long = "a".repeat(33);
        let cases: Vec<(&str, Value, Vec<&str>, Vec<&str>)> = vec![
            (
                "variant not an object",
                json!({"variant": "torch"}),
                vec!["variant: variant is not an object"],
                vec![],
            ),
            (
                "missing key",
                json!({"variant": {"facets": {"framework": "jax"}}}),
                vec!["variant: variant.key is required"],
                vec![],
            ),
            (
                "bad key, facets, priority, unknown field",
                json!({"variant": {"key": "Torch_CUDA", "facets": {"accel": "nvidia", "render": "x",
                    "framework": "Torch", "gpu": "a100"}, "priority": 2000, "tier": "a"}}),
                vec![
                    "variant: key \"Torch_CUDA\" is not 1-32 lowercase",
                    "facets.accel \"nvidia\" is not cpu, cuda or rocm",
                    "facets.framework \"Torch\" is not a lowercase token",
                    "unknown facet \"gpu\" (allowed: framework, accel, render)",
                    "facets.render \"x\" is not osmesa, egl or none",
                    "variant: priority 2000 is not an integer in [-1000, 1000]",
                    "variant: unknown key \"tier\"",
                ],
                vec![],
            ),
            (
                "key too long",
                json!({"variant": {"key": long}}),
                vec!["is not 1-32 lowercase"],
                vec![],
            ),
            (
                "fractional priority",
                json!({"variant": {"key": "a", "priority": 1.5}}),
                vec!["priority 1.5 is not an integer"],
                vec![],
            ),
            (
                "accel contradicts vendor",
                json!({"variant": {"key": "a", "facets": {"accel": "cuda"}, "requires": {"accel.vendor": "amd"}}}),
                vec!["facets.accel cuda contradicts requires accel.vendor amd"],
                vec![],
            ),
            (
                "accel without the vendor it implies",
                json!({"variant": {"key": "a", "facets": {"accel": "rocm"}, "requires": {}}}),
                vec!["facets.accel rocm contradicts requires accel.vendor (none)"],
                vec![],
            ),
            (
                "cpu accel with a vendor",
                json!({"variant": {"key": "a", "facets": {"accel": "cpu"}, "requires": {"accel.vendor": "nvidia"}}}),
                vec!["facets.accel cpu contradicts requires accel.vendor nvidia"],
                vec![],
            ),
            (
                "profiles not a list",
                json!({"profiles": {"key": "egl"}}),
                vec!["profiles: profiles is not a list"],
                vec![],
            ),
            (
                "duplicate keys",
                json!({"profiles": [{"key": "egl", "default": true}, {"key": "egl"}]}),
                vec!["profiles: key \"egl\" appears more than once"],
                vec![],
            ),
            (
                "no default is allowed: the first is",
                json!({"profiles": [{"key": "osmesa"}, {"key": "egl"}]}),
                vec![],
                vec!["the first, osmesa, is the default"],
            ),
            (
                "two defaults",
                json!({"profiles": [{"key": "osmesa", "default": true}, {"key": "egl", "default": true}]}),
                vec!["osmesa, egl are all marked default; at most one may be"],
                vec![],
            ),
            (
                "entries, keys, defaults",
                json!({"profiles": [{"default": true}, "egl", {"key": "Bad"}, {"key": "ok", "default": "yes"}]}),
                vec![
                    "profiles: profiles[0].key is required",
                    "profiles: profiles[1] is not an object",
                    "profiles: profiles[2].key \"Bad\" is not 1-32",
                    "profiles: ok: default \"yes\" is not true or false",
                ],
                vec!["the first, ok, is the default"],
            ),
            (
                "envVars, gpu, requires, unknown field",
                json!({"profiles": [{"key": "egl", "default": true,
                    "envVars": {"MUJOCO_GL": "egl", "1BAD": "x", "N": 3, "RLMESH_ADDRESS": "0.0.0.0:1"},
                    "gpu": {"count": 9}, "requires": {"accel.cuda": ">=12"}, "resources": {"memory": "8Gi"},
                    "type": "h100"}]}),
                vec![
                    "profiles: egl: envVars N 3 is not a string",
                    "profiles: egl: gpu.count 9 is not in [0, 8]",
                    "profiles: egl: requires accel.cuda needs accel.vendor",
                    "profiles: egl: unknown key \"type\"",
                ],
                vec![
                    "profiles: egl: envVars name \"1BAD\" is not a portable",
                    "profiles: egl: envVars sets RLMESH_ADDRESS",
                ],
            ),
        ];
        for (name, package, failed, warnings) in cases {
            let (_, report) = parse_compute_blocks(&object(package));
            assert_eq!(report.failed.len(), failed.len(), "{name}: {report:#?}");
            assert_eq!(report.warnings.len(), warnings.len(), "{name}: {report:#?}");
            for (message, needle) in report.failed.iter().zip(&failed) {
                assert!(
                    message.contains(needle),
                    "{name}: {needle:?} not in {message:?}"
                );
            }
            for (message, needle) in report.warnings.iter().zip(&warnings) {
                assert!(
                    message.contains(needle),
                    "{name}: {needle:?} not in {message:?}"
                );
            }
        }
    }

    #[test]
    fn inference_table() {
        let cases: Vec<(&str, ImageConfig, Option<&str>, &str, &str)> = vec![
            (
                "CUDA_VERSION",
                image(&["CUDA_VERSION=12.4.1"], None),
                Some("cuda"),
                "accel.cuda>=12.4, accel.vendor=nvidia",
                "cuda12",
            ),
            (
                "CUDA_VERSION wins over NVIDIA_REQUIRE_CUDA",
                image(
                    &["CUDA_VERSION=12.4.1", "NVIDIA_REQUIRE_CUDA=cuda>=12.2"],
                    None,
                ),
                Some("cuda"),
                "accel.cuda>=12.4, accel.vendor=nvidia",
                "cuda12",
            ),
            (
                "NVIDIA_REQUIRE_CUDA",
                image(
                    &["NVIDIA_REQUIRE_CUDA=brand=tesla,driver>=470 cuda>=11.8"],
                    None,
                ),
                Some("cuda"),
                "accel.cuda>=11.8, accel.vendor=nvidia",
                "cuda11",
            ),
            (
                "ROCM_VERSION",
                image(&["ROCM_VERSION=6.2.1"], None),
                Some("rocm"),
                "accel.vendor=amd",
                "rocm6",
            ),
            (
                "torch +cu121",
                with_torch(image(&[], None), json!({"torch": "2.3.0+cu121"})),
                Some("cuda"),
                "accel.cuda>=12.1, accel.vendor=nvidia",
                "cuda12",
            ),
            (
                "torch +cu118",
                with_torch(image(&[], None), json!({"Torch": "2.1.0+cu118"})),
                Some("cuda"),
                "accel.cuda>=11.8, accel.vendor=nvidia",
                "cuda11",
            ),
            (
                "torch +rocm6.0",
                with_torch(image(&[], None), json!({"torch": "2.3.0+rocm6.0"})),
                Some("rocm"),
                "accel.vendor=amd",
                "rocm6",
            ),
            (
                "torch +cpu",
                with_torch(image(&[], None), json!({"torch": "2.3.0+cpu"})),
                Some("cpu"),
                "",
                "cpu",
            ),
            (
                "nothing",
                image(&["PATH=/bin"], None),
                Some("cpu"),
                "",
                "cpu",
            ),
            (
                "both markers",
                image(&["CUDA_VERSION=12.4", "ROCM_VERSION=6.2"], None),
                None,
                "",
                "default",
            ),
        ];
        for (name, config, accel, requires, stack) in cases {
            let inferred = infer(&config);
            assert_eq!(
                inferred.facets.get("accel").map(String::as_str),
                accel,
                "{name}"
            );
            assert_eq!(format_requires(&inferred.requires), requires, "{name}");
            assert_eq!(inferred.stack_key(), stack, "{name}");
        }
        // The env markers beat the torch tag; the framework facet comes from describe.
        let inferred = infer(&with_torch(
            image(&["ROCM_VERSION=6.2"], None),
            json!({"jaxlib": "0.4.30", "torch": "2.3.0+cu121"}),
        ));
        assert_eq!(inferred.facets["accel"], "rocm");
        assert_eq!(inferred.facets["framework"], "torch");
        assert_eq!(inferred.evidence, "ROCM_VERSION=6.2");
    }

    #[test]
    fn undeclared_images_report_what_is_inferred() {
        let report = check_variant(&image(&["CUDA_VERSION=12.4.1"], None));
        assert_buckets(&report, &[], &[]);
        assert_eq!(
            report.passed,
            [
                "variant: none declared; inferred from CUDA_VERSION=12.4.1: facets accel=cuda; \
                 requires accel.cuda>=12.4, accel.vendor=nvidia",
                "rows: default",
            ]
        );
        let report = check_variant(&image(&[], None));
        assert_eq!(
            report.passed[0],
            "variant: none declared; inferred from no CUDA or ROCm markers: facets accel=cpu; no \
             requires (a CPU image)"
        );
        // A declared variant without requires gets the inferred ones.
        let report = check_variant(&image(
            &["ROCM_VERSION=6.2.1"],
            Some(
                json!({"schemaVersion": 1, "variant": {"key": "rocm6", "facets": {"framework": "torch"}}}),
            ),
        ));
        assert_eq!(
            report.passed[0],
            "variant: rocm6 (facets accel=rocm, framework=torch; requires accel.vendor=amd); \
             requires inferred from ROCM_VERSION=6.2.1"
        );
        // Profiles without a variant block are rows named for the profile.
        let mut config = image(
            &[],
            Some(
                json!({"schemaVersion": 1, "profiles": [{"key": "osmesa"}, {"key": "egl", "default": true}]}),
            ),
        );
        let report = check_variant(&config);
        assert_eq!(report.passed[1], "rows: osmesa, egl (default)");
        // A platform the fleet cannot select is named for it and excluded.
        config.architecture = "arm64".to_owned();
        config.labels.clear();
        assert_eq!(
            check_variant(&config).passed[1],
            "rows: default-arm64 (excluded: not linux/amd64)"
        );
    }

    #[test]
    fn declarations_are_checked_against_the_markers() {
        let variant = |requires: Value, facets: Value| json!({"schemaVersion": 1, "variant": {"key": "v", "requires": requires, "facets": facets}});
        let cuda = |package: Value| image(&["CUDA_VERSION=12.4.1"], Some(package));
        let report = check_variant(&cuda(variant(
            json!({"accel.vendor": "nvidia", "accel.cuda": "12.4"}),
            json!({}),
        )));
        assert_buckets(&report, &[], &[]);
        let report = check_variant(&cuda(variant(
            json!({"accel.vendor": "nvidia", "accel.cuda": ">=12.2"}),
            json!({}),
        )));
        assert_buckets(
            &report,
            &[],
            &["accel.cuda>=12.2 admits drivers older than the image's CUDA 12.4 runtime needs"],
        );
        let report = check_variant(&cuda(variant(
            json!({"accel.vendor": "nvidia", "accel.cuda": "<12"}),
            json!({}),
        )));
        assert_buckets(
            &report,
            &[],
            &["accel.cuda<12 admits no driver that can run the image's CUDA 12.4"],
        );
        let report = check_variant(&cuda(variant(
            json!({"accel.vendor": "amd"}),
            json!({"accel": "rocm"}),
        )));
        assert_buckets(
            &report,
            &[],
            &[
                "requires accel.vendor amd but the image carries CUDA 12.4",
                "facets.accel rocm but the image is built on cuda",
            ],
        );
        let rocm = image(
            &["ROCM_VERSION=6.2"],
            Some(variant(json!({"accel.vendor": "nvidia"}), json!({}))),
        );
        assert_buckets(
            &check_variant(&rocm),
            &[],
            &["requires accel.vendor nvidia but the image carries ROCm 6.2"],
        );
        let jax = with_torch(
            cuda(variant(
                json!({"accel.vendor": "nvidia", "accel.cuda": ">=12.4"}),
                json!({"framework": "jax"}),
            )),
            json!({"torch": "2.3.0"}),
        );
        assert_buckets(
            &check_variant(&jax),
            &[],
            &["facets.framework jax is not among describe.runtime.framework_versions"],
        );
        // A declared stack the markers do not show keeps only its vendor.
        let package = object(json!({"variant": {"key": "v", "facets": {"accel": "rocm"}}}));
        let (blocks, _) = parse_compute_blocks(&package);
        let (requires, facets) = effective(&blocks, &infer(&image(&["CUDA_VERSION=12.4.1"], None)));
        assert_eq!(format_requires(&requires), "accel.vendor=amd");
        assert_eq!(facets["accel"], "rocm");
    }

    #[test]
    fn profiles_override_and_must_still_hold_together() {
        let package = json!({
            "schemaVersion": 1,
            "variant": {"key": "cuda12", "requires": {"accel.vendor": "nvidia", "accel.cuda": ">=12.4"}},
            "profiles": [
                {"key": "osmesa", "default": true},
                {"key": "rocm", "requires": {"accel.vendor": "amd"}, "resources": {"memory": "8Gi"}},
            ]
        });
        let report = check_variant(&image(&["CUDA_VERSION=12.4.1"], Some(package)));
        assert_buckets(
            &report,
            &[],
            &[
                "profiles: rocm: with the variant's requires it inherits, accel.cuda is an NVIDIA requirement",
                "profiles: rocm: requires accel.vendor amd but the image carries CUDA 12.4",
            ],
        );
        assert_eq!(report.not_checked.len(), 1, "{report:#?}");
        assert!(report.not_checked[0].contains("resources are validated by the platform"));
    }

    #[test]
    fn row_keys_must_fit_the_key_pattern() {
        let package = json!({"schemaVersion": 1, "variant": {"key": "torch-cuda12-ampere-plus"},
            "profiles": [{"key": "osmesa-software", "default": true}]});
        let report = check_variant(&image(&[], Some(package)));
        assert_buckets(
            &report,
            &["rows: row key \"torch-cuda12-ampere-plus-osmesa-software\""],
            &[],
        );
        assert!(
            report.passed.iter().all(|m| !m.starts_with("rows:")),
            "{report:#?}"
        );
    }

    #[test]
    fn synthesized_keys() {
        let cuda = infer(&image(&["CUDA_VERSION=12.4.1"], None));
        assert_eq!(synth_key(&cuda, "linux", "amd64", "", 1), "default");
        assert_eq!(synth_key(&cuda, "linux", "amd64", "", 2), "cuda12");
        assert_eq!(
            synth_key(&cuda, "linux", "arm64", "v8", 1),
            "default-arm64v8"
        );
        assert_eq!(
            synth_key(&cuda, "linux", "arm64", "v8", 2),
            "cuda12-arm64v8"
        );
    }

    #[test]
    fn index_annotation_keys() {
        assert!(
            index_annotation_warnings(&object(json!({"schemaVersion": 1, "rev": 2, "name": "pi0",
            "description": "", "checkpoints": [], "compatibility": {}, "capabilities": [],
            "inputArtifacts": []})))
            .is_empty()
        );
        assert_eq!(
            index_annotation_warnings(&object(json!({"name": "pi0", "variant": {}, "tags": []}))),
            ["index annotation keys ignored (they belong on the child images): tags, variant"]
        );
    }
}
