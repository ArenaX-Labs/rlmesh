//! The `variant` and `profiles` blocks as written, and the version-level keys
//! of an index annotation.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};

use super::requires::{Requires, accel_vendor, parse_facets, parse_requires, valid_key};
use crate::image_check::{ADDRESS_ENV, CheckReport};

/// `variant.priority` lies in `[-MAX_PRIORITY, MAX_PRIORITY]`.
pub const MAX_PRIORITY: i64 = 1000;
/// The most GPUs a profile may request.
pub const MAX_PROFILE_GPUS: u64 = 8;

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
    /// (declared, or synthesized by [`synth_key`](super::synth_key)): one per profile, named
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
    if let Some(raw_requires) = raw.get("requires") {
        let (requires, problems) = parse_requires(raw_requires);
        report.failed.extend(
            problems
                .into_iter()
                .map(|p| format!("variant: requires {p}")),
        );
        variant.requires = Some(requires);
    }
    if let Some(priority) = raw.get("priority") {
        match priority
            .as_i64()
            .filter(|p| (-MAX_PRIORITY..=MAX_PRIORITY).contains(p))
        {
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
    let key = profile_key(index, raw, report)?;
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
    if let Some(vars) = raw.get("envVars") {
        profile.env_vars = parse_env_vars(&scope, vars, report);
    }
    if let Some(gpu) = raw.get("gpu") {
        profile.gpu_count = parse_gpu(&scope, gpu, report);
    }
    if let Some(raw_requires) = raw.get("requires") {
        let (requires, problems) = parse_requires(raw_requires);
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

/// A profile needs a valid `key`; without one the entry is dropped.
fn profile_key(index: usize, raw: &Map<String, Value>, report: &mut CheckReport) -> Option<String> {
    match raw.get("key").and_then(Value::as_str) {
        Some(key) if valid_key(key) => Some(key.to_owned()),
        Some(key) if !key.is_empty() => {
            report.failed.push(format!(
                "profiles: profiles[{index}].key {key:?} is not 1-32 lowercase letters, digits or \
                 dashes"
            ));
            None
        }
        _ => {
            report
                .failed
                .push(format!("profiles: profiles[{index}].key is required"));
            None
        }
    }
}

/// `envVars` is an object of strings; names should be portable, and the
/// platform's own address variable has no effect.
fn parse_env_vars(scope: &str, vars: &Value, report: &mut CheckReport) -> BTreeMap<String, String> {
    let mut env_vars = BTreeMap::new();
    let Value::Object(vars) = vars else {
        report
            .failed
            .push(format!("{scope} envVars is not an object of strings"));
        return env_vars;
    };
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
        env_vars.insert(name.clone(), value.to_owned());
    }
    env_vars
}

/// `gpu` is an object whose `count` lies in `[0, MAX_PROFILE_GPUS]`.
fn parse_gpu(scope: &str, gpu: &Value, report: &mut CheckReport) -> Option<u64> {
    let Value::Object(gpu) = gpu else {
        report.failed.push(format!("{scope} gpu is not an object"));
        return None;
    };
    let Some(count) = gpu.get("count") else {
        report.failed.push(format!("{scope} gpu has no count"));
        return None;
    };
    let valid = count.as_u64().filter(|n| *n <= MAX_PROFILE_GPUS);
    if valid.is_none() {
        report.failed.push(format!(
            "{scope} gpu.count {count} is not in [0, {MAX_PROFILE_GPUS}]"
        ));
    }
    valid
}

fn is_env_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes
        .next()
        .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        && bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_')
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
    use crate::variant::testing::*;
    use serde_json::json;

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
                "priority as a string",
                json!({"variant": {"key": "a", "priority": "10"}}),
                vec!["priority \"10\" is not an integer"],
                vec![],
            ),
            (
                "priority at the lower bound",
                json!({"variant": {"key": "a", "priority": -1000}}),
                vec![],
                vec![],
            ),
            (
                "priority at the upper bound",
                json!({"variant": {"key": "a", "priority": 1000}}),
                vec![],
                vec![],
            ),
            (
                "priority just past the bounds",
                json!({"variant": {"key": "a", "priority": 1001}}),
                vec!["priority 1001 is not an integer in [-1000, 1000]"],
                vec![],
            ),
            (
                "priority at i64::MIN does not overflow",
                json!({"variant": {"key": "a", "priority": i64::MIN}}),
                vec!["priority -9223372036854775808 is not an integer in [-1000, 1000]"],
                vec![],
            ),
            (
                "priority at i64::MAX",
                json!({"variant": {"key": "a", "priority": i64::MAX}}),
                vec!["priority 9223372036854775807 is not an integer in [-1000, 1000]"],
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
                "vram quantity",
                json!({"variant": {"key": "a", "requires": {"accel.vendor": "amd", "accel.vram": "192Gi"}}}),
                vec![],
                vec![],
            ),
            (
                "vram as a number",
                json!({"variant": {"key": "a", "requires": {"accel.vendor": "amd", "accel.vram": 16}}}),
                vec![
                    "variant: requires accel.vram 16 is not a quantity string; write it as a \
                      quantity string, e.g. \"16Gi\"",
                ],
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

    #[test]
    fn profile_gpu_and_env_vars_must_be_objects() {
        let mut report = CheckReport::default();
        assert_eq!(parse_gpu("p:", &json!(1), &mut report), None);
        assert_eq!(parse_gpu("p:", &json!({}), &mut report), None);
        assert_eq!(parse_gpu("p:", &json!({"count": -1}), &mut report), None);
        assert_eq!(parse_gpu("p:", &json!({"count": 2}), &mut report), Some(2));
        assert_eq!(
            parse_gpu("p:", &json!({"count": MAX_PROFILE_GPUS}), &mut report),
            Some(MAX_PROFILE_GPUS)
        );
        assert!(parse_env_vars("p:", &json!(["MUJOCO_GL=egl"]), &mut report).is_empty());
        assert_eq!(
            parse_env_vars("p:", &json!({"MUJOCO_GL": "egl"}), &mut report),
            BTreeMap::from([("MUJOCO_GL".to_owned(), "egl".to_owned())])
        );
        assert_eq!(
            report.failed,
            [
                "p: gpu is not an object",
                "p: gpu has no count",
                "p: gpu.count -1 is not in [0, 8]",
                "p: envVars is not an object of strings",
            ]
        );
        assert!(report.warnings.is_empty(), "{report:#?}");
    }
}
