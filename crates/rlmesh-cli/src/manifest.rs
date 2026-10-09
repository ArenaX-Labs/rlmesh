//! `rlmesh.toml`: a version's builds and its `dev.rlmesh.package` label,
//! written once in TOML instead of as JSON in a Dockerfile.
//!
//! The top level is the package label itself, under the label's own key
//! names (`name`, `[[checkpoints]]`, `envVars`, ...), plus the build keys
//! `dockerfile`, `context` and `build_args`. Each `[variant.KEY]` table is one
//! build, in priority order (the first is the version's default): its
//! `accel.*` keys are the variant's `requires`, under the same names, and it
//! may set `facets`, `priority`, its own build keys, and a `package` table
//! overriding top-level package keys for that build. `[profile.KEY]` tables
//! are shared by every variant. A file with no `[variant.*]` table is one
//! build, keyed `default`, whose `accel.*` keys sit at the top level; with
//! none, its label has no variant block at all, as a hand-written one need
//! not.
//!
//! Each variant's label is rendered as JSON and checked by the same
//! [`parse_compute_blocks`] that checks a hand-written label, so the TOML
//! accepts exactly what the label does.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value};

use crate::variant::{
    ComputeBlocks, DEFAULT_VARIANT_KEY, VERSION_PACKAGE_KEYS, parse_compute_blocks,
};

/// The file `rlmesh registry publish` reads when given no SOURCE.
pub const FILE_NAME: &str = "rlmesh.toml";

const BUILD_KEYS: [&str; 3] = ["dockerfile", "context", "build_args"];
const VARIANT_KEYS: [&str; 7] = [
    "dockerfile",
    "context",
    "build_args",
    "accel",
    "facets",
    "priority",
    "package",
];
const PROFILE_KEYS: [&str; 6] = ["default", "accel", "facets", "envVars", "gpu", "resources"];

/// A parsed `rlmesh.toml`.
#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    /// The builds, in priority order; the first is the version's default.
    pub variants: Vec<VariantBuild>,
    /// The version-level package keys (see [`VERSION_PACKAGE_KEYS`]) set at
    /// the top level, for the index annotation; empty when none are.
    pub version: Map<String, Value>,
}

/// One variant: how to build it and the label it carries.
#[derive(Debug, Clone, PartialEq)]
pub struct VariantBuild {
    pub key: String,
    /// Resolved against the manifest's directory.
    pub dockerfile: PathBuf,
    /// Resolved against the manifest's directory.
    pub context: PathBuf,
    pub build_args: BTreeMap<String, String>,
    /// The full `dev.rlmesh.package` label.
    pub label: Map<String, Value>,
    /// The label's `variant` and `profiles` blocks as parsed.
    pub blocks: ComputeBlocks,
}

/// Read and check the manifest at `path` (`-` reads stdin, relative to the
/// current directory). Returns it with its warnings; every problem found
/// fails it at once.
pub fn load(path: &Path) -> Result<(Manifest, Vec<String>)> {
    let (raw, dir) = if path == Path::new("-") {
        let mut raw = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut raw)
            .context("reading the manifest from stdin")?;
        (raw, PathBuf::new())
    } else {
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let dir = path.parent().map_or_else(PathBuf::new, Path::to_path_buf);
        (raw, dir)
    };
    let origin = if path == Path::new("-") {
        "stdin".to_owned()
    } else {
        path.display().to_string()
    };
    parse(&raw, &dir, &origin)
}

/// Parse a manifest whose relative paths resolve against `dir`; `origin`
/// prefixes every message.
pub fn parse(raw: &str, dir: &Path, origin: &str) -> Result<(Manifest, Vec<String>)> {
    let table: toml::Table =
        toml::from_str(raw).with_context(|| format!("{origin} is not valid TOML"))?;
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    let mut package = Map::new();
    let mut build = toml::Table::new();
    let mut variants = None;
    let mut profiles = None;
    let mut top_accel = None;
    for (key, value) in table {
        match key.as_str() {
            "variant" => variants = Some(value),
            "profile" => profiles = Some(value),
            "accel" => top_accel = Some(value),
            "profiles" => {
                errors.push("profiles: write each profile as a [profile.KEY] table".to_owned())
            }
            "schemaVersion" if value.as_integer() != Some(1) => {
                errors.push(format!(
                    "schemaVersion {value} is not the supported version 1"
                ));
            }
            key if BUILD_KEYS.contains(&key) => {
                build.insert(key.to_owned(), value);
            }
            _ => {
                package.insert(key, json(value));
            }
        }
    }
    package.insert("schemaVersion".to_owned(), Value::from(1));
    let defaults = Build::parse("", &build, None, dir, &mut errors);

    let profiles = match profiles {
        None => Vec::new(),
        Some(toml::Value::Table(profiles)) => profiles
            .into_iter()
            .map(|(key, value)| profile(&key, value, &mut errors))
            .collect(),
        Some(_) => {
            errors.push("profile: write each profile as a [profile.KEY] table".to_owned());
            Vec::new()
        }
    };

    // A single build that declares no requirements writes no variant block,
    // as a hand-written label would: the platform keys it `default` and
    // names its profiles' rows by their bare keys (`egl`, not `default-egl`).
    let undeclared = variants.is_none() && top_accel.is_none();
    let tables: Vec<(String, toml::Table)> = match variants {
        None => {
            let mut table = toml::Table::new();
            if let Some(accel) = top_accel.take() {
                table.insert("accel".to_owned(), accel);
            }
            vec![(DEFAULT_VARIANT_KEY.to_owned(), table)]
        }
        Some(toml::Value::Table(variants)) if !variants.is_empty() => variants
            .into_iter()
            .filter_map(|(key, value)| match value {
                toml::Value::Table(table) => Some((key, table)),
                _ => {
                    errors.push(format!(
                        "variant.{key}: write it as a [variant.{key}] table"
                    ));
                    None
                }
            })
            .collect(),
        Some(_) => {
            errors.push("variant: write each variant as a [variant.KEY] table".to_owned());
            Vec::new()
        }
    };
    if top_accel.is_some() {
        errors.push(
            "accel: with [variant.*] tables, each variant sets its own accel.* keys".to_owned(),
        );
    }

    let mut built = Vec::new();
    for (key, mut table) in tables {
        let at = format!("variant.{key}");
        for field in table.keys().filter(|f| !VARIANT_KEYS.contains(&f.as_str())) {
            errors.push(format!("{at}: unknown key {field:?}"));
        }
        let build = Build::parse(&at, &table, Some(&defaults), dir, &mut errors);
        let mut label = package.clone();
        if let Some(overrides) = table.remove("package") {
            match json(overrides) {
                Value::Object(overrides) => {
                    for (name, value) in overrides {
                        if VERSION_PACKAGE_KEYS.contains(&name.as_str()) {
                            errors.push(format!(
                                "{at}.package: {name} is the whole version's, so it is set \
                                 once at the top level"
                            ));
                        } else if matches!(name.as_str(), "variant" | "profiles" | "schemaVersion")
                        {
                            errors.push(format!("{at}.package: {name} cannot be overridden"));
                        } else {
                            label.insert(name, value);
                        }
                    }
                }
                _ => errors.push(format!("{at}.package: write it as a [{at}.package] table")),
            }
        }
        let mut variant = Map::new();
        variant.insert("key".to_owned(), Value::from(key.clone()));
        for field in ["facets", "priority"] {
            if let Some(value) = table.remove(field) {
                variant.insert(field.to_owned(), json(value));
            }
        }
        if let Some(accel) = table.remove("accel") {
            variant.insert("requires".to_owned(), requires(&at, accel, &mut errors));
        }
        if !undeclared {
            label.insert("variant".to_owned(), Value::Object(variant));
        }
        if !profiles.is_empty() {
            label.insert("profiles".to_owned(), Value::Array(profiles.clone()));
        }

        let (blocks, report) = parse_compute_blocks(&label);
        // A profile's message is the same under every variant: say it once.
        let said = |list: &mut Vec<String>, message: String| {
            let message = match message.strip_prefix("variant: ") {
                Some(rest) => format!("{at}: {rest}"),
                None => message,
            };
            if !list.contains(&message) {
                list.push(message);
            }
        };
        for message in report.failed {
            said(&mut errors, message);
        }
        for message in report.warnings {
            said(&mut warnings, message);
        }
        built.push(VariantBuild {
            key,
            dockerfile: build.dockerfile,
            context: build.context,
            build_args: build.args,
            label,
            blocks,
        });
    }

    if !errors.is_empty() {
        bail!(
            "{origin} has {} problem{}:\n{}",
            errors.len(),
            if errors.len() == 1 { "" } else { "s" },
            errors
                .iter()
                .map(|error| format!("  - {error}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
    let version = package
        .iter()
        .filter(|(key, _)| VERSION_PACKAGE_KEYS.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let warnings = warnings
        .into_iter()
        .map(|warning| format!("{origin}: {warning}"))
        .collect();
    Ok((
        Manifest {
            variants: built,
            version,
        },
        warnings,
    ))
}

/// The build keys of the top level (`defaults` absent) or of one variant,
/// which falls back to the top level's, key by key for `build_args`.
struct Build {
    dockerfile: PathBuf,
    context: PathBuf,
    args: BTreeMap<String, String>,
}

impl Build {
    fn parse(
        at: &str,
        table: &toml::Table,
        defaults: Option<&Build>,
        dir: &Path,
        errors: &mut Vec<String>,
    ) -> Build {
        let name = |field: &str| {
            if at.is_empty() {
                field.to_owned()
            } else {
                format!("{at}.{field}")
            }
        };
        let path = |field: &str, errors: &mut Vec<String>| match table.get(field) {
            None => None,
            Some(toml::Value::String(path)) if !path.is_empty() => Some(dir.join(path)),
            Some(_) => {
                errors.push(format!("{}: must be a path", name(field)));
                None
            }
        };
        let dockerfile = path("dockerfile", errors)
            .or_else(|| defaults.map(|d| d.dockerfile.clone()))
            .unwrap_or_else(|| dir.join("Dockerfile"));
        let context = path("context", errors)
            .or_else(|| defaults.map(|d| d.context.clone()))
            .unwrap_or_else(|| {
                if dir.as_os_str().is_empty() {
                    PathBuf::from(".")
                } else {
                    dir.to_path_buf()
                }
            });
        let mut args = defaults.map(|d| d.args.clone()).unwrap_or_default();
        match table.get("build_args") {
            None => {}
            Some(toml::Value::Table(values)) => {
                for (key, value) in values {
                    match value {
                        toml::Value::String(value) => {
                            args.insert(key.clone(), value.clone());
                        }
                        _ => errors.push(format!("{}.{key}: must be a string", name("build_args"))),
                    }
                }
            }
            Some(_) => errors.push(format!("{}: must be a table", name("build_args"))),
        }
        Build {
            dockerfile,
            context,
            args,
        }
    }
}

/// One `[profile.KEY]` table as a `profiles[]` entry.
fn profile(key: &str, value: toml::Value, errors: &mut Vec<String>) -> Value {
    let at = format!("profile.{key}");
    let toml::Value::Table(table) = value else {
        errors.push(format!("{at}: write it as a [{at}] table"));
        return Value::Null;
    };
    let mut entry = Map::new();
    entry.insert("key".to_owned(), Value::from(key));
    for (field, value) in table {
        match field.as_str() {
            "accel" => {
                entry.insert("requires".to_owned(), requires(&at, value, errors));
            }
            field if PROFILE_KEYS.contains(&field) => {
                entry.insert(field.to_owned(), json(value));
            }
            _ => errors.push(format!("{at}: unknown key {field:?}")),
        }
    }
    Value::Object(entry)
}

/// An `accel` table as the label's `requires`: `accel.vendor = "nvidia"`
/// is the requirement `"accel.vendor": "nvidia"`.
fn requires(at: &str, accel: toml::Value, errors: &mut Vec<String>) -> Value {
    let toml::Value::Table(accel) = accel else {
        errors.push(format!(
            "{at}.accel: write requirements as accel.KEY = VALUE (e.g. accel.vendor = \"nvidia\")"
        ));
        return Value::Object(Map::new());
    };
    Value::Object(
        accel
            .into_iter()
            .map(|(key, value)| (format!("accel.{key}"), json(value)))
            .collect(),
    )
}

fn json(value: toml::Value) -> Value {
    match value {
        toml::Value::String(s) => Value::from(s),
        toml::Value::Integer(i) => Value::from(i),
        toml::Value::Float(f) => Value::from(f),
        toml::Value::Boolean(b) => Value::from(b),
        toml::Value::Datetime(d) => Value::from(d.to_string()),
        toml::Value::Array(items) => Value::Array(items.into_iter().map(json).collect()),
        toml::Value::Table(table) => {
            Value::Object(table.into_iter().map(|(k, v)| (k, json(v))).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ok(raw: &str) -> (Manifest, Vec<String>) {
        parse(raw, Path::new("app"), "rlmesh.toml").unwrap()
    }

    fn problems(raw: &str) -> String {
        format!(
            "{:#}",
            parse(raw, Path::new("app"), "rlmesh.toml").unwrap_err()
        )
    }

    const TWO: &str = r#"
name = "pi0 LIBERO"
tags = ["pi0"]
dockerfile = "Dockerfile.gpu"
build_args.PYTHON = "3.11"

[[checkpoints]]
name = "libero"
uri = "hf://model/acme/pi0-libero"
default = true

[variant.cuda12]
build_args.TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
accel.vendor = "nvidia"
accel.compute = ">=8.0,<10.0"
accel.vram = "16Gi"
facets.framework = "torch"

[variant.cpu]
dockerfile = "Dockerfile.cpu"
priority = -5
package.tags = ["pi0", "cpu"]

[profile.osmesa]
default = true
envVars.MUJOCO_GL = "osmesa"

[profile.egl]
accel.vendor = "nvidia"
facets.render = "egl"
envVars.MUJOCO_GL = "egl"
gpu.count = 1
"#;

    #[test]
    fn each_variant_renders_the_label_a_hand_written_one_would_be() {
        let (manifest, warnings) = ok(TWO);
        assert!(warnings.is_empty(), "{warnings:?}");
        let keys: Vec<&str> = manifest.variants.iter().map(|v| v.key.as_str()).collect();
        assert_eq!(keys, ["cuda12", "cpu"], "table order is priority order");

        let profiles = json!([
            {"key": "osmesa", "default": true, "envVars": {"MUJOCO_GL": "osmesa"}},
            {"key": "egl", "requires": {"accel.vendor": "nvidia"}, "facets": {"render": "egl"},
             "envVars": {"MUJOCO_GL": "egl"}, "gpu": {"count": 1}}
        ]);
        let checkpoints =
            json!([{"name": "libero", "uri": "hf://model/acme/pi0-libero", "default": true}]);
        assert_eq!(
            Value::Object(manifest.variants[0].label.clone()),
            json!({
                "schemaVersion": 1, "name": "pi0 LIBERO", "tags": ["pi0"],
                "checkpoints": checkpoints,
                "variant": {"key": "cuda12", "facets": {"framework": "torch"},
                            "requires": {"accel.vendor": "nvidia", "accel.compute": ">=8.0,<10.0",
                                         "accel.vram": "16Gi"}},
                "profiles": profiles,
            })
        );
        assert_eq!(
            Value::Object(manifest.variants[1].label.clone()),
            json!({
                "schemaVersion": 1, "name": "pi0 LIBERO", "tags": ["pi0", "cpu"],
                "checkpoints": checkpoints,
                "variant": {"key": "cpu", "priority": -5},
                "profiles": profiles,
            })
        );
        assert_eq!(
            Value::Object(manifest.version),
            json!({"name": "pi0 LIBERO", "checkpoints": checkpoints})
        );
        assert_eq!(
            manifest.variants[0].blocks.row_keys("cuda12"),
            ["cuda12-osmesa", "cuda12-egl"]
        );
    }

    #[test]
    fn build_keys_fall_back_to_the_top_level_and_resolve_against_its_directory() {
        let (manifest, _) = ok(TWO);
        let (cuda, cpu) = (&manifest.variants[0], &manifest.variants[1]);
        assert_eq!(cuda.dockerfile, Path::new("app/Dockerfile.gpu"));
        assert_eq!(cpu.dockerfile, Path::new("app/Dockerfile.cpu"));
        assert_eq!(cuda.context, Path::new("app"));
        assert_eq!(
            cuda.build_args,
            BTreeMap::from([
                ("PYTHON".to_owned(), "3.11".to_owned()),
                (
                    "TORCH_INDEX".to_owned(),
                    "https://download.pytorch.org/whl/cu126".to_owned()
                ),
            ])
        );
        assert_eq!(
            cpu.build_args,
            BTreeMap::from([("PYTHON".to_owned(), "3.11".to_owned())])
        );
    }

    #[test]
    fn a_file_without_variants_is_one_default_build() {
        let (manifest, _) =
            ok("name = \"pi0\"\naccel.vendor = \"nvidia\"\naccel.vram = \"24Gi\"\n");
        assert_eq!(manifest.variants.len(), 1);
        let only = &manifest.variants[0];
        assert_eq!(only.key, "default");
        assert_eq!(only.dockerfile, Path::new("app/Dockerfile"));
        assert_eq!(
            only.label["variant"],
            json!({"key": "default", "requires": {"accel.vendor": "nvidia", "accel.vram": "24Gi"}})
        );

        let (manifest, _) = ok("");
        assert_eq!(
            Value::Object(manifest.variants[0].label.clone()),
            json!({"schemaVersion": 1})
        );
        assert!(manifest.version.is_empty());
    }

    #[test]
    fn a_single_build_without_requirements_keeps_bare_profile_rows() {
        let (manifest, _) = ok("[profile.osmesa]\ndefault = true\n[profile.egl]\ngpu.count = 1\n");
        let only = &manifest.variants[0];
        assert_eq!(only.key, "default");
        assert!(!only.label.contains_key("variant"), "{:?}", only.label);
        assert_eq!(only.blocks.row_keys(&only.key), ["osmesa", "egl"]);
    }

    #[test]
    fn every_problem_is_reported_against_its_toml_path() {
        let message = problems(
            r#"
accel.vendor = "nvidia"
[variant.cuda12]
accel.vendor = "nvidia"
accel.cuda = "12.4 or newer"
accel.vram = "16GB"
dockerfil = "Dockerfile"
package.name = "other"
[variant.Bad]
[profile.egl]
envVars.MUJOCO_GL = "egl"
gpus = 1
"#,
        );
        for expected in [
            "accel: with [variant.*] tables",
            "variant.cuda12: unknown key \"dockerfil\"",
            "variant.cuda12: requires accel.cuda",
            "variant.cuda12: requires accel.vram",
            "variant.cuda12.package: name is the whole version's",
            "variant.Bad: key \"Bad\" is not 1-32 lowercase",
            "profile.egl: unknown key \"gpus\"",
        ] {
            assert!(
                message.contains(expected),
                "{expected:?} not in:\n{message}"
            );
        }
        assert!(
            message.starts_with("rlmesh.toml has 7 problems"),
            "{message}"
        );
    }

    #[test]
    fn a_profile_problem_is_said_once_not_per_variant() {
        let message = problems(
            "[variant.a]\n[variant.b]\n[profile.x]\ndefault = true\n[profile.y]\ndefault = true\n",
        );
        assert_eq!(
            message.matches("are all marked default").count(),
            1,
            "{message}"
        );
    }

    #[test]
    fn hand_written_label_blocks_are_refused_for_their_tables() {
        let message = problems("profiles = []\nvariant = \"cpu\"\nschemaVersion = 2\n");
        assert!(message.contains("[profile.KEY]"), "{message}");
        assert!(message.contains("[variant.KEY]"), "{message}");
        assert!(message.contains("schemaVersion 2"), "{message}");
    }
}
