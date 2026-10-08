//! Checking the children as one version: each package label, the set's row
//! keys and kinds, and the index annotation.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value};

use super::media::OCI_INDEX;
use super::source::Child;
use crate::image_check::{
    self, CheckReport, DESCRIBE_LABEL, ImageConfig, Kind, PACKAGE_LABEL, parse_serve_command,
};
use crate::variant::{self, ComputeBlocks, Requires};

/// A child with its package label read.
#[derive(Debug, Clone)]
pub(crate) struct VariantImage {
    pub child: Child,
    pub key: String,
    pub kind: Option<Kind>,
    pub blocks: ComputeBlocks,
    /// The requires the platform records for the base row: declared, or
    /// inferred from the image's markers.
    pub requires: Requires,
    pub requires_inferred: bool,
    pub facets: BTreeMap<String, String>,
    /// The row keys the platform derives (one per profile).
    pub rows: Vec<String>,
}

impl VariantImage {
    pub(super) fn selectable(&self) -> bool {
        self.child.selectable()
    }
}

/// What the image serves: the describe label's kind, else the serve command's.
fn image_kind(config: &ImageConfig) -> Option<Kind> {
    config
        .labels
        .get(DESCRIBE_LABEL)
        .and_then(|raw| image_check::parse_describe_label(raw).ok())
        .and_then(|label| label.kind)
        .or_else(|| parse_serve_command(config).kind())
}

/// The version's default row: the first `linux/amd64` child, at its default
/// profile, as `(variant index, row index)`.
pub(crate) fn default_row(variants: &[VariantImage]) -> Option<(usize, usize)> {
    variants
        .iter()
        .position(VariantImage::selectable)
        .map(|index| (index, variants[index].blocks.default_profile().unwrap_or(0)))
}

/// Read and check each child's package label, and the set as a whole: every
/// child needs a label with a `variant.key`, its declaration must pass the
/// platform's checks, row keys are unique, kinds agree, and at least one
/// child is `linux/amd64`. Returns the variants, the warnings worth printing,
/// and every error found (all of them, so one run reports everything to fix).
pub(crate) fn check_variants(
    children: Vec<Child>,
) -> (Vec<VariantImage>, Vec<String>, Vec<String>) {
    let mut variants = Vec::new();
    let mut warnings = Vec::new();
    let mut errors = Vec::new();
    for child in children {
        variants.extend(check_child(child, &mut warnings, &mut errors));
    }
    errors.extend(duplicate_keys(&variants));
    errors.extend(colliding_rows(&variants));
    errors.extend(mixed_kinds(&variants));
    errors.extend(unschedulable(&variants));
    (variants, warnings, errors)
}

/// Check one child on its own, appending its problems in order; `None` when
/// it has no usable `variant.key`.
fn check_child(
    child: Child,
    warnings: &mut Vec<String>,
    errors: &mut Vec<String>,
) -> Option<VariantImage> {
    let source = child.source.clone();
    let package = match package_object(&child) {
        Ok(package) => package,
        Err(error) => {
            errors.push(error);
            return None;
        }
    };
    errors.extend(unsupported_schema_version(&source, &package));
    let report: CheckReport = variant::check_variant(&child.config);
    errors.extend(report.failed.iter().map(|m| format!("{source}: {m}")));
    warnings.extend(report.warnings.iter().map(|m| format!("{source}: {m}")));
    let (blocks, _) = variant::parse_compute_blocks(&package);
    let Some(key) = blocks.variant.as_ref().and_then(|v| v.key.clone()) else {
        errors.extend(missing_variant_block(&source, &blocks));
        return None;
    };
    let kind = image_kind(&child.config);
    warnings.extend(unknown_kind(&source, kind));
    warnings.extend(excluded_platform(&child));
    Some(variant_image(child, key, kind, blocks))
}

/// Each child carries a non-empty package label holding a JSON object.
fn package_object(child: &Child) -> Result<Map<String, Value>, String> {
    let source = &child.source;
    let Some(raw) = child
        .config
        .labels
        .get(PACKAGE_LABEL)
        .filter(|raw| !raw.trim().is_empty())
    else {
        return Err(format!(
            "{source}: no {PACKAGE_LABEL} label; each child needs one declaring its \
             variant.key"
        ));
    };
    match serde_json::from_str(raw) {
        Ok(Value::Object(package)) => Ok(package),
        Ok(_) => Err(format!("{source}: {PACKAGE_LABEL} must be a JSON object")),
        Err(err) => Err(format!(
            "{source}: {PACKAGE_LABEL} is not valid JSON: {err}"
        )),
    }
}

/// The package label is `schemaVersion` 1.
fn unsupported_schema_version(source: &str, package: &Map<String, Value>) -> Option<String> {
    (package.get("schemaVersion").and_then(Value::as_u64) != Some(1)).then(|| {
        format!(
            "{source}: {PACKAGE_LABEL} schemaVersion {} is not the supported version 1",
            package.get("schemaVersion").unwrap_or(&Value::Null)
        )
    })
}

/// Each child declares a `variant` block; a block without a valid key is
/// already reported by the declaration checks.
fn missing_variant_block(source: &str, blocks: &ComputeBlocks) -> Option<String> {
    blocks.variant.is_none().then(|| {
        format!(
            "{source}: {PACKAGE_LABEL} declares no variant block; each child of a \
             version is addressed by its variant.key"
        )
    })
}

/// Warn when neither the describe label nor the command says what the image
/// serves.
fn unknown_kind(source: &str, kind: Option<Kind>) -> Option<String> {
    kind.is_none().then(|| {
        format!(
            "{source}: kind unknown (no {DESCRIBE_LABEL} label and the command does not \
             run -m rlmesh.serve); the platform's probe decides it"
        )
    })
}

/// Warn that a child the fleet cannot run is recorded but never selected.
fn excluded_platform(child: &Child) -> Option<String> {
    (!child.selectable()).then(|| {
        format!(
            "{}: {} image; recorded with status excluded and never selected, since \
             the fleet runs linux/amd64",
            child.source, child.platform
        )
    })
}

/// The variant a checked child records: its requires and facets as the
/// platform derives them, and its row keys.
fn variant_image(
    child: Child,
    key: String,
    kind: Option<Kind>,
    blocks: ComputeBlocks,
) -> VariantImage {
    let inferred = variant::infer(&child.config);
    let (requires, facets) = variant::effective(&blocks, &inferred);
    let requires_inferred = blocks
        .variant
        .as_ref()
        .is_some_and(|variant| variant.requires.is_none());
    let rows = blocks.row_keys(&key);
    VariantImage {
        child,
        key,
        kind,
        blocks,
        requires,
        requires_inferred,
        facets,
        rows,
    }
}

/// `variant.key`s are unique within a version.
fn duplicate_keys(variants: &[VariantImage]) -> Vec<String> {
    let mut by_key: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for variant in variants {
        by_key
            .entry(&variant.key)
            .or_default()
            .push(&variant.child.source);
    }
    by_key
        .iter()
        .filter(|(_, sources)| sources.len() > 1)
        .map(|(key, sources)| {
            format!(
                "variant.key {key:?} is declared by {}; keys must be unique within a version",
                sources.join(" and ")
            )
        })
        .collect()
}

/// Derived row keys are unique within a version: a profile row must not
/// collide with another child's key or row.
fn colliding_rows(variants: &[VariantImage]) -> Vec<String> {
    let mut by_row: BTreeMap<&str, Vec<&VariantImage>> = BTreeMap::new();
    for variant in variants {
        for row in &variant.rows {
            by_row.entry(row).or_default().push(variant);
        }
    }
    by_row
        .iter()
        .filter_map(|(row, owners)| {
            let keys: BTreeSet<&str> = owners.iter().map(|v| v.key.as_str()).collect();
            // A shared variant.key is reported by `duplicate_keys`; this is a
            // profile row colliding with another child's key or row.
            (owners.len() > 1 && keys.len() > 1).then(|| {
                format!(
                    "row key {row:?} is derived by {}; rows are <variant.key>-<profile.key>, so \
                     rename a key",
                    owners
                        .iter()
                        .map(|v| v.child.source.as_str())
                        .collect::<Vec<_>>()
                        .join(" and ")
                )
            })
        })
        .collect()
}

/// Every variant of a version serves the same kind (unknown kinds aside).
fn mixed_kinds(variants: &[VariantImage]) -> Option<String> {
    let mut by_kind: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for variant in variants {
        if let Some(kind) = variant.kind {
            by_kind
                .entry(kind.name())
                .or_default()
                .push(&variant.child.source);
        }
    }
    (by_kind.len() > 1).then(|| {
        format!(
            "mixed kinds: {}; every variant of a version serves the same kind",
            by_kind
                .iter()
                .map(|(kind, sources)| format!("{kind} ({})", sources.join(", ")))
                .collect::<Vec<_>>()
                .join(" vs ")
        )
    })
}

/// At least one child is `linux/amd64`, so the version has a default row.
fn unschedulable(variants: &[VariantImage]) -> Option<String> {
    (!variants.is_empty() && default_row(variants).is_none())
        .then(|| "no child is linux/amd64, so nothing in this version can be scheduled".to_owned())
}

/// Read `--index-package`: a JSON object of version-level package data.
/// `schemaVersion` defaults to 1. The platform reads only the version-level
/// keys off the index ([`variant::VERSION_PACKAGE_KEYS`]); anything else is
/// kept as written but warned about. Returns compact JSON and the warnings.
pub(crate) fn index_package(raw: &str, origin: &str) -> Result<(String, Vec<String>)> {
    let mut value: Value =
        serde_json::from_str(raw).with_context(|| format!("{origin} is not valid JSON"))?;
    let Value::Object(package) = &mut value else {
        bail!("{origin} must be a JSON object");
    };
    match package.get("schemaVersion") {
        None => {
            package.insert("schemaVersion".to_owned(), Value::from(1));
        }
        Some(version) if version.as_u64() == Some(1) => {}
        Some(version) => bail!("{origin} schemaVersion {version} is not the supported version 1"),
    }
    let warnings = variant::index_annotation_warnings(package)
        .into_iter()
        .map(|warning| format!("{origin}: {warning}"))
        .collect();
    Ok((serde_json::to_string(&value)?, warnings))
}

/// Warn for each child that sets a version-level key differently from the
/// index annotation, which wins on the platform.
pub(super) fn index_disagreements(annotation: &str, variants: &[VariantImage]) -> Vec<String> {
    let Ok(Value::Object(index)) = serde_json::from_str::<Value>(annotation) else {
        return Vec::new();
    };
    variants
        .iter()
        .filter_map(|variant| {
            let child = variant::package_object(&variant.child.config)?;
            let differs: Vec<&str> = variant::VERSION_PACKAGE_KEYS
                .iter()
                .copied()
                .filter(|key| matches!((child.get(*key), index.get(*key)), (Some(a), Some(b)) if a != b))
                .collect();
            (!differs.is_empty()).then(|| {
                format!(
                    "{}: sets {} differently from the index annotation, which wins",
                    variant.child.source,
                    differs.join(", ")
                )
            })
        })
        .collect()
}

/// Why `index` would not carry the `--index-package` annotation, if it would
/// not: it is not an OCI image index (a Docker manifest list has no
/// annotations), or the annotation is missing from it.
pub(super) fn annotation_dropped(index: &Value) -> Option<String> {
    let media_type = index
        .get("mediaType")
        .and_then(Value::as_str)
        .unwrap_or("(no media type)");
    if media_type != OCI_INDEX {
        return Some(format!(
            "a {media_type}, not an OCI image index, so the --index-package annotation is \
             dropped; rebuild the sources with OCI media types (docker buildx build --push)"
        ));
    }
    let kept = index
        .get("annotations")
        .and_then(|annotations| annotations.get(PACKAGE_LABEL))
        .is_some();
    (!kept).then(|| format!("an index without the {PACKAGE_LABEL} annotation --index-package sets"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::publish::source::resolve_source;
    use crate::publish::testing::*;
    use serde_json::json;

    #[test]
    fn every_problem_is_reported_before_anything_is_created() {
        let mut registry = three_variants();
        // Same key as the CUDA build.
        registry.add(
            "reg.example/ns/pi0:dup",
            None,
            'd',
            oci_config(
                &[],
                MODEL,
                Some(json!({"schemaVersion": 1, "variant": {"key": "torch-cuda12"}})),
            ),
        );
        // An env among models.
        registry.add(
            "reg.example/ns/pi0:env",
            None,
            'e',
            oci_config(
                &[],
                ENV,
                Some(json!({"schemaVersion": 1, "variant": {"key": "sim"}})),
            ),
        );
        registry.add(
            "reg.example/ns/pi0:nolabel",
            None,
            'n',
            oci_config(&[], MODEL, None),
        );
        // An empty label (`--build-arg RLMESH_PACKAGE=` left unset) reads as none.
        let mut empty: Value = serde_json::from_str(&oci_config(&[], MODEL, None)).unwrap();
        empty["config"]["Labels"][PACKAGE_LABEL] = Value::from("");
        registry.add("reg.example/ns/pi0:empty", None, 'm', empty.to_string());
        registry.add(
            "reg.example/ns/pi0:nokey",
            None,
            'k',
            oci_config(
                &[],
                MODEL,
                Some(json!({"schemaVersion": 1, "profiles": []})),
            ),
        );
        registry.add(
            "reg.example/ns/pi0:badreq",
            None,
            'b',
            oci_config(
                &[],
                MODEL,
                Some(json!({"schemaVersion": 1,
                "variant": {"key": "bad", "requires": {"accel.cuda": ">12"}}})),
            ),
        );
        let windows = buildkit_index(&digest('w'), ("windows", "amd64"));
        registry.insert("reg.example/ns/pi0:win", &windows);
        let mut multi = buildkit_index(&digest('x'), ("linux", "amd64"));
        multi["manifests"].as_array_mut().unwrap().push(
            json!({"digest": digest('y'), "platform": {"os": "linux", "architecture": "arm64"}}),
        );
        registry.insert("reg.example/ns/pi0:multi", &multi);

        let sources = [
            "reg.example/ns/pi0:v3-cuda12",
            "reg.example/ns/pi0:dup",
            "reg.example/ns/pi0:env",
            "reg.example/ns/pi0:nolabel",
            "reg.example/ns/pi0:empty",
            "reg.example/ns/pi0:nokey",
            "reg.example/ns/pi0:badreq",
            "reg.example/ns/pi0:win",
            "reg.example/ns/pi0:multi",
            "reg.example/ns/pi0:missing",
        ];
        let mut args = args("reg.example/ns/pi0:v3", &sources);
        args.dry_run = true;
        let (result, out) = run(&registry, &args);
        let error = format!("{:#}", result.unwrap_err());
        assert!(registry.created.borrow().is_empty(), "nothing is created");
        for needle in [
            "cannot publish reg.example/ns/pi0:v3:",
            "reg.example/ns/pi0:win holds a windows/amd64 image; the platform runs linux images only",
            "reg.example/ns/pi0:multi is an index of 2 images (linux/amd64, linux/arm64)",
            "reg.example/ns/pi0:missing: not found",
            "reg.example/ns/pi0:nolabel: no dev.rlmesh.package label",
            "reg.example/ns/pi0:empty: no dev.rlmesh.package label",
            "reg.example/ns/pi0:nokey: dev.rlmesh.package declares no variant block",
            "reg.example/ns/pi0:badreq: variant: requires accel.cuda needs accel.vendor",
            "variant.key \"torch-cuda12\" is declared by reg.example/ns/pi0:v3-cuda12 and reg.example/ns/pi0:dup",
            "mixed kinds: env (reg.example/ns/pi0:env) vs model (",
        ] {
            assert!(error.contains(needle), "{needle:?} not in:\n{error}");
        }
        // The variants that did resolve are still summarized.
        assert!(out.lines().any(|l| l.starts_with("sim ")), "{out}");
    }

    #[test]
    fn non_linux_bare_manifests_and_unknown_kinds() {
        let mut registry = FakeRegistry::default();
        let mut config: Value = serde_json::from_str(&oci_config(&[], &[], None)).unwrap();
        config["os"] = Value::from("windows");
        registry.add("r/x:win", None, 'w', config.to_string());
        let child = resolve_source(&registry, "r/x:win").unwrap_err();
        assert!(
            format!("{child:#}").contains("is a windows/amd64 image"),
            "{child:#}"
        );

        registry.add(
            "r/x:custom",
            None,
            'c',
            oci_config(
                &[],
                &["./serve.sh"],
                Some(json!({"schemaVersion": 1, "variant": {"key": "a"}})),
            ),
        );
        let child = resolve_source(&registry, "r/x:custom").unwrap();
        let (variants, warnings, errors) = check_variants(vec![child]);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(variants[0].kind, None);
        assert!(warnings[0].contains("kind unknown"), "{warnings:?}");
    }

    #[test]
    fn index_package_is_version_level_only() {
        assert_eq!(
            index_package(r#"{"checkpoints":[]}"#, "f").unwrap(),
            (r#"{"checkpoints":[],"schemaVersion":1}"#.to_owned(), vec![])
        );
        let (_, warnings) =
            index_package(r#"{"rev":3,"variant":{"key":"a"},"tags":[]}"#, "f").unwrap();
        assert_eq!(
            warnings,
            ["f: index annotation keys ignored (they belong on the child images): tags, variant"]
        );
        for (raw, needle) in [
            ("[]", "must be a JSON object"),
            ("{", "not valid JSON"),
            (r#"{"schemaVersion":2}"#, "schemaVersion 2"),
        ] {
            let error = format!("{:#}", index_package(raw, "f").unwrap_err());
            assert!(error.contains(needle), "{raw}: {error}");
        }
    }

    #[test]
    fn the_index_annotation_wins_over_children_and_is_warned_about() {
        let mut registry = FakeRegistry::default();
        registry.add(
            "r/x:a",
            None,
            'a',
            oci_config(
                &[],
                MODEL,
                Some(json!({"schemaVersion": 1, "name": "old",
                "variant": {"key": "a"}})),
            ),
        );
        let dir = tempfile::tempdir().unwrap();
        let package = dir.path().join("version.json");
        std::fs::write(&package, r#"{"name":"pi0","tags":["gpu"]}"#).unwrap();
        let mut publish = args("r/x:v1", &["r/x:a"]);
        publish.dry_run = true;
        publish.index_package = Some(package);
        let (result, out) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert!(
            out.contains("index annotation keys ignored (they belong on the child images): tags"),
            "{out}"
        );
        assert!(
            out.contains("r/x:a: sets name differently from the index annotation, which wins"),
            "{out}"
        );
    }

    #[test]
    fn rows_are_unique_across_the_version_and_one_child_must_be_schedulable() {
        let mut registry = FakeRegistry::default();
        // "cuda12" with profile "egl" derives "cuda12-egl", another child's key.
        registry.add(
            "r/x:a",
            None,
            'a',
            oci_config(
                &[],
                MODEL,
                Some(json!({"schemaVersion": 1, "variant": {"key": "cuda12"},
                "profiles": [{"key": "egl", "default": true}]})),
            ),
        );
        registry.add(
            "r/x:b",
            None,
            'b',
            oci_config(
                &[],
                MODEL,
                Some(json!({"schemaVersion": 1, "variant": {"key": "cuda12-egl"}})),
            ),
        );
        let (result, _) = run(&registry, &args("r/x:v1", &["r/x:a", "r/x:b"]));
        let error = format!("{:#}", result.unwrap_err());
        assert!(
            error.contains("row key \"cuda12-egl\" is derived by r/x:a and r/x:b"),
            "{error}"
        );

        // Only an arm64 child: excluded, so nothing can be scheduled.
        let mut arm: Value = serde_json::from_str(&oci_config(
            &[],
            MODEL,
            Some(json!({"schemaVersion": 1, "variant": {"key": "arm"}})),
        ))
        .unwrap();
        arm["architecture"] = Value::from("arm64");
        registry.add("r/x:arm", None, 'r', arm.to_string());
        let (result, out) = run(&registry, &args("r/x:v1", &["r/x:arm"]));
        let error = format!("{:#}", result.unwrap_err());
        assert!(error.contains("no child is linux/amd64"), "{error}");
        assert!(
            out.contains("linux/amd64") && out.contains("status excluded"),
            "{out}"
        );
        assert!(out.contains("linux/arm64 (excluded)"), "{out}");
    }

    #[test]
    fn the_assembled_index_must_keep_the_annotation() {
        assert_eq!(
            annotation_dropped(&json!({"mediaType": OCI_INDEX,
                "annotations": {PACKAGE_LABEL: "{}"}})),
            None
        );
        let list = annotation_dropped(
            &json!({"mediaType": "application/vnd.docker.distribution.manifest.list.v2+json"}),
        )
        .unwrap();
        assert!(
            list.contains("not an OCI image index, so the --index-package annotation is dropped"),
            "{list}"
        );
        let bare = annotation_dropped(&json!({"mediaType": OCI_INDEX})).unwrap();
        assert!(
            bare.contains("without the dev.rlmesh.package annotation"),
            "{bare}"
        );
    }
}
