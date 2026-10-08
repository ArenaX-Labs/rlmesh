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
//! AMD targets; `accel.vram` is the minimum VRAM per GPU as a quantity string
//! (`"16Gi"`, [`parse_quantity`]); every key but `accel.vendor` needs
//! `accel.vendor`. A profile's `requires` and `facets` override the variant's
//! key by key. An image without a variant block gets its requires inferred
//! from its CUDA/ROCm markers ([`infer`](fn@infer)).
//!
//! The managed platform implements exactly these rules (its
//! `variant_requires_schema`, `variant_requires_vs_env`, and row naming), so
//! like [`crate::image_check`] the parsing rules, the row keys, and the
//! report's message prefixes (`variant:`, `profiles:`, `rows:`) are a
//! contract.

mod constraint;
mod declaration;
mod infer;
mod quantity;
mod requires;
#[cfg(test)]
mod testing;

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};

use crate::image_check::{CheckReport, ImageConfig, PACKAGE_LABEL};

pub use constraint::{Clause, Comparator, parse_constraint};
pub use declaration::{
    ComputeBlocks, MAX_PRIORITY, MAX_PROFILE_GPUS, Profile, VERSION_PACKAGE_KEYS, Variant,
    index_annotation_warnings, parse_compute_blocks,
};
pub use infer::{Inferred, infer};
pub use quantity::{QuantityError, format_quantity, parse_quantity};
pub use requires::{
    ACCEL_FACETS, FACET_KEYS, MAX_KEY_LEN, RENDER_FACETS, REQUIRE_KEYS, Requirement, Requires,
    VENDORS, format_facets, format_requires, parse_requirement, valid_key,
};

use infer::{check_against_markers, marker_vendor_warning};
use requires::{accel_vendor, requires_vendor, vendor_problems};

/// The row key of an undeclared image.
pub const DEFAULT_VARIANT_KEY: &str = "default";

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
        if profile.resources {
            report.not_checked.push(format!(
                "profiles: {}: resources are validated by the platform against its ceilings",
                profile.key
            ));
        }
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

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;
    use serde_json::json;

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
                    "accel.vram": "24Gi"
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
                 accel.vendor=nvidia, accel.vram>=24Gi; priority 10)",
                "rows: cuda12-osmesa (default), cuda12-egl",
            ]
        );
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

        // Resources are noted whether or not the profile overrides requires.
        let package = json!({
            "schemaVersion": 1,
            "variant": {"key": "cpu"},
            "profiles": [{"key": "big", "default": true, "resources": {"memory": "64Gi"}}]
        });
        let report = check_variant(&image(&[], Some(package)));
        assert_buckets(&report, &[], &[]);
        assert_eq!(
            report.not_checked,
            ["profiles: big: resources are validated by the platform against its ceilings"]
        );
    }

    #[test]
    fn profiles_override_vram() {
        let package = |variant: Value, profile: Value| {
            json!({
                "schemaVersion": 1,
                "variant": {"key": "cuda12", "requires": variant},
                "profiles": [{"key": "big", "default": true, "requires": profile}]
            })
        };
        for (variant, profile, merged) in [
            (
                json!({"accel.vendor": "nvidia", "accel.vram": "16G"}),
                json!({"accel.vendor": "nvidia", "accel.vram": "80Gi"}),
                "accel.vendor=nvidia, accel.vram>=80Gi",
            ),
            (
                json!({"accel.vendor": "nvidia", "accel.vram": "16Gi"}),
                json!({"accel.vendor": "nvidia", "accel.vram": "24Gi"}),
                "accel.vendor=nvidia, accel.vram>=24Gi",
            ),
        ] {
            let (blocks, report) = parse_compute_blocks(&object(package(variant, profile)));
            assert!(report.failed.is_empty(), "{report:#?}");
            let mut requires = blocks.variant.unwrap().requires.unwrap();
            requires.extend(blocks.profiles[0].requires.clone());
            assert_eq!(format_requires(&requires), merged);
        }
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
}
