//! What an image's CUDA/ROCm markers and describe label say about its
//! accelerator, and declarations those markers contradict.

use std::collections::BTreeMap;

use serde_json::Value;

use super::DEFAULT_VARIANT_KEY;
use super::constraint::{admits_below, caps_below, parse_constraint, parse_version};
use super::declaration::Variant;
use super::requires::{Requirement, Requires, requires_vendor};
use crate::image_check::{CheckReport, DESCRIBE_LABEL, ImageConfig};

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
    let mut out = Inferred::default();
    let mut evidence = Vec::new();
    read_toolkit_markers(config, &mut out, &mut evidence);
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
    {
        read_torch_build_tag(torch, &mut out, &mut evidence);
    }
    derive_accel(&mut out, &mut evidence);
    out.evidence = evidence.join(", ");
    out
}

/// The toolkit environment markers: `CUDA_VERSION`, else the floor in
/// `NVIDIA_REQUIRE_CUDA`, and `ROCM_VERSION`.
fn read_toolkit_markers(config: &ImageConfig, out: &mut Inferred, evidence: &mut Vec<String>) {
    let env = |key: &str| {
        config
            .env_value(key)
            .map(str::trim)
            .filter(|v| !v.is_empty())
    };
    if let Some(version) = env("CUDA_VERSION") {
        out.cuda = major_minor(version);
        evidence.push(format!("CUDA_VERSION={version}"));
    } else if let Some(floor) = env("NVIDIA_REQUIRE_CUDA").and_then(cuda_floor) {
        evidence.push(format!("NVIDIA_REQUIRE_CUDA cuda>={floor}"));
        out.cuda = Some(floor);
    }
    if let Some(version) = env("ROCM_VERSION") {
        out.rocm = major_minor(version);
        evidence.push(format!("ROCM_VERSION={version}"));
    }
}

/// The first `cuda>=X.Y` anywhere in an `NVIDIA_REQUIRE_CUDA` value, as the
/// platform reads it.
fn cuda_floor(value: &str) -> Option<String> {
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
}

/// Without toolkit markers, a torch build tag (`+cu121`, `+rocm6.0`) names
/// the stack.
fn read_torch_build_tag(torch: &str, out: &mut Inferred, evidence: &mut Vec<String>) {
    let Some((_, tag)) = torch.rsplit_once('+') else {
        return;
    };
    let (stack, version) = if let Some(v) = tag.strip_prefix("cu") {
        ("cu", v)
    } else if let Some(v) = tag.strip_prefix("rocm") {
        ("rocm", v)
    } else {
        ("", "")
    };
    if version.is_empty() || !version.bytes().all(|b| b.is_ascii_digit() || b == b'.') {
        return;
    }
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

/// The `accel` facet and requires from the stack found: CUDA, ROCm, neither
/// (a CPU image), or both (nothing).
fn derive_accel(out: &mut Inferred, evidence: &mut Vec<String>) {
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
}

pub(super) fn marker_vendor_warning(
    scope: &str,
    vendor: &str,
    inferred: &Inferred,
    report: &mut CheckReport,
) {
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
pub(super) fn check_against_markers(
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
        if caps_below(clauses, &image) {
            report.warnings.push(format!(
                "variant: requires accel.cuda{requirement} admits no driver that can run the \
                 image's CUDA {cuda} runtime"
            ));
        } else if admits_below(clauses, &image) {
            report.warnings.push(format!(
                "variant: requires accel.cuda{requirement} admits drivers older than the image's \
                 CUDA {cuda} runtime needs; declare \">={cuda}\""
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variant::testing::*;
    use crate::variant::{check_variant, effective, format_requires, parse_compute_blocks};
    use serde_json::json;

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
        // The whole constraint decides: no lower bound admits older drivers,
        // and a cap below the runtime is the stronger complaint.
        for (constraint, warnings) in [
            (
                "<13",
                vec!["accel.cuda<13 admits drivers older than the image's CUDA 12.4 runtime needs"],
            ),
            ("<=12.6", vec!["accel.cuda<=12.6 admits drivers older"]),
            ("==12.4", vec![]),
            ("=12.4.0", vec![]),
            (">12.4", vec![]),
            (">=12.4,<13", vec![]),
            (">12.3", vec!["accel.cuda>12.3 admits drivers older"]),
            (
                "==12.2",
                vec!["accel.cuda==12.2 admits no driver that can run the image's CUDA 12.4"],
            ),
            (
                ">=12.2,<12.3",
                vec!["accel.cuda>=12.2,<12.3 admits no driver that can run"],
            ),
        ] {
            let report = check_variant(&cuda(variant(
                json!({"accel.vendor": "nvidia", "accel.cuda": constraint}),
                json!({}),
            )));
            assert_buckets(&report, &[], &warnings);
        }
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
    fn the_cuda_floor_is_the_first_whole_one() {
        for (value, floor) in [
            ("cuda>=12.4", Some("12.4")),
            (
                "brand=tesla,driver>=470 cuda>=11.8 brand=nvidia",
                Some("11.8"),
            ),
            // A floor without a minor version is skipped for the next one.
            ("cuda>=12 cuda>=11.8", Some("11.8")),
            ("cuda>=12.4rc1", Some("12.4")),
            ("cuda>=12.", None),
            ("driver>=470", None),
            ("", None),
        ] {
            assert_eq!(cuda_floor(value).as_deref(), floor, "{value:?}");
        }
    }

    #[test]
    fn the_evidence_names_each_marker_read() {
        for (config, evidence) in [
            (
                image(&["NVIDIA_REQUIRE_CUDA=cuda>=11.8"], None),
                "NVIDIA_REQUIRE_CUDA cuda>=11.8",
            ),
            (
                image(&["CUDA_VERSION=12.4", "ROCM_VERSION=6.2"], None),
                "CUDA_VERSION=12.4, ROCM_VERSION=6.2, both CUDA and ROCm markers, so nothing",
            ),
            (
                with_torch(image(&[], None), json!({"torch": "2.3.0+cu121"})),
                "torch 2.3.0+cu121",
            ),
            // Tags that name no stack version are not evidence.
            (
                with_torch(image(&[], None), json!({"torch": "2.3.0+cu12x"})),
                "no CUDA or ROCm markers",
            ),
            (
                with_torch(image(&[], None), json!({"torch": "2.3.0"})),
                "no CUDA or ROCm markers",
            ),
        ] {
            assert_eq!(infer(&config).evidence, evidence);
        }
    }
}
