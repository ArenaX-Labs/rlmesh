//! The hardware stamp: what machine and software an evaluation ran on.

use std::process::Command;

use serde_json::Value;

use super::schema::AttemptStamp;

/// Shells out to nvidia-smi (amd-smi or rocm-smi when it shows no GPU):
/// call from a blocking context. The image references and the managed ABI
/// are the platform's to fill in.
pub fn collect() -> AttemptStamp {
    let query = run(
        "nvidia-smi",
        &["--query-gpu=name,driver_version", "--format=csv,noheader"],
    );
    let mut gpu = query
        .as_deref()
        .and_then(|out| out.lines().next())
        .map(|line| line.split(',').map(|part| part.trim().to_string()));
    let gpu_model = gpu
        .as_mut()
        .and_then(Iterator::next)
        .filter(|s| !s.is_empty())
        .or_else(amd_gpu_model);
    let nvidia_driver = gpu
        .as_mut()
        .and_then(Iterator::next)
        .filter(|s| !s.is_empty())
        .or_else(|| {
            std::fs::read_to_string("/proc/driver/nvidia/version")
                .ok()?
                .split_whitespace()
                .find(|word| {
                    word.chars().next().is_some_and(|c| c.is_ascii_digit()) && word.contains('.')
                })
                .map(str::to_string)
        });
    // The driver's CUDA version is only reported by the bare table; skip it
    // entirely when the query already showed no usable nvidia-smi.
    let cuda = query.as_ref().and_then(|_| {
        let out = run("nvidia-smi", &[])?;
        let rest = out.split("CUDA Version:").nth(1)?;
        rest.split_whitespace().next().map(str::to_string)
    });
    AttemptStamp {
        runner_version: crate::VERSION.to_string(),
        oss_crate_versions: [("rlmesh".to_string(), crate::VERSION.to_string())].into(),
        git_sha: crate::build_info().git.map(str::to_string),
        node: std::fs::read_to_string("/etc/hostname")
            .ok()
            .or_else(|| run("uname", &["-n"]))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty()),
        gpu_model,
        nvidia_driver,
        cuda,
        // The release a ROCm install records; "6.3.3-74" reads "6.3.3".
        rocm: std::fs::read_to_string("/opt/rocm/.info/version")
            .ok()
            .and_then(|v| v.trim().split('-').next().map(str::to_string))
            .filter(|v| !v.is_empty()),
        // /proc on Linux; uname on hosts without it (macOS).
        kernel: std::fs::read_to_string("/proc/sys/kernel/osrelease")
            .ok()
            .or_else(|| run("uname", &["-r"]))
            .map(|s| s.trim().to_string()),
        ..AttemptStamp::default()
    }
}

/// The AMD GPU a ROCm node shows: amd-smi's market name (its PCI device id
/// when the name is unset), else rocm-smi's card series on releases that
/// predate amd-smi. The platform folds either through its hardware taxonomy.
fn amd_gpu_model() -> Option<String> {
    run("amd-smi", &["static", "--asic", "--json"])
        .and_then(|out| amd_smi_model(&out))
        .or_else(|| {
            run("rocm-smi", &["--showproductname", "--json"]).and_then(|out| rocm_smi_model(&out))
        })
}

fn known(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("n/a"))
        .map(str::to_string)
}

/// amd-smi prints a list of GPUs, or `{"gpu_data": [...]}` from ROCm 6.4.
fn amd_smi_model(out: &str) -> Option<String> {
    let value: Value = serde_json::from_str(out).ok()?;
    let gpus = value
        .as_array()
        .or_else(|| value.get("gpu_data").and_then(Value::as_array))?;
    let asic = gpus.first()?.get("asic")?;
    known(asic.get("market_name")).or_else(|| known(asic.get("device_id")))
}

/// rocm-smi prints `{"card0": {"Card Series": ..., "Card Model": "0x74a1"}}`;
/// the key's case changed across releases.
fn rocm_smi_model(out: &str) -> Option<String> {
    let value: Value = serde_json::from_str(out).ok()?;
    let mut cards: Vec<_> = value
        .as_object()?
        .iter()
        .filter(|(name, _)| name.starts_with("card"))
        .collect();
    cards.sort_by(|a, b| a.0.cmp(b.0));
    let card = cards.first()?.1.as_object()?;
    let field = |name: &str| {
        known(
            card.iter()
                .find(|(key, _)| key.eq_ignore_ascii_case(name))
                .map(|(_, v)| v),
        )
    };
    field("Card Series").or_else(|| field("Card Model"))
}

fn run(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::{amd_smi_model, rocm_smi_model};

    #[test]
    fn amd_smi_names_the_gpu() {
        let list = r#"[{"gpu": 0, "asic": {"market_name": "AMD Instinct MI300X", "device_id": "0x74a1", "target_graphics_version": "gfx942"}}]"#;
        assert_eq!(amd_smi_model(list).as_deref(), Some("AMD Instinct MI300X"));
        let wrapped =
            r#"{"gpu_data": [{"gpu": 0, "asic": {"market_name": "N/A", "device_id": "0x740c"}}]}"#;
        assert_eq!(amd_smi_model(wrapped).as_deref(), Some("0x740c"));
        assert_eq!(amd_smi_model("[]"), None);
        assert_eq!(amd_smi_model("not json"), None);
    }

    #[test]
    fn rocm_smi_names_the_first_card() {
        let out = r#"{"card1": {"Card Series": "Other"}, "card0": {"Card series": "Aqua Vanjaram [Instinct MI300X]", "Card model": "0x74a1"}, "system": {}}"#;
        assert_eq!(
            rocm_smi_model(out).as_deref(),
            Some("Aqua Vanjaram [Instinct MI300X]")
        );
        let model_only = r#"{"card0": {"Card Model": "0x738c"}}"#;
        assert_eq!(rocm_smi_model(model_only).as_deref(), Some("0x738c"));
        assert_eq!(rocm_smi_model(r#"{"system": {}}"#), None);
    }
}
