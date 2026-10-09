//! The variant summary a publish prints.

use std::io::Write;

use anyhow::Result;

use super::validate::{VariantImage, default_row};
use crate::image_check::Kind;
use crate::manifest::VariantBuild;
use crate::render::Style;
use crate::variant;

pub(super) fn write_table(
    stdout: &mut impl Write,
    style: Style,
    variants: &[VariantImage],
) -> Result<()> {
    let headers = [
        "VARIANT", "KIND", "PLATFORM", "PRIORITY", "FACETS", "REQUIRES", "ROWS", "IMAGE",
    ];
    let default = default_row(variants);
    let rows: Vec<Vec<String>> = variants
        .iter()
        .enumerate()
        .map(|(index, variant)| {
            let priority = variant
                .blocks
                .variant
                .as_ref()
                .map_or(0, |declared| declared.priority);
            let or_dash = |value: String| {
                if value.is_empty() {
                    "-".to_owned()
                } else {
                    value
                }
            };
            let requires = match (
                variant::format_requires(&variant.requires),
                variant.requires_inferred,
            ) {
                (requires, _) if requires.is_empty() => "-".to_owned(),
                (requires, true) => format!("{requires} (inferred)"),
                (requires, false) => requires,
            };
            let row_keys = variant
                .rows
                .iter()
                .enumerate()
                .map(|(row, key)| {
                    if default == Some((index, row)) {
                        format!("{key}*")
                    } else {
                        key.clone()
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            [
                variant.key.clone(),
                variant.kind.map_or("?", Kind::name).to_owned(),
                if variant.selectable() {
                    variant.child.platform.clone()
                } else {
                    format!("{} (excluded)", variant.child.platform)
                },
                priority.to_string(),
                or_dash(variant::format_facets(&variant.facets)),
                requires,
                row_keys,
                format!(
                    "{} {}",
                    variant.child.source,
                    short_digest(&variant.child.image_digest)
                ),
            ]
            .into()
        })
        .collect();
    write_columns(stdout, style, &headers, &rows)?;
    if default.is_some() {
        writeln!(
            stdout,
            "{}",
            style.muted(
                "* the version's default row: the first linux/amd64 child, at its default profile"
            )
        )?;
    }
    Ok(())
}

/// The builds a publish from `rlmesh.toml` would run, before any exists.
pub(super) fn write_build_table(
    stdout: &mut impl Write,
    style: Style,
    variants: &[VariantBuild],
) -> Result<()> {
    let rows: Vec<Vec<String>> = variants
        .iter()
        .enumerate()
        .map(|(index, build)| {
            let declared = build.blocks.variant.as_ref();
            let requires = declared
                .and_then(|variant| variant.requires.as_ref())
                .map(variant::format_requires)
                .filter(|requires| !requires.is_empty())
                .unwrap_or_else(|| "(inferred from the image)".to_owned());
            let mut rows = build.blocks.row_keys(&build.key);
            if index == 0 {
                let default = build.blocks.default_profile().unwrap_or(0);
                rows[default].push('*');
            }
            let args = build
                .build_args
                .iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join(", ");
            vec![
                build.key.clone(),
                declared.map_or(0, |variant| variant.priority).to_string(),
                requires,
                rows.join(", "),
                build.dockerfile.display().to_string(),
                if args.is_empty() {
                    "-".to_owned()
                } else {
                    args
                },
            ]
        })
        .collect();
    write_columns(
        stdout,
        style,
        &[
            "VARIANT",
            "PRIORITY",
            "REQUIRES",
            "ROWS",
            "DOCKERFILE",
            "BUILD ARGS",
        ],
        &rows,
    )?;
    writeln!(
        stdout,
        "{}",
        style.muted("* the version's default row: the first variant, at its default profile")
    )?;
    Ok(())
}

/// Left-aligned columns under a bold header.
fn write_columns(
    stdout: &mut impl Write,
    style: Style,
    headers: &[&str],
    rows: &[Vec<String>],
) -> Result<()> {
    let widths: Vec<usize> = (0..headers.len())
        .map(|column| {
            rows.iter()
                .map(|row| row[column].chars().count())
                .chain([headers[column].len()])
                .max()
                .unwrap_or(0)
        })
        .collect();
    let line = |cells: &[String]| {
        cells
            .iter()
            .zip(&widths)
            .map(|(cell, width)| format!("{cell:<width$}"))
            .collect::<Vec<_>>()
            .join("  ")
            .trim_end()
            .to_owned()
    };
    let header: Vec<String> = headers.iter().map(|h| (*h).to_owned()).collect();
    writeln!(stdout, "{}", style.bold(&line(&header)))?;
    for row in rows {
        writeln!(stdout, "{}", line(row))?;
    }
    Ok(())
}

fn short_digest(digest: &str) -> String {
    let hex = digest.strip_prefix("sha256:").unwrap_or(digest);
    format!("({})", &hex[..hex.len().min(12)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::publish::source::resolve_source;
    use crate::publish::testing::*;
    use crate::publish::validate::check_variants;
    use serde_json::{Value, json};

    /// The `three_variants` sources named, plus an undeclared-requires CUDA
    /// child (`v3-cu`) and an arm64 one (`v3-arm`), checked.
    fn variants(sources: &[&str]) -> Vec<VariantImage> {
        let mut registry = three_variants();
        registry.add(
            "reg.example/ns/pi0:v3-cu",
            None,
            'u',
            oci_config(
                &["CUDA_VERSION=12.4.1"],
                MODEL,
                Some(json!({"schemaVersion": 1, "variant": {"key": "cu"}})),
            ),
        );
        let mut arm: Value = serde_json::from_str(&keyed("arm")).unwrap();
        arm["architecture"] = Value::from("arm64");
        registry.add("reg.example/ns/pi0:v3-arm", None, 'f', arm.to_string());
        let children = sources
            .iter()
            .map(|source| resolve_source(&registry, source).unwrap())
            .collect();
        check_variants(children).0
    }

    fn table(style: Style, variants: &[VariantImage]) -> String {
        let mut out = Vec::new();
        write_table(&mut out, style, variants).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn the_table_marks_the_default_row_and_inferred_requires() {
        let variants = variants(&[
            "reg.example/ns/pi0:v3-arm",
            "reg.example/ns/pi0:v3-jax",
            "reg.example/ns/pi0:v3-cuda12",
            "reg.example/ns/pi0:v3-rocm6",
            "reg.example/ns/pi0:v3-cu",
        ]);
        // The default row is the first linux/amd64 child's default profile;
        // an excluded child comes first but cannot be it.
        assert_eq!(
            table(Style::for_terminal(false), &variants),
            concat!(
                "VARIANT       KIND   PLATFORM                PRIORITY  FACETS                       \
                 REQUIRES                                          ROWS                  IMAGE\n",
                "arm           model  linux/arm64 (excluded)  0         accel=cpu                    \
                 -                                                 arm                   \
                 reg.example/ns/pi0:v3-arm (ffffffffffff)\n",
                "jax           model  linux/amd64             0         accel=cpu, framework=jax     \
                 -                                                 jax-osmesa*, jax-egl  \
                 reg.example/ns/pi0:v3-jax (jjjjjjjjjjjj)\n",
                "torch-cuda12  model  linux/amd64             10        accel=cuda, framework=torch  \
                 accel.cuda>=12.4, accel.vendor=nvidia             torch-cuda12          \
                 reg.example/ns/pi0:v3-cuda12 (cccccccccccc)\n",
                "torch-rocm6   model  linux/amd64             5         accel=rocm                   \
                 accel.gfx in [gfx942,gfx90a], accel.vendor=amd    torch-rocm6           \
                 reg.example/ns/pi0:v3-rocm6 (rrrrrrrrrrrr)\n",
                "cu            model  linux/amd64             0         accel=cuda                   \
                 accel.cuda>=12.4, accel.vendor=nvidia (inferred)  cu                    \
                 reg.example/ns/pi0:v3-cu (uuuuuuuuuuuu)\n",
                "* the version's default row: the first linux/amd64 child, at its default profile\n",
            )
        );
    }

    #[test]
    fn styling_paints_only_the_header_and_the_footnote() {
        let variants = variants(&["reg.example/ns/pi0:v3-cuda12"]);
        let header = "VARIANT       KIND   PLATFORM     PRIORITY  FACETS                       \
                      REQUIRES                               ROWS           IMAGE";
        let row = "torch-cuda12  model  linux/amd64  10        accel=cuda, framework=torch  \
                   accel.cuda>=12.4, accel.vendor=nvidia  torch-cuda12*  \
                   reg.example/ns/pi0:v3-cuda12 (cccccccccccc)";
        let footnote =
            "* the version's default row: the first linux/amd64 child, at its default profile";
        assert_eq!(
            table(Style::for_terminal(false), &variants),
            format!("{header}\n{row}\n{footnote}\n")
        );
        assert_eq!(
            table(Style::colored(), &variants),
            format!("\x1b[1m{header}\x1b[0m\n{row}\n\x1b[2m{footnote}\x1b[0m\n")
        );
    }

    #[test]
    fn a_table_without_a_default_row_has_no_footnote() {
        let variants = variants(&["reg.example/ns/pi0:v3-arm"]);
        assert_eq!(
            table(Style::for_terminal(false), &variants),
            concat!(
                "VARIANT  KIND   PLATFORM                PRIORITY  FACETS     REQUIRES  ROWS  IMAGE\n",
                "arm      model  linux/arm64 (excluded)  0         accel=cpu  -         arm   \
                 reg.example/ns/pi0:v3-arm (ffffffffffff)\n",
            )
        );
    }

    #[test]
    fn vram_renders_as_a_quantity() {
        let mut registry = FakeRegistry::default();
        for (tag, seed, requires) in [
            (
                "v3-new",
                'n',
                json!({"accel.vendor": "nvidia", "accel.vram": "80Gi"}),
            ),
            (
                "v3-dec",
                'd',
                json!({"accel.vendor": "nvidia", "accel.vram": "24000M"}),
            ),
        ] {
            registry.add(
                &format!("reg.example/ns/pi0:{tag}"),
                None,
                seed,
                oci_config(
                    &[],
                    MODEL,
                    Some(json!({"schemaVersion": 1, "variant": {"key": tag,
                        "facets": {"accel": "cuda"}, "requires": requires}})),
                ),
            );
        }
        let children = ["reg.example/ns/pi0:v3-new", "reg.example/ns/pi0:v3-dec"]
            .iter()
            .map(|source| resolve_source(&registry, source).unwrap())
            .collect();
        let (variants, _, _) = check_variants(children);
        let table = table(Style::for_terminal(false), &variants);
        let requires: Vec<&str> = table
            .lines()
            .skip(1)
            .filter_map(|line| line.split("  ").find(|cell| cell.contains("accel.vram")))
            .collect();
        assert_eq!(
            requires,
            [
                "accel.vendor=nvidia, accel.vram>=80Gi",
                "accel.vendor=nvidia, accel.vram>=24G"
            ],
            "{table}"
        );
    }

    #[test]
    fn short_digests_are_twelve_hex_characters() {
        assert_eq!(short_digest(&digest('c')), "(cccccccccccc)");
        assert_eq!(short_digest("sha256:abc"), "(abc)");
        assert_eq!(short_digest("0123456789abcdef"), "(0123456789ab)");
    }
}
