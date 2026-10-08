//! The variant summary a publish prints.

use std::io::Write;

use anyhow::Result;

use super::validate::{VariantImage, default_row};
use crate::image_check::Kind;
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
    let rows: Vec<[String; 8]> = variants
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
        })
        .collect();
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
    for row in &rows {
        writeln!(stdout, "{}", line(row))?;
    }
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

fn short_digest(digest: &str) -> String {
    let hex = digest.strip_prefix("sha256:").unwrap_or(digest);
    format!("({})", &hex[..hex.len().min(12)])
}
