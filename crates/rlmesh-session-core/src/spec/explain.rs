//! What people read: refusals (§6.7) and binding tables (§6.8).
//!
//! Every refusal names the code, the entry, what was found and a fix:
//!
//! ```text
//! binding refused: my_vla on ur10e (2 problems)
//!
//!   [ambiguous] input "image" (kind camera, at "*/cam") matches 2 parts:
//!       wrist/cam   camera 640x480 @30 Hz
//!       scene/cam   camera 1280x720 @15 Hz
//!     fix: rlmesh.run(..., at={"image": "scene/cam"})
//! ```
//!
//! A binding table has one row per entry (direction, part, status, plan and
//! what `predict` sees), then each input's delivery contract, source and
//! delivery rates and freshness bound, then each action's grid, chunk,
//! validity, edge control rate and action-age cap, each marked when it was
//! defaulted or derived rather than declared.
//!
//! Numbers read at three significant digits, never exponential, trailing
//! zeros trimmed (`66.7`, `1000`, `0.25`).

#[allow(unused_imports)]
use crate::prelude::*;

use core::fmt;
use core::time::Duration;

use crate::wire::part::CommandSpec;

use crate::spec::bind::{Binding, Dir, NamesClash, Problem, Refusal, Status};
use crate::spec::contract;

/// `x` to three significant digits (integers above 100 kept whole).
pub fn num(x: f64) -> String {
    if !x.is_finite() {
        return x.to_string();
    }
    let a = x.abs();
    let decimals = if a >= 100.0 || a == 0.0 {
        0
    } else if a >= 10.0 {
        1
    } else if a >= 1.0 {
        2
    } else {
        3
    };
    let text = format!("{x:.decimals$}");
    let text = if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    };
    if text == "-0" { "0".into() } else { text }
}

/// A rate in Hz, as a number.
pub fn hz(x: f64) -> String {
    num(x)
}

/// A duration in milliseconds, as a number.
pub fn ms(d: Duration) -> String {
    num(d.as_secs_f64() * 1e3)
}

/// A duration in seconds, as a number.
pub fn seconds(d: Duration) -> String {
    num(d.as_secs_f64())
}

/// Rows as aligned columns, two spaces apart, trailing space trimmed.
fn columns(rows: &[Vec<String>], indent: &str) -> Vec<String> {
    let n = rows.iter().map(Vec::len).max().unwrap_or(0);
    let widths: Vec<usize> = (0..n)
        .map(|i| {
            rows.iter()
                .filter_map(|r| r.get(i))
                .map(|c| c.chars().count())
                .max()
                .unwrap_or(0)
        })
        .collect();
    rows.iter()
        .map(|row| {
            let mut line = indent.to_string();
            for (i, cell) in row.iter().enumerate() {
                line.push_str(cell);
                if i + 1 < row.len() {
                    line.push_str(&" ".repeat(widths[i] - cell.chars().count() + 2));
                }
            }
            line.trim_end().to_string()
        })
        .collect()
}

fn header(
    policy: &str,
    robot: &str,
    under: Option<&String>,
    captures: &[(String, String)],
) -> String {
    let mut text = format!("{policy} on {robot}");
    if let Some(under) = under {
        text.push_str(&format!(" under {under:?}"));
    }
    if !captures.is_empty() {
        let given: Vec<String> = captures.iter().map(|(k, v)| format!("{k}={v:?}")).collect();
        text.push_str(&format!(" with {}", given.join(", ")));
    }
    text
}

/// The block for entries whose model and robot names share nothing.
fn names_block(
    clash: &NamesClash,
    entries: &[&Problem],
    run: &crate::spec::bind::RunCall,
) -> Vec<String> {
    let subjects: Vec<String> = entries
        .iter()
        .map(|p| {
            format!(
                "{} {:?}",
                if p.dir == Dir::In { "input" } else { "action" },
                p.entry
            )
        })
        .collect();
    let mut lines = vec![
        format!(
            "  [incompatible] {} -> {}:",
            subjects.join(" and "),
            clash.part
        ),
        format!("      model names: {}", clash.model.join(", ")),
        format!("      robot names: {}", clash.robot.join(", ")),
        "    no name in common, so they cannot be reordered.".to_string(),
    ];
    if clash.model.len() == clash.robot.len() {
        let which = if entries.len() > 1 { "both" } else { "it" };
        lines.push(format!(
            "    fix: if index i means the same joint on both, bind {which} positionally:"
        ));
        lines.push(format!(
            "         {}",
            run.with_positional(entries.iter().map(|p| p.entry.as_str()))
        ));
    } else {
        lines.push(format!(
            "    The model names {} joints and {} has {}, so binding by index cannot work either.",
            clash.model.len(),
            clash.part,
            clash.robot.len()
        ));
    }
    lines
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.problems.len();
        writeln!(
            f,
            "binding refused: {} ({n} problem{})",
            header(
                &self.policy,
                self.robot.as_str(),
                self.under.as_ref(),
                &self.captures
            ),
            if n == 1 { "" } else { "s" }
        )?;
        let mut done = vec![false; n];
        for (i, p) in self.problems.iter().enumerate() {
            if done[i] {
                continue;
            }
            writeln!(f)?;
            let lines = match &p.names {
                Some(clash) => {
                    let group: Vec<usize> = (i..n)
                        .filter(|j| self.problems[*j].names.as_ref() == Some(clash))
                        .collect();
                    for j in &group {
                        done[*j] = true;
                    }
                    let entries: Vec<&Problem> = group.iter().map(|j| &self.problems[*j]).collect();
                    names_block(clash, &entries, &self.run)
                }
                None => {
                    let mut lines = vec![format!("  [{}] {}", p.code, p.head)];
                    lines.extend(p.body.iter().cloned());
                    lines
                }
            };
            for line in lines {
                writeln!(f, "{line}")?;
            }
        }
        Ok(())
    }
}

fn marked(value: String, declared: bool, word: &str) -> String {
    if declared {
        value
    } else {
        format!("{value} ({word})")
    }
}

fn edge_rate(command: Option<&CommandSpec>, hz: f64, control_hz: f64) -> String {
    match command {
        Some(_) => marked(
            format!("{} Hz", num(control_hz)),
            control_hz != hz,
            "default",
        ),
        None => "-".into(),
    }
}

impl Binding {
    /// The human-readable table (§6.8).
    pub fn table(&self) -> String {
        let count = |n: usize, word: &str| format!("{n} {word}{}", if n == 1 { "" } else { "s" });
        let mut out = format!(
            "binding {}: {}, {}, fingerprint {}\nreplan {} Hz\n\n",
            header(
                &self.policy,
                self.robot.as_str(),
                self.under.as_ref(),
                &self.captures
            ),
            count(self.inputs.len(), "input"),
            count(self.actions.len(), "action"),
            self.fingerprint,
            num(self.replan_hz)
        );
        let mut rows = vec![
            ["entry", "dir", "part", "status", "plan", "delivers"]
                .map(String::from)
                .to_vec(),
        ];
        for r in &self.inputs {
            rows.push(vec![
                r.entry.clone(),
                Dir::In.short_name().into(),
                match (&r.part, r.applied) {
                    (Some(p), true) => p.path.applied(),
                    (Some(p), false) => p.path.to_string(),
                    (None, _) => "none".into(),
                },
                r.status.name().into(),
                r.plan.to_string(),
                r.delivers.to_string(),
            ]);
        }
        for r in &self.actions {
            rows.push(vec![
                r.entry.clone(),
                Dir::Out.short_name().into(),
                r.part
                    .as_ref()
                    .map_or_else(|| "none".into(), |p| p.path.to_string()),
                r.status.name().into(),
                r.plan.to_string(),
                r.delivers.to_string(),
            ]);
        }
        for line in columns(&rows, "  ") {
            out.push_str(&line);
            out.push('\n');
        }
        if !self.inputs.is_empty() {
            out.push('\n');
            let mut rows = vec![
                ["input", "contract", "source", "delivery", "freshness"]
                    .map(String::from)
                    .to_vec(),
            ];
            for r in &self.inputs {
                let contract = r
                    .channel
                    .and_then(|c| self.contracts.iter().find(|k| k.channel == c));
                let source = match &r.part {
                    Some(p) if p.event => "event".to_string(),
                    Some(_) => format!("{} Hz", num(r.source_hz)),
                    None => "-".into(),
                };
                rows.push(vec![
                    r.entry.clone(),
                    contract.map_or_else(|| "-".into(), contract::label),
                    source,
                    contract.map_or_else(|| "-".into(), |c| format!("{} Hz", num(c.delivery_hz))),
                    r.max_age.map_or_else(
                        || "-".into(),
                        |m| marked(format!("{} ms", ms(m)), r.max_age_explicit, "derived"),
                    ),
                ]);
            }
            for line in columns(&rows, "  ") {
                out.push_str(&line);
                out.push('\n');
            }
        }
        if !self.actions.is_empty() {
            out.push('\n');
            let mut rows = vec![
                ["action", "grid", "chunk", "valid", "edge", "age cap"]
                    .map(String::from)
                    .to_vec(),
            ];
            for r in &self.actions {
                let command = r.part.as_ref().and_then(|p| p.command.as_ref());
                rows.push(vec![
                    r.entry.clone(),
                    marked(format!("{} Hz", num(r.grid_hz)), r.grid_declared, "default"),
                    format!(
                        "{} × {} ms = {} ms",
                        r.horizon,
                        ms(r.dt),
                        ms(r.dt * r.horizon)
                    ),
                    if r.status == Status::Dropped {
                        "-".into()
                    } else {
                        format!("{} ms", ms(r.valid))
                    },
                    r.part
                        .as_ref()
                        .map_or_else(|| "-".into(), |p| edge_rate(command, p.hz, p.control_hz)),
                    command.map_or_else(
                        || "-".into(),
                        |c| {
                            marked(
                                format!("{} ms", ms(c.max_action_age)),
                                c.max_action_age != CommandSpec::DEFAULT_MAX_ACTION_AGE,
                                "default",
                            )
                        },
                    ),
                ]);
            }
            for line in columns(&rows, "  ") {
                out.push_str(&line);
                out.push('\n');
            }
        }
        if !self.warnings.is_empty() {
            out.push_str("\n  warnings\n");
            for w in &self.warnings {
                out.push_str(&format!("    {w}\n"));
            }
        }
        out
    }
}

impl fmt::Display for Binding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.table())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_read_at_three_significant_digits() {
        for (x, want) in [
            (66.666_666, "66.7"),
            (1000.0, "1000"),
            (15.0, "15"),
            (0.25, "0.25"),
            (2.0 / 3.0, "0.667"),
            (12.5, "12.5"),
            (0.0, "0"),
            (-0.000_1, "0"),
            (-1.5, "-1.5"),
        ] {
            assert_eq!(num(x), want, "{x}");
        }
        assert_eq!(ms(Duration::from_nanos(66_666_667)), "66.7");
        assert_eq!(ms(Duration::from_millis(667)), "667");
    }
}
