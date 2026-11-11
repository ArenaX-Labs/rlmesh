//! Structural candidates, captures, duplicate targets and the emitted
//! table with its delivery contracts (steps 3, 4, 7 and 8).

use super::problem::{BindError, Code, Problem, problem};
use super::table::ActionRow;
use super::table::{Binding, Dir, InputRow, Status};
use super::{Binder, Candidate, MAX_ASSIGNMENTS, Slot, summary, two_columns};
use crate::prelude::*;

use alloc::collections::BTreeMap;
use core::time::Duration;

use crate::wire::part::PartSpec;
use crate::wire::{ChannelId, Fingerprint, StreamContract};

use crate::spec::canonical;
use crate::spec::contract;
use crate::spec::kinds::Kind;
use crate::spec::policy::{Allow, Entry};
use crate::spec::selector::{Selector, below};

impl<'a> Binder<'a> {
    fn slots(&self) -> Vec<Slot<'a>> {
        let mut slots = Vec::new();
        let make = |dir, entry: &'a Entry, input, action| {
            let override_at = self.options.at.get(&entry.name);
            let positional = self.options.positional.contains(&entry.name);
            let selector = match override_at {
                Some(path) => Some(Selector::literal(path)),
                None => entry.at.clone(),
            };
            let robot = selector
                .as_ref()
                .and_then(|s| s.robot.clone())
                .unwrap_or_else(|| self.robot.clone());
            Slot {
                dir,
                entry,
                input,
                action,
                selector,
                overridden: override_at.is_some() || positional,
                robot,
            }
        };
        for input in &self.policy.inputs {
            slots.push(make(Dir::In, &input.entry, Some(input), None));
        }
        for action in &self.policy.actions {
            slots.push(make(Dir::Out, &action.entry, None, Some(action)));
        }
        slots
    }

    fn kind_matches(entry: &Entry, part: &PartSpec) -> bool {
        if entry.kind == Kind::Custom {
            part.kind == entry.kind_name
        } else {
            part.kind == entry.kind.name()
        }
    }

    /// (candidates, considered) of a slot.
    fn candidates(&self, slot: &Slot<'a>) -> (Vec<Candidate<'a>>, Vec<&'a PartSpec>) {
        let scope: Vec<&str> = self.scope.iter().map(String::as_str).collect();
        let mut candidates = Vec::new();
        let mut considered = Vec::new();
        // A qualified selector resolves in its robot's own map, view or not.
        let qualified = slot.selector.as_ref().is_some_and(|s| s.robot.is_some());
        let pool: Vec<&'a PartSpec> = match self.view {
            Some(view) if !qualified => view.iter().collect(),
            _ => self
                .parts
                .iter()
                .filter(|p| p.owner == slot.robot)
                .collect(),
        };
        for part in pool {
            if !Self::kind_matches(slot.entry, part) {
                continue;
            }
            let matched = match &slot.selector {
                Some(sel) => sel.matches(part.path.as_str(), &scope),
                None => below(part.path.as_str(), &scope).then(Vec::new),
            };
            match matched {
                Some(captures) => candidates.push(Candidate { part, captures }),
                None => considered.push(part),
            }
        }
        (candidates, considered)
    }

    pub(super) fn bind(mut self) -> Result<Binding, BindError> {
        let slots = self.slots();
        let found: Vec<(Vec<Candidate<'a>>, Vec<&'a PartSpec>)> =
            slots.iter().map(|s| self.candidates(s)).collect();
        let given: Vec<(String, String)> = self
            .options
            .captures
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let all_captures: Vec<String> = {
            let mut names: Vec<String> = Vec::new();
            for slot in &slots {
                for name in slot.selector.iter().flat_map(|s| s.captures()) {
                    if !names.iter().any(|n| n == name) {
                        names.push(name.to_string());
                    }
                }
            }
            names
        };
        if let Some((name, _)) = given.iter().find(|(n, _)| !all_captures.contains(n)) {
            return Err(BindError::Options(format!(
                "capture {name:?} is not in any selector of {}",
                self.options.name
            )));
        }

        // Step 4: captures, from the required entries.
        let required: Vec<usize> = (0..slots.len()).filter(|i| slots[*i].required()).collect();
        let mut free: Vec<String> = Vec::new();
        for i in &required {
            for name in slots[*i].selector.iter().flat_map(|s| s.captures()) {
                if !given.iter().any(|(n, _)| n == name) && !free.iter().any(|f| f == name) {
                    free.push(name.to_string());
                }
            }
        }
        let fits_given = |c: &Candidate<'_>| {
            c.captures
                .iter()
                .all(|(n, v)| given.iter().all(|(gn, gv)| gn != n || gv == v))
        };
        let domains: Vec<Vec<String>> = free
            .iter()
            .map(|name| {
                let mut domain: Option<Vec<String>> = None;
                for i in &required {
                    if !slots[*i]
                        .selector
                        .iter()
                        .any(|s| s.captures().contains(&name.as_str()))
                    {
                        continue;
                    }
                    let mut values: Vec<String> = Vec::new();
                    for c in found[*i].0.iter().filter(|c| fits_given(c)) {
                        if let Some((_, v)) = c.captures.iter().find(|(n, _)| n == name)
                            && !values.contains(v)
                        {
                            values.push(v.clone());
                        }
                    }
                    domain = Some(match domain {
                        None => values,
                        Some(d) => d.into_iter().filter(|v| values.contains(v)).collect(),
                    });
                }
                domain.unwrap_or_default()
            })
            .collect();
        let count: usize = domains.iter().map(Vec::len).product();
        if count > MAX_ASSIGNMENTS {
            let first = &slots[required.first().copied().unwrap_or(0)];
            let names: Vec<String> = free.iter().map(|n| format!("{{{n}}}")).collect();
            self.problems.push(problem(
                Code::PatternTooBroad,
                first,
                format!(
                    "captures {} allow {count} assignments (at most {MAX_ASSIGNMENTS})",
                    names.join(", ")
                ),
                vec![format!(
                    "    fix: give a capture explicitly, e.g. {}",
                    self.run
                        .with_captures(&[(free[0].clone(), domains[0][0].clone())])
                )],
            ));
            return Err(self.refusal());
        }
        let mut assignments: Vec<Vec<(String, String)>> = vec![given.clone()];
        for (name, domain) in free.iter().zip(&domains) {
            assignments = assignments
                .into_iter()
                .flat_map(|a| {
                    domain.iter().map(move |v| {
                        let mut a = a.clone();
                        a.push((name.clone(), v.clone()));
                        a
                    })
                })
                .collect();
        }
        let tally = |assignment: &[(String, String)]| -> Vec<usize> {
            required
                .iter()
                .map(|i| found[*i].0.iter().filter(|c| c.fits(assignment)).count())
                .collect()
        };
        let survivors: Vec<&Vec<(String, String)>> = assignments
            .iter()
            .filter(|a| tally(a).iter().all(|n| *n == 1))
            .collect();
        let chosen: Vec<(String, String)> = match survivors.as_slice() {
            [one] => (*one).clone(),
            [] => assignments
                .iter()
                .max_by_key(|a| {
                    let ones = tally(a).iter().filter(|n| **n == 1).count();
                    // max_by_key keeps the last maximum; prefer the first.
                    (
                        ones,
                        core::cmp::Reverse(assignments.iter().position(|b| b == *a)),
                    )
                })
                .cloned()
                .unwrap_or_else(|| given.clone()),
            many => {
                let first = &slots[required[0]];
                let names: Vec<String> = free.iter().map(|n| format!("{{{n}}}")).collect();
                let rows: Vec<(String, String)> = many
                    .iter()
                    .map(|a| {
                        let label: Vec<String> = a
                            .iter()
                            .filter(|(n, _)| free.contains(n))
                            .map(|(n, v)| format!("{n}={v:?}"))
                            .collect();
                        let mut paths: Vec<String> = Vec::new();
                        let captured = required.iter().filter(|i| {
                            slots[**i].selector.iter().any(|s| !s.captures().is_empty())
                        });
                        for i in captured {
                            if let Some(c) = found[*i].0.iter().find(|c| c.fits(a)) {
                                let path = c.part.path.to_string();
                                if !paths.contains(&path) {
                                    paths.push(path);
                                }
                            }
                        }
                        (label.join(", "), format!("-> {}", paths.join(", ")))
                    })
                    .collect();
                let head = if free.len() == 1 {
                    format!(
                        "{} has {} values that bind every required entry:",
                        names[0],
                        many.len()
                    )
                } else {
                    format!(
                        "{} have {} assignments that bind every required entry:",
                        names.join(", "),
                        many.len()
                    )
                };
                let mut body: Vec<String> = two_columns(&rows, 2)
                    .into_iter()
                    .map(|l| format!("      {l}"))
                    .collect();
                let pick: Vec<(String, String)> = many[0]
                    .iter()
                    .filter(|(n, _)| free.contains(n))
                    .cloned()
                    .collect();
                body.push(format!("    fix: {}", self.run.with_captures(&pick)));
                self.problems
                    .push(problem(Code::AmbiguousCapture, first, head, body));
                return Err(self.refusal());
            }
        };

        // Steps 3-5: each slot's one candidate under the chosen assignment.
        let mut bound: Vec<Option<&'a PartSpec>> = vec![None; slots.len()];
        let mut absent: Vec<bool> = vec![false; slots.len()];
        let unique: Vec<Option<&'a PartSpec>> = slots
            .iter()
            .zip(&found)
            .map(|(_, (cands, _))| {
                let fitting: Vec<&Candidate<'a>> =
                    cands.iter().filter(|c| c.fits(&chosen)).collect();
                match fitting.as_slice() {
                    [one] => Some(one.part),
                    _ => None,
                }
            })
            .collect();
        for (i, slot) in slots.iter().enumerate() {
            let (cands, considered) = &found[i];
            let unassigned: Vec<&str> = slot
                .selector
                .iter()
                .flat_map(|s| s.captures())
                .filter(|n| !chosen.iter().any(|(c, _)| c == n))
                .collect();
            if !slot.required() && !unassigned.is_empty() {
                self.problems.push(problem(
                    Code::Unbound,
                    slot,
                    format!(
                        "{}: {{{}}} is used only by optional entries, so it is not inferred.",
                        slot.subject(),
                        unassigned[0]
                    ),
                    vec![format!(
                        "    fix: {}",
                        self.run
                            .with_captures(&[(unassigned[0].to_string(), "...".to_string())])
                    )],
                ));
                continue;
            }
            let fitting: Vec<&Candidate<'a>> = cands.iter().filter(|c| c.fits(&chosen)).collect();
            match fitting.as_slice() {
                [one] => bound[i] = Some(one.part),
                [] if !slot.required() => {
                    let allow = match slot.dir {
                        Dir::In => Allow::Blank,
                        Dir::Out => Allow::Drop,
                    };
                    if self.allowed(slot.entry, allow) {
                        absent[i] = true;
                    } else {
                        self.problems.push(problem(
                            Code::NotAllowed,
                            slot,
                            format!(
                                "{}: {} has no {} part {}",
                                slot.name(),
                                slot.robot,
                                slot.entry.kind_name,
                                slot.selector.as_ref().map_or_else(String::new, |s| format!(
                                    "at {:?}",
                                    s.to_string()
                                ))
                            )
                            .trim_end()
                            .to_string(),
                            vec![
                                format!("    needs: {allow} (it is optional)"),
                                format!("    fix: add {:?} to allow", allow.name()),
                            ],
                        ));
                    }
                }
                [] => self.unbound(slot, considered),
                many => {
                    let rows: Vec<(String, String)> = many
                        .iter()
                        .map(|c| (c.part.path.to_string(), summary(c.part)))
                        .collect();
                    let taken: Vec<&PartSpec> = unique
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .filter_map(|(_, p)| *p)
                        .collect();
                    let suggest = many
                        .iter()
                        .find(|c| !taken.iter().any(|t| core::ptr::eq(*t, c.part)))
                        .unwrap_or(&many[0]);
                    let mut body: Vec<String> = two_columns(&rows, 3)
                        .into_iter()
                        .map(|l| format!("      {l}"))
                        .collect();
                    body.push(format!(
                        "    fix: {}",
                        self.run
                            .with_at(&slot.entry.name, suggest.part.path.as_str())
                    ));
                    self.problems.push(problem(
                        Code::Ambiguous,
                        slot,
                        format!("{} matches {} parts:", slot.subject(), many.len()),
                        body,
                    ));
                }
            }
        }

        // Step 6: compatibility.
        let replan = self.policy.hz;
        let mut inputs = Vec::new();
        let mut actions = Vec::new();
        for (i, slot) in slots.iter().enumerate() {
            let status = if absent[i] {
                if slot.dir == Dir::In {
                    Status::Blank
                } else {
                    Status::Dropped
                }
            } else if slot.overridden {
                Status::Override
            } else {
                Status::Bound
            };
            match (slot.input, slot.action) {
                (Some(input), _) => {
                    if absent[i] {
                        if let Some(row) = self.blank(slot, input) {
                            inputs.push(row);
                        }
                    } else if let Some(part) = bound[i]
                        && let Some(row) = self.input(slot, input, part, status, replan)
                    {
                        inputs.push(row);
                    }
                }
                (_, Some(action)) => {
                    if absent[i] {
                        actions.push(self.dropped(slot, action, replan));
                    } else if let Some(part) = bound[i]
                        && let Some(row) = self.action(slot, action, part, status, replan)
                    {
                        actions.push(row);
                    }
                }
                _ => {}
            }
        }

        // Step 7: duplicate targets.
        self.duplicates(&slots, &actions);

        // Contracts, one per bound input stream.
        let contracts = self.contracts(&slots, &inputs, replan);

        // Problems are reported in entry order.
        let order = |p: &Problem| {
            slots
                .iter()
                .position(|s| s.dir == p.dir && s.entry.name == p.entry)
                .unwrap_or(usize::MAX)
        };
        self.problems.sort_by_key(|p| order(p));
        if !self.problems.is_empty() {
            return Err(self.refusal());
        }
        // A view's rows hold their parts as declared, under their own paths:
        // what the streams' keys are made of.
        if self.view.is_some() {
            let declared = |part: &mut Option<PartSpec>| {
                if let Some(p) = part
                    && let Some(own) = self.parts.iter().find(|o| o.channel == p.channel)
                {
                    *p = own.clone();
                }
            };
            inputs.iter_mut().for_each(|r| declared(&mut r.part));
            actions.iter_mut().for_each(|r| declared(&mut r.part));
        }
        let mut binding = Binding {
            policy: self.options.name.clone(),
            robot: self.robot.clone(),
            places: self.places.to_vec(),
            under: self.options.under.as_ref().map(ToString::to_string),
            captures: chosen,
            replan_hz: replan,
            inputs,
            actions,
            contracts,
            warnings: self.warnings,
            fingerprint: Fingerprint::default(),
        };
        binding.fingerprint = canonical::fingerprint(&binding.table_json());
        Ok(binding)
    }

    fn duplicates(&mut self, slots: &[Slot<'a>], actions: &[ActionRow]) {
        for (i, a) in actions.iter().enumerate() {
            let Some(pa) = &a.part else { continue };
            for b in &actions[..i] {
                let Some(pb) = &b.part else { continue };
                let group = |p: &PartSpec| p.command.as_ref().and_then(|c| c.group.clone());
                let same_group =
                    pa.owner == pb.owner && group(pa).is_some() && group(pa) == group(pb);
                if pa.channel != pb.channel && !same_group {
                    continue;
                }
                let Some(slot) = slots
                    .iter()
                    .find(|s| s.dir == Dir::Out && s.entry.name == a.entry)
                else {
                    continue;
                };
                let head = if pa.channel == pb.channel {
                    format!(
                        "actions {:?} and {:?} both command {}.",
                        b.entry,
                        a.entry,
                        self.part_name(pa)
                    )
                } else {
                    format!(
                        "actions {:?} and {:?} command {} and {}, members of one command group {:?}.",
                        b.entry,
                        a.entry,
                        self.part_name(pb),
                        self.part_name(pa),
                        group(pa).unwrap_or_default()
                    )
                };
                self.problems.push(problem(
                    Code::DuplicateTarget,
                    slot,
                    head,
                    vec![
                        "    fix: keep one of them; a part (or a group) has one commanding entry."
                            .into(),
                    ],
                ));
            }
        }
    }

    fn contracts(
        &mut self,
        slots: &[Slot<'a>],
        inputs: &[InputRow],
        replan: f64,
    ) -> Vec<StreamContract> {
        let mut by_stream: BTreeMap<ChannelId, Vec<&InputRow>> = BTreeMap::new();
        let mut order: Vec<ChannelId> = Vec::new();
        for row in inputs {
            if let Some(channel) = row.channel {
                if !by_stream.contains_key(&channel) {
                    order.push(channel);
                }
                by_stream.entry(channel).or_default().push(row);
            }
        }
        let mut contracts = Vec::new();
        let mut rings = Vec::new();
        for channel in order {
            let rows = &by_stream[&channel];
            let Some(part) = rows[0].part.as_ref() else {
                continue;
            };
            let applied = rows[0].applied;
            let history = rows.iter().map(|r| r.history).max().unwrap_or(1);
            let mode = contract::mode(part, rows[0].source_hz, replan, history);
            let bounds: Vec<(Duration, bool)> = rows
                .iter()
                .filter_map(|r| r.max_age.map(|m| (m, r.max_age_explicit)))
                .collect();
            let explicit = bounds.iter().any(|(_, e)| *e);
            let max_age = bounds
                .iter()
                .filter(|(_, e)| *e == explicit)
                .map(|(m, _)| *m)
                .min();
            let c = StreamContract {
                channel,
                mode,
                delivery_hz: replan,
                history,
                max_age,
                max_age_explicit: explicit,
            };
            if let Err(why) = contract::admit(&c, part, applied) {
                let row = rows
                    .iter()
                    .find(|r| r.history == history && (r.max_age_explicit || !explicit))
                    .unwrap_or(&rows[0]);
                if let Some(slot) = slots
                    .iter()
                    .find(|s| s.dir == Dir::In && s.entry.name == row.entry)
                {
                    let why = why.split_once(": ").map_or(why.as_str(), |(_, w)| w);
                    let name = if applied {
                        self.applied_name(part)
                    } else {
                        self.part_name(part)
                    };
                    let head = format!(
                        "{}: {}.",
                        slot.name(),
                        why.replace("it publishes", &format!("{name} publishes"))
                    );
                    let robots_bound = why.contains("max_age_ms")
                        && slots
                            .iter()
                            .filter_map(|s| s.input)
                            .filter(|i| rows.iter().any(|r| r.entry == i.entry.name))
                            .all(|i| i.max_age.is_none());
                    let body = if robots_bound {
                        vec![format!(
                            "    The bound is the robot's own max_age_ms on {name}; a slower delivery cannot meet it.",
                        )]
                    } else {
                        Vec::new()
                    };
                    self.problems
                        .push(problem(Code::Incompatible, slot, head, body));
                }
                continue;
            }
            rings.push((c.clone(), part.clone(), applied));
            contracts.push(c);
        }
        let bytes = contract::ring_bytes(rings.iter().map(|(c, p, a)| (c, p, *a)));
        if bytes > contract::MAX_RING_BYTES
            && let Some(slot) = slots.iter().find(|s| s.dir == Dir::In)
        {
            self.problems.push(problem(
                Code::Incompatible,
                slot,
                format!(
                    "the windows of {} hold {} MiB of samples (at most 512 MiB per controller).",
                    self.options.name,
                    bytes >> 20
                ),
                Vec::new(),
            ));
        }
        contracts
    }
}
