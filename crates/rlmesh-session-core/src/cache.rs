//! The controller's input cache and history windows (§8.8), sans-IO.
//!
//! [`InputCache`] keeps one slot per bound stream (shared by the entries
//! bound to it) plus a ring as deep as the deepest window on it, fed sample
//! by sample and snapshotted at each policy start. Every time is passed in
//! (`now_ns`, session time). The rules (snapshot, freshness, stale required
//! inputs, never-received inputs, windows and gaps, epochs, provenance and
//! applied streams) are documented on `rlmesh_session::cache`, whose
//! `Inputs` feeds this cache from the network.

use std::collections::{BTreeMap, VecDeque};
use std::time::Duration;

use crate::spec::Binding;
use crate::wire::{Applied, ChannelId, Epoch, InputRange};
use bytes::Bytes;

/// One sample as the cache keeps it.
#[derive(Clone, Debug, PartialEq)]
pub struct Cached {
    /// Its seq.
    pub seq: u64,
    /// Its capture time in session time; `None` without a clock mapping
    /// for its publisher (never fresh).
    pub capture_ns: Option<i64>,
    /// The publisher's mapping error ε_pub.
    pub error_ns: u64,
    /// The packed value (empty for an applied tick that dispatched
    /// nothing).
    pub payload: Bytes,
    /// Applied metadata, on an applied stream.
    pub applied: Option<Applied>,
}

impl Cached {
    fn has_value(&self) -> bool {
        self.applied.is_none_or(|a| a.present)
    }
}

/// What the cache knows of one bound input.
#[derive(Clone, Debug, PartialEq)]
struct EntryRule {
    entry: String,
    channel: Option<ChannelId>,
    history: usize,
    max_age: Option<Duration>,
    hold: bool,
    warmup: bool,
    event: bool,
}

#[derive(Debug, Default)]
struct Ring {
    epoch: Epoch,
    depth: usize,
    /// An event stream: its last value survives a new epoch.
    event: bool,
    /// In seq order.
    samples: VecDeque<Cached>,
    /// The newest sample carrying a value (an applied stream's last
    /// dispatched value outlives the ring).
    held: Option<(u64, Bytes)>,
}

impl Ring {
    fn push(&mut self, epoch: Epoch, sample: Cached) {
        if epoch < self.epoch {
            return;
        }
        if epoch > self.epoch {
            self.epoch = epoch;
            if !self.event {
                self.samples.clear();
                self.held = None;
            }
        }
        if sample.has_value() && self.held.as_ref().is_none_or(|(seq, _)| sample.seq > *seq) {
            self.held = Some((sample.seq, sample.payload.clone()));
        }
        // Live samples arrive in order; recovered ones may fill a gap.
        match self.samples.binary_search_by_key(&sample.seq, |s| s.seq) {
            Ok(_) => {}
            Err(i) if i == 0 && self.samples.len() >= self.depth => {}
            Err(i) => {
                self.samples.insert(i, sample);
                while self.samples.len() > self.depth {
                    self.samples.pop_front();
                }
            }
        }
    }
}

/// One input in a snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct InputSnapshot {
    /// The entry.
    pub entry: String,
    /// Its samples, oldest first: one, or the window (empty when blank).
    pub samples: Vec<Cached>,
    /// The newest sample's conservative age (`None`: blank, or unmapped).
    pub age_ns: Option<i64>,
    /// Older than its bound, unmapped or from the future (delivered only
    /// under `allow: hold`).
    pub stale: bool,
    /// Stamped in the future beyond the error bounds.
    pub future: bool,
    /// Window slots filled by repeating the first sample (`allow: warmup`).
    pub padded: u32,
    /// Bound to a constant (an absent optional part).
    pub blank: bool,
}

/// What one policy start sees (`obs` and `obs.meta`).
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    /// Every input, in policy order.
    pub inputs: Vec<InputSnapshot>,
    /// The newest capture minus the oldest, over the inputs' newest samples.
    pub skew_ns: i64,
    /// Per stream, the seq range used: a chunk's `based_on`.
    pub based_on: Vec<InputRange>,
}

impl Snapshot {
    /// The input of `entry`.
    pub fn input(&self, entry: &str) -> Option<&InputSnapshot> {
        self.inputs.iter().find(|i| i.entry == entry)
    }
}

/// Why a start is skipped, by entry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Skip {
    /// Never received in this epoch, or a window not yet full
    /// (`waiting_for`).
    pub waiting: Vec<String>,
    /// Stale without `allow: hold` (`stale_input`).
    pub stale: Vec<String>,
    /// A seq gap inside the window (`window_gap`).
    pub gaps: Vec<String>,
}

impl Skip {
    fn is_empty(&self) -> bool {
        self.waiting.is_empty() && self.stale.is_empty() && self.gaps.is_empty()
    }
}

/// The sans-IO cache of one binding's inputs.
#[derive(Debug)]
pub struct InputCache {
    rules: Vec<EntryRule>,
    rings: BTreeMap<ChannelId, Ring>,
}

impl InputCache {
    /// The cache for `binding`'s inputs.
    pub fn new(binding: &Binding) -> Self {
        let rules: Vec<EntryRule> = binding
            .inputs
            .iter()
            .map(|r| EntryRule {
                entry: r.entry.clone(),
                channel: r.channel,
                history: r.history.max(1) as usize,
                max_age: r.max_age,
                hold: r.hold,
                warmup: r.warmup,
                event: r.part.as_ref().is_some_and(|p| p.event),
            })
            .collect();
        let mut rings: BTreeMap<ChannelId, Ring> = BTreeMap::new();
        for rule in &rules {
            if let Some(channel) = rule.channel {
                let ring = rings.entry(channel).or_default();
                ring.depth = ring.depth.max(rule.history);
                ring.event |= rule.event;
            }
        }
        Self { rules, rings }
    }

    /// The bound streams.
    pub fn channels(&self) -> impl Iterator<Item = ChannelId> + '_ {
        self.rings.keys().copied()
    }

    /// The deepest window on `channel` (its ring depth).
    pub fn depth(&self, channel: ChannelId) -> usize {
        self.rings.get(&channel).map_or(0, |r| r.depth)
    }

    /// The session moved to `epoch`: state streams start empty, event
    /// streams keep their last value; older samples are dropped from now on.
    pub fn advance(&mut self, epoch: Epoch) {
        for ring in self.rings.values_mut() {
            if ring.epoch < epoch {
                ring.epoch = epoch;
                if !ring.event {
                    ring.samples.clear();
                    ring.held = None;
                }
            }
        }
    }

    /// Takes one sample of `channel` from `epoch`.
    pub fn push(&mut self, channel: ChannelId, epoch: Epoch, sample: Cached) {
        if let Some(ring) = self.rings.get_mut(&channel) {
            ring.push(epoch, sample);
        }
    }

    /// The seqs a window on `channel` misses: `(after, until)` of the first
    /// gap among the newest `depth` samples, for a history query.
    pub fn gap(&self, channel: ChannelId) -> Option<(Epoch, u64, u64)> {
        let ring = self.rings.get(&channel)?;
        let tail: Vec<&Cached> = ring.samples.iter().rev().take(ring.depth).collect();
        tail.windows(2)
            .find(|w| w[0].seq != w[1].seq + 1)
            .map(|w| (ring.epoch, w[1].seq, w[0].seq - 1))
    }

    /// The snapshot at a policy start at session time `now_ns`, with this
    /// controller's mapping error `self_error_ns`: every input, or why the
    /// start is skipped.
    pub fn snapshot(&self, now_ns: i64, self_error_ns: u64) -> Result<Snapshot, Skip> {
        let mut skip = Skip::default();
        let mut inputs = Vec::with_capacity(self.rules.len());
        let mut used: BTreeMap<ChannelId, (u64, u64)> = BTreeMap::new();
        for rule in &self.rules {
            let Some(channel) = rule.channel else {
                inputs.push(InputSnapshot {
                    entry: rule.entry.clone(),
                    samples: Vec::new(),
                    age_ns: None,
                    stale: false,
                    future: false,
                    padded: 0,
                    blank: true,
                });
                continue;
            };
            let Some(ring) = self.rings.get(&channel) else {
                continue;
            };
            let Some(newest) = ring.samples.back() else {
                skip.waiting.push(rule.entry.clone());
                continue;
            };
            // Freshness of the newest sample (§8.7, §9.1).
            let slack =
                i64::try_from(newest.error_ns.saturating_add(self_error_ns)).unwrap_or(i64::MAX);
            let (age_ns, future) = match newest.capture_ns {
                Some(capture) => (
                    Some(now_ns.saturating_sub(capture).saturating_add(slack)),
                    capture > now_ns.saturating_add(slack),
                ),
                None => (None, false),
            };
            let stale = future
                || match (rule.max_age, age_ns) {
                    (None, _) => false,
                    (Some(_), None) => true,
                    (Some(bound), Some(age)) => {
                        u128::try_from(age).is_ok_and(|age| age > bound.as_nanos())
                    }
                };
            if stale && !rule.hold {
                skip.stale.push(rule.entry.clone());
                continue;
            }
            // The window, by seq.
            let want = rule.history;
            let mut window: Vec<Cached> = ring
                .samples
                .iter()
                .rev()
                .take(want)
                .rev()
                .cloned()
                .collect();
            if window.windows(2).any(|w| w[1].seq != w[0].seq + 1) {
                skip.gaps.push(rule.entry.clone());
                continue;
            }
            // Applied ticks that dispatched nothing keep the previous value.
            let older = ring.samples.len() - window.len();
            let first = window.first().map_or(0, |s| s.seq);
            let mut carried = ring
                .samples
                .iter()
                .take(older)
                .rev()
                .find(|s| s.has_value())
                .map(|s| s.payload.clone())
                .or_else(|| {
                    ring.held
                        .as_ref()
                        .filter(|(seq, _)| *seq < first)
                        .map(|(_, value)| value.clone())
                });
            for sample in &mut window {
                if sample.has_value() {
                    carried = Some(sample.payload.clone());
                } else if let Some(value) = &carried {
                    sample.payload = value.clone();
                }
            }
            if window
                .iter()
                .any(|s| !s.has_value() && s.payload.is_empty())
            {
                skip.waiting.push(rule.entry.clone());
                continue;
            }
            let mut padded = 0;
            if window.len() < want {
                if !rule.warmup {
                    skip.waiting.push(rule.entry.clone());
                    continue;
                }
                padded = (want - window.len()) as u32;
                let first = window[0].clone();
                window.splice(0..0, std::iter::repeat_n(first, padded as usize));
            }
            let (first, last) = (window[padded as usize].seq, newest.seq);
            used.entry(channel)
                .and_modify(|r| *r = (r.0.min(first), r.1.max(last)))
                .or_insert((first, last));
            inputs.push(InputSnapshot {
                entry: rule.entry.clone(),
                samples: window,
                age_ns,
                stale,
                future,
                padded,
                blank: false,
            });
        }
        if !skip.is_empty() {
            return Err(skip);
        }
        let captures: Vec<i64> = inputs
            .iter()
            .filter_map(|i| i.samples.last().and_then(|s| s.capture_ns))
            .collect();
        let skew_ns = match (captures.iter().max(), captures.iter().min()) {
            (Some(max), Some(min)) => max - min,
            _ => 0,
        };
        Ok(Snapshot {
            inputs,
            skew_ns,
            based_on: used
                .into_iter()
                .map(|(channel, (seq_first, seq_last))| InputRange {
                    channel,
                    seq_first,
                    seq_last,
                })
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{Policy, RunOptions, bind, normalise};
    use crate::wire::version::SPEC_VERSION;
    use crate::wire::{EdgeMode, LeaseToken, ParticipantId};

    const MS: i64 = 1_000_000;

    /// At 200 Hz: ft at 1 kHz windowed 4 deep with warmup (derived bound
    /// 15 ms), q at 1 kHz latest (bound 10 ms, hold allowed when asked), the
    /// arm's applied stream, and a blank camera.
    fn cache(allow_hold: bool) -> (InputCache, ChannelId, ChannelId, ChannelId) {
        let mut parts = normalise(
            &ParticipantId::new("robot").expect("id"),
            SPEC_VERSION,
            r#"{"ft": {"kind": "wrench", "frame": "tool0", "hz": 1000},
                "arm": {"kind": "joints", "count": 2, "hz": 1000, "command": "position"},
                "cam": {"kind": "camera", "width": 4, "height": 4, "hz": 30},
                "task": {"kind": "text"}}"#,
        )
        .expect("robot");
        for (i, p) in parts.iter_mut().enumerate() {
            p.channel = ChannelId(10 * (i as u32 + 1));
            p.applied_channel = p
                .command
                .as_ref()
                .map(|_| ChannelId(10 * (i as u32 + 1) + 1));
        }
        let hold = if allow_hold {
            r#", "allow": ["hold"]"#
        } else {
            ""
        };
        let policy = Policy::parse(&format!(
            r#"{{"hz": 200,
                "inputs": {{"ft": {{"kind": "wrench", "at": "ft", "history": 4, "allow": ["warmup"]}},
                           "q": {{"kind": "joints", "at": "arm", "fields": ["pos"], "max_age_ms": 10{hold}}},
                           "last": {{"kind": "joints", "at": "arm", "source": "applied"}},
                           "prompt": {{"kind": "text", "at": "task"}},
                           "low": {{"kind": "camera", "at": "low", "optional": true, "size": [4, 4],
                                    "allow": ["blank"]}}}}}}"#
        ))
        .expect("policy");
        let binding = bind(&policy, &parts, &RunOptions::named("p")).expect("bound");
        let mut cache = InputCache::new(&binding);
        cache.push(ChannelId(40), Epoch(1), sample(1, 0, b'a'));
        (cache, ChannelId(10), ChannelId(20), ChannelId(21))
    }

    fn sample(seq: u64, capture_ms: i64, value: u8) -> Cached {
        Cached {
            seq,
            capture_ns: Some(capture_ms * MS),
            error_ns: 0,
            payload: Bytes::from(vec![value; 8]),
            applied: None,
        }
    }

    fn applied(seq: u64, capture_ms: i64, value: Option<u8>) -> Cached {
        Cached {
            payload: value.map_or_else(Bytes::new, |v| Bytes::from(vec![v; 8])),
            applied: Some(Applied {
                mode: if value.is_some() {
                    EdgeMode::Driving
                } else {
                    EdgeMode::Idle
                },
                present: value.is_some(),
                lease_token: LeaseToken(1),
                chunk_seq: 1,
                action_index: 0,
            }),
            ..sample(seq, capture_ms, 0)
        }
    }

    #[test]
    fn the_first_start_waits_for_every_input_and_blanks_never_wait() {
        let (mut cache, ft, q, last) = cache(false);
        let skip = cache.snapshot(10 * MS, 0).expect_err("nothing yet");
        assert_eq!(skip.waiting, ["ft", "q", "last"]);
        cache.push(ft, Epoch(1), sample(1, 9, 1));
        cache.push(q, Epoch(1), sample(1, 9, 2));
        cache.push(last, Epoch(1), applied(1, 9, Some(3)));
        let snap = cache.snapshot(10 * MS, 0).expect("warmup pads ft");
        let ft_input = snap.input("ft").expect("ft");
        assert_eq!((ft_input.samples.len(), ft_input.padded), (4, 3));
        assert!(snap.input("low").expect("low").blank);
        assert_eq!(
            snap.based_on,
            [
                InputRange::one(ft, 1),
                InputRange::one(q, 1),
                InputRange::one(last, 1),
                InputRange::one(ChannelId(40), 1)
            ]
        );
    }

    #[test]
    fn a_stale_required_input_skips_the_start_unless_held() {
        let (mut cache, ft, q, last) = cache(false);
        cache.push(ft, Epoch(1), sample(1, 0, 1));
        cache.push(q, Epoch(1), sample(1, 0, 2));
        cache.push(last, Epoch(1), applied(1, 0, Some(3)));
        // q's bound is 10 ms; at 11 ms its conservative age (plus 0.5 ms of
        // mapping error on each side) is past it.
        assert!(cache.snapshot(9 * MS, 0).is_ok());
        let mut late = sample(2, 0, 2);
        late.error_ns = 500_000;
        cache.push(q, Epoch(1), late);
        assert_eq!(
            cache
                .snapshot(9 * MS + 600_000, 500_000)
                .expect_err("stale")
                .stale,
            ["q"]
        );
        let (mut held, ft, q, last) = self::cache(true);
        held.push(ft, Epoch(1), sample(1, 0, 1));
        held.push(q, Epoch(1), sample(1, 0, 2));
        held.push(last, Epoch(1), applied(1, 0, Some(3)));
        let snap = held.snapshot(12 * MS, 0).expect("held");
        assert!(snap.input("q").expect("q").stale, "delivered, flagged");
        // Unmapped or future samples are never fresh.
        let mut unmapped = sample(2, 0, 2);
        unmapped.capture_ns = None;
        held.push(q, Epoch(1), unmapped);
        assert!(
            held.snapshot(MS, 0)
                .expect("held")
                .input("q")
                .expect("q")
                .stale
        );
        held.push(q, Epoch(1), sample(3, 100, 2));
        let q_input = held
            .snapshot(MS, 0)
            .expect("held")
            .input("q")
            .cloned()
            .expect("q");
        assert!(q_input.future && q_input.stale);
    }

    #[test]
    fn a_gap_inside_a_window_skips_until_it_is_filled_or_a_new_window_exists() {
        let (mut cache, ft, q, last) = cache(false);
        cache.push(q, Epoch(1), sample(1, 9, 2));
        cache.push(last, Epoch(1), applied(1, 9, Some(3)));
        for seq in [1, 2, 4, 5] {
            cache.push(ft, Epoch(1), sample(seq, 9, seq as u8));
        }
        assert_eq!(cache.snapshot(10 * MS, 0).expect_err("gap").gaps, ["ft"]);
        assert_eq!(cache.gap(ft), Some((Epoch(1), 2, 3)));
        // The host store fills it.
        cache.push(ft, Epoch(1), sample(3, 9, 3));
        let window: Vec<u64> = cache
            .snapshot(10 * MS, 0)
            .expect("filled")
            .input("ft")
            .expect("ft")
            .samples
            .iter()
            .map(|s| s.seq)
            .collect();
        assert_eq!(window, [2, 3, 4, 5]);
        // Or a complete new window: 7..=10 after a lost 6.
        for seq in 7..=10 {
            cache.push(ft, Epoch(1), sample(seq, 9, 0));
        }
        assert!(cache.snapshot(10 * MS, 0).is_ok());
        assert_eq!(cache.gap(ft), None);
    }

    #[test]
    fn epochs_never_seed_each_other_and_applied_gaps_carry_the_value() {
        let (mut cache, ft, q, last) = cache(false);
        for seq in 1..=4 {
            cache.push(ft, Epoch(1), sample(seq, 9, 0));
        }
        cache.push(q, Epoch(1), sample(1, 9, 2));
        cache.push(last, Epoch(1), applied(1, 9, Some(7)));
        cache.push(last, Epoch(1), applied(2, 9, None));
        let snap = cache.snapshot(10 * MS, 0).expect("ok");
        assert_eq!(
            snap.input("last").expect("last").samples[0].payload,
            Bytes::from(vec![7_u8; 8]),
            "a tick that dispatched nothing keeps the previous value"
        );
        // An arm: state streams start empty and older samples are dropped;
        // the prompt (an event) is kept.
        cache.advance(Epoch(2));
        assert_eq!(
            cache.snapshot(10 * MS, 0).expect_err("cleared").waiting,
            ["ft", "q", "last"]
        );
        cache.push(ft, Epoch(1), sample(5, 9, 0));
        cache.push(ft, Epoch(2), sample(6, 9, 9));
        cache.push(q, Epoch(2), sample(2, 9, 2));
        cache.push(last, Epoch(2), applied(3, 9, Some(1)));
        let snap = cache.snapshot(10 * MS, 0).expect("warmup");
        let ft_input = snap.input("ft").cloned().expect("ft");
        assert_eq!((ft_input.padded, ft_input.samples[3].seq), (3, 6));
        assert_eq!(snap.input("prompt").expect("prompt").samples[0].seq, 1);
    }
}
