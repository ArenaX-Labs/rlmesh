//! Replaying a recorded session: scrubbing anywhere shows what playing
//! straight through shows, and a paused live session shows where it was.

mod common;

use std::time::Duration;

use common::{MS, World, pid};
use rlmesh_cockpit::replay::History;
use rlmesh_cockpit::{ModelConfig, Replay, View};

#[test]
fn scrubbing_a_replay_shows_what_playing_straight_through_shows() {
    // Long enough for several checkpoints, with a preemption and a loss
    // (a cumulative count) between them.
    let mut w = World::ur10e();
    w.run(12_000);
    w.grant("operator");
    w.run(6_000);
    w.lose(w.channel("arm"));
    w.run(7_000);
    let mut recorded = std::mem::take(&mut w.recorded);
    recorded.sort_by_key(|(at, _)| *at);
    let open = || Replay::new(recorded.clone(), pid("cockpit"), ModelConfig::default());
    let view = |r: &Replay| format!("{:?}", r.model().view(r.at_ns()));

    let mut played = open();
    let (first, last) = played.bounds();
    played.playing = true;
    while played.playing {
        played.tick(Duration::from_millis(16));
    }
    assert_eq!(played.at_ns(), last);
    assert!(view(&played).contains("lost: 1"), "the loss is shown");

    let mut scrubbed = open();
    for t in [last, first + 15_000 * MS, first + 3_000 * MS, last] {
        scrubbed.seek(t);
    }
    assert_eq!(view(&scrubbed), view(&played));

    // Stepping back one message and forward again lands where it was.
    scrubbed.step(false);
    assert!(scrubbed.at_ns() < last);
    scrubbed.step(true);
    assert_eq!(view(&scrubbed), view(&played));
}

#[test]
fn pausing_live_shows_the_session_from_its_last_seconds_alone() {
    let mut w = World::ur10e();
    w.run(4_000);
    w.grant("operator");
    w.run(8_000);
    // Values are decoded at most every 20 ms, from wherever playing
    // started, so only which parts show a value is compared.
    let mut history = History::new(Duration::from_secs(2), usize::MAX);
    for (at, input) in &w.recorded {
        history.push(*at, input);
    }
    let paused = history.replay(pid("cockpit"), ModelConfig::default());
    let shown = |v: View| {
        let tree: Vec<_> = v
            .tree
            .iter()
            .map(|r| (r.label.clone(), r.value.is_some()))
            .collect();
        (v.header.mode, v.header.epoch, v.bindings.len(), tree)
    };
    let live = shown(w.model.view(w.now));
    assert!(!live.3.is_empty());
    assert_eq!(shown(paused.model().view(paused.at_ns())), live);
}
