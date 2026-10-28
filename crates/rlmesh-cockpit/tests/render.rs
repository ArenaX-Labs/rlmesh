//! The cockpit's UI, headless: egui_kittest drives the real widgets
//! (accessibility tree, clicks) and renders frames offscreen with wgpu
//! (lavapipe without a GPU). No display, no browser.
//!
//! Rendered frames go to `$RLMESH_COCKPIT_SNAPSHOTS` (default: this test
//! binary's target tmpdir, `cockpit/`). Without any wgpu adapter the
//! rendering test says so and passes, unless `RLMESH_COCKPIT_REQUIRE_RENDER`
//! is set.
//!
//! `live_session` (ignored) attaches to a running session instead:
//! `RLMESH_COCKPIT_ADDRESS=... cargo test -p rlmesh-cockpit --test render -- --ignored`.

// The helpers report where frames went, or why none were rendered.
#![allow(clippy::print_stderr)]

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use common::{World, pid};
use egui::vec2;
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use rlmesh_cockpit::model::Verb;
use rlmesh_cockpit::ui::{self, ESTOP_RED, Tab, UiState};
use rlmesh_cockpit::{Feed, Model, ModelConfig, View};
use rlmesh_session::wire::{Command, EdgeMode};

type State = (View, UiState, Vec<Verb>);

fn harness(view: View) -> Harness<'static, State> {
    Harness::builder()
        .with_size(vec2(1600.0, 950.0))
        .build_ui_state(
            |ui, (view, state, verbs): &mut State| verbs.extend(ui::show(ui, view, state)),
            (view, UiState::default(), Vec::new()),
        )
}

fn out_dir() -> PathBuf {
    let dir = std::env::var_os("RLMESH_COCKPIT_SNAPSHOTS").map_or_else(
        || PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("cockpit"),
        PathBuf::from,
    );
    std::fs::create_dir_all(&dir).expect("snapshot dir");
    dir
}

/// Renders every tab to `{prefix}-{tab}.png`; `false` when no adapter.
fn render_tabs(harness: &mut Harness<'static, State>, prefix: &str) -> bool {
    for (tab, name) in [
        (Tab::Measured, "measured"),
        (Tab::Timeline, "timeline"),
        (Tab::Bindings, "bindings"),
    ] {
        harness.state_mut().1.tab = tab;
        harness.run();
        let rendered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| harness.render()));
        let image = match rendered {
            Ok(Ok(image)) => image,
            failed => {
                let why = match failed {
                    Ok(Err(err)) => err,
                    Err(panic) => panic
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_string()))
                        .unwrap_or_default(),
                    Ok(Ok(_)) => unreachable!(),
                };
                assert!(
                    std::env::var_os("RLMESH_COCKPIT_REQUIRE_RENDER").is_none(),
                    "render: {why}"
                );
                eprintln!("no wgpu adapter to render with ({why}); skipping the rendered frames");
                return false;
            }
        };
        let path = out_dir().join(format!("{prefix}-{name}.png"));
        image.save(&path).expect("save");
        // The e-stop button is drawn on every tab, red and large.
        let red = image
            .pixels()
            .filter(|p| {
                let [r, g, b, _] = p.0;
                r.abs_diff(ESTOP_RED.r()) < 8
                    && g.abs_diff(ESTOP_RED.g()) < 8
                    && b.abs_diff(ESTOP_RED.b()) < 8
            })
            .count();
        assert!(
            red > 3_000,
            "{name}: {red} e-stop pixels in {}",
            path.display()
        );
        eprintln!("rendered {}", path.display());
    }
    true
}

#[test]
fn every_tab_renders_headless_with_the_estop_visible() {
    let mut w = World::ur10e();
    w.run(3_000);
    // A takeover, a fenced chunk, then the edge holding: something in
    // every panel.
    w.grant("operator");
    w.run(400);
    w.edge_mode = EdgeMode::Hold;
    w.run(60);
    w.edge_mode = EdgeMode::Driving;
    w.run(500);
    let mut h = harness(w.model.view(w.now));
    h.run();
    // The four panels are there, by their accessible labels.
    for label in [
        "Live channels",
        "Declared vs measured",
        "Lease and mode timeline",
        "Bindings",
        "E-STOP",
    ] {
        h.get_by_label(label);
    }
    h.get_by_label("Streams");
    h.get_by_label("Edges");
    render_tabs(&mut h, "fixture");
}

#[test]
fn estop_is_one_click_and_arm_asks_first() {
    let mut w = World::ur10e();
    w.run(100);
    let mut h = harness(w.model.view(w.now));
    h.run();
    h.get_by_label("E-STOP").click();
    h.run();
    assert_eq!(h.state().2, [Verb::Estop], "one click, no confirmation");

    // ACTIVE: arm is not offered.
    assert!(!h.state().0.header.can_arm);
    w.verb("cockpit", Command::Disarm).expect("disarm");
    w.publish_snapshot();
    let mut h = harness(w.model.view(w.now));
    h.run();
    h.get_by_label("Arm…").click();
    h.run();
    assert!(h.state().2.is_empty(), "arm waits for the confirmation");
    h.get_by_label("Arm the session?");
    h.get_by_label("Cancel").click();
    h.run();
    assert!(
        h.state().2.is_empty() && !h.state().1.confirm_arm,
        "cancelled"
    );
    h.get_by_label("Arm…").click();
    h.run();
    h.get_by_label("Arm").click();
    h.run();
    assert_eq!(h.state().2, [Verb::Arm]);
    assert!(!h.state().1.confirm_arm);
}

#[test]
#[ignore = "needs a live session: RLMESH_COCKPIT_ADDRESS"]
fn live_session() {
    let address: rlmesh_session::SessionAddress = std::env::var("RLMESH_COCKPIT_ADDRESS")
        .expect("RLMESH_COCKPIT_ADDRESS")
        .parse()
        .expect("address");
    let seconds: f64 = std::env::var("RLMESH_COCKPIT_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8.0);
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    let _entered = runtime.enter();
    let participant = Arc::new(
        runtime
            .block_on(address.join(
                "cockpit-snapshot",
                vec![rlmesh_session::wire::Role::Observer],
                rlmesh_session::wire::PriorityClass::Operator,
                Duration::from_secs(10),
            ))
            .expect("join"),
    );
    let feed = runtime
        .block_on(Feed::start(participant, Arc::new(|| {})))
        .expect("feed");
    let mut model = Model::new(pid("cockpit-snapshot"), ModelConfig::default());
    let end = std::time::Instant::now() + Duration::from_secs_f64(seconds);
    while std::time::Instant::now() < end {
        feed.drain(&mut model, None);
        std::thread::sleep(Duration::from_millis(50));
    }
    feed.drain(&mut model, None);
    let mut h = harness(model.view(feed.now_ns()));
    h.run();
    assert!(
        render_tabs(&mut h, "live"),
        "rendering needs a wgpu adapter"
    );
}
