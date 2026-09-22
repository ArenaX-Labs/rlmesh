//! Golden behavior fingerprint for workflow edition 2026.06.
//!
//! Each scenario drives the runtime through one paragraph of
//! `docs/editions/2026.06.md` with the shared fake harness and serializes what
//! the edition promises an observer -- the episode ledger, the hook-event order
//! under each episode id, the model lifecycle, the seeds a reset threads, the
//! refusals -- into a deterministic JSON record compared byte for byte against a
//! committed fixture. 2026.06 is sealed: if a test here fails you are BREAKING
//! THE BEHAVIOR CONTRACT of an edition already in the field, and the fix is to
//! seal a NEW edition, never to update the fixture. Nothing time-dependent
//! (wall-clock durations, telemetry latencies, lane interleavings a sleep
//! decides) is recorded, and runtime-minted UUIDv7 episode ids appear as
//! first-seen placeholders, so the record is a pure function of the edition's
//! semantics.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use rlmesh_proto::core::v1::AutoresetMode;
use rlmesh_proto::spaces::v1::{MetaMap, meta_value};
use rlmesh_proto::{Edition, EditionDefaults};
use rlmesh_runtime::{RuntimeDriver, RuntimeError, RuntimeReport, RuntimeSessionSpec};
use serde_json::{Value, json};

mod common;

use common::*;

/// Characters of a refusal message pinned by the fingerprint: enough to name
/// the rule, short enough to stop before the build-stamped edition list that
/// spells the current cohort.
const REFUSAL_PREFIX_CHARS: usize = 100;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/2026.06")
        .join(format!("{name}.json"))
}

/// Compare one scenario's record with its committed fixture. `RLMESH_UPDATE_FIXTURES=1`
/// rewrites the fixtures instead of asserting; it exists to seal an edition once,
/// not to absorb a behavior change.
fn assert_frozen(name: &str, record: &Value) {
    let got = format!(
        "{}\n",
        serde_json::to_string_pretty(record).expect("the scenario record serializes")
    );
    let path = fixture_path(name);
    if std::env::var("RLMESH_UPDATE_FIXTURES").as_deref() == Ok("1") {
        let parent = path.parent().expect("fixture path has a parent");
        std::fs::create_dir_all(parent).expect("the fixture directory is writable");
        std::fs::write(&path, &got).expect("the fixture is writable");
        return;
    }
    let want = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("{name}: cannot read fixture {}: {err}", path.display()));
    assert_eq!(
        got,
        want,
        "{name}: workflow edition 2026.06 behavior changed (this is a sealed-edition break, not a \
         golden to update); the fix is a new edition, never an edit of {}",
        path.display()
    );
}

/// Stable stand-ins for the runtime-minted UUIDv7 episode ids, numbered in the
/// order the record first mentions them.
#[derive(Default)]
struct Ids(Vec<String>);

impl Ids {
    fn label(&mut self, id: &str) -> String {
        let position = match self.0.iter().position(|seen| seen == id) {
            Some(position) => position,
            None => {
                self.0.push(id.to_string());
                self.0.len() - 1
            }
        };
        format!("ep-{}", position + 1)
    }

    fn labels(&mut self, ids: &[String]) -> Vec<String> {
        ids.iter().map(|id| self.label(id)).collect()
    }

    /// First-seen position of a label, for ordering a set of ids.
    fn rank(&self, label: &str) -> usize {
        label
            .trim_start_matches("ep-")
            .parse::<usize>()
            .unwrap_or_default()
    }
}

/// The episode ledger a finished run reports, minus `duration_ms` (wall clock).
fn report_record(report: &RuntimeReport) -> Value {
    let episodes: Vec<Value> = report
        .episodes
        .iter()
        .map(|episode| {
            json!({
                "episode_index": episode.episode_index,
                "env_index": episode.env_index,
                "seed": episode.seed,
                "trial_index": episode.trial_index,
                "step_count": episode.step_count,
                "cumulative_reward": episode.cumulative_reward,
                "terminated": episode.terminated,
                "truncated": episode.truncated,
                "success": episode.success,
            })
        })
        .collect();
    json!({
        "total_steps": report.total_steps,
        "total_episodes": report.total_episodes,
        "episodes": episodes,
    })
}

/// Every per-episode hook event in arrival order, by the ids it named.
fn hook_events(hooks: &RecordingHooks, ids: &mut Ids) -> Value {
    let events = hooks
        .events
        .lock()
        .expect("event recorder lock poisoned")
        .clone();
    let rows: Vec<Value> = events
        .into_iter()
        .map(|(kind, episode_ids)| json!({"event": kind, "episodes": ids.labels(&episode_ids)}))
        .collect();
    Value::Array(rows)
}

/// Every model call in arrival order: the predict/evict lifecycle per episode.
/// A predict names its lanes in lane order, but one ResetAdapter evicts a *set*
/// of ended ids whose order inside the request is decided by when each episode
/// ended relative to an in-flight predict -- a race, not an edition promise --
/// so evicted ids are recorded in first-seen order. That sort relies on
/// `hook_events` running first in every scenario record, so each id is already
/// ranked by the order the hooks named it: a scenario whose first mention of an
/// id were an evict batch would rank ids by the very race this hides.
fn model_lifecycle(model: &TestModel, ids: &mut Ids) -> Value {
    let calls = model
        .lifecycle
        .lock()
        .expect("lifecycle lock poisoned")
        .clone();
    let rows: Vec<Value> = calls
        .into_iter()
        .map(|(call, episode_ids)| {
            let mut labels = ids.labels(&episode_ids);
            if call == "evict" {
                labels.sort_by_key(|label| ids.rank(label));
            }
            json!({"call": call, "episodes": labels})
        })
        .collect();
    Value::Array(rows)
}

/// Each predict's history rows and its own observation, as `(step, first byte)`.
fn ledger(model: &TestModel) -> Value {
    let entries = model.ledger.lock().expect("ledger poisoned").clone();
    let rows: Vec<Value> = entries
        .into_iter()
        .map(|(history, own)| {
            json!({
                "history": history.iter().map(|(step, byte)| json!([step, byte])).collect::<Vec<_>>(),
                "own": json!([own.0, own.1]),
            })
        })
        .collect();
    Value::Array(rows)
}

fn meta_text(value: &rlmesh_proto::spaces::v1::MetaValue) -> Value {
    match value.kind.as_ref() {
        Some(meta_value::Kind::Text(text)) => json!(text),
        Some(meta_value::Kind::Integer(number)) => json!(number),
        Some(meta_value::Kind::Number(number)) => json!(number),
        Some(meta_value::Kind::Bool(flag)) => json!(flag),
        _ => json!("<non-scalar>"),
    }
}

/// One `MetaMap` as sorted `[key, value]` pairs (`null` when absent).
fn meta_record(meta: &Option<MetaMap>) -> Value {
    let Some(meta) = meta else {
        return Value::Null;
    };
    let mut entries: Vec<(&String, Value)> = meta
        .entries
        .iter()
        .map(|(key, value)| (key, meta_text(value)))
        .collect();
    entries.sort_by(|left, right| left.0.cmp(right.0));
    Value::Array(
        entries
            .into_iter()
            .map(|(key, value)| json!([key, value]))
            .collect(),
    )
}

fn meta_records(metas: &[Option<MetaMap>]) -> Value {
    Value::Array(metas.iter().map(meta_record).collect())
}

/// Every observation the runtime emitted, by the episodes it was attributed to.
fn observations(hooks: &RecordingHooks, ids: &mut Ids) -> Value {
    let emitted = hooks
        .emitted_observations
        .lock()
        .expect("emitted observation recorder lock poisoned")
        .clone();
    let rows: Vec<Value> = emitted
        .into_iter()
        .map(|observation| {
            json!({
                "episodes": ids.labels(&observation.episode_ids),
                "observation": observation.observation,
                "raw_observation": observation.raw_observation,
                "infos": meta_record(&observation.infos),
            })
        })
        .collect();
    Value::Array(rows)
}

/// The refusal a spec the runtime must not drive produces, truncated to its
/// build-independent prefix.
async fn refusal_prefix(spec: RuntimeSessionSpec) -> Value {
    let error = RuntimeDriver::new(
        spec,
        TestEnv::default(),
        TestModel::default(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .expect_err("the runtime refuses the spec");
    assert!(
        matches!(error, RuntimeError::InvalidSpec(_)),
        "a refused spec fails validation, not the run: {error}"
    );
    json!(
        error
            .to_string()
            .chars()
            .take(REFUSAL_PREFIX_CHARS)
            .collect::<String>()
    )
}

/// The same refusal for an edition name that never reaches a typed spec:
/// `workflow_edition` is an [`rlmesh_proto::Edition`], so a name outside the
/// retained list is refused one step earlier, at the shared parse boundary every
/// string-carrying caller goes through. The text is the one `validate` uses, in
/// the same error, so the contract an operator sees is unchanged.
fn unparsed_edition_refusal_prefix(edition: &str) -> Value {
    let message = rlmesh_proto::parse_retained_edition(edition)
        .expect_err("the runtime refuses an edition outside the retained list");
    json!(
        RuntimeError::InvalidSpec(message)
            .to_string()
            .chars()
            .take(REFUSAL_PREFIX_CHARS)
            .collect::<String>()
    )
}

/// Every [`EditionDefaults`] field the driver reads, mapped to the scenario that
/// pins its value. The destructuring is exhaustive on purpose: a new field does
/// not compile until it is named here alongside the fixture that observes it.
#[test]
fn every_edition_defaults_field_is_pinned_by_a_scenario() {
    let EditionDefaults {
        // 05-default-truncation-bound: an episode with no explicit cap truncates
        // at this bound under driver-owned resets.
        default_max_episode_steps,
        // 09-episode-id-and-trial-authority delivers it; 03 proves NEXT_STEP
        // mints no ordinal at all.
        trial_index_option_key,
        // 01-disabled-single-env-episode (driver-owned) vs 02-next-step-roll-timing
        // (env-owned) split on exactly this set.
        driver_owned_reset_modes,
        // 01-disabled-single-env-episode reports through `is_success`; the
        // `success` spelling and its numeric coercion ride the vector lanes of
        // 02 and 03. Which keys count, and in what priority, is the promise —
        // pinned as the whole list here.
        success_info_keys,
        // A served env's promise, not the driver's: the wire adapters
        // (`rlmesh::env::wire`) report under it at the session's pinned edition.
        // No driver scenario observes it, so its value is pinned here.
        conformance_warning_info_key,
    } = *rlmesh_proto::defaults(Edition::E2026_06);

    assert_eq!(default_max_episode_steps, 100_000);
    assert_eq!(trial_index_option_key, "trial_index");
    assert_eq!(
        driver_owned_reset_modes,
        [AutoresetMode::Disabled, AutoresetMode::Unspecified]
    );
    assert_eq!(success_info_keys, ["is_success", "success", "task_success"]);
    assert_eq!(conformance_warning_info_key, "rlmesh.conformance.warning");
}

// ---------------------------------------------------------------------------
// 1. A DISABLED single-env episode: reset -> predict -> step -> terminate.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scenario_01_disabled_single_env_episode() {
    // The env stamps Gymnasium's `is_success` on the terminal step, so the
    // ledger's derived `success` is pinned true here; the `success` spelling and
    // its numeric coercion to false ride the vector lanes of scenarios 2 and 3.
    let env = TestEnv {
        final_info: Some(meta_map("is_success", meta_value::Kind::Bool(true))),
        ..Default::default()
    };
    let model = TestModel::default();
    let hooks = Arc::new(RecordingHooks::default());
    let report = RuntimeDriver::new(
        one_episode_spec(),
        env.clone(),
        model.clone(),
        hooks.clone(),
    )
    .run()
    .await
    .expect("the single-episode route completes");

    let ids = &mut Ids::default();
    let record = json!({
        "report": report_record(&report),
        "hook_events": hook_events(&hooks, ids),
        "model_lifecycle": model_lifecycle(&model, ids),
        "observations": observations(&hooks, ids),
        "step_infos": meta_records(&hooks.step_infos.lock().expect("step info lock poisoned")),
        "env": {
            "reset_seeds": json!(*env.reset_seeds.lock().expect("reset seed lock poisoned")),
            "reset_options": meta_records(&env.reset_options.lock().expect("reset option lock poisoned")),
            "closed": env.closed.load(Ordering::SeqCst),
        },
        "model_closed": model.closed.load(Ordering::SeqCst),
        "session_ended": hooks.ended.load(Ordering::SeqCst),
    });
    assert_frozen("01-disabled-single-env-episode", &record);
}

// ---------------------------------------------------------------------------
// 2. NEXT_STEP roll timing: the terminal step keeps the old id, the roll step
//    at t+1 belongs to the runtime-minted new one.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scenario_02_next_step_roll_timing() {
    // Four lanes of different lengths: no driver reset after the cold start.
    let wide_env = VectorTestEnv::new(vec![2, 3, 2, 4]);
    let wide_model = TestModel::default();
    let wide_hooks = Arc::new(RecordingHooks::default());
    let wide = RuntimeDriver::new(
        vector_spec(4, 8),
        wide_env.clone(),
        wide_model.clone(),
        wide_hooks.clone(),
    )
    .run()
    .await
    .expect("the vector route completes");

    // Two lanes that both end at step 1: the step-2 response is the new
    // episodes' reset observation, and its infos belong to them.
    let roll_env = VectorTestEnv::new(vec![1, 1]);
    let roll_hooks = Arc::new(RecordingHooks::default());
    RuntimeDriver::new(
        vector_spec(2, 4),
        roll_env,
        TestModel::default(),
        roll_hooks.clone(),
    )
    .run()
    .await
    .expect("the rolling route completes");

    // Lanes of 1 and 3 steps under a budget of 4: an ended id is observed and
    // stepped once more (the autoreset step), then completed last.
    let budget_hooks = Arc::new(RecordingHooks::default());
    let budget_model = TestModel::default();
    let budget = RuntimeDriver::new(
        vector_spec(2, 4),
        VectorTestEnv::new(vec![1, 3]),
        budget_model.clone(),
        budget_hooks.clone(),
    )
    .run()
    .await
    .expect("the budgeted route completes");

    let wide_ids = &mut Ids::default();
    let roll_ids = &mut Ids::default();
    let budget_ids = &mut Ids::default();
    let record = json!({
        "four_lanes": {
            "report": report_record(&wide),
            "env_resets": wide_env.reset_seeds.lock().expect("reset seed lock poisoned").len(),
            "predicts": wide_model.predicts.load(Ordering::SeqCst),
            "hook_events": hook_events(&wide_hooks, wide_ids),
        },
        "simultaneous_roll": {
            "step_infos": meta_records(&roll_hooks.step_infos.lock().expect("step info lock poisoned")),
            "observations": observations(&roll_hooks, roll_ids),
        },
        "budget_ends_a_roll": {
            "report": report_record(&budget),
            "hook_events": hook_events(&budget_hooks, budget_ids),
            "model_lifecycle": model_lifecycle(&budget_model, budget_ids),
        },
    });
    assert_frozen("02-next-step-roll-timing", &record);
}

// ---------------------------------------------------------------------------
// 3. NEXT_STEP mints no trial ordinal: the env owns its resets.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scenario_03_next_step_mints_no_trial_ordinal() {
    let mut runs = Vec::new();
    for base in [None, Some(0)] {
        let env = VectorTestEnv::new(vec![2, 3]);
        let hooks = Arc::new(RecordingHooks::default());
        let mut spec = vector_spec(2, 4);
        spec.trial_index_base = base;
        declare_reset_options(&mut spec, &["trial_index"]);
        let report = RuntimeDriver::new(spec, env.clone(), TestModel::default(), hooks.clone())
            .run()
            .await
            .expect("the declared NEXT_STEP route completes");
        runs.push(json!({
            "trial_index_base": base,
            "report": report_record(&report),
            "reset_options": meta_records(&env.reset_options.lock().expect("reset option lock poisoned")),
            "started_trials": json!(*hooks.started_trials.lock().expect("started trial lock poisoned")),
            "completed_trials": json!(*hooks.completed_trials.lock().expect("completed trial lock poisoned")),
        }));
    }
    assert_frozen("03-next-step-mints-no-trial-ordinal", &json!(runs));
}

// ---------------------------------------------------------------------------
// 4. A runtime truncation is not double counted when the env echoes it.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scenario_04_runtime_truncation_ledger() {
    let env = EchoingEnv::default();
    let hooks = Arc::new(RecordingHooks::default());
    let spec = RuntimeSessionSpec {
        max_episodes: Some(3),
        max_episode_steps: Some(4),
        ..one_episode_spec()
    };
    let report = RuntimeDriver::new(spec, env.clone(), TestModel::default(), hooks.clone())
        .run()
        .await
        .expect("the truncating route completes");

    let (steps, resets) = {
        let state = env.inner.lock().expect("echoing env lock poisoned");
        (state.steps, state.resets)
    };
    let ids = &mut Ids::default();
    let record = json!({
        "report": report_record(&report),
        "env": {"steps": steps, "resets": resets},
        "hook_events": hook_events(&hooks, ids),
        "distinct_completions": hooks.completed_ids().len(),
    });
    assert_frozen("04-runtime-truncation-ledger", &record);
}

// ---------------------------------------------------------------------------
// 5. The built-in per-episode step bound when the spec sets none.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scenario_05_default_truncation_bound() {
    // An env that never terminates, no explicit cap, driver-owned resets: the
    // edition's built-in bound is the only thing that can end the episode.
    let env = EchoingEnv::default();
    let report = RuntimeDriver::new(
        one_episode_spec(),
        env.clone(),
        TestModel {
            // Chunks of 100 keep the bound's cost in predicts down; the env
            // still steps every step, which is what the bound counts.
            replay_frames: 99,
            ..Default::default()
        },
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .expect("the unbounded env is truncated at the built-in bound");

    let steps = env.inner.lock().expect("echoing env lock poisoned").steps;
    let record = json!({
        "report": report_record(&report),
        "env_steps": steps,
    });
    assert_frozen("05-default-truncation-bound", &record);
}

// ---------------------------------------------------------------------------
// 6. Chunk replay: buffered frames replay without a predict, a stale chunk is
//    discarded at an episode boundary, and a chunked vector route warns once.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scenario_06_chunk_replay() {
    let replay_env = TestEnv {
        terminal_after: 6,
        ..Default::default()
    };
    let replay_model = TestModel {
        replay_frames: 2,
        ..Default::default()
    };
    let replay_hooks = Arc::new(RecordingHooks::default());
    let replay = RuntimeDriver::new(
        one_episode_spec(),
        replay_env.clone(),
        replay_model.clone(),
        replay_hooks.clone(),
    )
    .run()
    .await
    .expect("the chunked route completes");

    // Two 3-step episodes, chunk size 3, lead 1: the prefetch fired at the
    // first episode's tail is stale and must not leak across the boundary.
    // Only boundary-independent facts are recorded -- how many speculative
    // predicts a prefetch lands is a race, what each episode re-plans from is not.
    let stale_model = TestModel {
        replay_frames: 2,
        predict_delay: Some(Duration::from_millis(20)),
        ..Default::default()
    };
    let stale_spec = RuntimeSessionSpec {
        max_episodes: Some(2),
        ..one_episode_spec()
    };
    let stale = RuntimeDriver::new(
        stale_spec,
        TestEnv {
            terminal_after: 3,
            ..Default::default()
        },
        stale_model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .with_prefetch(1)
    .run()
    .await
    .expect("the prefetching route completes");
    let first_plans: Vec<u8> = ledger_per_episode(&stale_model)
        .iter()
        .filter_map(|(_, entries)| entries.first().map(|(_, own)| own.1))
        .collect();

    let record = json!({
        "replay": {
            "report": report_record(&replay),
            "predicts": replay_model.predicts.load(Ordering::SeqCst),
            "env_steps": replay_env.step_count.load(Ordering::SeqCst),
            "observations_emitted": replay_hooks.emitted_observations.lock().expect("observation lock poisoned").len(),
            "actions_received": replay_hooks.actions.load(Ordering::SeqCst),
            "ledger": ledger(&replay_model),
        },
        "stale_chunk_across_a_boundary": {
            "report": report_record(&stale),
            "episodes_in_ledger": first_plans.len(),
            "first_plan_observation_per_episode": first_plans,
        },
        "whole_batch_tripwire": {
            "two_lanes": chunked_run_logs(vec![2, 3]).await.matches("chunk replay is whole-batch").count(),
            "one_lane": chunked_run_logs(vec![2]).await.matches("chunk replay is whole-batch").count(),
        },
    });
    assert_frozen("06-chunk-replay", &record);
}

// ---------------------------------------------------------------------------
// 7. Observation history: every env step reaches a history route exactly once,
//    in order; a route without history carries no rows and no step.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scenario_07_observation_history() {
    let mut leads = Vec::new();
    for lead in 0u32..3 {
        let model = TestModel {
            replay_frames: 3,
            wants_history: true,
            ..Default::default()
        };
        let report = RuntimeDriver::new(
            one_episode_spec(),
            TestEnv {
                terminal_after: 9,
                ..Default::default()
            },
            model.clone(),
            Arc::new(RecordingHooks::default()),
        )
        .with_prefetch(lead)
        .run()
        .await
        .expect("the history route completes");
        let entries = model.ledger.lock().expect("ledger poisoned").clone();
        leads.push(json!({
            "prefetch_lead": lead,
            "report": report_record(&report),
            "predicts": model.predicts.load(Ordering::SeqCst),
            "ledger": ledger(&model),
            "delivered_steps": delivered_steps(&entries),
        }));
    }

    // Two 3-step episodes with a prefetch across each reset: the reset
    // observation is promoted to the re-armed request's own, never delivered
    // twice. The per-episode step sequence is a race-free consequence; how far
    // into the episode the last predict fires is not, so only the invariant
    // (each step at most once, contiguous from the episode's first) is pinned.
    let across_model = TestModel {
        replay_frames: 2,
        wants_history: true,
        predict_delay: Some(Duration::from_millis(20)),
        ..Default::default()
    };
    let across_spec = RuntimeSessionSpec {
        max_episodes: Some(2),
        ..one_episode_spec()
    };
    let across = RuntimeDriver::new(
        across_spec,
        TestEnv {
            terminal_after: 3,
            ..Default::default()
        },
        across_model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .with_prefetch(1)
    .run()
    .await
    .expect("the route prefetching across a reset completes");
    let per_episode = ledger_per_episode(&across_model);
    let contiguous_once = per_episode.iter().all(|(_, entries)| {
        let steps = delivered_steps(entries);
        let first = steps.first().copied().unwrap_or_default();
        steps == (first..first + steps.len() as i64).collect::<Vec<_>>()
    });

    // The same chunking without history: no rows, and no step stamped.
    let bare_model = TestModel {
        replay_frames: 3,
        ..Default::default()
    };
    RuntimeDriver::new(
        one_episode_spec(),
        TestEnv {
            terminal_after: 9,
            ..Default::default()
        },
        bare_model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .expect("the route without history completes");
    let bare = bare_model.ledger.lock().expect("ledger poisoned").clone();

    let record = json!({
        "prefetch_leads": leads,
        "across_a_reset": {
            "report": report_record(&across),
            "episodes_in_ledger": per_episode.len(),
            "each_step_once_and_contiguous": contiguous_once,
        },
        "without_history": {
            "predicts": bare.len(),
            "rows_per_predict": bare.iter().map(|(rows, _)| rows.len()).collect::<Vec<_>>(),
            "step_per_predict": bare.iter().map(|(_, own)| own.0).collect::<Vec<_>>(),
        },
    });
    assert_frozen("07-observation-history", &record);
}

// ---------------------------------------------------------------------------
// 8. Lane (`subset_step`) accounting: the budgeted slots are scored whatever
//    the lane timing, and surplus lanes idle.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scenario_08_lane_accounting() {
    // Lane timing decides WHICH lane claims a slot, never which slots are
    // scored or what seed each carries, so the record is sorted by slot.
    let env = LaneTestEnv::new(vec![
        (2, Duration::from_millis(20)),
        (5, Duration::from_millis(40)),
        (1, Duration::ZERO),
    ]);
    let model = TestModel::default();
    let report = RuntimeDriver::new(
        lane_spec(3, 7, (100..110).collect()),
        env.clone(),
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .expect("the lane route completes");
    let mut by_slot = report.episodes.clone();
    by_slot.sort_by_key(|episode| episode.episode_index);
    let slots: Vec<Value> = by_slot
        .iter()
        .map(|episode| json!([episode.episode_index, episode.seed, episode.step_count]))
        .collect();
    let mut lanes_used: Vec<i32> = report
        .episodes
        .iter()
        .map(|episode| episode.env_index)
        .collect();
    lanes_used.sort_unstable();
    lanes_used.dedup();
    let mut reset_seeds: Vec<i64> = env
        .inner
        .lock()
        .expect("lane env state lock poisoned")
        .resets
        .iter()
        .filter_map(|(_, seed)| *seed)
        .collect();
    reset_seeds.sort_unstable();

    // Four lanes, two slots: two lanes never reset at all.
    let idle_env = LaneTestEnv::new(vec![(1, Duration::ZERO); 4]);
    let idle_hooks = Arc::new(RecordingHooks::default());
    let idle = RuntimeDriver::new(
        lane_spec(4, 2, Vec::new()),
        idle_env.clone(),
        TestModel::default(),
        idle_hooks.clone(),
    )
    .run()
    .await
    .expect("the surplus-lane route completes");
    let idle_events: Vec<Vec<&'static str>> = idle_hooks
        .completed_ids()
        .iter()
        .map(|id| idle_hooks.events_for(id))
        .collect();

    let record = json!({
        "budgeted_slots": {
            "total_episodes": report.total_episodes,
            "slot_seed_steps": slots,
            "lanes_used": lanes_used,
            "reset_seeds": reset_seeds,
            "env_closes": env.closed.load(Ordering::SeqCst),
            "model_closed": model.closed.load(Ordering::SeqCst),
        },
        "surplus_lanes_idle": {
            "total_episodes": idle.total_episodes,
            "env_resets": idle_env.inner.lock().expect("lane env state lock poisoned").resets.len(),
            "events_per_completed_episode": idle_events,
        },
    });
    assert_frozen("08-lane-accounting", &record);
}

// ---------------------------------------------------------------------------
// 9. The runtime is the episode-id and trial-ordinal authority, and its reset
//    seeds are a pure function of the spec.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scenario_09_episode_id_and_trial_authority() {
    let model = TestModel::default();
    RuntimeDriver::new(
        one_episode_spec(),
        TestEnv::default(),
        model.clone(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .expect("the single-episode route completes");
    let evicted = model
        .reset_adapters
        .lock()
        .expect("reset_adapter recorder lock poisoned")
        .clone();
    let minted: Vec<Value> = evicted
        .iter()
        .flatten()
        .map(|id| {
            json!({
                "len": id.len(),
                "hyphens": id.matches('-').count(),
                "version_nibble": id.chars().nth(14),
            })
        })
        .collect();

    // The same spec run twice threads the same seeds: the derivation mixes only
    // base_seed, session id, reset generation and lane.
    let seeded = RuntimeSessionSpec {
        base_seed: Some(1234),
        max_episodes: Some(2),
        ..one_episode_spec()
    };
    let mut runs = Vec::new();
    let mut hook_seeds = Vec::new();
    for _ in 0..2 {
        let env = TestEnv::default();
        let hooks = Arc::new(RecordingHooks::default());
        RuntimeDriver::new(
            seeded.clone(),
            env.clone(),
            TestModel::default(),
            hooks.clone(),
        )
        .run()
        .await
        .expect("the seeded route completes");
        runs.push(
            env.reset_seeds
                .lock()
                .expect("reset seed lock poisoned")
                .clone(),
        );
        hook_seeds.push(json!({
            "started": json!(*hooks.started_seeds.lock().expect("started seed lock poisoned")),
            "completed": json!(*hooks.completed_seeds.lock().expect("completed seed lock poisoned")),
        }));
    }

    // Trial ordinals: minted per driver-owned reset, delivered only to an env
    // that declared the key, reported either way.
    let declared_env = TestEnv::default();
    let mut declared_spec = trial_spec(3, &["trial_index"]);
    declared_spec.trial_index_base = Some(100);
    let declared = RuntimeDriver::new(
        declared_spec,
        declared_env.clone(),
        TestModel::default(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .expect("the declaring route completes");
    let undeclared_env = TestEnv::default();
    let undeclared = RuntimeDriver::new(
        trial_spec(2, &[]),
        undeclared_env.clone(),
        TestModel::default(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .expect("the undeclared route completes");

    let record = json!({
        "reset_adapter_calls": evicted.len(),
        "minted_episode_ids": minted,
        "deterministic_seeds": {
            "reset_seeds": runs,
            "hook_seeds": hook_seeds,
        },
        "declared_env": {
            "delivered_ordinals": declared_env
                .reset_options
                .lock()
                .expect("reset option lock poisoned")
                .iter()
                .map(recorded_trial)
                .collect::<Vec<_>>(),
            "reported_ordinals": declared.episodes.iter().map(|episode| episode.trial_index).collect::<Vec<_>>(),
        },
        "undeclared_env": {
            "reset_options": meta_records(&undeclared_env.reset_options.lock().expect("reset option lock poisoned")),
            "reported_ordinals": undeclared.episodes.iter().map(|episode| episode.trial_index).collect::<Vec<_>>(),
        },
    });
    assert_frozen("09-episode-id-and-trial-authority", &record);
}

// ---------------------------------------------------------------------------
// 10. The refusal set: what a 2026.06 runtime declines to drive at all.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scenario_10_refusal_set() {
    let same_step = RuntimeSessionSpec {
        env_contract: rlmesh_proto::core::v1::EnvContract {
            autoreset_mode: rlmesh_proto::core::v1::AutoresetMode::SameStep as i32,
            ..one_episode_spec().env_contract
        },
        ..one_episode_spec()
    };
    let unknown_autoreset = RuntimeSessionSpec {
        env_contract: rlmesh_proto::core::v1::EnvContract {
            autoreset_mode: 99,
            ..one_episode_spec().env_contract
        },
        ..one_episode_spec()
    };
    let lockstep_vector = RuntimeSessionSpec {
        env_contract: rlmesh_proto::core::v1::EnvContract {
            autoreset_mode: rlmesh_proto::core::v1::AutoresetMode::Disabled as i32,
            ..vector_spec(2, 2).env_contract
        },
        ..vector_spec(2, 2)
    };
    let record = json!({
        "same_step_autoreset": refusal_prefix(same_step).await,
        "unknown_autoreset_mode": refusal_prefix(unknown_autoreset).await,
        "lockstep_vector_without_next_step": refusal_prefix(lockstep_vector).await,
        "edition_outside_the_retained_list": unparsed_edition_refusal_prefix("2020.01"),
    });
    assert_frozen("10-refusal-set", &record);
}
