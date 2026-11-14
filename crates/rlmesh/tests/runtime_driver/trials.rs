//! Trial ordinals and reset options on the wire.

use super::*;

#[tokio::test]
async fn declared_env_receives_the_trial_ordinal_on_every_reset() {
    let env = TestEnv::default();
    let hooks = Arc::new(RecordingHooks::default());
    let mut spec = trial_spec(3, &["trial_index"]);
    spec.trial_index_base = Some(100);

    let report = new_driver(spec, env.clone(), TestModel::default(), hooks.clone())
        .run()
        .await
        .unwrap();

    // One reset per episode (driver-owned resets), each carrying the next ordinal.
    let options = env.reset_options.lock().unwrap().clone();
    assert_eq!(
        options.iter().map(recorded_trial).collect::<Vec<_>>(),
        vec![Some(100), Some(101), Some(102)],
    );
    // The window rule: a shard walks exactly its `max_episodes` ordinals, so the
    // next shard's base is base + M.
    assert_eq!(report.total_episodes, 3);
    assert_eq!(
        report
            .episodes
            .iter()
            .map(|episode| episode.trial_index)
            .collect::<Vec<_>>(),
        vec![Some(100), Some(101), Some(102)],
    );
    assert_eq!(
        *hooks.started_trials.lock().unwrap(),
        vec![Some(100), Some(101), Some(102)],
    );
    assert_eq!(
        *hooks.completed_trials.lock().unwrap(),
        vec![Some(100), Some(101), Some(102)],
    );
}

#[tokio::test]
async fn undeclared_env_is_not_sent_the_ordinal_but_the_report_still_carries_it() {
    let env = TestEnv::default();
    let mut spec = trial_spec(2, &["something_else"]);
    spec.trial_index_base = Some(7);

    let report = new_driver(
        spec,
        env.clone(),
        TestModel::default(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    assert!(
        env.reset_options
            .lock()
            .unwrap()
            .iter()
            .all(|options| options.is_none()),
        "an env that declared no trial_index must not receive one",
    );
    assert_eq!(
        report
            .episodes
            .iter()
            .map(|episode| episode.trial_index)
            .collect::<Vec<_>>(),
        vec![Some(7), Some(8)],
        "the ordinal is minted either way, so a coverage audit can read the sweep",
    );
}

#[tokio::test]
async fn no_trial_base_walks_the_ordinals_from_zero() {
    // The ordinal is on by default: with `trial_index_base` unset a declaring env
    // receives 0, 1, 2 ... and the report / hooks carry the same sweep.
    let env = TestEnv::default();
    let hooks = Arc::new(RecordingHooks::default());
    let spec = trial_spec(3, &["trial_index"]);
    assert_eq!(spec.trial_index_base, None);

    let report = new_driver(spec, env.clone(), TestModel::default(), hooks.clone())
        .run()
        .await
        .unwrap();

    let options = env.reset_options.lock().unwrap().clone();
    assert_eq!(
        options.iter().map(recorded_trial).collect::<Vec<_>>(),
        vec![Some(0), Some(1), Some(2)],
    );
    assert_eq!(
        report
            .episodes
            .iter()
            .map(|episode| episode.trial_index)
            .collect::<Vec<_>>(),
        vec![Some(0), Some(1), Some(2)],
    );
    assert_eq!(
        *hooks.started_trials.lock().unwrap(),
        vec![Some(0), Some(1), Some(2)],
    );
    assert_eq!(
        *hooks.completed_trials.lock().unwrap(),
        vec![Some(0), Some(1), Some(2)],
    );
}

#[tokio::test]
async fn no_trial_base_still_withholds_the_ordinal_from_an_undeclared_env() {
    // Minted by default, delivered only on declaration: a plain env keeps its
    // options-less reset, while the report still records the sweep.
    let env = TestEnv::default();
    let report = new_driver(
        trial_spec(2, &[]),
        env.clone(),
        TestModel::default(),
        Arc::new(RecordingHooks::default()),
    )
    .run()
    .await
    .unwrap();

    assert!(
        env.reset_options
            .lock()
            .unwrap()
            .iter()
            .all(|options| options.is_none()),
        "an env that declared no trial_index must not receive one",
    );
    assert_eq!(
        report
            .episodes
            .iter()
            .map(|episode| episode.trial_index)
            .collect::<Vec<_>>(),
        vec![Some(0), Some(1)],
    );
}

#[tokio::test]
async fn next_step_autoreset_mints_no_ordinal() {
    // Under NEXT_STEP the env restarts its own lanes: the default base validates
    // (an explicit 0 as well, the Python surface always passes an integer) but
    // no ordinal is minted, so neither the declaring env nor the report sees one.
    for base in [None, Some(0)] {
        let env = VectorTestEnv::new(vec![2, 3]);
        let hooks = Arc::new(RecordingHooks::default());
        let mut spec = vector_spec(2, 4);
        spec.trial_index_base = base;
        declare_reset_options(&mut spec, &["trial_index"]);

        let report = new_driver(spec, env.clone(), TestModel::default(), hooks.clone())
            .run()
            .await
            .unwrap();

        assert!(report.total_episodes >= 4);
        assert!(
            env.reset_options
                .lock()
                .unwrap()
                .iter()
                .all(|options| options.is_none()),
            "base {base:?}: no ordinal reaches an env that owns its resets",
        );
        assert!(
            report
                .episodes
                .iter()
                .all(|episode| episode.trial_index.is_none()),
            "base {base:?}: no ordinal is reported",
        );
        assert!(
            hooks
                .started_trials
                .lock()
                .unwrap()
                .iter()
                .all(Option::is_none)
        );
    }
}
#[tokio::test]
async fn the_driver_sends_the_shared_reset_options_encoding_on_the_wire() {
    // The literal encoding, not a restatement of the fn the driver calls: a
    // single-lane reset carries the bare integer and a multi-lane reset the
    // list in lane order.
    let env = TestEnv::default();
    let hooks = Arc::new(RecordingHooks::default());
    let mut spec = trial_spec(2, &["trial_index"]);
    spec.trial_index_base = Some(41);

    new_driver(spec, env.clone(), TestModel::default(), hooks)
        .run()
        .await
        .unwrap();

    let options = env.reset_options.lock().unwrap().clone();
    let trials = options
        .iter()
        .map(|options| {
            options
                .as_ref()
                .expect("a declaring env receives options")
                .get("trial_index")
                .cloned()
                .expect("trial_index")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        trials,
        vec![
            rlmesh_spaces::MetaValue::Int(41),
            rlmesh_spaces::MetaValue::Int(42)
        ],
        "a single-lane reset sends the bare integer",
    );
}

#[test]
fn a_multi_lane_reset_sends_the_trial_ordinals_as_a_list_in_lane_order() {
    let mut spec = trial_spec(1, &["trial_index"]);
    declare_reset_options(&mut spec, &["trial_index"]);

    let options = rlmesh::binding::runtime::reset_options_for(
        &spec.env_contract,
        rlmesh::binding::runtime::TRIAL_INDEX_OPTION,
        &[41, 42],
    )
    .expect("a declaring env receives options");

    use rlmesh_spaces::MetaValue::{Int, List};
    assert_eq!(
        options.get("trial_index"),
        Some(&List(vec![Int(41), Int(42)]))
    );
}
