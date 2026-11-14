//! Corner selection, lane math and chunk capping, driven by a counting model.

use std::sync::atomic::{AtomicUsize, Ordering};

use rlmesh_adapters::v1::Value;

use super::adapt::annotate_predict_error;
use super::corners::{bucket_fuses, dispatch_route_corners};
use super::*;
use crate::Error;
use crate::model::types::ModelRouteContext;

/// Counts corner calls and emits `native_chunk`-frame `Value::List` chunks
/// from the chunk corners (or single Numbers from the plain corners), so
/// corner selection, lane math, and chunk capping are observable without an
/// adapter resolver. `short` drops one batched output to exercise the
/// lane-count guard.
struct CountingPredict {
    batch: bool,
    chunk: bool,
    chunk_batch: bool,
    native_chunk: usize,
    short: bool,
    predict_calls: AtomicUsize,
    chunk_calls: AtomicUsize,
    batch_calls: AtomicUsize,
    chunk_batch_calls: AtomicUsize,
    /// The row-aligned episode ids each batched-corner call was handed.
    batched_ids: Mutex<Vec<Vec<String>>>,
}

impl CountingPredict {
    fn new(
        batch: bool,
        chunk: bool,
        chunk_batch: bool,
        native_chunk: usize,
        short: bool,
    ) -> Arc<Self> {
        Arc::new(Self {
            batch,
            chunk,
            chunk_batch,
            native_chunk,
            short,
            predict_calls: AtomicUsize::new(0),
            chunk_calls: AtomicUsize::new(0),
            batch_calls: AtomicUsize::new(0),
            chunk_batch_calls: AtomicUsize::new(0),
            batched_ids: Mutex::new(Vec::new()),
        })
    }

    fn native_chunk_value(&self) -> Value {
        Value::List(
            (0..self.native_chunk)
                .map(|frame| Value::Number(frame as f64))
                .collect(),
        )
    }
}

impl PredictFn for CountingPredict {
    fn predict(&self, _model_input: Value, _episode: Option<&EpisodeInfo>) -> Result<Value> {
        self.predict_calls.fetch_add(1, Ordering::SeqCst);
        Ok(Value::Number(0.0))
    }

    fn predict_spec_less(&self, observation: ModelObservation) -> Result<Vec<SpaceValue>> {
        Ok((0..observation.num_envs)
            .map(|_| SpaceValue::Discrete(0))
            .collect())
    }

    fn has_chunk(&self) -> bool {
        self.chunk
    }

    fn predict_chunk(
        &self,
        _model_input: Value,
        _horizon: u32,
        _episode: Option<&EpisodeInfo>,
    ) -> Result<Option<Value>> {
        self.chunk_calls.fetch_add(1, Ordering::SeqCst);
        Ok(Some(self.native_chunk_value()))
    }

    fn has_batch(&self) -> bool {
        self.batch
    }

    fn predict_batch(&self, inputs: Vec<Value>, episodes: &[EpisodeInfo]) -> Result<Vec<Value>> {
        self.batch_calls.fetch_add(1, Ordering::SeqCst);
        self.batched_ids
            .lock()
            .expect("batched ids poisoned")
            .push(episodes.iter().map(|e| e.episode_id.clone()).collect());
        let keep = inputs.len() - usize::from(self.short);
        Ok((0..keep).map(|_| Value::Number(1.0)).collect())
    }

    fn has_chunk_batch(&self) -> bool {
        self.chunk_batch
    }

    fn predict_chunk_batch(
        &self,
        inputs: Vec<Value>,
        _execution_horizon: u32,
        episodes: &[EpisodeInfo],
    ) -> Result<Vec<Value>> {
        self.chunk_batch_calls.fetch_add(1, Ordering::SeqCst);
        self.batched_ids
            .lock()
            .expect("batched ids poisoned")
            .push(episodes.iter().map(|e| e.episode_id.clone()).collect());
        let keep = inputs.len() - usize::from(self.short);
        Ok((0..keep).map(|_| self.native_chunk_value()).collect())
    }

    fn allow_fusion(&self) -> bool {
        true
    }
}

fn lanes(count: usize) -> Vec<Value> {
    (0..count).map(|lane| Value::Number(lane as f64)).collect()
}

/// Row-aligned episode identity for `count` lanes: `ep0..ep{count-1}`.
fn episodes(count: usize) -> Vec<EpisodeInfo> {
    (0..count)
        .map(|lane| EpisodeInfo {
            episode_id: format!("ep{lane}"),
            seed: Some(lane as i64),
            env_id: String::new(),
        })
        .collect()
}

#[test]
fn dispatch_prefers_chunk_batch_and_caps_to_horizon() {
    let counting = CountingPredict::new(true, false, true, 10, false);
    let predict: Arc<dyn PredictFn> = Arc::clone(&counting) as Arc<dyn PredictFn>;

    let frames = dispatch_route_corners(
        &predict,
        lanes(5),
        &episodes(5),
        4,
        5,
        &ShortChunkWarned::default(),
    )
    .expect("frames");

    assert_eq!(counting.chunk_batch_calls.load(Ordering::SeqCst), 1);
    assert_eq!(counting.batch_calls.load(Ordering::SeqCst), 0);
    assert_eq!(frames.len(), 5);
    assert!(
        frames.iter().all(|lane| lane.len() == 4),
        "each lane keeps the horizon prefix of its native chunk"
    );
}

#[test]
fn dispatch_horizon_one_uses_plain_batch() {
    let counting = CountingPredict::new(true, false, true, 10, false);
    let predict: Arc<dyn PredictFn> = Arc::clone(&counting) as Arc<dyn PredictFn>;

    let frames = dispatch_route_corners(
        &predict,
        lanes(3),
        &episodes(3),
        1,
        3,
        &ShortChunkWarned::default(),
    )
    .expect("frames");

    assert_eq!(counting.batch_calls.load(Ordering::SeqCst), 1);
    assert_eq!(counting.chunk_batch_calls.load(Ordering::SeqCst), 0);
    assert!(frames.iter().all(|lane| lane.len() == 1));
}

/// The chunk-preserving corner precedence a declined fusion falls back to:
/// a chunk + batch model (no batched chunk corner) at horizon > 1 keeps its
/// per-lane chunk corner and full replay frames.
#[test]
fn dispatch_horizon_gt_one_prefers_per_lane_chunk_over_batch() {
    let counting = CountingPredict::new(true, true, false, 10, false);
    let predict: Arc<dyn PredictFn> = Arc::clone(&counting) as Arc<dyn PredictFn>;

    let frames = dispatch_route_corners(
        &predict,
        lanes(3),
        &episodes(3),
        4,
        3,
        &ShortChunkWarned::default(),
    )
    .expect("frames");

    assert_eq!(counting.chunk_calls.load(Ordering::SeqCst), 3);
    assert_eq!(counting.batch_calls.load(Ordering::SeqCst), 0);
    assert!(
        frames.iter().all(|lane| lane.len() == 4),
        "per-lane chunks keep the horizon prefix"
    );
}

#[test]
fn dispatch_reports_lane_count_mismatch() {
    let counting = CountingPredict::new(true, false, true, 10, true);
    let predict: Arc<dyn PredictFn> = Arc::clone(&counting) as Arc<dyn PredictFn>;

    let error = dispatch_route_corners(
        &predict,
        lanes(4),
        &episodes(4),
        4,
        4,
        &ShortChunkWarned::default(),
    )
    .expect_err("short fails");

    assert!(
        error.to_string().contains("lanes"),
        "unexpected error: {error}"
    );
}

/// The batched corners see the SAME row-aligned episode identity the per-lane
/// ones do: a fused batch is N independent episodes, so identity is per row.
#[test]
fn batched_corners_receive_row_aligned_episodes() {
    let counting = CountingPredict::new(true, false, true, 10, false);
    let predict: Arc<dyn PredictFn> = Arc::clone(&counting) as Arc<dyn PredictFn>;

    dispatch_route_corners(
        &predict,
        lanes(3),
        &episodes(3),
        4,
        3,
        &ShortChunkWarned::default(),
    )
    .expect("chunk batch");
    dispatch_route_corners(
        &predict,
        lanes(2),
        &episodes(2),
        1,
        2,
        &ShortChunkWarned::default(),
    )
    .expect("batch");

    assert_eq!(
        *counting.batched_ids.lock().expect("batched ids poisoned"),
        vec![
            vec!["ep0".to_string(), "ep1".to_string(), "ep2".to_string()],
            vec!["ep0".to_string(), "ep1".to_string()],
        ]
    );
}

/// A request whose episode rows do not match its lanes is malformed: fail
/// rather than mis-attribute a lane's state to another episode.
#[test]
fn dispatch_rejects_episode_rows_that_do_not_match_the_lanes() {
    let counting = CountingPredict::new(true, false, false, 10, false);
    let predict: Arc<dyn PredictFn> = Arc::clone(&counting) as Arc<dyn PredictFn>;

    let error = dispatch_route_corners(
        &predict,
        lanes(3),
        &episodes(2),
        1,
        3,
        &ShortChunkWarned::default(),
    )
    .expect_err("row-count mismatch fails");

    assert!(
        error.to_string().contains("episode rows"),
        "unexpected error: {error}"
    );
    assert_eq!(counting.batch_calls.load(Ordering::SeqCst), 0);
}

/// The behavior-parity fusion gate: a bucket fuses ONLY when the batched
/// corner it would run is the corner a direct predict picks, so grouping
/// never changes which model function runs.
#[test]
fn bucket_fuses_only_when_the_direct_corner_is_batched() {
    let chunk_batch = CountingPredict::new(true, true, true, 10, false);
    assert!(bucket_fuses(chunk_batch.as_ref(), 4));
    assert!(bucket_fuses(chunk_batch.as_ref(), 1));

    let chunk_and_batch = CountingPredict::new(true, true, false, 10, false);
    assert!(
        !bucket_fuses(chunk_and_batch.as_ref(), 4),
        "direct picks the per-lane chunk corner; fusing would drop chunking"
    );
    assert!(bucket_fuses(chunk_and_batch.as_ref(), 1));

    let batch_only = CountingPredict::new(true, false, false, 10, false);
    assert!(
        bucket_fuses(batch_only.as_ref(), 4),
        "direct already degrades to single-step batch (warned at configure)"
    );

    let chunk_batch_only = CountingPredict::new(false, false, true, 10, false);
    assert!(
        !bucket_fuses(chunk_batch_only.as_ref(), 1),
        "direct picks the per-lane predict loop at horizon 1"
    );
    assert!(bucket_fuses(chunk_batch_only.as_ref(), 4));

    let per_lane_only = CountingPredict::new(false, false, false, 10, false);
    assert!(!bucket_fuses(per_lane_only.as_ref(), 1));
    assert!(!bucket_fuses(per_lane_only.as_ref(), 4));
}

/// A fused corner failure is cloned per group (not rebuilt from its
/// message), so the model's recoverable flag survives to the wire and the
/// annotation carries the group's own input signature.
#[test]
fn fused_error_broadcast_preserves_recoverability() {
    let error = Error::model_recoverable("transient OOM, retry");

    let annotated = annotate_predict_error(error.clone(), "sig: float32[8]; lanes: 2");

    assert!(annotated.is_recoverable(), "recoverable flag must survive");
    match annotated {
        Error::Model(model) => assert_eq!(
            model.message,
            "transient OOM, retry\nsig: float32[8]; lanes: 2"
        ),
        other => panic!("expected Error::Model, got {other:?}"),
    }
}

/// Spec-less routes bypass the engine corners entirely: each group serves
/// through the preserved raw path, one result per group, in order.
#[tokio::test]
async fn grouped_predict_serves_spec_less_groups_per_group() {
    let counting = CountingPredict::new(true, false, true, 10, false);
    let mut handler = AdaptedModelHandler::new(Arc::clone(&counting) as Arc<dyn PredictFn>, None);
    let observations = (0..3)
        .map(|index| ModelObservation {
            history: Vec::new(),
            step: None,
            observation: None,
            route: ModelRouteContext {
                env_id: format!("env-{index}"),
                episodes: vec![EpisodeInfo {
                    episode_id: format!("ep-{index}"),
                    seed: None,
                    env_id: String::new(),
                }],
                ..Default::default()
            },
            num_envs: 1,
            env_contract: None,
        })
        .collect();

    let results = handler.predict_grouped(observations).await;

    assert_eq!(results.len(), 3, "one result per group, in order");
    for result in results {
        let frames = result.expect("spec-less group serves");
        assert_eq!(frames.actions.len(), 1);
        assert!(frames.replay.is_empty());
    }
    assert_eq!(counting.batch_calls.load(Ordering::SeqCst), 0);
    assert_eq!(counting.chunk_batch_calls.load(Ordering::SeqCst), 0);
}
