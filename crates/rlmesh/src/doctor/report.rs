use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;
const KNEE_GAIN: f64 = 1.10;
const KNEE_LATENCY_CEILING_MS: f64 = 250.0;
const KNEE_LATENCY_OVER_SINGLE: f64 = 2.0;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeReport {
    pub schema_version: u32,
    pub subject: ProbeSubject,
    pub started_at: DateTime<Utc>,
    pub ended_at: DateTime<Utc>,
    pub env: ProbeEnv,
    pub peer: ProbePeer,
    pub checks: Vec<ProbeCheck>,
    pub observed: Observed,
    pub baseline: Vec<BaselinePoint>,
    pub recommendation: Recommendation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preview: Option<ProbePreview>,
}

/// The short random-action clip an environment probe captured from its
/// scene camera, for the catalog card.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbePreview {
    pub video: String,
    pub image: String,
    pub frames: u32,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProbeSubject {
    #[default]
    Model,
    Environment,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeEnv {
    pub repo: String,
    pub env_id: String,
    pub digest: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbePeer {
    pub package_version: String,
    pub language: String,
    pub language_version: String,
    pub capabilities: Vec<String>,
    /// The describe envelope the SDK stamps on its handshake `PeerInfo.extra`.
    /// An image pushed without a describe label is completed from this: the
    /// control plane stores it on the digest and runs the static tier on it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub describe: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckStatus {
    Pass,
    Warn,
    Fail,
    Skip,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeCheck {
    pub name: String,
    pub status: CheckStatus,
    pub detail: String,
    pub tags: Vec<String>,
    pub metrics: BTreeMap<String, f64>,
}

impl ProbeCheck {
    pub fn new(name: &str, status: CheckStatus, detail: impl Into<String>) -> Self {
        Self {
            name: name.to_string(),
            status,
            detail: detail.into(),
            tags: Vec::new(),
            metrics: BTreeMap::new(),
        }
    }

    pub fn skipped(name: &str, reason: &str) -> Self {
        Self::new(name, CheckStatus::Skip, reason)
    }

    pub fn tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    pub fn metric(mut self, key: &str, value: f64) -> Self {
        self.metrics.insert(key.to_string(), value);
        self
    }

    /// Append to the detail, for a check whose second leg qualifies what the
    /// first one measured.
    pub fn detail_suffix(mut self, suffix: String) -> Self {
        self.detail.push_str(&suffix);
        self
    }
}

/// One camera resolution a model image input asserts, as the env's dial must
/// be set for it. `role` is the image role the assertion rides.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderRequest {
    pub role: String,
    pub height: u32,
    pub width: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Observed {
    /// The model's native chunk length K: how many per-step actions one chunk
    /// corner call returns. `None` under the elastic contract (K is unknown:
    /// the model honoured every horizon asked for, so nothing bounds it).
    ///
    /// `actionHorizon` is the pre-rc.9 name and still decodes, so an older
    /// report keeps its number; new reports emit `nativeChunk` only.
    #[serde(
        default,
        alias = "actionHorizon",
        skip_serializing_if = "Option::is_none"
    )]
    pub native_chunk: Option<u32>,
    /// The model returned exactly as many actions as were asked for at every
    /// horizon tried, so no ceiling was reached and `native_chunk` is unset — a
    /// declared chunk is a floor, not a lie.
    #[serde(
        default,
        alias = "actionHorizonElastic",
        skip_serializing_if = "Option::is_none"
    )]
    pub native_chunk_elastic: Option<bool>,
    /// `native_chunk` came from the model's own `ResolveAdapterResponse`, not
    /// from the probe's sweep: the number is a contract, and
    /// `native_chunk_contract` gated it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_chunk_declared: Option<bool>,
    /// The camera resolutions the model's image inputs assert, in spec document
    /// order. Read straight off the model spec, so it survives a resolve
    /// failure; the control plane binds the env's camera dial from it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub render_requests: Vec<RenderRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_batching: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_ok_batch: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deterministic: Option<bool>,
    /// Episodes seeded alike answered identically: the model honours
    /// `EpisodeInfo.seed`, so a run reproduces even when `deterministic` is
    /// false because the model samples.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seeded_reproducible: Option<bool>,
    /// Episodes seeded *differently* answered differently: the seed actually
    /// reaches the sampler. `None` for a model with a zero determinism floor,
    /// where two seeds agreeing proves nothing — only a sampling model can
    /// demonstrate that the seed moves its output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_sensitive: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stateless: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env_reset_reproducible: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env_state_reproducible: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env_image_reproducible: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BaselinePoint {
    pub batch_size: u32,
    pub iterations: u32,
    pub p50_ms: f64,
    pub p95_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint_p50_ms: Option<f64>,
    pub predicts_per_s: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Recommendation {
    /// The model's native chunk K, when the probe established one. Replaces the
    /// old `maxActionChunk`, which read as "the largest horizon you may ask
    /// for" and was silently the largest one the probe happened to try: a
    /// ceiling was indistinguishable from a sweep that never reached it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_chunk: Option<u32>,
    /// No ceiling was found, so any horizon is servable and `native_chunk` says
    /// nothing. A consumer must never clamp an elastic model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_chunk_elastic: Option<bool>,
    /// The camera resolutions the model asserts, for the env's dial.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub render_requests: Vec<RenderRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_batch_size: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_window_ms: Option<u64>,
}

impl Recommendation {
    pub fn derive(observed: &Observed, baseline: &[BaselinePoint]) -> Self {
        Self {
            // A chunk of 1 is "no chunking": carrying it would read as a cap.
            native_chunk: observed.native_chunk.filter(|chunk| *chunk > 1),
            native_chunk_elastic: observed.native_chunk_elastic.filter(|elastic| *elastic),
            render_requests: observed.render_requests.clone(),
            max_batch_size: knee_batch_size(baseline, observed.max_ok_batch),
            batch_window_ms: Some(crate::BatchOptions::default().batch_window_ms),
        }
    }
}

/// Picks the sweep point with the best lane throughput whose p50 stays within
/// the latency budget: 250 ms, or twice the single-predict p50 when that is
/// slower. A larger batch only takes over by gaining at least 10% over the
/// best so far, and a point that falls short is stepped over rather than
/// ending the walk, so one warm-up outlier cannot hide the gain past it.
pub fn knee_batch_size(baseline: &[BaselinePoint], max_ok_batch: Option<u32>) -> Option<u32> {
    let first = baseline.first()?;
    let ceiling_ms = KNEE_LATENCY_CEILING_MS.max(first.p50_ms * KNEE_LATENCY_OVER_SINGLE);
    let chosen = baseline
        .iter()
        .filter(|point| max_ok_batch.is_none_or(|cap| point.batch_size <= cap))
        .fold(first, |best, point| {
            if point.p50_ms <= ceiling_ms && point.predicts_per_s >= best.predicts_per_s * KNEE_GAIN
            {
                point
            } else {
                best
            }
        });
    Some(chosen.batch_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(batch_size: u32, predicts_per_s: f64) -> BaselinePoint {
        timed_point(batch_size, predicts_per_s, 1.0)
    }

    fn timed_point(batch_size: u32, predicts_per_s: f64, p50_ms: f64) -> BaselinePoint {
        BaselinePoint {
            batch_size,
            iterations: 20,
            p50_ms,
            p95_ms: p50_ms * 2.0,
            endpoint_p50_ms: None,
            predicts_per_s,
        }
    }

    /// The xvla/libero sweep on a B300: batch 4 is a warm-up outlier and the
    /// real gain sits past it.
    fn xvla_sweep() -> [BaselinePoint; 6] {
        [
            timed_point(1, 49.0, 3.9),
            timed_point(2, 136.0, 6.1),
            timed_point(4, 83.0, 42.9),
            timed_point(8, 110.0, 62.0),
            timed_point(16, 154.0, 83.0),
            timed_point(32, 175.0, 122.0),
        ]
    }

    /// The smolvla/abot sweep: p50 grows linearly with the batch, so no size
    /// buys throughput.
    fn smolvla_sweep() -> [BaselinePoint; 4] {
        [
            timed_point(1, 5.2, 192.0),
            timed_point(2, 5.1, 391.0),
            timed_point(4, 5.15, 776.0),
            timed_point(8, 4.9, 1630.0),
        ]
    }

    #[test]
    fn knee_walks_past_a_single_outlier_to_the_best_lane_throughput() {
        let knee = knee_batch_size(&xvla_sweep(), Some(32)).unwrap();
        assert!(
            knee >= 16,
            "xvla knee {knee} must reach the gain past the batch-4 outlier"
        );
        assert_eq!(knee, 32);
        let dip_then_recovery = [
            point(1, 10.0),
            point(2, 18.0),
            point(4, 30.0),
            point(8, 31.0),
            point(16, 60.0),
        ];
        assert_eq!(knee_batch_size(&dip_then_recovery, Some(16)), Some(16));
    }

    #[test]
    fn knee_stays_where_later_points_gain_under_ten_percent() {
        let baseline = [
            point(1, 10.0),
            point(2, 18.0),
            point(4, 30.0),
            point(8, 31.0),
            point(16, 32.0),
        ];
        assert_eq!(knee_batch_size(&baseline, Some(16)), Some(4));
        assert_eq!(knee_batch_size(&smolvla_sweep(), Some(8)), Some(1));
    }

    #[test]
    fn knee_ignores_points_past_the_latency_budget() {
        let fast_but_late = [
            timed_point(1, 10.0, 100.0),
            timed_point(2, 19.0, 105.0),
            timed_point(4, 36.0, 300.0),
        ];
        assert_eq!(knee_batch_size(&fast_but_late, None), Some(2));
        let slow_single = [
            timed_point(1, 2.0, 400.0),
            timed_point(2, 3.8, 420.0),
            timed_point(4, 7.0, 900.0),
        ];
        assert_eq!(knee_batch_size(&slow_single, None), Some(2));
    }

    #[test]
    fn knee_is_capped_by_max_ok_batch_and_empty_sweep_yields_nothing() {
        let baseline = [point(1, 10.0), point(2, 20.0), point(4, 40.0)];
        assert_eq!(knee_batch_size(&baseline, Some(2)), Some(2));
        assert_eq!(knee_batch_size(&baseline, None), Some(4));
        assert_eq!(knee_batch_size(&[], Some(1)), None);
        assert_eq!(knee_batch_size(&[point(1, 5.0)], Some(1)), Some(1));
    }

    #[test]
    fn recommendation_only_carries_a_chunk_for_chunking_models() {
        let chunking = Observed {
            native_chunk: Some(8),
            ..Observed::default()
        };
        assert_eq!(Recommendation::derive(&chunking, &[]).native_chunk, Some(8));
        let single_step = Observed {
            native_chunk: Some(1),
            ..Observed::default()
        };
        assert_eq!(Recommendation::derive(&single_step, &[]).native_chunk, None);
        assert_eq!(
            Recommendation::derive(&Observed::default(), &[]).native_chunk,
            None
        );
    }

    /// An elastic model has no ceiling: `nativeChunk` must stay absent so a
    /// consumer cannot mistake a swept number for a cap, and the elastic flag
    /// must ride along to say why.
    #[test]
    fn an_elastic_model_recommends_no_chunk_but_says_it_is_elastic() {
        let elastic = Observed {
            native_chunk: None,
            native_chunk_elastic: Some(true),
            ..Observed::default()
        };
        let recommendation = Recommendation::derive(&elastic, &[]);
        assert_eq!(recommendation.native_chunk, None);
        assert_eq!(recommendation.native_chunk_elastic, Some(true));
        let json = serde_json::to_value(&recommendation).expect("serialize");
        assert!(json.get("nativeChunk").is_none(), "{json}");
        assert!(json.get("maxActionChunk").is_none(), "{json}");
        assert_eq!(json["nativeChunkElastic"], true);

        // A bounded model says the opposite, and says nothing about elasticity.
        let bounded = Observed {
            native_chunk: Some(20),
            native_chunk_elastic: Some(false),
            ..Observed::default()
        };
        let json = serde_json::to_value(Recommendation::derive(&bounded, &[])).expect("serialize");
        assert_eq!(json["nativeChunk"], 20);
        assert!(json.get("nativeChunkElastic").is_none(), "{json}");
    }

    /// Reports written before rc.9 named the same number `actionHorizon`; the
    /// alias keeps them readable without emitting the old key again.
    #[test]
    fn the_pre_rc9_chunk_keys_still_decode() {
        let legacy = serde_json::json!({
            "actionHorizon": 20,
            "actionHorizonElastic": false,
        });
        let observed: Observed = serde_json::from_value(legacy).expect("decode");
        assert_eq!(observed.native_chunk, Some(20));
        assert_eq!(observed.native_chunk_elastic, Some(false));
        let json = serde_json::to_value(&observed).expect("serialize");
        assert_eq!(json["nativeChunk"], 20);
        assert!(json.get("actionHorizon").is_none(), "{json}");
    }

    /// The render requests are carried verbatim into the recommendation the
    /// control plane reads to move the env's camera dial.
    #[test]
    fn render_requests_reach_the_recommendation() {
        let observed = Observed {
            render_requests: vec![RenderRequest {
                role: "image/primary".to_string(),
                height: 256,
                width: 256,
            }],
            ..Observed::default()
        };
        let json = serde_json::to_value(Recommendation::derive(&observed, &[])).expect("serialize");
        assert_eq!(
            json["renderRequests"],
            serde_json::json!([{"role": "image/primary", "height": 256, "width": 256}])
        );
    }

    #[test]
    fn report_round_trips_through_json_with_contract_field_names() {
        let observed = Observed {
            native_chunk: Some(8),
            native_chunk_elastic: Some(false),
            native_chunk_declared: Some(true),
            render_requests: vec![RenderRequest {
                role: "image/primary".to_string(),
                height: 224,
                width: 224,
            }],
            supports_batching: Some(true),
            max_ok_batch: Some(16),
            deterministic: Some(true),
            seeded_reproducible: Some(true),
            seed_sensitive: Some(true),
            stateless: Some(true),
            env_reset_reproducible: Some(true),
            env_state_reproducible: Some(true),
            env_image_reproducible: Some(false),
        };
        let baseline = vec![point(1, 24.3), point(2, 40.0)];
        let report = ProbeReport {
            schema_version: SCHEMA_VERSION,
            subject: ProbeSubject::Model,
            started_at: Utc::now(),
            ended_at: Utc::now(),
            env: ProbeEnv {
                repo: "prebuilt/libero".to_string(),
                env_id: "libero".to_string(),
                digest: "sha256:abc".to_string(),
            },
            peer: ProbePeer {
                package_version: "0.1.0rc5".to_string(),
                language: "python".to_string(),
                language_version: "3.11".to_string(),
                capabilities: vec!["rlmesh.model.concurrent_predict.v1".to_string()],
                describe: None,
            },
            checks: vec![
                ProbeCheck::new("handshake", CheckStatus::Pass, "ok").metric("connectMs", 12.0),
                ProbeCheck::skipped("resolve", "handshake failed").tag("no_chunk"),
            ],
            recommendation: Recommendation::derive(&observed, &baseline),
            observed,
            baseline,
            preview: Some(ProbePreview {
                video: "rlmesh/system/probe-preview.mp4".to_string(),
                image: "rlmesh/system/probe-preview.png".to_string(),
                frames: 60,
                width: 320,
                height: 240,
                fps: 20,
            }),
        };

        let json = serde_json::to_value(&report).expect("serialize");
        assert_eq!(json["schemaVersion"], 1);
        assert_eq!(json["subject"], "model");
        assert_eq!(json["env"]["envId"], "libero");
        assert_eq!(json["peer"]["packageVersion"], "0.1.0rc5");
        assert_eq!(json["checks"][0]["status"], "pass");
        assert_eq!(json["checks"][1]["status"], "skip");
        assert_eq!(json["observed"]["nativeChunk"], 8);
        assert_eq!(json["observed"]["nativeChunkElastic"], false);
        assert_eq!(json["observed"]["nativeChunkDeclared"], true);
        assert_eq!(json["observed"]["renderRequests"][0]["height"], 224);
        assert_eq!(json["observed"]["seedSensitive"], true);
        assert_eq!(json["observed"]["envResetReproducible"], true);
        assert_eq!(json["observed"]["envStateReproducible"], true);
        assert_eq!(json["observed"]["envImageReproducible"], false);
        assert_eq!(json["baseline"][0]["predictsPerS"], 24.3);
        assert!(json["baseline"][0].get("endpointP50Ms").is_none());
        assert_eq!(json["recommendation"]["nativeChunk"], 8);
        assert_eq!(json["recommendation"]["maxBatchSize"], 2);
        assert_eq!(json["recommendation"]["batchWindowMs"], 5);
        assert_eq!(json["preview"]["video"], "rlmesh/system/probe-preview.mp4");
        assert_eq!(json["preview"]["image"], "rlmesh/system/probe-preview.png");
        assert_eq!(json["preview"]["frames"], 60);
        assert_eq!(json["preview"]["width"], 320);
        assert_eq!(json["preview"]["height"], 240);
        assert_eq!(json["preview"]["fps"], 20);

        let decoded: ProbeReport = serde_json::from_value(json).expect("deserialize");
        assert_eq!(decoded, report);

        let without_preview = ProbeReport {
            preview: None,
            ..report
        };
        let json = serde_json::to_value(&without_preview).expect("serialize");
        assert!(json.get("preview").is_none());
        let decoded: ProbeReport = serde_json::from_value(json).expect("deserialize");
        assert_eq!(decoded, without_preview);
    }

    #[test]
    fn subject_serializes_lowercase() {
        assert_eq!(
            serde_json::to_value(ProbeSubject::Environment).unwrap(),
            "environment"
        );
        assert_eq!(serde_json::to_value(ProbeSubject::Model).unwrap(), "model");
    }

    #[test]
    fn empty_observed_omits_every_key() {
        let json = serde_json::to_value(Observed::default()).expect("serialize");
        assert_eq!(json, serde_json::json!({}));
        let json = serde_json::to_value(Recommendation::default()).expect("serialize");
        assert_eq!(json, serde_json::json!({}));
    }
}
