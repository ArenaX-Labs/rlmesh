//! The blocking predict pipelines: one route's direct predict and a grouped
//! request's fused cross-route batch.

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;
use rlmesh_adapters::v1::{ObsPlan, Value};

use super::adapt::{
    annotate_predict_error, assemble_route_inputs, finish_route_frames, inputs_summary,
};
use super::corners::{ShortChunkWarned, bucket_fuses, dispatch_corners, dispatch_route_corners};
use super::route::RouteEntry;
use crate::model::handler::PredictFrames;
use crate::model::predict_fn::{PredictFn, RouteConfig};
use crate::model::types::{EpisodeInfo, ModelObservation};
use crate::{Error, Result};

/// The spec'd per-lane loop (CPU + the model's predict callback), run on a
/// blocking worker thread. Holds the per-route entry lock across input
/// assembly (the frame buffers mutate in place there) and again, after the
/// model call, across the per-frame apply (the action window records there),
/// never across the call itself; the horizon it dispatches with is the
/// runtime-chosen execution horizon pinned on `ResolveAdapter`, not the model
/// spec.
///
/// Emits each lane's action chunk as [`PredictFrames`]: frame 0 per lane plus the
/// future-step frames the runtime driver replays. With execution horizon 1 every
/// lane yields a single frame and `replay` is empty — the unchanged single-action
/// path. The engine no longer replays internally; it returns the executed prefix of
/// the model's native chunk and re-plans on every call (the runtime decides when).
pub(super) fn predict_route(
    entry: &Arc<Mutex<RouteEntry>>,
    predict: &Arc<dyn PredictFn>,
    observation: ModelObservation,
    adapter_ns: &AtomicU64,
    short_chunk_warned: &ShortChunkWarned,
) -> Result<PredictFrames> {
    let num_envs = observation.num_envs;
    let (inputs, config) = {
        let mut guard = entry.lock().expect("route entry poisoned");
        let inputs = assemble_route_inputs(&mut guard, &observation, adapter_ns)?;
        (inputs, Arc::clone(&guard.config))
    };
    let lane_raw_steps = dispatch_route_corners(
        predict,
        inputs,
        &observation.route.episodes,
        config.execution_horizon,
        num_envs,
        short_chunk_warned,
    )?;
    finish_route_frames(
        entry,
        lane_raw_steps,
        &observation.route.episodes,
        num_envs,
        adapter_ns,
    )
}

/// One grouped predict group's serving lane, classified once (on the async
/// side) from the routes map: a spec-less route carries the horizon pinned for
/// it (there is no [`RouteEntry`] to stamp it on); a spec'd route carries its
/// entry.
pub(super) enum GroupLane {
    SpecLess { horizon: u32 },
    Routed { entry: Arc<Mutex<RouteEntry>> },
}

/// Whether a grouped request's spec'd groups may assemble and finish on the
/// rayon pool: every routed group names a distinct route (so no two threads
/// contend one entry lock in undefined order) and no route's plan runs a
/// Python custom transform (which would take the GIL from the pool).
fn assembles_in_parallel(lanes: &[GroupLane]) -> bool {
    let mut seen: HashSet<*const Mutex<RouteEntry>> = HashSet::new();
    let mut routed = 0usize;
    for lane in lanes {
        if let GroupLane::Routed { entry } = lane {
            routed += 1;
            if !seen.insert(Arc::as_ptr(entry)) {
                return false;
            }
            let guard = entry.lock().expect("route entry poisoned");
            if guard
                .config
                .adapter
                .obs_plans
                .iter()
                .any(|plan| matches!(plan, ObsPlan::Custom(_)))
            {
                return false;
            }
        }
    }
    routed > 1
}

/// Runs a parallel adapter section and charges `adapter_ns` its wall time.
/// The per-lane kernels still stamp their own spans, into `scratch`, which is
/// dropped: on the rayon pool those spans overlap, and summing them reported
/// thread time (about 2.5x the wall at 32 lanes) as the endpoint's `adapter`
/// phase, which is meant to be the time the request spent in adapter work.
fn charge_wall<T>(adapter_ns: &AtomicU64, section: impl FnOnce(&AtomicU64) -> T) -> T {
    let scratch = AtomicU64::new(0);
    let started = Instant::now();
    let out = section(&scratch);
    adapter_ns.fetch_add(rlmesh_proto::elapsed_ns(started), Ordering::Relaxed);
    out
}

/// The fused grouped predict (the batched forward across routes), run on a
/// blocking worker thread.
///
/// A spec-less group has no adapter or chunk semantics of its own and serves
/// inline through the preserved raw path, matching `predict_chunked`. Each
/// spec'd group assembles its inputs under a short-lived per-route entry lock
/// (the frame-stack buffers mutate there); the lock drops before any model
/// call, so a route repeated within one request re-locks sequentially instead
/// of self-deadlocking, and a panic can poison at most the route being
/// assembled. Prepared groups are bucketed by their pinned execution horizon
/// (in practice one runner pins one value, so this is a single bucket). A
/// bucket fuses into ONE batched corner call — lanes concatenated in group
/// order, split back by lane count, finished per group with replay frames
/// intact — only when [`bucket_fuses`] proves that corner is the one a direct
/// predict would pick; otherwise the bucket serves per group through the same
/// dispatch as a direct predict. Lanes from different routes are independent
/// by the batched-corner contract, so a fused cross-route batch is
/// semantically identical to a vectorized route's lanes. Per-group errors stay
/// per-group; a fused corner failure is cloned to every group in its bucket
/// (variant and recoverability preserved) and annotated with that group's own
/// assembled-input signature.
pub(super) fn predict_grouped_fused(
    lanes: Vec<GroupLane>,
    observations: Vec<ModelObservation>,
    predict: &Arc<dyn PredictFn>,
    adapter_ns: &AtomicU64,
    short_chunk_warned: &ShortChunkWarned,
) -> Vec<Result<PredictFrames>> {
    struct Prepared {
        index: usize,
        inputs: Vec<Value>,
        episodes: Vec<EpisodeInfo>,
        entry: Arc<Mutex<RouteEntry>>,
        config: Arc<RouteConfig>,
        num_envs: usize,
    }

    // A spec'd group's assembly is host work under its own route entry lock,
    // independent of every other group's, so the groups assemble in parallel
    // (indexed, so results keep request order) — unless two groups name the
    // same route (their buffer mutations would then interleave in no defined
    // order; the runtime never groups one route twice) or a plan runs Python
    // (a custom transform takes the GIL, which serialises the pool anyway).
    // A spec-less group calls the model itself and stays on this thread.
    enum Staged {
        SpecLess {
            observation: ModelObservation,
            horizon: u32,
        },
        Routed {
            assembled: Result<(Vec<Value>, Arc<RouteConfig>)>,
            episodes: Vec<EpisodeInfo>,
            entry: Arc<Mutex<RouteEntry>>,
            num_envs: usize,
        },
    }
    let parallel = assembles_in_parallel(&lanes);
    let stage =
        |(lane, observation): (GroupLane, ModelObservation), adapter_ns: &AtomicU64| match lane {
            GroupLane::SpecLess { horizon } => Staged::SpecLess {
                observation,
                horizon,
            },
            GroupLane::Routed { entry } => {
                let num_envs = observation.num_envs;
                let assembled = {
                    let mut guard = entry.lock().expect("route entry poisoned");
                    assemble_route_inputs(&mut guard, &observation, adapter_ns)
                        .map(|inputs| (inputs, Arc::clone(&guard.config)))
                };
                Staged::Routed {
                    assembled,
                    episodes: observation.route.episodes,
                    entry,
                    num_envs,
                }
            }
        };
    let staged: Vec<Staged> = if parallel {
        charge_wall(adapter_ns, |ns| {
            lanes
                .into_par_iter()
                .zip(observations.into_par_iter())
                .map(|pair| stage(pair, ns))
                .collect()
        })
    } else {
        lanes
            .into_iter()
            .zip(observations)
            .map(|pair| stage(pair, adapter_ns))
            .collect()
    };

    let mut results: Vec<Option<Result<PredictFrames>>> = Vec::with_capacity(staged.len());
    let mut prepared: Vec<Prepared> = Vec::new();
    for (index, staged) in staged.into_iter().enumerate() {
        match staged {
            Staged::SpecLess {
                observation,
                horizon,
            } => results.push(Some(
                predict.predict_spec_less_chunked(observation, horizon),
            )),
            Staged::Routed {
                assembled: Ok((inputs, config)),
                episodes,
                entry,
                num_envs,
            } => {
                results.push(None);
                prepared.push(Prepared {
                    index,
                    inputs,
                    episodes,
                    entry,
                    config,
                    num_envs,
                });
            }
            Staged::Routed {
                assembled: Err(error),
                ..
            } => results.push(Some(Err(error))),
        }
    }

    let mut buckets: BTreeMap<u32, Vec<Prepared>> = BTreeMap::new();
    for group in prepared {
        buckets
            .entry(group.config.execution_horizon.max(1))
            .or_default()
            .push(group);
    }

    for (horizon, groups) in buckets {
        if bucket_fuses(predict.as_ref(), horizon) {
            struct FusedGroup {
                index: usize,
                lane_count: usize,
                summary: String,
                entry: Arc<Mutex<RouteEntry>>,
                episodes: Vec<EpisodeInfo>,
                num_envs: usize,
            }
            let mut flat: Vec<Value> = Vec::new();
            let mut flat_episodes: Vec<EpisodeInfo> = Vec::new();
            let mut fused: Vec<FusedGroup> = Vec::with_capacity(groups.len());
            for group in groups {
                fused.push(FusedGroup {
                    index: group.index,
                    lane_count: group.inputs.len(),
                    summary: inputs_summary(&group.inputs),
                    entry: group.entry,
                    episodes: group.episodes.clone(),
                    num_envs: group.num_envs,
                });
                flat.extend(group.inputs);
                flat_episodes.extend(group.episodes);
            }
            let total = flat.len();
            // Each group's episodes were assembled alongside its own inputs, so a
            // disagreement here is an engine bug, not a bad request.
            let fused_result = if flat_episodes.len() == total {
                dispatch_corners(
                    predict,
                    flat,
                    &flat_episodes,
                    horizon,
                    total,
                    short_chunk_warned,
                )
            } else {
                Err(Error::Internal(format!(
                    "fused predict concatenated {} episode rows for {total} lanes",
                    flat_episodes.len()
                )))
            };
            match fused_result {
                Ok(all_frames) => {
                    // Split the fused chunk per group in order, then apply each
                    // group's action frames under its own entry lock — in
                    // parallel under the same conditions as assembly.
                    let mut frames = all_frames.into_iter();
                    let split: Vec<(FusedGroup, Vec<Vec<Value>>)> = fused
                        .into_iter()
                        .map(|group| {
                            let group_frames = frames.by_ref().take(group.lane_count).collect();
                            (group, group_frames)
                        })
                        .collect();
                    let finish = |(group, group_frames): (FusedGroup, Vec<Vec<Value>>),
                                  adapter_ns: &AtomicU64| {
                        let result = finish_route_frames(
                            &group.entry,
                            group_frames,
                            &group.episodes,
                            group.num_envs,
                            adapter_ns,
                        );
                        (group.index, result)
                    };
                    let finished: Vec<(usize, Result<PredictFrames>)> = if parallel {
                        charge_wall(adapter_ns, |ns| {
                            split.into_par_iter().map(|pair| finish(pair, ns)).collect()
                        })
                    } else {
                        split
                            .into_iter()
                            .map(|pair| finish(pair, adapter_ns))
                            .collect()
                    };
                    for (index, result) in finished {
                        results[index] = Some(result);
                    }
                }
                Err(error) => {
                    for group in fused {
                        results[group.index] =
                            Some(Err(annotate_predict_error(error.clone(), &group.summary)));
                    }
                }
            }
        } else {
            // Each non-fused group is exactly one route, so its row-aligned
            // episode identity travels with it — the same dispatch a direct
            // predict of that route would run.
            for group in groups {
                results[group.index] = Some(
                    dispatch_route_corners(
                        predict,
                        group.inputs,
                        &group.episodes,
                        horizon,
                        group.num_envs,
                        short_chunk_warned,
                    )
                    .and_then(|raw| {
                        finish_route_frames(
                            &group.entry,
                            raw,
                            &group.episodes,
                            group.num_envs,
                            adapter_ns,
                        )
                    }),
                );
            }
        }
    }

    results
        .into_iter()
        .map(|result| {
            result.unwrap_or_else(|| {
                Err(Error::Internal(
                    "grouped predict left a prepared group unserved".to_string(),
                ))
            })
        })
        .collect()
}
