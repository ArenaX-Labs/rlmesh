#![cfg(feature = "lockstep")]
//! A model served on a session: routes resolve once, a predict carries
//! several routes' rows and gets one outcome each, chunks come back as
//! frames.

#![allow(clippy::unwrap_used)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::boxed;
use rlmesh::binding::model::{ModelHandler, ModelObservation, PredictFrames};
use rlmesh::lockstep::map::contract_to_json;
use rlmesh::lockstep::{ModelBinding, ModelConnection, ModelService};
use rlmesh_session::wire::{ModelEpisode, ModelGroup, ModelReply, ModelRequest, ModelResolve};
use rlmesh_spaces::leaves::{decode_leaf_slab, encode_leaf_slab};
use rlmesh_spaces::{AutoresetMode, DType, EnvContract, SpaceValue, Tensor};

/// Acts with its observation; chunks of three (the observation, then +1,
/// +2) when asked for a chunk.
struct Echo;

fn plus(value: &SpaceValue, n: f32) -> SpaceValue {
    let SpaceValue::Box(t) = value else {
        panic!("a box")
    };
    let data: Vec<f32> = t
        .to_contiguous_bytes()
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()) + n)
        .collect();
    let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    SpaceValue::Box(Tensor::from_vec(bytes, vec![2], DType::Float32).unwrap())
}

#[rlmesh::binding::async_trait]
impl ModelHandler for Echo {
    async fn predict(&mut self, observation: ModelObservation) -> rlmesh::Result<Vec<SpaceValue>> {
        observation.decoded_lanes()
    }

    async fn predict_chunked(
        &mut self,
        observation: ModelObservation,
    ) -> rlmesh::Result<PredictFrames> {
        let lanes = observation.decoded_lanes()?;
        Ok(PredictFrames {
            replay: (1..3)
                .map(|n| lanes.iter().map(|v| plus(v, n as f32)).collect())
                .collect(),
            actions: lanes,
        })
    }
}

fn contract() -> EnvContract {
    EnvContract {
        id: "Echo-v0".into(),
        action_space: Some(boxed(DType::Float32, &[2])),
        observation_space: Some(boxed(DType::Float32, &[2])),
        metadata: None,
        render_mode: String::new(),
        num_envs: 1,
        autoreset_mode: AutoresetMode::Disabled,
    }
}

fn value(v: f32) -> SpaceValue {
    let bytes = [v, -v].iter().flat_map(|x| x.to_le_bytes()).collect();
    SpaceValue::Box(Tensor::from_vec(bytes, vec![2], DType::Float32).unwrap())
}

fn group(route: &str, values: &[f32]) -> ModelGroup {
    let rows: Vec<SpaceValue> = values.iter().map(|v| value(*v)).collect();
    ModelGroup {
        route: route.into(),
        episodes: (0..rows.len())
            .map(|i| ModelEpisode {
                id: format!("{route}-{i}"),
                seed: Some(i as i64),
            })
            .collect(),
        observation: encode_leaf_slab(&rows, contract().observation_space.as_ref().unwrap())
            .unwrap(),
        history: Vec::new(),
        step: Some(0),
    }
}

fn resolve(route: &str, horizon: u32) -> ModelRequest {
    ModelRequest::Resolve(ModelResolve {
        route: route.into(),
        contract_json: contract_to_json(&contract()).to_string(),
        execution_horizon: horizon,
        delivers_history: false,
        workflow_edition: String::new(),
        reads_advisories: false,
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_served_model_answers_each_group_of_a_predict_with_its_own_outcome() {
    let binding = ModelBinding::bind(":0/echo").await.unwrap();
    let address = binding.address();
    let cancel = tokio_util::sync::CancellationToken::new();
    let service = Arc::new(ModelService::new(Echo, "{\"kind\":\"model\"}".into()));
    let served = tokio::spawn(binding.serve(service, Duration::from_secs(10), cancel.clone()));
    let model = ModelConnection::connect(&address, "driver", Duration::from_secs(10))
        .await
        .unwrap();
    let call = |request: ModelRequest| {
        let link = &model.link;
        async move { link.call(&request, None).await.unwrap().0 }
    };

    assert_eq!(
        call(ModelRequest::Describe).await,
        ModelReply::Described("{\"kind\":\"model\"}".into())
    );
    assert!(matches!(
        call(resolve("a", 3)).await,
        ModelReply::Resolved(_)
    ));
    // Two lanes of route a, and route b, never resolved: b's fault stays
    // b's.
    let ModelReply::Predicted(outcomes) = call(ModelRequest::Predict(vec![
        group("a", &[1.0, 2.0]),
        group("b", &[3.0]),
    ]))
    .await
    else {
        panic!("a predict reply")
    };
    let space = contract().action_space.unwrap();
    let frames = outcomes[0].as_ref().unwrap();
    assert_eq!(frames.len(), 3, "frame 0 and two replay frames");
    assert_eq!(
        decode_leaf_slab(&frames[0], &space, 2).unwrap(),
        vec![value(1.0), value(2.0)]
    );
    assert_eq!(
        decode_leaf_slab(&frames[2], &space, 2).unwrap(),
        vec![plus(&value(1.0), 2.0), plus(&value(2.0), 2.0)]
    );
    let fault = outcomes[1].as_ref().unwrap_err();
    assert!(
        fault.message.contains("\"b\" was not resolved"),
        "{fault:?}"
    );

    // Each reply carries what the model measured, as v1's endpoint phases.
    let (_, timing) = model
        .link
        .call(&ModelRequest::Predict(vec![group("a", &[1.0])]), None)
        .await
        .unwrap();
    let timing = timing.expect("a predict is timed");
    assert!(
        timing.total_ns.is_some() && timing.user_ns.is_some(),
        "{timing:?}"
    );
    assert!(timing.total_ns >= timing.user_ns, "{timing:?}");
    assert_eq!(timing.in_flight, Some(1));

    // A released route is gone.
    assert_eq!(
        call(ModelRequest::Reset {
            route: "a".into(),
            episode_ids: vec!["a-0".into()],
        })
        .await,
        ModelReply::Done
    );
    assert_eq!(
        call(ModelRequest::Release { route: "a".into() }).await,
        ModelReply::Done
    );
    let ModelReply::Predicted(outcomes) =
        call(ModelRequest::Predict(vec![group("a", &[1.0])])).await
    else {
        panic!("a predict reply")
    };
    assert!(outcomes[0].is_err());

    // An edition this build does not implement is refused at resolve.
    let ModelRequest::Resolve(mut pinned) = resolve("c", 1) else {
        unreachable!()
    };
    pinned.workflow_edition = "1999.01".into();
    let ModelReply::Fault(fault) = call(ModelRequest::Resolve(pinned)).await else {
        panic!("a fault")
    };
    assert!(fault.message.contains("1999.01"), "{fault:?}");

    model.close().await;
    cancel.cancel();
    served.await.unwrap().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_served_model_is_a_handler_a_driver_drives_like_a_local_one() {
    use rlmesh::binding::model::{EpisodeInfo, ModelRouteContext, ResolveOptions};
    use rlmesh::lockstep::ServedModel;

    let binding = ModelBinding::bind(":0/echo").await.unwrap();
    let address = binding.address();
    let cancel = tokio_util::sync::CancellationToken::new();
    let service = Arc::new(ModelService::new(Echo, String::new()));
    let served = tokio::spawn(binding.serve(service, Duration::from_secs(10), cancel.clone()));

    let mut model = ServedModel::new(address, Duration::from_secs(10));
    let routes = model.route_setup().unwrap();
    routes
        .resolve_adapter(
            "a",
            &contract(),
            ResolveOptions {
                execution_horizon: 3,
                delivers_history: false,
                workflow_edition: None,
            },
        )
        .await
        .unwrap();
    let group = group("a", &[1.0, 2.0]);
    let observation = ModelObservation {
        observation: Some(group.observation),
        route: ModelRouteContext {
            env_id: "a".into(),
            episodes: vec![EpisodeInfo::default(), EpisodeInfo::default()],
            ..ModelRouteContext::default()
        },
        num_envs: 2,
        ..ModelObservation::default()
    };
    let frames = model.predict_chunked(observation).await.unwrap();
    assert_eq!(frames.actions, vec![value(1.0), value(2.0)]);
    assert_eq!(frames.replay.len(), 2);
    let unknown = ModelObservation {
        route: ModelRouteContext {
            env_id: "b".into(),
            episodes: vec![EpisodeInfo::default()],
            ..ModelRouteContext::default()
        },
        num_envs: 1,
        ..ModelObservation::default()
    };
    let err = model.predict(unknown).await.unwrap_err();
    assert!(err.to_string().contains("\"b\" was not resolved"), "{err}");

    model.on_close().await.unwrap();
    cancel.cancel();
    served.await.unwrap().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_served_model_answers_in_the_session_it_is_invited_into() {
    use rlmesh::binding::model::ResolveOptions;
    use rlmesh::lockstep::ServedModel;

    let binding = ModelBinding::bind(":0/echo").await.unwrap();
    let address = binding.address();
    let cancel = tokio_util::sync::CancellationToken::new();
    let service = Arc::new(ModelService::new(Echo, String::new()));
    let served = tokio::spawn(binding.serve(service, Duration::from_secs(2), cancel.clone()));
    let mut target = rlmesh_session::address::host_at(
        "127.0.0.1:0".parse().unwrap(),
        "bench",
        Duration::from_secs(1),
    )
    .await
    .unwrap();
    let session = target.address.to_string();

    let mut model = ServedModel::new(address.clone(), Duration::from_secs(10));
    model.join(&session).await.unwrap();
    let resolve = |model: &ServedModel| {
        let routes = model.route_setup().unwrap();
        async move {
            routes
                .resolve_adapter(
                    "a",
                    &contract(),
                    ResolveOptions {
                        execution_horizon: 1,
                        delivers_history: false,
                        workflow_edition: None,
                    },
                )
                .await
        }
    };
    resolve(&model).await.unwrap();
    // The model now answers in the target session too.
    let there = ModelConnection::connect(
        &format!("{session}/echo"),
        "observer",
        Duration::from_secs(5),
    )
    .await
    .unwrap();
    there.close().await;
    model.on_close().await.unwrap();

    // Its own session needs no invitation.
    let own = ModelConnection::connect(&address, "observer", Duration::from_secs(5))
        .await
        .unwrap();
    assert!(own.answers_in(address.rsplit_once('/').unwrap().0));
    assert!(!own.answers_in(&session));
    own.close().await;

    // A session the model can't reach is refused, naming it.
    let mut model = ServedModel::new(address, Duration::from_secs(10));
    model.join("127.0.0.1:1/nowhere").await.unwrap();
    let err = resolve(&model).await.unwrap_err();
    assert!(err.to_string().contains("can't reach"), "{err}");

    cancel.cancel();
    served.await.unwrap().unwrap();
    target.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_served_model_declares_its_workflow_editions_in_the_roster() {
    let binding = ModelBinding::bind(":0/echo").await.unwrap();
    let address = binding.address();
    let cancel = tokio_util::sync::CancellationToken::new();
    let declared = rlmesh::current_workflow_edition();
    let service = Arc::new(ModelService::new(Echo, String::new()).declaring(Some(declared)));
    let served = tokio::spawn(binding.serve(service, Duration::from_secs(10), cancel.clone()));
    let model = ModelConnection::connect(&address, "driver", Duration::from_secs(10))
        .await
        .unwrap();

    let offer = model.workflow_editions().expect("the model's editions");
    assert_eq!(offer.preferred.as_deref(), Some(declared));
    assert!(!offer.editions.is_empty(), "{offer:?}");

    model.close().await;
    cancel.cancel();
    served.await.unwrap().unwrap();
}

/// A session host's four class ports on 127.0.0.1, held until the sockets
/// drop: bound with `SO_REUSEADDR` but not listening, so a connect is
/// refused and no other socket can take them, while the host's own
/// listeners (Zenoh sets `SO_REUSEADDR` too) bind and listen on them.
fn reserve_session_ports() -> (u16, Vec<tokio::net::TcpSocket>) {
    let reserve = |port: u16| {
        let socket = tokio::net::TcpSocket::new_v4().ok()?;
        socket.set_reuseaddr(true).ok()?;
        socket.bind(([127, 0, 0, 1], port).into()).ok()?;
        Some(socket)
    };
    for _ in 0..64 {
        let first = reserve(0).unwrap();
        let base = first.local_addr().unwrap().port();
        let rest: Option<Vec<_>> = (1..4)
            .map(|i| base.checked_add(i).and_then(reserve))
            .collect();
        if let Some(rest) = rest {
            return (base, std::iter::once(first).chain(rest).collect());
        }
    }
    panic!("no four consecutive free ports on 127.0.0.1")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_driver_waits_for_a_model_not_up_yet_and_never_hosts_its_session() {
    let (port, reserved) = reserve_session_ports();
    let address = format!("127.0.0.1:{port}/echo");
    let connecting = tokio::spawn({
        let address = address.clone();
        async move { ModelConnection::connect(&address, "driver", Duration::from_secs(10)).await }
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    let listening = std::net::TcpStream::connect(("127.0.0.1", port)).is_ok();
    assert!(!listening, "the driver hosted the model's session");

    let binding = ModelBinding::bind(&address).await.unwrap();
    drop(reserved);
    let cancel = tokio_util::sync::CancellationToken::new();
    let service = Arc::new(ModelService::new(Echo, String::new()));
    let served = tokio::spawn(binding.serve(service, Duration::from_secs(10), cancel.clone()));
    let first = connecting.await.unwrap().unwrap();
    first.close().await;
    // The model's session is its own: a driver leaving it leaves it serving.
    let again = ModelConnection::connect(&address, "driver", Duration::from_secs(10))
        .await
        .unwrap();
    again.close().await;
    cancel.cancel();
    served.await.unwrap().unwrap();
}

/// Fuses: one handler call answers every group, each with its own lanes.
/// Records the groups of each call and refuses an observation of `-1`.
#[derive(Clone, Default)]
struct Fused(Arc<std::sync::Mutex<Vec<usize>>>);

impl Fused {
    fn calls(&self) -> Vec<usize> {
        self.0.lock().unwrap().clone()
    }
}

#[rlmesh::binding::async_trait]
impl ModelHandler for Fused {
    async fn predict(&mut self, observation: ModelObservation) -> rlmesh::Result<Vec<SpaceValue>> {
        observation.decoded_lanes()
    }

    async fn predict_grouped(
        &mut self,
        observations: Vec<ModelObservation>,
    ) -> Vec<rlmesh::Result<PredictFrames>> {
        self.0.lock().unwrap().push(observations.len());
        observations
            .into_iter()
            .map(|observation| {
                let lanes = observation.decoded_lanes()?;
                if lanes.contains(&value(-1.0)) {
                    return Err(rlmesh::Error::model_recoverable("refused -1"));
                }
                Ok(PredictFrames {
                    actions: lanes,
                    replay: Vec::new(),
                })
            })
            .collect()
    }

    fn fuses_groups(&self) -> bool {
        true
    }
}

fn batched<H: ModelHandler + 'static>(
    handler: H,
    options: rlmesh::BatchOptions,
) -> Arc<ModelService<H>> {
    Arc::new(
        ModelService::new(handler, String::new())
            .batching(options)
            .unwrap(),
    )
}

/// Predicts one group each, all at once, and returns each reply.
async fn predict_together<H: ModelHandler + 'static>(
    service: &Arc<ModelService<H>>,
    groups: Vec<ModelGroup>,
) -> Vec<ModelReply> {
    let calls = groups.into_iter().map(|group| {
        let service = Arc::clone(service);
        tokio::spawn(async move { service.answer(ModelRequest::Predict(vec![group])).await.0 })
    });
    let mut replies = Vec::new();
    for call in calls.collect::<Vec<_>>() {
        replies.push(call.await.unwrap());
    }
    replies
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn predicts_from_several_sessions_coalesce_into_one_handler_call() {
    use rlmesh::binding::model::{EpisodeInfo, ModelRouteContext, ResolveOptions};
    use rlmesh::lockstep::ServedModel;

    let records = tempfile::tempdir().unwrap();
    let path = records.path().join("batches.jsonl");
    let binding = ModelBinding::bind(":0/fused").await.unwrap();
    let address = binding.address();
    let cancel = tokio_util::sync::CancellationToken::new();
    let fused = Fused::default();
    let service = batched(
        fused.clone(),
        rlmesh::BatchOptions {
            batch_window_ms: 500,
            batch_records: Some(path.clone()),
            ..Default::default()
        },
    );
    let served = tokio::spawn(binding.serve(service, Duration::from_secs(10), cancel.clone()));

    // Two drivers, each in a session of its own that it invites the model
    // into, each predicting its own route.
    let mut hosts = Vec::new();
    let mut drivers = Vec::new();
    for (i, route) in ["a", "b"].into_iter().enumerate() {
        let host = rlmesh_session::address::host_at(
            "127.0.0.1:0".parse().unwrap(),
            &format!("run{i}"),
            Duration::from_secs(1),
        )
        .await
        .unwrap();
        let mut model = ServedModel::new(address.clone(), Duration::from_secs(10));
        model.join(&host.address.to_string()).await.unwrap();
        model
            .route_setup()
            .unwrap()
            .resolve_adapter(
                route,
                &contract(),
                ResolveOptions {
                    execution_horizon: 1,
                    delivers_history: false,
                    workflow_edition: None,
                },
            )
            .await
            .unwrap();
        hosts.push(host);
        drivers.push((route, model));
    }
    let predicts = drivers.into_iter().map(|(route, mut model)| {
        tokio::spawn(async move {
            let observation = ModelObservation {
                observation: Some(group(route, &[1.0]).observation),
                route: ModelRouteContext {
                    env_id: route.into(),
                    episodes: vec![EpisodeInfo::default()],
                    ..ModelRouteContext::default()
                },
                num_envs: 1,
                ..ModelObservation::default()
            };
            let actions = model.predict(observation).await;
            model.on_close().await.unwrap();
            actions
        })
    });
    for predict in predicts.collect::<Vec<_>>() {
        assert_eq!(predict.await.unwrap().unwrap(), vec![value(1.0)]);
    }
    assert_eq!(fused.calls(), vec![2], "one handler call for both sessions");

    cancel.cancel();
    served.await.unwrap().unwrap();
    for mut host in hosts {
        host.close().await.unwrap();
    }
    // One record, in the platform's batches.jsonl shape.
    let text = std::fs::read_to_string(&path).unwrap();
    let records: Vec<serde_json::Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records.len(), 1, "{text}");
    let record = records[0].as_object().unwrap();
    let mut keys: Vec<&str> = record.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "at",
            "componentId",
            "endpoint",
            "endpointTotalMs",
            "fill",
            "forwardWallMs",
            "inFlightAtDispatch",
            "maxSize",
            "ok",
            "queueWaitMs",
            "size"
        ]
    );
    assert_eq!(record["size"], 2);
    assert_eq!(record["componentId"], "fused");
    assert_eq!(record["ok"], true);
    assert!(record["queueWaitMs"]["max"].as_f64().unwrap() > 0.0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn max_batch_size_and_the_byte_budget_split_batches() {
    // Each group carries 8 observation bytes.
    for (options, want) in [
        (
            rlmesh::BatchOptions {
                max_batch_size: 2,
                batch_window_ms: 300,
                ..Default::default()
            },
            vec![1, 2],
        ),
        (
            rlmesh::BatchOptions {
                batch_window_ms: 300,
                batch_byte_budget: 20,
                ..Default::default()
            },
            vec![1, 2],
        ),
    ] {
        let fused = Fused::default();
        let service = batched(fused.clone(), options);
        for route in ["a", "b", "c"] {
            assert!(matches!(
                service.answer(resolve(route, 1)).await.0,
                ModelReply::Resolved(_)
            ));
        }
        let replies = predict_together(
            &service,
            vec![group("a", &[1.0]), group("b", &[2.0]), group("c", &[3.0])],
        )
        .await;
        assert!(
            replies
                .iter()
                .all(|reply| matches!(reply, ModelReply::Predicted(o) if o[0].is_ok()))
        );
        let mut calls = fused.calls();
        calls.sort_unstable();
        assert_eq!(calls, want);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_batch_closes_once_every_route_has_a_request_in_it() {
    let fused = Fused::default();
    let service = batched(
        fused.clone(),
        rlmesh::BatchOptions {
            batch_window_ms: 5_000,
            ..Default::default()
        },
    );
    service.answer(resolve("a", 1)).await;
    service.answer(resolve("b", 1)).await;
    let started = std::time::Instant::now();
    let replies = predict_together(&service, vec![group("a", &[1.0]), group("b", &[2.0])]).await;
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "it waited the window"
    );
    assert!(
        replies
            .iter()
            .all(|reply| matches!(reply, ModelReply::Predicted(o) if o[0].is_ok()))
    );
    assert_eq!(fused.calls(), vec![2]);

    // A queued alone; b's release leaves it holding every route.
    let lone = tokio::spawn({
        let service = Arc::clone(&service);
        async move {
            service
                .answer(ModelRequest::Predict(vec![group("a", &[3.0])]))
                .await
                .0
        }
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    let started = std::time::Instant::now();
    service
        .answer(ModelRequest::Release { route: "b".into() })
        .await;
    let reply = lone.await.unwrap();
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "it waited the window after b left"
    );
    assert!(matches!(reply, ModelReply::Predicted(o) if o[0].is_ok()));
    assert_eq!(fused.calls(), vec![2, 1]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_window_bounds_how_long_a_lone_request_waits() {
    let service = batched(
        Fused::default(),
        rlmesh::BatchOptions {
            batch_window_ms: 200,
            ..Default::default()
        },
    );
    // b never predicts, so a's batch waits the window for it.
    service.answer(resolve("a", 1)).await;
    service.answer(resolve("b", 1)).await;
    let started = std::time::Instant::now();
    let (reply, timing) = service
        .answer(ModelRequest::Predict(vec![group("a", &[1.0])]))
        .await;
    let waited = started.elapsed();
    assert!(matches!(reply, ModelReply::Predicted(o) if o[0].is_ok()));
    assert!(
        waited >= Duration::from_millis(200) && waited < Duration::from_secs(2),
        "{waited:?}"
    );
    assert!(timing.queue_ns.unwrap() >= 200_000_000, "{timing:?}");
}

/// Fuses as [`Fused`] does, taking 200 ms per call.
#[derive(Clone, Default)]
struct Slow(Fused);

#[rlmesh::binding::async_trait]
impl ModelHandler for Slow {
    async fn predict(&mut self, observation: ModelObservation) -> rlmesh::Result<Vec<SpaceValue>> {
        self.0.predict(observation).await
    }

    async fn predict_grouped(
        &mut self,
        observations: Vec<ModelObservation>,
    ) -> Vec<rlmesh::Result<PredictFrames>> {
        tokio::time::sleep(Duration::from_millis(200)).await;
        self.0.predict_grouped(observations).await
    }

    fn fuses_groups(&self) -> bool {
        true
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_batchs_inference_time_is_split_across_its_requests() {
    let slow = Slow::default();
    let service = batched(
        slow.clone(),
        rlmesh::BatchOptions {
            batch_window_ms: 300,
            ..Default::default()
        },
    );
    service.answer(resolve("a", 1)).await;
    service.answer(resolve("b", 1)).await;
    let calls = ["a", "b"].map(|route| {
        let service = Arc::clone(&service);
        tokio::spawn(async move {
            service
                .answer(ModelRequest::Predict(vec![group(route, &[1.0])]))
                .await
                .1
        })
    });
    for call in calls {
        let user = call.await.unwrap().user_ns.unwrap();
        // Half of the one 200 ms forward each, not all of it.
        assert!((100_000_000..200_000_000).contains(&user), "user_ns {user}");
    }
    assert_eq!(slow.0.calls(), vec![2], "one handler call for both");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_model_that_does_not_fuse_is_not_batched() {
    // Echo runs a grouped predict one route at a time.
    let service = batched(
        Echo,
        rlmesh::BatchOptions {
            batch_window_ms: 5_000,
            ..Default::default()
        },
    );
    service.answer(resolve("a", 1)).await;
    service.answer(resolve("b", 1)).await;
    let started = std::time::Instant::now();
    let replies = predict_together(&service, vec![group("a", &[1.0]), group("b", &[2.0])]).await;
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "it waited a window"
    );
    let space = contract().action_space.unwrap();
    for (reply, want) in replies.iter().zip([1.0, 2.0]) {
        let ModelReply::Predicted(outcomes) = reply else {
            panic!("a predict reply")
        };
        let frames = outcomes[0].as_ref().unwrap();
        assert_eq!(
            decode_leaf_slab(&frames[0], &space, 1).unwrap(),
            vec![value(want)]
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_requests_error_does_not_fail_the_others_in_its_batch() {
    let fused = Fused::default();
    let service = batched(
        fused.clone(),
        rlmesh::BatchOptions {
            batch_window_ms: 300,
            ..Default::default()
        },
    );
    service.answer(resolve("a", 1)).await;
    service.answer(resolve("b", 1)).await;
    // b's rows are refused by the model and c was never resolved; a's
    // stand.
    let replies = predict_together(
        &service,
        vec![group("a", &[1.0]), group("b", &[-1.0]), group("c", &[3.0])],
    )
    .await;
    assert_eq!(fused.calls(), vec![2], "one call, for the resolved routes");
    let outcome = |i: usize| match &replies[i] {
        ModelReply::Predicted(outcomes) => outcomes[0].clone(),
        other => panic!("a predict reply, not {other:?}"),
    };
    assert!(outcome(0).is_ok());
    let refused = outcome(1).unwrap_err();
    assert!(
        refused.recoverable && refused.message.contains("refused -1"),
        "{refused:?}"
    );
    assert!(
        outcome(2)
            .unwrap_err()
            .message
            .contains("\"c\" was not resolved")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn drivers_inviting_a_model_into_one_session_at_once_get_one_answer_per_predict_and_rejoin() {
    let binding = ModelBinding::bind(":0/echo").await.unwrap();
    let address = binding.address();
    let cancel = tokio_util::sync::CancellationToken::new();
    let handler = Fused::default();
    let service = Arc::new(ModelService::new(handler.clone(), String::new()));
    let served = tokio::spawn(binding.serve(service, Duration::from_secs(2), cancel.clone()));
    let mut target = rlmesh_session::address::host_at(
        "127.0.0.1:0".parse().unwrap(),
        "bench",
        Duration::from_secs(1),
    )
    .await
    .unwrap();
    let session = target.address.to_string();

    // Two spellings of one session.
    let invites = (0..4).map(|i| {
        let address = address.clone();
        let session = if i % 2 == 0 {
            session.clone()
        } else {
            format!("rlmesh://{session}")
        };
        tokio::spawn(async move {
            let id = format!("driver-{i}");
            ModelConnection::connect(&address, &id, Duration::from_secs(5))
                .await
                .unwrap()
                .invite(&session, &id, Duration::from_secs(5))
                .await
                .unwrap()
        })
    });
    let mut drivers = Vec::new();
    for invite in invites.collect::<Vec<_>>() {
        drivers.push(invite.await.unwrap());
    }
    let link = &drivers[0].link;
    link.call(&resolve("a", 1), None).await.unwrap();
    let (reply, _) = link
        .call(&ModelRequest::Predict(vec![group("a", &[1.0])]), None)
        .await
        .unwrap();
    assert!(matches!(reply, ModelReply::Predicted(_)), "{reply:?}");
    // A second answerer would predict again, after the first reply.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(handler.calls(), [1]);

    for driver in drivers {
        driver.close().await;
    }
    // Invited again while it leaves: it joins once it has left.
    let again = ModelConnection::connect(&address, "driver-again", Duration::from_secs(5))
        .await
        .unwrap()
        .invite(&session, "driver-again", Duration::from_secs(5))
        .await
        .unwrap();
    again.link.call(&resolve("a", 1), None).await.unwrap();
    let (reply, _) = again
        .link
        .call(&ModelRequest::Predict(vec![group("a", &[1.0])]), None)
        .await
        .unwrap();
    assert!(matches!(reply, ModelReply::Predicted(_)), "{reply:?}");
    again.close().await;
    cancel.cancel();
    served.await.unwrap().unwrap();
    target.close().await.unwrap();
}

/// Closes the service it serves in, from inside one of its own callbacks.
type CloseFn = Arc<
    dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = rlmesh::Result<()>> + Send>>
        + Send
        + Sync,
>;

/// Logs its forwards and its close. A forward says it started and waits
/// for the test to let it end, or panics on an observation of `-1`; it
/// closes its service from inside when `close_in` names the callback.
#[derive(Clone)]
struct Held {
    log: Arc<std::sync::Mutex<Vec<&'static str>>>,
    started: Arc<tokio::sync::Notify>,
    go: Arc<tokio::sync::Semaphore>,
    fuses: bool,
    close_in: Option<&'static str>,
    close: Arc<std::sync::OnceLock<CloseFn>>,
}

impl Held {
    fn new(fuses: bool, close_in: Option<&'static str>) -> Self {
        Self {
            log: Arc::default(),
            started: Arc::default(),
            go: Arc::new(tokio::sync::Semaphore::new(0)),
            fuses,
            close_in,
            close: Arc::default(),
        }
    }

    fn log(&self) -> Vec<&'static str> {
        self.log.lock().unwrap().clone()
    }

    async fn close_from(&self, callback: &str) {
        if self.close_in == Some(callback) {
            let refused = (self.close.get().unwrap())().await.is_err();
            self.log
                .lock()
                .unwrap()
                .push(if refused { "close refused" } else { "closed" });
        }
    }
}

#[rlmesh::binding::async_trait]
impl ModelHandler for Held {
    async fn predict(&mut self, observation: ModelObservation) -> rlmesh::Result<Vec<SpaceValue>> {
        let lanes = observation.decoded_lanes()?;
        assert!(!lanes.contains(&value(-1.0)), "the handler panicked");
        self.log.lock().unwrap().push("start");
        self.started.notify_one();
        self.close_from("predict").await;
        self.go.acquire().await.unwrap().forget();
        self.log.lock().unwrap().push("end");
        Ok(lanes)
    }

    fn fuses_groups(&self) -> bool {
        self.fuses
    }

    async fn on_close(&mut self) -> rlmesh::Result<()> {
        self.log.lock().unwrap().push("on_close");
        self.close_from("on_close").await;
        Ok(())
    }
}

fn held(handler: &Held) -> Arc<ModelService<Held>> {
    let service = batched(
        handler.clone(),
        rlmesh::BatchOptions {
            batch_window_ms: 200,
            ..Default::default()
        },
    );
    let weak = Arc::downgrade(&service);
    let close: CloseFn = Arc::new(move || {
        let service = weak.upgrade().unwrap();
        Box::pin(async move { service.close(Duration::from_secs(10)).await })
    });
    let _ = handler.close.set(close);
    service
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_close_waits_for_accepted_work_however_its_request_ends_and_closes_once() {
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Ends {
        Answered,
        /// Cancelled once its forward started.
        Cancelled,
        /// Cancelled while its batch still assembles.
        TimedOut,
        Panicked,
    }
    for (fuses, ends, want) in [
        (false, Ends::Answered, &["start", "end", "on_close"][..]),
        (true, Ends::Answered, &["start", "end", "on_close"]),
        // Its forward is dropped with it.
        (false, Ends::Cancelled, &["start", "on_close"]),
        // Its batch forwards on: the close waits for that.
        (true, Ends::Cancelled, &["start", "end", "on_close"]),
        (true, Ends::TimedOut, &["start", "end", "on_close"]),
        (false, Ends::Panicked, &["on_close"]),
        (true, Ends::Panicked, &["on_close"]),
    ] {
        let case = format!("fuses {fuses}, {ends:?}");
        let handler = Held::new(fuses, None);
        let service = held(&handler);
        service.answer(resolve("a", 1)).await;
        let observation = if ends == Ends::Panicked { -1.0 } else { 1.0 };
        let started = handler.started.notified();
        let accept = || {
            let service = Arc::clone(&service);
            tokio::spawn(async move {
                service
                    .answer(ModelRequest::Predict(vec![group("a", &[observation])]))
                    .await
                    .0
            })
        };
        let answered = match ends {
            Ends::Panicked => {
                let reply = accept().await;
                // Unbatched, the panic is the request's own.
                assert!(
                    reply.is_err()
                        || matches!(reply, Ok(ModelReply::Predicted(ref o)) if o[0].is_err()),
                    "{case}: {reply:?}"
                );
                None
            }
            Ends::Cancelled => {
                let accepted = accept();
                started.await;
                accepted.abort();
                assert!(accepted.await.unwrap_err().is_cancelled(), "{case}");
                None
            }
            Ends::TimedOut => {
                // b never predicts, so a's batch waits its window.
                service.answer(resolve("b", 1)).await;
                let predict = service.answer(ModelRequest::Predict(vec![group("a", &[1.0])]));
                assert!(tokio::time::timeout(Duration::ZERO, predict).await.is_err());
                None
            }
            Ends::Answered => {
                let accepted = accept();
                started.await;
                Some(accepted)
            }
        };
        if answered.is_some() || (ends != Ends::Panicked && fuses) {
            // The work is still running: the close waits for it, up to its
            // deadline, and refuses anything new meanwhile.
            let early = service.close(Duration::from_millis(100)).await;
            assert!(
                matches!(early, Err(rlmesh::Error::Timeout(_))),
                "{case}: {early:?}"
            );
            let (late, _) = service
                .answer(ModelRequest::Predict(vec![group("a", &[1.0])]))
                .await;
            assert!(matches!(late, ModelReply::Fault(_)), "{case}: {late:?}");
            assert!(!handler.log().contains(&"on_close"), "{case}");
            handler.go.add_permits(1);
        }
        let (first, second) = tokio::join!(
            service.close(Duration::from_secs(10)),
            service.close(Duration::from_secs(10))
        );
        first.unwrap();
        second.unwrap();
        service.close(Duration::from_secs(10)).await.unwrap();
        assert_eq!(handler.log(), want, "{case}");
        if let Some(answered) = answered {
            let reply = answered.await.unwrap();
            assert!(
                matches!(reply, ModelReply::Predicted(ref o) if o[0].is_ok()),
                "{case}: {reply:?}"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_close_from_the_models_own_callback_is_refused_rather_than_waiting_on_itself() {
    for (fuses, close_in, want) in [
        (
            false,
            "predict",
            &["start", "close refused", "end", "on_close"][..],
        ),
        (
            true,
            "predict",
            &["start", "close refused", "end", "on_close"],
        ),
        (
            false,
            "on_close",
            &["start", "end", "on_close", "close refused"],
        ),
    ] {
        let handler = Held::new(fuses, Some(close_in));
        handler.go.add_permits(1);
        let service = held(&handler);
        service.answer(resolve("a", 1)).await;
        let (reply, _) = service
            .answer(ModelRequest::Predict(vec![group("a", &[1.0])]))
            .await;
        assert!(
            matches!(reply, ModelReply::Predicted(ref o) if o[0].is_ok()),
            "{reply:?}"
        );
        service.close(Duration::from_secs(10)).await.unwrap();
        assert_eq!(handler.log(), want, "fuses {fuses}, from {close_in}");
    }
}
