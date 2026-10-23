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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_driver_waits_for_a_model_not_up_yet_and_never_hosts_its_session() {
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let address = format!("127.0.0.1:{port}/echo");
    let connecting = tokio::spawn({
        let address = address.clone();
        async move { ModelConnection::connect(&address, "driver", Duration::from_secs(10)).await }
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    let listening = std::net::TcpStream::connect(("127.0.0.1", port)).is_ok();
    assert!(!listening, "the driver hosted the model's session");

    let binding = ModelBinding::bind(&address).await.unwrap();
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
