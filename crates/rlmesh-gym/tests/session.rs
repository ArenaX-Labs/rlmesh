//! A lockstep env in a session: one lease grant, then per-step checks of
//! the epoch, the lease token and the e-stop: the direct link in one
//! process, the lockstep session otherwise.

#![allow(clippy::unwrap_used)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::CatalogEnv;
use rlmesh_gym::{ConnectOptions, GymError, LockstepAddress};
use rlmesh_proto::env::v1::{ResetRequest, StepRequest};
use rlmesh_proto::spaces::v1::SpaceValue;
use rlmesh_session::{Participant, ParticipantConfig, Transport, TransportConfig};
use rlmesh_spaces::AutoresetMode;
use rlmesh_wire::{LeaseToken, ParticipantId, PriorityClass, Role};

fn step(seq: u8) -> StepRequest {
    StepRequest {
        action: Some(SpaceValue {
            leaves: vec![bytes::Bytes::from(vec![seq; 28])],
        }),
        ..Default::default()
    }
}

fn reset() -> ResetRequest {
    ResetRequest {
        seeds: vec![3],
        episode_ids: vec!["e1".into()],
        ..Default::default()
    }
}

async fn operator(address: &str) -> Participant {
    let address = LockstepAddress::parse(address).unwrap();
    let (tenant, session) = address.session.unwrap();
    let transport = Transport::open(
        &TransportConfig::per_class_tcp(Some("127.0.0.1:0".parse().unwrap()), Some(address.base))
            .unwrap(),
    )
    .await
    .unwrap();
    Participant::attach(
        transport,
        ParticipantConfig::new(
            tenant,
            session,
            ParticipantId::new("operator").unwrap(),
            vec![Role::Controller],
            PriorityClass::Safety,
        ),
    )
    .await
    .unwrap()
}

fn refused(result: Result<impl std::fmt::Debug, GymError>, why: &str) {
    let err = result.expect_err("refused");
    assert!(err.to_string().contains(why), "{err} (wanted {why:?})");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_in_process_driver_calls_the_env_directly_and_the_env_checks_epoch_and_estop() {
    let binding = rlmesh_gym::bind("lockstep://127.0.0.1:0/catalog")
        .await
        .unwrap();
    let address = binding.address();
    let cancel = tokio_util::sync::CancellationToken::new();
    let served = tokio::spawn(binding.serve(
        Arc::new(CatalogEnv::new(1, AutoresetMode::Disabled, false)),
        rlmesh_gym::ServeOptions::default(),
        cancel.clone(),
    ));
    let direct = rlmesh_gym::connect(&address, ConnectOptions::default())
        .await
        .unwrap();
    // A second driver over the lockstep session (it takes the lease over
    // only when the first releases it: the same priority class is busy).
    let networked = rlmesh_gym::connect(
        &address,
        ConnectOptions {
            direct: false,
            id: Some("driver-net".into()),
            ..Default::default()
        },
    )
    .await;
    assert!(networked.is_err(), "the lease is held by the first driver");

    direct.reset_v1(reset()).await.unwrap();
    let reply = direct.step_v1(step(1)).await.unwrap();
    assert_eq!(reply.rewards.len(), 1);
    assert_eq!(direct.info().contract.id, "Catalog-v0");

    // A stale epoch is refused locally, without the host.
    let (epoch, token) = direct.authority();
    direct.set_authority(rlmesh_wire::Epoch(epoch.0 + 7), token);
    refused(direct.step_v1(step(2)).await, "stale");
    direct.set_authority(epoch, token);
    direct.step_v1(step(3)).await.unwrap();

    // E-stop: refused between steps.
    let op = operator(&address).await;
    op.estop("test").await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    refused(direct.step_v1(step(2)).await, "e-stopped");
    cancel.cancel();
    served.await.unwrap().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_driver_over_the_lockstep_session_steps_and_a_stale_token_is_refused() {
    let binding = rlmesh_gym::bind("lockstep://127.0.0.1:0/catalog")
        .await
        .unwrap();
    let address = binding.address();
    let cancel = tokio_util::sync::CancellationToken::new();
    let served = tokio::spawn(binding.serve(
        Arc::new(CatalogEnv::new(2, AutoresetMode::NextStep, false)),
        rlmesh_gym::ServeOptions::default(),
        cancel.clone(),
    ));
    let env = rlmesh_gym::connect(
        &address,
        ConnectOptions {
            direct: false,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    env.reset_v1(ResetRequest {
        episode_ids: vec!["a".into(), "b".into()],
        ..Default::default()
    })
    .await
    .unwrap();
    let mut two = step(1);
    two.action = Some(SpaceValue {
        leaves: vec![bytes::Bytes::from(vec![1_u8; 56])],
    });
    two.episode_ids = vec!["a".into(), "b".into()];
    let reply = env.step_v1(two.clone()).await.unwrap();
    assert_eq!(reply.rewards.len(), 2);
    assert_eq!(reply.terminated_mask.len(), 2);

    // Keep the epoch, forge the token: the env refuses it from its own view.
    let (epoch, _) = env.authority();
    env.set_authority(epoch, LeaseToken(0xdead));
    refused(env.step_v1(two.clone()).await, "lease token");
    // A malformed action (one lane's worth) is refused as malformed.
    let mut short = two;
    short.action = Some(SpaceValue {
        leaves: vec![bytes::Bytes::from(vec![1_u8; 28])],
    });
    refused(env.step_v1(short).await, "lanes");
    cancel.cancel();
    served.await.unwrap().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_recorded_lockstep_env_tees_every_step_per_lane() {
    let path = std::env::temp_dir().join(format!("rlmesh-lockstep-{}.mcap", std::process::id()));
    let binding = rlmesh_gym::bind("lockstep://127.0.0.1:0/catalog")
        .await
        .unwrap();
    let address = binding.address();
    let cancel = tokio_util::sync::CancellationToken::new();
    let served = tokio::spawn(binding.serve(
        Arc::new(CatalogEnv::new(2, AutoresetMode::NextStep, false)),
        rlmesh_gym::ServeOptions {
            record: Some(path.clone()),
            ..Default::default()
        },
        cancel.clone(),
    ));
    let env = rlmesh_gym::connect(&address, ConnectOptions::default())
        .await
        .unwrap();
    env.reset_v1(ResetRequest {
        episode_ids: vec!["a".into(), "b".into()],
        ..Default::default()
    })
    .await
    .unwrap();
    for i in 0..3_u8 {
        let mut two = step(i);
        two.action = Some(SpaceValue {
            leaves: vec![bytes::Bytes::from(vec![i; 56])],
        });
        two.episode_ids = vec!["a".into(), "b".into()];
        env.step_v1(two).await.unwrap();
    }
    drop(env);
    cancel.cancel();
    served.await.unwrap().unwrap();
    let summary = rlmesh_record::summarize(&path).unwrap();
    let count = |topic: &str| {
        summary
            .topics
            .iter()
            .find(|t| t.topic == topic)
            .map_or(0, |t| t.messages)
    };
    for lane in 0..2 {
        assert_eq!(
            count(&format!("/catalog/lane{lane}/scene/cam")),
            4,
            "reset + 3 steps"
        );
        assert_eq!(
            count(&format!("/catalog/lane{lane}/action/_applied")),
            3,
            "3 commands"
        );
    }
    assert!(summary.topics.iter().all(|t| t.undecodable == 0));
    let _ = std::fs::remove_file(&path);
}
