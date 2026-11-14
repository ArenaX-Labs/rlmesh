//! Several drivers of one served model in one process, on the thread that
//! blocks on the program's future, as an embedding runner drives one model
//! per route from its main future.

#![allow(clippy::unwrap_used)]

mod common;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Poll;
use std::time::Duration;

use common::boxed;
use rlmesh::binding::model::{
    EpisodeInfo, ModelHandler, ModelObservation, ModelRouteContext, ResolveOptions,
};
use rlmesh::lockstep::{ModelBinding, ModelService, ServedModel};
use rlmesh_session::wire::{ModelEpisode, ModelGroup};
use rlmesh_spaces::leaves::encode_leaf_slab;
use rlmesh_spaces::{AutoresetMode, DType, EnvContract, SpaceValue, Tensor};

/// Set in the child process that serves the model.
const CHILD: &str = "RLMESH_TEST_SERVED_MODEL_CHILD";

/// Answers each predict with its observation, after a pause when told to
/// (`RLMESH_TEST_SERVED_MODEL_PAUSE`, milliseconds).
struct Echo(Duration);

#[rlmesh::binding::async_trait]
impl ModelHandler for Echo {
    async fn predict(&mut self, observation: ModelObservation) -> rlmesh::Result<Vec<SpaceValue>> {
        tokio::time::sleep(self.0).await;
        observation.decoded_lanes()
    }
}

/// The child half of these tests: serves the model in a session it hosts
/// under the launch identity its parent passed, as a model's container
/// does, and prints only its port.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn served_model_child() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    let pause = std::env::var("RLMESH_TEST_SERVED_MODEL_PAUSE")
        .ok()
        .and_then(|ms| ms.parse().ok())
        .map(Duration::from_millis)
        .unwrap_or_default();
    let binding = ModelBinding::bind("127.0.0.1:0/model").await.unwrap();
    let address = rlmesh_session::address::Address::participant(&binding.address()).unwrap();
    let port = address.host.and_then(|host| host.port).unwrap();
    println!("PORT {port}");
    let service = Arc::new(ModelService::new(Echo(pause), String::new()));
    binding
        .serve(
            service,
            Duration::from_secs(10),
            tokio_util::sync::CancellationToken::new(),
        )
        .await
        .unwrap();
}

/// A model served by a child process, killed when dropped.
struct Model {
    child: std::process::Child,
    port: String,
    key: String,
}

impl Model {
    /// Starts the child, its predicts paused `pause`.
    fn start(pause: Duration) -> Self {
        use std::io::BufRead;
        let (identity, key) = rlmesh_session::address::launch_identity().unwrap();
        // The identity reaches the child through its environment, never argv.
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "served_model_child", "--nocapture"])
            .env(CHILD, "1")
            .env(
                "RLMESH_TEST_SERVED_MODEL_PAUSE",
                pause.as_millis().to_string(),
            )
            .env(
                rlmesh_session::address::HOST_IDENTITY_VAR,
                identity.as_str(),
            )
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut lines = std::io::BufReader::new(child.stdout.take().unwrap()).lines();
        let port = loop {
            let line = lines.next().expect("the child serves").unwrap();
            if let Some(port) = line.strip_prefix("PORT ") {
                break port.to_string();
            }
        };
        Self {
            child,
            port,
            key: key.to_string(),
        }
    }

    /// The model's keyed address.
    fn address(&self) -> String {
        format!("127.0.0.1:{}#{}/model", self.port, self.key)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
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

fn observation(route: &str) -> ModelObservation {
    let bytes = [1.0_f32, -1.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let row = SpaceValue::Box(Tensor::from_vec(bytes, vec![2], DType::Float32).unwrap());
    let group = ModelGroup {
        route: route.into(),
        episodes: vec![ModelEpisode {
            id: format!("{route}-0"),
            seed: Some(0),
        }],
        observation: encode_leaf_slab(&[row], contract().observation_space.as_ref().unwrap())
            .unwrap(),
        history: Vec::new(),
        step: Some(0),
    };
    ModelObservation {
        observation: Some(group.observation),
        route: ModelRouteContext {
            env_id: route.into(),
            episodes: vec![EpisodeInfo::default()],
            ..ModelRouteContext::default()
        },
        num_envs: 1,
        ..ModelObservation::default()
    }
}

async fn resolved(address: &str, route: &str) -> ServedModel {
    resolve(ServedModel::new(address, Duration::from_secs(20)), route).await
}

async fn resolve(model: ServedModel, route: &str) -> ServedModel {
    model
        .route_setup()
        .unwrap()
        .resolve_adapter(
            route,
            &contract(),
            ResolveOptions {
                execution_horizon: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    model
}

/// One round on the calling thread: `b`'s predict in flight and its reply
/// in, then, in the same poll, `a` dropped, as a runner drops a finished
/// route's model while another route waits on its own. Polled first, `b`
/// is not polled again in that poll: only its reply's wakeup polls it
/// again.
async fn round(address: &str) {
    let a = resolved(address, "a").await;
    let mut b = resolved(address, "b").await;
    let mut a = Some(a);
    let mut predict: Pin<Box<dyn Future<Output = _>>> = Box::pin(b.predict(observation("b")));
    let actions = std::future::poll_fn(|cx| {
        let polled = predict.as_mut().poll(cx);
        if polled.is_pending()
            && let Some(a) = a.take()
        {
            // The reply arrives meanwhile: its wakeup is pending here.
            std::thread::sleep(Duration::from_millis(50));
            drop(a);
            return Poll::Pending;
        }
        polled
    })
    .await
    .unwrap();
    assert_eq!(actions.len(), 1);
    drop(predict);
    b.on_close().await.unwrap();
}

#[test]
fn dropping_a_driver_never_takes_the_reply_of_another_on_the_same_thread() {
    let model = Model::start(Duration::ZERO);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    // Each round on a thread of its own, which blocks on it as a program's
    // main thread does: one whose wakeup was taken never finishes.
    const ROUNDS: usize = 5;
    for n in 0..ROUNDS {
        let (done, finished) = std::sync::mpsc::channel();
        let (handle, address) = (runtime.handle().clone(), model.address());
        std::thread::spawn(move || {
            handle.block_on(round(&address));
            let _ = done.send(());
        });
        assert!(
            finished.recv_timeout(Duration::from_secs(30)).is_ok(),
            "round {n} of {ROUNDS}: the predict's reply was lost while another driver dropped"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_predict_waits_its_own_deadline_and_fails_at_once_when_the_model_is_gone() {
    use rlmesh::ext::ErrorExt;

    // Its predicts take far longer than the test: only a deadline or the
    // model going away ends one.
    let mut model = Model::start(Duration::from_secs(600));
    let silent = ServedModel::new(model.address(), Duration::from_secs(20))
        .with_request_timeout(Duration::from_secs(1));
    let mut silent = resolve(silent, "a").await;
    let started = std::time::Instant::now();
    let err = silent.predict(observation("a")).await.unwrap_err();
    let waited = started.elapsed();
    assert!(
        (Duration::from_secs(1)..Duration::from_secs(5)).contains(&waited),
        "{waited:?}: {err}"
    );
    assert!(matches!(err, rlmesh::Error::Timeout(_)), "{err}");
    assert!(err.is_transient(), "{err}");

    // A model its host does not have yet may still be loading: a retry may
    // find it, so a caller that retries does not stop for good.
    let address = model.address();
    let (host, _) = address.rsplit_once('/').unwrap();
    let absent = ServedModel::new(format!("{host}/absent"), Duration::from_secs(2));
    let err = absent
        .route_setup()
        .unwrap()
        .resolve_adapter(
            "a",
            &contract(),
            ResolveOptions {
                execution_horizon: 1,
                ..Default::default()
            },
        )
        .await
        .expect_err("no such model");
    assert!(err.is_transient(), "{err}");
    assert!(err.to_string().contains("model absent to answer"), "{err}");

    let mut driver = resolved(&model.address(), "b").await;
    let predict = tokio::spawn(async move { driver.predict(observation("b")).await });
    tokio::time::sleep(Duration::from_millis(500)).await;
    let killed = std::time::Instant::now();
    model.child.kill().unwrap();
    let outcome = tokio::time::timeout(Duration::from_secs(30), predict)
        .await
        .expect("the predict ends once the model is gone")
        .unwrap();
    let ended = killed.elapsed();
    let err = outcome.expect_err("no model answered");
    assert!(
        ended < Duration::from_secs(5),
        "ended {ended:?} after: {err}"
    );
    assert!(err.is_transient(), "{err}");
    assert!(err.to_string().contains("went away"), "{err}");
}

#[test]
fn a_wrong_key_refuses_every_driver_on_the_thread_at_once_and_for_good() {
    use rlmesh::ext::ErrorExt;

    let model = Model::start(Duration::ZERO);
    let (_, wrong) = rlmesh_session::address::launch_identity().unwrap();
    let wrong = wrong.to_string();
    let address = format!("127.0.0.1:{}#{wrong}/model", model.port);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    // Three routes' joins in one future on this thread, as a runner joins
    // one model per route from its main future, with a join timeout far
    // longer than the test.
    let started = std::time::Instant::now();
    let refused = runtime.block_on(async {
        let joins = (0..3).map(|route| -> Pin<Box<dyn Future<Output = _>>> {
            let model = ServedModel::new(address.clone(), Duration::from_secs(600));
            Box::pin(async move {
                model
                    .route_setup()
                    .unwrap()
                    .resolve_adapter(&format!("{route}"), &contract(), ResolveOptions::default())
                    .await
                    .map(|_| ())
            })
        });
        tokio::time::timeout(Duration::from_secs(30), futures_join(joins.collect()))
            .await
            .expect("every driver is refused, none waits out its timeout")
    });
    let took = started.elapsed();
    assert!(took < Duration::from_secs(5), "refused in {took:?}");
    // A predict that joins first is refused as finally as a resolve.
    let predicted = runtime.block_on(async {
        let mut model = ServedModel::new(address.clone(), Duration::from_secs(600));
        model.predict(observation("0")).await.map(|_| ())
    });
    for outcome in refused.into_iter().chain([predicted]) {
        let err = outcome.expect_err("a wrong key is refused");
        let text = err.to_string();
        assert!(!err.is_transient(), "{text}");
        assert!(text.contains("refused") && text.contains("key"), "{text}");
        assert!(!text.contains(&wrong[..20]), "the error repeats the key");
    }
}

/// Every future's outcome, all polled by the one future the caller awaits.
async fn futures_join<T>(futures: Vec<Pin<Box<dyn Future<Output = T> + '_>>>) -> Vec<T> {
    let mut futures: Vec<_> = futures.into_iter().map(Some).collect();
    let mut out: Vec<Option<T>> = futures.iter().map(|_| None).collect();
    std::future::poll_fn(|cx| {
        for (future, slot) in futures.iter_mut().zip(out.iter_mut()) {
            if let Some(polled) = future.as_mut()
                && let Poll::Ready(value) = polled.as_mut().poll(cx)
            {
                *slot = Some(value);
                *future = None;
            }
        }
        if out.iter().all(Option::is_some) {
            Poll::Ready(out.iter_mut().map(|o| o.take().unwrap()).collect())
        } else {
            Poll::Pending
        }
    })
    .await
}
