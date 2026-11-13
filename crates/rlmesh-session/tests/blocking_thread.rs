//! A transport opened or dropped on a thread that blocks on a future (a
//! program's main thread, a binding's call) takes none of the wakeups of the
//! other futures polled there.

#![allow(clippy::unwrap_used)]

use std::future::Future;
use std::task::Poll;
use std::time::Duration;

use rlmesh_session::transport::{Transport, TransportConfig, TransportTuning, open_local_listener};

/// Polls `sibling` once with a timer pending whose wakeup has fired, as a
/// runner's main future polls one route while another waits; then waits
/// only for that timer. A sibling that took the wakeup leaves it waiting
/// forever.
fn timer_survives(sibling: impl Future<Output = ()> + Send + 'static) -> bool {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let (done, finished) = std::sync::mpsc::channel();
    let handle = runtime.handle().clone();
    std::thread::spawn(move || {
        handle.block_on(async move {
            let mut timer = Box::pin(tokio::time::sleep(Duration::from_millis(5)));
            let mut sibling = Some(Box::pin(sibling));
            std::future::poll_fn(|cx| {
                if timer.as_mut().poll(cx).is_ready() {
                    return Poll::Ready(());
                }
                if let Some(mut polled) = sibling.take() {
                    // The timer fires meanwhile: its wakeup is pending here.
                    std::thread::sleep(Duration::from_millis(50));
                    // Kept pending or not, it is not polled again.
                    let _ = polled.as_mut().poll(cx);
                    std::mem::forget(polled);
                }
                Poll::Pending
            })
            .await;
        });
        let _ = done.send(());
    });
    let survived = finished.recv_timeout(Duration::from_secs(10)).is_ok();
    std::mem::forget(runtime);
    survived
}

fn client_of(base: std::net::SocketAddr) -> TransportConfig {
    TransportConfig::per_class_tcp(base).with_tuning(TransportTuning {
        client: true,
        ..TransportTuning::default()
    })
}

#[test]
fn opening_a_transport_takes_no_wakeup_of_another_future() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let (host, base) = runtime
        .block_on(open_local_listener([127, 0, 0, 1].into()))
        .unwrap();
    let opened = timer_survives(async move {
        let _ = Transport::open(&client_of(base)).await;
    });
    assert!(opened, "a transport's open took a pending wakeup");
    drop(host);
}

#[test]
fn dropping_a_transport_takes_no_wakeup_of_another_future() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let (host, base) = runtime
        .block_on(open_local_listener([127, 0, 0, 1].into()))
        .unwrap();
    let client = runtime.block_on(Transport::open(&client_of(base))).unwrap();
    let dropped = timer_survives(async move { drop(client) });
    assert!(dropped, "a transport's close took a pending wakeup");
    drop(host);
}
