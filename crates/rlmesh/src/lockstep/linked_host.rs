//! For tests: a session host whose participants open links of their own
//! (the lockstep session included), and a check that dropping what holds
//! them takes no wakeup of another future on the dropping thread.

use std::future::Future;
use std::net::SocketAddr;
use std::task::Poll;
use std::time::Duration;

use rlmesh_session::address::SessionAddress;
use rlmesh_session::transport::{Transport, TransportConfig, open_local_listeners};
use rlmesh_session::wire::{ParticipantId, PriorityClass, Role, SessionKeys};
use rlmesh_session::{HostConfig, Participant, ParticipantConfig, SessionHost};

/// A host on free local ports, not registered as this process's: a
/// participant joining it links to it rather than riding its sessions.
pub(crate) struct LinkedHost {
    transport: Transport,
    host: SessionHost,
    /// Its address (only the base and the session matter here).
    pub(crate) address: SessionAddress,
}

impl LinkedHost {
    pub(crate) async fn start() -> Self {
        let (transport, base) = open_local_listeners([127, 0, 0, 1].into())
            .await
            .expect("listeners");
        let address = SessionAddress::new(base, "linked").expect("address");
        let mut config = HostConfig::new(address.tenant.clone(), address.session);
        config.health_log = None;
        let host = SessionHost::start(transport.clone(), config)
            .await
            .expect("host");
        Self {
            transport,
            host,
            address,
        }
    }

    /// Attaches `id` over links of its own.
    pub(crate) async fn attach(
        &self,
        id: &str,
        role: Role,
        priority: PriorityClass,
    ) -> Participant {
        let base: SocketAddr = self.address.base;
        let transport = Transport::open(
            &TransportConfig::per_class_tcp(
                Some(SocketAddr::from(([127, 0, 0, 1], 0))),
                Some(base),
            )
            .expect("config"),
        )
        .await
        .expect("links");
        let mut config = ParticipantConfig::new(
            self.address.tenant.clone(),
            self.address.session,
            ParticipantId::new(id).expect("id"),
            vec![role],
            priority,
        );
        config.credential = Some(self.host.credential());
        Participant::attach(transport, config)
            .await
            .expect("attach")
    }

    /// The session's keys.
    pub(crate) fn keys(&self) -> SessionKeys {
        SessionKeys::new(&self.address.tenant, self.address.session)
    }

    /// Waits until something answers a query on `key` (a refusal counts).
    pub(crate) async fn answered(&self, key: &str) {
        loop {
            let replies = self
                .transport
                .control()
                .get(key)
                .timeout(Duration::from_millis(200))
                .await
                .expect("query");
            if replies.recv_async().await.is_ok() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

/// Polls `sibling` once, on a thread blocking on a future of a multi-thread
/// runtime, with a timer pending whose wakeup has fired; then waits only for
/// that timer. A sibling that took the wakeup (a Zenoh close in a nested
/// `block_on` on the same thread) leaves it waiting forever.
pub(crate) fn timer_survives(sibling: impl Future<Output = ()> + Send + 'static) -> bool {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime");
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
