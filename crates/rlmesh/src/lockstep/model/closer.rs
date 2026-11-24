//! Flights and the closer: the count of accepted requests still running, and
//! the one close that waits for them before closing the handler.

use std::sync::Arc;
use std::time::Duration;

use tracing::warn;

use super::ModelService;
use crate::Error;
use crate::model::handler::ModelHandler;

#[derive(Default)]
pub(super) struct Flights {
    count: std::sync::atomic::AtomicU32,
    landed: tokio::sync::Notify,
}

pub(super) struct Flight(Arc<Flights>);

impl Flights {
    pub(super) fn take(self: &Arc<Self>) -> (Flight, u32) {
        let count = self.count.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        (Flight(Arc::clone(self)), count)
    }

    async fn landed(&self) {
        loop {
            let landed = self.landed.notified();
            tokio::pin!(landed);
            landed.as_mut().enable();
            if self.count.load(std::sync::atomic::Ordering::SeqCst) == 0 {
                return;
            }
            landed.await;
        }
    }
}

impl Drop for Flight {
    fn drop(&mut self) {
        if self
            .0
            .count
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst)
            == 1
        {
            self.0.landed.notify_waiters();
        }
    }
}
impl<H: ModelHandler + 'static> ModelService<H> {
    /// Refuses new requests and waits up to `timeout` for the model to
    /// close: the first close starts the one closer, which waits for the
    /// accepted requests to land, then drops every route and closes the
    /// handler; every close returns its outcome. Past `timeout` this fails and
    /// the closer carries on, so `on_close` never runs beside a request. A close
    /// from one of the model's own requests or its `on_close` waits on itself
    /// and fails with [`Error::Timeout`].
    pub async fn close(self: &Arc<Self>, timeout: Duration) -> Result<(), Error> {
        let mut closed = self.closed.subscribe();
        if !self.closing.swap(true, std::sync::atomic::Ordering::SeqCst) {
            tokio::spawn(Arc::clone(self).closer());
        }
        match tokio::time::timeout(timeout, closed.wait_for(Option::is_some)).await {
            Ok(Ok(outcome)) => outcome.clone().unwrap_or(Ok(())),
            Ok(Err(_)) => Err(Error::Runtime("the model's closer is gone".into())),
            Err(_) => {
                warn!(
                    ?timeout,
                    "model requests still running at close; the handler closes once they end"
                );
                Err(Error::Timeout(timeout))
            }
        }
    }

    async fn closer(self: Arc<Self>) {
        self.flights.landed().await;
        if let Some(batching) = &self.batching {
            batching
                .queue
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
        }
        let service = Arc::clone(&self);
        let shut = tokio::spawn(async move { service.shut().await });
        let outcome = shut
            .await
            .unwrap_or_else(|e| Err(Error::model(format!("closing the model failed: {e}"))));
        self.closed.send_replace(Some(outcome));
    }

    async fn shut(&self) -> Result<(), Error> {
        let routes: Vec<String> = self.routes.lock().await.drain().map(|(r, _)| r).collect();
        let mut shut = Ok(());
        if let Some(setup) = &self.setup {
            for route in &routes {
                shut = shut.and(setup.release_adapter(route).await);
            }
        }
        shut.and(self.handler.lock().await.on_close().await)
    }
}
