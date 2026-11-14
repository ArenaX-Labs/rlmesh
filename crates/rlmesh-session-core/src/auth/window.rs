//! The host's per-generation request window: request ids and the reply
//! cache that makes a retried verb idempotent.

use alloc::collections::BTreeMap;

/// How many request ids the host remembers per participant generation.
pub const REQUEST_WINDOW: u64 = 64;

/// What the host does with a request id ([`RequestWindow::admit`]).
#[derive(Debug, PartialEq, Eq)]
pub enum Admit<'a, R> {
    /// New: apply the verb once, then [`RequestWindow::complete`] it.
    Execute,
    /// A retry of an answered request: send this reply again.
    Done(&'a R),
    /// A retry of a request still executing (a grant waiting for its
    /// robots' keys): it joins that operation and gets its one reply.
    Pending,
    /// Below the window: never applied (`EXPIRED`).
    Expired,
}

/// One participant generation's request ids: a window of
/// [`REQUEST_WINDOW`] ids from a floor, each absent, pending or answered
/// (with its reply). The host applies verbs one at a time; only a grant
/// stays pending past its turn, while its robots install its keys.
#[derive(Debug)]
pub struct RequestWindow<R> {
    floor: u64,
    done: BTreeMap<u64, R>,
    pending: alloc::collections::BTreeSet<u64>,
}

impl<R> Default for RequestWindow<R> {
    fn default() -> Self {
        Self {
            floor: 1,
            done: BTreeMap::new(),
            pending: alloc::collections::BTreeSet::new(),
        }
    }
}

impl<R> RequestWindow<R> {
    /// Admits request `id`. An id above the window moves the floor up (and
    /// forgets the replies below it).
    pub fn admit(&mut self, id: u64) -> Admit<'_, R> {
        if id < self.floor {
            return Admit::Expired;
        }
        if id - self.floor >= REQUEST_WINDOW {
            self.floor = id - (REQUEST_WINDOW - 1);
            self.done = self.done.split_off(&self.floor);
        }
        if self.pending.contains(&id) {
            return Admit::Pending;
        }
        match self.done.get(&id) {
            Some(reply) => Admit::Done(reply),
            None => Admit::Execute,
        }
    }

    /// `id` is executing past its turn: retries join it until
    /// [`complete`](Self::complete).
    pub fn begin(&mut self, id: u64) {
        self.pending.insert(id);
    }

    /// Records the reply to `id`, which a retry gets again.
    pub fn complete(&mut self, id: u64, reply: R) {
        self.pending.remove(&id);
        if id >= self.floor {
            self.done.insert(id, reply);
        }
    }
}
