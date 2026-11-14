//! The host also keeps the session's **history store**: for every
//! declared stream (bulk streams only when configured) it subscribes to the
//! samples and keeps a bounded ring per stream, in the current epoch only,
//! served on `h/{robot}/{path}`. A controller queries it on attach, on a
//! standby swap and to recover a gap, never per tick: steady-state windows
//! are assembled from the live stream. A query names the epoch, an exclusive
//! `after_seq`, an inclusive `until_seq` watermark taken from the live
//! stream and a limit; the reply is one or more [`SampleBatch`] pages, the
//! first carrying `truncated` (retention ran out before the range) and the
//! gaps the store knows of. A query must be sealed under the requester's
//! `k_up`, and its pages are sealed under its `k_down` with the query's
//! request id; the store opens every sample under its stream's key
//! and keeps only the session's current epoch.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, mpsc};

use crate::wire::codec;
use crate::wire::part::{MAX_BATCH, MAX_CHUNK_BYTES};
use crate::wire::{
    Applied, ChannelId, ChannelPriority, Epoch, HistoryQuery, ParticipantId, SampleBatch,
    SampleMeta, SeqRange, SessionKeys, StreamSpec,
};
use bytes::Bytes;
use rlmesh_session_core::auth::Keyring;
use rlmesh_session_core::crypto::Kind;
use rlmesh_session_core::crypto::sealed::Header;
use tracing::{debug, warn};
use zenoh::Wait;
use zenoh::query::Query;

use crate::authority::Access;
use crate::clock::Clock;
use crate::host::peer::SharedPeers;
use crate::host::{HostActor, lock};
use crate::transport::Transport;

/// The most gap ranges a stream's history remembers.
const MAX_GAPS: usize = 64;

/// One stored sample.
#[derive(Clone)]
struct Stored {
    seq: u64,
    capture_ns: i64,
    tick: u64,
    applied: Option<Applied>,
    payload: Bytes,
}

/// One stream's ring, in its newest epoch.
struct RingState {
    epoch: Epoch,
    /// The first seq seen in this epoch: anything older was never here.
    first_seq: u64,
    samples: VecDeque<Stored>,
    gaps: VecDeque<SeqRange>,
}

pub(super) struct Ring {
    spec: StreamSpec,
    state: Mutex<RingState>,
    /// The session's epoch: the only one a sample is kept in.
    epoch: Arc<AtomicU64>,
    /// Samples kept.
    depth: usize,
}

impl Ring {
    /// Records a live sample, opened under its stream's key.
    fn record(&self, keys: &Mutex<Keyring>, keyexpr: &str, payload: &zenoh::bytes::ZBytes) {
        let bytes = payload.to_bytes().into_owned();
        let opened = lock(keys).open_sample(keyexpr, bytes, &self.spec.owner, self.spec.applied);
        match opened {
            Ok(sample) => self.record_one(sample),
            Err(err) => debug!(channel = %self.spec.channel, %err, "history sample dropped"),
        }
    }

    fn record_one(&self, sample: crate::wire::Sample) {
        if sample.channel != self.spec.channel || self.spec.check_payload(&sample.payload).is_err()
        {
            return;
        }
        // A forged future epoch can't reset the ring: only the session's.
        if sample.epoch != Epoch(self.epoch.load(Ordering::Relaxed)) {
            return;
        }
        let depth = self.depth;
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if sample.epoch != state.epoch {
            // A new epoch never seeds from an older one.
            *state = RingState {
                epoch: sample.epoch,
                first_seq: sample.seq,
                samples: VecDeque::with_capacity(depth.min(1024)),
                gaps: VecDeque::new(),
            };
        }
        let last = state.samples.back().map(|s| s.seq);
        match last {
            Some(last) if sample.seq <= last => return,
            Some(last) if sample.seq > last + 1 => {
                if state.gaps.len() == MAX_GAPS {
                    state.gaps.pop_front();
                }
                state.gaps.push_back(SeqRange {
                    first: last + 1,
                    last: sample.seq - 1,
                });
            }
            _ => {}
        }
        if state.samples.len() >= depth {
            state.samples.pop_front();
        }
        state.samples.push_back(Stored {
            seq: sample.seq,
            capture_ns: sample.capture_ns,
            tick: sample.tick,
            applied: sample.applied,
            payload: sample.payload,
        });
        let oldest = state.samples.front().map_or(0, |s| s.seq);
        while state.gaps.front().is_some_and(|g| g.last < oldest) {
            state.gaps.pop_front();
        }
    }

    /// The pages answering `query`.
    fn pages(&self, query: &HistoryQuery, cap: usize, now_ns: i64) -> Vec<SampleBatch> {
        let Ok(state) = self.state.lock() else {
            return Vec::new();
        };
        let empty = |epoch| SampleBatch {
            channel: self.spec.channel,
            epoch,
            publish_ns: now_ns,
            samples: Vec::new(),
            payload: Bytes::new(),
            truncated: false,
            gaps: Vec::new(),
            request_id: 0,
        };
        if query.epoch != state.epoch {
            // Another epoch's samples are never served: an empty page says
            // which epoch the store is in.
            return vec![empty(state.epoch)];
        }
        let until = query.until_seq.unwrap_or(u64::MAX);
        let limit = query
            .limit
            .map_or(cap, |l| usize::try_from(l).unwrap_or(cap).min(cap));
        let selected: Vec<&Stored> = state
            .samples
            .iter()
            .filter(|s| s.seq > query.after_seq && s.seq <= until)
            .collect();
        let selected = &selected[selected.len().saturating_sub(limit)..];
        let oldest = state.samples.front().map_or(u64::MAX, |s| s.seq);
        // Retention ran out: the range starts before the oldest sample kept,
        // and that sample is not the epoch's first.
        let truncated = query.after_seq.saturating_add(1) < oldest && oldest > state.first_seq;
        let (low, high) = (
            selected.first().map_or(query.after_seq, |s| s.seq),
            selected.last().map_or(until, |s| s.seq),
        );
        let gaps: Vec<SeqRange> = state
            .gaps
            .iter()
            .filter(|g| g.last >= low && g.first <= high)
            .copied()
            .collect();
        let per_page = self
            .spec
            .layout
            .stride()
            .filter(|_| self.spec.batchable())
            .and_then(|stride| MAX_CHUNK_BYTES.checked_div(stride))
            .map_or(1, |n| n.clamp(1, MAX_BATCH));
        let mut pages: Vec<SampleBatch> = selected
            .chunks(per_page)
            .map(|chunk| {
                let mut payload = Vec::with_capacity(chunk.iter().map(|s| s.payload.len()).sum());
                for s in chunk {
                    payload.extend_from_slice(&s.payload);
                }
                SampleBatch {
                    samples: chunk
                        .iter()
                        .map(|s| SampleMeta {
                            seq: s.seq,
                            capture_ns: s.capture_ns,
                            tick: s.tick,
                            applied: s.applied,
                        })
                        .collect(),
                    payload: Bytes::from(payload),
                    ..empty(state.epoch)
                }
            })
            .collect();
        if pages.is_empty() {
            pages.push(empty(state.epoch));
        }
        pages[0].truncated = truncated;
        pages[0].gaps = gaps;
        pages
    }
}

/// History rings by `(robot, stream path)`, shared with the history
/// queryable.
/// A stream by `(robot, key path)`.
pub(super) type StreamKey = (ParticipantId, String);

#[derive(Clone)]
pub(super) struct HistoryStore {
    pub(super) depth: usize,
    pub(super) clock: Option<Arc<dyn Clock>>,
    pub(super) rings: Arc<RwLock<HashMap<StreamKey, Arc<Ring>>>>,
    /// Requesters' keys: a query is sealed under its sender's `k_up`.
    pub(super) peers: SharedPeers,
    /// Who may read what: the authority's, kept in step by the host.
    pub(super) access: Arc<RwLock<Access>>,
}

impl HistoryStore {
    /// Replies to an `h/{robot}/{path}` query, sealed under the `k_up` of a
    /// generation in effect, with its pages sealed under that generation's
    /// `k_down`. Anything else is dropped unanswered.
    pub(super) fn serve(&self, keys: &SessionKeys, query: &Query) {
        let key = query.key_expr().as_str();
        let Some((robot, path)) = keys.history_of(key) else {
            return;
        };
        let ring = self
            .rings
            .read()
            .ok()
            .and_then(|r| r.get(&(robot.clone(), path.to_string())).cloned());
        let Some(ring) = ring else {
            return;
        };
        let mut raw = query
            .payload()
            .map(|p| p.to_bytes().into_owned())
            .unwrap_or_default();
        let Some(peer) = Header::parse(&raw).ok().and_then(|h| {
            let peers = self.peers.read().ok()?;
            let peer = peers.by_key.get(&h.key_id)?;
            peers
                .current
                .get(&peer.participant)
                .filter(|current| Arc::ptr_eq(current, peer))
                .cloned()
        }) else {
            debug!(%key, "a history query not sealed by a member refused");
            return;
        };
        // Read permission, before producing anything.
        let reads = self
            .access
            .read()
            .is_ok_and(|a| a.may_read(&peer.identity.device, &robot, path));
        if !reads {
            let device = &peer.identity.device;
            debug!(%key, %device, "a history query without read permission refused");
            return;
        }
        let request = match lock(&peer.up).open(key, &mut raw) {
            Ok(opened) if opened.header.kind == Kind::HistoryQuery => {
                codec::decode_history_query(opened.body)
            }
            Ok(opened) => {
                debug!(%key, kind = ?opened.header.kind, "not a history query");
                return;
            }
            Err(err) => {
                debug!(%key, %err, "history query dropped");
                return;
            }
        };
        let request = match request {
            Ok(request) => request,
            Err(err) => {
                warn!(%key, %err, "undecodable history query");
                return;
            }
        };
        let now = self.clock.as_ref().map_or(0, |c| c.now_ns());
        for page in ring.pages(&request, self.depth, now) {
            let page = SampleBatch {
                request_id: request.request_id,
                ..page
            };
            let sealed = codec::encode_batch(&page)
                .map_err(|err| err.to_string())
                .and_then(|encoded| {
                    lock(&peer.down)
                        .seal(Kind::HistoryPage, key, &encoded.to_vec())
                        .map_err(|err| err.to_string())
                });
            let sealed = match sealed {
                Ok(sealed) => sealed,
                Err(err) => {
                    warn!(%key, %err, "history page");
                    return;
                }
            };
            if let Err(err) = query.reply(key, sealed).wait() {
                warn!(%key, %err, "history reply failed");
                return;
            }
        }
    }

    fn insert(&self, key: StreamKey, ring: Arc<Ring>) {
        if let Ok(mut rings) = self.rings.write() {
            rings.insert(key, ring);
        }
    }

    fn remove(&self, key: &StreamKey) {
        if let Ok(mut rings) = self.rings.write() {
            rings.remove(key);
        }
    }

    pub(super) fn clear(&self) {
        if let Ok(mut rings) = self.rings.write() {
            rings.clear();
        }
    }
}

/// A recorded stream: its ring and the subscription feeding it.
struct Recorded {
    channel: ChannelId,
    _subscriber: zenoh::pubsub::Subscriber<()>,
}

/// The streams to record, by key.
type Wanted = BTreeMap<StreamKey, StreamSpec>;

/// The history subscriptions, kept on their own thread: declaring or
/// undeclaring one tells every linked peer, and a link that stopped reading
/// can hold that for seconds.
pub(super) struct HistorySync {
    wanted: mpsc::Sender<Wanted>,
    last: Wanted,
    worker: std::thread::JoinHandle<()>,
}

impl HistorySync {
    /// Starts the thread feeding `store`, opening samples with `host_keys`.
    pub(super) fn start(
        transport: Transport,
        keys: SessionKeys,
        store: HistoryStore,
        epoch: Arc<AtomicU64>,
        host_keys: Arc<Mutex<Keyring>>,
    ) -> std::io::Result<Self> {
        let (wanted, updates) = mpsc::channel();
        let subscriptions = Subscriptions {
            transport,
            keys,
            store,
            epoch,
            host_keys,
            recorded: HashMap::new(),
        };
        let worker = std::thread::Builder::new()
            .name("rlmesh-history".into())
            .spawn(move || subscriptions.run(&updates))?;
        Ok(Self {
            wanted,
            last: Wanted::new(),
            worker,
        })
    }

    /// Hands over the streams to record, if they changed.
    fn set(&mut self, wanted: Wanted) {
        if wanted != self.last {
            self.last.clone_from(&wanted);
            let _ = self.wanted.send(wanted);
        }
    }

    /// Undeclares every subscription and stops the thread.
    pub(super) async fn stop(self) {
        drop(self.wanted);
        let worker = self.worker;
        let _ = tokio::task::spawn_blocking(move || worker.join()).await;
    }
}

struct Subscriptions {
    transport: Transport,
    keys: SessionKeys,
    store: HistoryStore,
    epoch: Arc<AtomicU64>,
    host_keys: Arc<Mutex<Keyring>>,
    recorded: HashMap<StreamKey, Recorded>,
}

impl Subscriptions {
    fn run(mut self, updates: &mpsc::Receiver<Wanted>) {
        while let Ok(mut wanted) = updates.recv() {
            // Only the newest list matters.
            while let Ok(newer) = updates.try_recv() {
                wanted = newer;
            }
            self.apply(wanted);
        }
    }

    /// Keeps one history subscription per recordable stream, re-subscribing
    /// when a path's stream id changes.
    fn apply(&mut self, wanted: Wanted) {
        let store = &self.store;
        self.recorded.retain(|key, recorded| {
            let keep = wanted
                .get(key)
                .is_some_and(|s| s.channel == recorded.channel);
            if !keep {
                store.remove(key);
            }
            keep
        });
        for (key, spec) in wanted {
            if self.recorded.contains_key(&key) {
                continue;
            }
            let ring = Arc::new(Ring {
                state: Mutex::new(RingState {
                    epoch: Epoch(0),
                    first_seq: 0,
                    samples: VecDeque::new(),
                    gaps: VecDeque::new(),
                }),
                spec: spec.clone(),
                epoch: self.epoch.clone(),
                depth: self.store.depth,
            });
            let data = self.keys.data(&spec.owner, &spec.path);
            let subscriber = self
                .transport
                .for_priority(spec.priority)
                .declare_subscriber(data.clone())
                .callback({
                    let (ring, keys) = (ring.clone(), self.host_keys.clone());
                    move |sample| ring.record(&keys, &data, sample.payload())
                })
                .wait();
            match subscriber {
                Ok(subscriber) => {
                    debug!(channel = %spec.channel, path = %spec.path, "history ring");
                    self.store.insert(key.clone(), ring);
                    self.recorded.insert(
                        key,
                        Recorded {
                            channel: spec.channel,
                            _subscriber: subscriber,
                        },
                    );
                }
                Err(err) => {
                    warn!(part = %spec.qualified(), %err, "history subscription failed");
                }
            }
        }
    }
}

impl HostActor {
    /// Tells the history thread which streams to record.
    pub(super) fn sync_history(&mut self) {
        let Some(history) = &mut self.history else {
            return;
        };
        let bulk = self.config.history.bulk;
        let wanted: Wanted = self
            .authority
            .parts()
            .flat_map(|part| {
                [
                    part.stream().ok(),
                    part.applied_stream().and_then(Result::ok),
                ]
            })
            .flatten()
            .filter(|s| bulk || s.priority != ChannelPriority::Bulk)
            .map(|s| ((s.owner.clone(), s.path.clone()), s))
            .collect();
        history.set(wanted);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{Encoding, FieldSpec, Layout, Sample};

    fn ring(fields: Vec<FieldSpec>) -> Ring {
        Ring {
            spec: StreamSpec {
                channel: ChannelId(3),
                owner: ParticipantId::new("robot").expect("id"),
                path: "ft".into(),
                layout: Layout::of(&fields).expect("layout"),
                fields,
                encoding: Encoding::Raw,
                priority: ChannelPriority::Realtime,
                event: false,
                max_age: None,
                applied: false,
            },
            state: Mutex::new(RingState {
                epoch: Epoch(0),
                first_seq: 0,
                samples: VecDeque::new(),
                gaps: VecDeque::new(),
            }),
            epoch: Arc::new(AtomicU64::new(2)),
            depth: 6,
        }
    }

    fn put(ring: &Ring, epoch: u64, seq: u64) {
        let sample = Sample {
            channel: ChannelId(3),
            epoch: Epoch(epoch),
            seq,
            capture_ns: 0,
            publish_ns: 0,
            tick: 0,
            payload: Bytes::from(seq.to_le_bytes().to_vec()),
            applied: None,
        };
        ring.record_one(sample);
    }

    fn seqs(pages: &[SampleBatch]) -> Vec<u64> {
        pages
            .iter()
            .flat_map(|p| p.samples.iter().map(|s| s.seq))
            .collect()
    }

    #[test]
    fn the_store_pages_by_epoch_watermark_and_limit_with_gaps_and_truncation() {
        let ring = ring(vec![FieldSpec::new("x", crate::wire::DType::U64, &[])]);
        for seq in [1, 2, 3, 6, 7, 8, 9, 10] {
            put(&ring, 2, seq);
        }
        put(&ring, 1, 11);
        put(&ring, 3, 11);
        let query = |after, until, limit| HistoryQuery {
            epoch: Epoch(2),
            after_seq: after,
            until_seq: until,
            limit,
            request_id: 0,
        };
        // Kept: 3, 6..=10 (depth 6), with 4..=5 known missing.
        let pages = ring.pages(&query(0, None, None), 1024, 0);
        assert_eq!(seqs(&pages), vec![3, 6, 7, 8, 9, 10]);
        assert!(pages[0].truncated, "1 and 2 were evicted");
        assert_eq!(pages[0].gaps, vec![SeqRange { first: 4, last: 5 }]);
        let split = pages[0].split(Some(8)).expect("split");
        assert_eq!(&split[1].payload[..], &6_u64.to_le_bytes());
        // A watermark and a limit: the newest of the range.
        let pages = ring.pages(&query(5, Some(9), Some(2)), 1024, 0);
        assert_eq!(seqs(&pages), vec![8, 9]);
        assert!(!pages[0].truncated && pages[0].gaps.is_empty());
        // Another epoch: an empty page naming the store's epoch.
        let pages = ring.pages(&HistoryQuery::default(), 1024, 0);
        assert!(pages[0].samples.is_empty());
        assert_eq!(pages[0].epoch, Epoch(2));
        // Only the session's epoch is kept: a new one empties the ring, and
        // it never seeds from the old one.
        ring.epoch.store(3, Ordering::Relaxed);
        put(&ring, 3, 1);
        let pages = ring.pages(
            &HistoryQuery {
                epoch: Epoch(3),
                ..HistoryQuery::default()
            },
            1024,
            0,
        );
        assert_eq!(seqs(&pages), vec![1]);
        assert!(!pages[0].truncated);
    }

    #[test]
    fn large_ranges_split_into_pages() {
        let mut ring = ring(vec![FieldSpec::new("x", crate::wire::DType::U64, &[])]);
        ring.depth = 1024;
        ring.epoch.store(1, Ordering::Relaxed);
        for seq in 1..=600 {
            put(&ring, 1, seq);
        }
        let pages = ring.pages(
            &HistoryQuery {
                epoch: Epoch(1),
                ..HistoryQuery::default()
            },
            1024,
            0,
        );
        assert_eq!(
            pages.iter().map(|p| p.samples.len()).collect::<Vec<_>>(),
            vec![MAX_BATCH, MAX_BATCH, 600 - 2 * MAX_BATCH]
        );
        assert_eq!(seqs(&pages), (1..=600).collect::<Vec<_>>());
    }
}
