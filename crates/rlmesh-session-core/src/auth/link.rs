//! A participant's link to the host: its sealed verbs and heartbeats, and
//! the replies, echoes and signed broadcasts it opens.

use super::credential::Attached;
use super::window::REQUEST_WINDOW;
use super::{KeyError, Keyring};
use crate::crypto::sealed::{HEADER_LEN, Header, TAG_LEN};
use crate::crypto::{
    Checked, Kind, OpenError, SealError, SealReceiver, SealSender, SignedChecker, SignedError,
    SignedKind, SignedReceiver,
};
use crate::prelude::*;
use crate::wire::codec;
use crate::wire::error::{Unread, WireError};
use crate::wire::heartbeat::{Heartbeat, HeartbeatEcho};
use crate::wire::tick::VirtualTick;
use crate::wire::{
    ActionChunk, AuthoritySnapshot, AuthorityView, Command, CtlError, CtlReply, CtlStatus,
    DeviceView, Epoch, EstopSignal, HandshakeAccept, HistoryQuery, HostHealth, KeysAck, LeaseToken,
    LockstepRequest, LockstepResponse, ParticipantId, Request, Sample, SampleBatch, SampleMeta,
    ViewReply,
};
use bytes::Bytes;

/// Why a [`Link`] refused to send or dropped what it received.
#[derive(Debug, thiserror::Error)]
pub enum LinkError {
    /// Sealing failed (an exhausted counter, a key expression too long).
    #[error(transparent)]
    Seal(#[from] SealError),
    /// A message under `k_down` failed to open: forged, replayed or not
    /// ours.
    #[error(transparent)]
    Open(#[from] OpenError),
    /// A broadcast failed to verify: unsigned, another instance, or not
    /// newer than the last taken.
    #[error(transparent)]
    Signed(#[from] SignedError),
    /// It authenticated but did not decode.
    #[error(transparent)]
    Wire(#[from] WireError),
    /// A delivered key's message: unknown key, out of scope, or none to
    /// send under.
    #[error(transparent)]
    Key(#[from] KeyError),
    /// A history page answering another query.
    #[error("a page for history query {0}, not this one")]
    OtherQuery(u64),
    /// Request `0` is still unanswered, and a new id would put it below the
    /// host's window.
    #[error("request {0} is still unanswered: {REQUEST_WINDOW} requests are outstanding")]
    Outstanding(u64),
}

/// What a reply on `ctl` was ([`Link::reply`], or [`Link::reply_view`]
/// for a robot that takes its device view).
#[derive(Debug, Clone, PartialEq)]
pub enum ReplyOutcome<R = CtlReply> {
    /// The host's sealed answer to the request.
    Answer(Box<R>),
    /// The host's sealed answer to another request (an earlier attempt).
    Other(u64),
    /// An unsealed `UNAUTHENTICATED` refusal: only a hint, since anyone can
    /// answer `ctl`; worth reporting when no answer comes.
    Hint(CtlError),
}

/// Verifies a `host/state` broadcast received on `keyexpr` under `host`.
///
/// # Errors
/// [`LinkError`].
pub fn open_state(
    host: &mut SignedReceiver,
    keyexpr: &str,
    msg: &[u8],
) -> Result<AuthoritySnapshot, LinkError> {
    Ok(codec::decode_state(host.verify(keyexpr, msg)?.body)?)
}

/// Verifies a robot's `host/view/{robot}` broadcast received on `keyexpr`
/// under `host`: signed by the host instance, newer than the last view taken,
/// a device view (kind 22) and the key's robot's. Decoded without the
/// authority state's decoder.
///
/// # Errors
/// [`LinkError`].
pub fn open_view(
    host: &mut SignedReceiver,
    keyexpr: &str,
    msg: &[u8],
) -> Result<DeviceView, LinkError> {
    let verified = host.verify(keyexpr, msg)?;
    if verified.header.kind != SignedKind::DeviceView {
        return Err(SignedError::UnknownKind(verified.header.kind.as_u8()).into());
    }
    decode_view_of(keyexpr, verified.body)
}

/// A verified view's body. Its own function so the decoder's frame isn't
/// underneath the signature check, a board's deepest call.
#[inline(never)]
fn decode_view_of(keyexpr: &str, body: &[u8]) -> Result<DeviceView, LinkError> {
    let view = codec::decode_view(body)?;
    if keyexpr.rsplit('/').next() != Some(view.robot.as_str()) {
        return Err(WireError::malformed("device view", "another robot's").into());
    }
    Ok(view)
}

/// Verifies a `host/tick` broadcast received on `keyexpr` under `host`.
///
/// # Errors
/// [`LinkError`].
pub fn open_tick(
    host: &mut SignedReceiver,
    keyexpr: &str,
    msg: &[u8],
) -> Result<VirtualTick, LinkError> {
    Ok(codec::decode_tick(host.verify(keyexpr, msg)?.body)?)
}

/// Verifies the host's `host/health` broadcast received on `keyexpr` under
/// `host`: signed by the host instance, newer than the last one taken, and a
/// health message (kind 23).
///
/// # Errors
/// [`LinkError`].
pub fn open_health(
    host: &mut SignedReceiver,
    keyexpr: &str,
    msg: &[u8],
) -> Result<HostHealth, LinkError> {
    let verified = host.verify(keyexpr, msg)?;
    if verified.header.kind != SignedKind::HostHealth {
        return Err(SignedError::UnknownKind(verified.header.kind.as_u8()).into());
    }
    Ok(codec::decode_health(verified.body)?)
}

/// The unsealed `UNAUTHENTICATED` refusal `msg` is, if it is one: its
/// status and message only ([`codec::decode_unauthenticated`]), since
/// anyone can answer `ctl`. Its own function so its frame isn't under the
/// real reply's decoder.
#[inline(never)]
fn unauthenticated_refusal(msg: &[u8]) -> Option<CtlError> {
    codec::decode_unauthenticated(msg)
}

/// A verified view's body decoded into `into`, its parts skipped: it must
/// be the robot the key names.
#[inline(never)]
fn decode_view_into_of(keyexpr: &str, body: &[u8], into: &mut DeviceView) -> Result<(), LinkError> {
    codec::decode_view_into(body, into, false)?;
    if keyexpr.rsplit('/').next() != Some(into.robot.as_str()) {
        return Err(WireError::unread("device view", Unread::Malformed("another robot's")).into());
    }
    Ok(())
}

/// A participant generation's link to the host:
/// `k_up` for its verbs and heartbeats, `k_down` for the replies and echoes
/// it opens, its request ids, and the host's key, instance and last
/// publication for the broadcasts it verifies. Sans-IO: key expressions
/// come in, sealed bytes go out.
#[derive(Debug)]
pub struct Link {
    participant: ParticipantId,
    accept: HandshakeAccept,
    up: SealSender,
    down: SealReceiver,
    host: SignedReceiver,
    last_request: u64,
    /// Request ids sent and not yet answered: a caller that timed out
    /// leaves its id here, to retry until answered.
    unanswered: alloc::collections::BTreeSet<u64>,
    last_history: u64,
    keyring: Keyring,
}

impl Link {
    /// The link a completed handshake as `participant` left.
    pub fn new(participant: ParticipantId, attached: Attached) -> Self {
        Self {
            keyring: Keyring::new(Some(participant.clone()), attached.accept.generation),
            participant,
            accept: attached.accept,
            up: attached.up,
            down: attached.down,
            host: attached.host,
            last_request: 0,
            unanswered: alloc::collections::BTreeSet::new(),
            last_history: 0,
        }
    }

    /// The keys the host delivered to this generation.
    pub fn keyring(&self) -> &Keyring {
        &self.keyring
    }

    /// The keys, for two-phase sealing and opening of large samples
    /// ([`Keyring::begin_sample`], [`Keyring::begin_open_sample`]).
    pub fn keyring_mut(&mut self) -> &mut Keyring {
        &mut self.keyring
    }

    /// Opens a `KeyDelivery` on `keyexpr` (`p/{self}/keys`) in place and
    /// applies it at `now_ns` (this participant's clock): the `KeysAck` to
    /// send.
    ///
    /// # Errors
    /// [`LinkError`]: it is dropped.
    pub fn keys(
        &mut self,
        keyexpr: &str,
        msg: &mut [u8],
        now_ns: i64,
    ) -> Result<KeysAck, LinkError> {
        let opened = self.down.open(keyexpr, msg)?;
        if opened.header.kind != Kind::KeyDelivery {
            return Err(OpenError::WrongKind(opened.header.kind).into());
        }
        let delivery = codec::decode_key_delivery(opened.body)?;
        Ok(self.keyring.apply(&delivery, now_ns))
    }

    /// Takes a newer authority snapshot, at `now_ns`: lease keys whose
    /// token ended go, and every key once a newer generation took over.
    pub fn observe(&mut self, snapshot: &impl AuthorityView, now_ns: i64) {
        self.keyring.observe(snapshot, now_ns);
    }

    /// Seals `chunk` for `robot` on `keyexpr` under its lease's `k_cmd`.
    ///
    /// # Errors
    /// [`LinkError::Key`] without that key.
    pub fn seal_chunk(
        &mut self,
        keyexpr: &str,
        robot: &ParticipantId,
        chunk: &ActionChunk,
    ) -> Result<Vec<u8>, LinkError> {
        Ok(self.keyring.seal_chunk(keyexpr, robot, chunk)?)
    }

    /// Opens a chunk to this robot on `keyexpr`: under a `k_cmd` whose part,
    /// token and sender match it.
    ///
    /// # Errors
    /// [`LinkError::Key`]: it is dropped.
    pub fn open_chunk(&mut self, keyexpr: &str, msg: Vec<u8>) -> Result<ActionChunk, LinkError> {
        Ok(self.keyring.open_chunk(keyexpr, msg)?)
    }

    /// Seals one of this robot's samples (an `_applied` entry when
    /// `applied`) on `keyexpr`.
    ///
    /// # Errors
    /// [`LinkError::Key`] before the stream's key arrived.
    pub fn seal_sample(
        &mut self,
        keyexpr: &str,
        sample: &Sample,
        applied: bool,
    ) -> Result<Vec<u8>, LinkError> {
        Ok(self.keyring.seal_sample(keyexpr, sample, applied)?)
    }

    /// Opens a sample of `robot` on `keyexpr`.
    ///
    /// # Errors
    /// [`LinkError::Key`]: it is dropped.
    pub fn open_sample(
        &mut self,
        keyexpr: &str,
        msg: Vec<u8>,
        robot: &ParticipantId,
        applied: bool,
    ) -> Result<Sample, LinkError> {
        Ok(self.keyring.open_sample(keyexpr, msg, robot, applied)?)
    }

    /// Seals a history query on `keyexpr` (`h/{robot}/{path}`) under
    /// `k_up`, with a new request id: the id and the bytes.
    ///
    /// # Errors
    /// [`LinkError::Seal`].
    pub fn history_query(
        &mut self,
        keyexpr: &str,
        query: HistoryQuery,
    ) -> Result<(u64, Vec<u8>), LinkError> {
        let request_id = self.last_history + 1;
        let query = HistoryQuery {
            request_id,
            ..query
        };
        let sealed = self.up.seal(
            Kind::HistoryQuery,
            keyexpr,
            &codec::encode_history_query(&query),
        )?;
        self.last_history = request_id;
        Ok((request_id, sealed))
    }

    /// Opens one reply to history query `request_id`, received on
    /// `history_key`: the host's page under `k_down`, or an event stream's
    /// last sample its robot sealed on `data_key` under the stream's key (a
    /// page of one).
    ///
    /// # Errors
    /// [`LinkError`]: it is dropped.
    pub fn history_page(
        &mut self,
        history_key: &str,
        data_key: &str,
        mut msg: Vec<u8>,
        request_id: u64,
        robot: &ParticipantId,
    ) -> Result<SampleBatch, LinkError> {
        let kind = Header::parse(&msg)?.kind;
        if matches!(kind, Kind::Sample | Kind::Applied) {
            let sample = self.open_sample(data_key, msg, robot, kind == Kind::Applied)?;
            return Ok(SampleBatch {
                channel: sample.channel,
                epoch: sample.epoch,
                publish_ns: sample.publish_ns,
                samples: vec![SampleMeta {
                    seq: sample.seq,
                    capture_ns: sample.capture_ns,
                    tick: sample.tick,
                    applied: sample.applied,
                }],
                payload: sample.payload,
                truncated: false,
                gaps: Vec::new(),
                request_id,
            });
        }
        let opened = self.down.open(history_key, &mut msg)?;
        if opened.header.kind != Kind::HistoryPage {
            return Err(OpenError::WrongKind(opened.header.kind).into());
        }
        let end = msg.len() - TAG_LEN;
        let page = codec::decode_batch(Bytes::from(msg).slice(HEADER_LEN..end))?;
        if page.request_id != request_id {
            return Err(LinkError::OtherQuery(page.request_id));
        }
        Ok(page)
    }

    /// Seals a lockstep request to `robot` under token `token` on
    /// `keyexpr` (`l/{robot}`).
    ///
    /// # Errors
    /// [`LinkError::Key`] without the lease's keys.
    pub fn seal_lockstep_request(
        &mut self,
        keyexpr: &str,
        robot: &ParticipantId,
        token: LeaseToken,
        request: &LockstepRequest,
    ) -> Result<Vec<u8>, LinkError> {
        Ok(self
            .keyring
            .seal_lockstep_request(keyexpr, robot, token, request)?)
    }

    /// Opens a lockstep request to this robot: it and its key's token.
    ///
    /// # Errors
    /// [`LinkError::Key`]: it is dropped.
    pub fn open_lockstep_request(
        &mut self,
        keyexpr: &str,
        msg: Vec<u8>,
    ) -> Result<(LockstepRequest, LeaseToken), LinkError> {
        Ok(self.keyring.open_lockstep_request(keyexpr, msg)?)
    }

    /// Seals this robot's response under token `token`'s `k_obs`.
    ///
    /// # Errors
    /// [`LinkError::Key`] without it.
    pub fn seal_lockstep_response(
        &mut self,
        keyexpr: &str,
        token: LeaseToken,
        response: &LockstepResponse,
    ) -> Result<Vec<u8>, LinkError> {
        Ok(self
            .keyring
            .seal_lockstep_response(keyexpr, token, response)?)
    }

    /// Opens `robot`'s response under token `token`'s `k_obs`.
    ///
    /// # Errors
    /// [`LinkError::Key`]: it is dropped.
    pub fn open_lockstep_response(
        &mut self,
        keyexpr: &str,
        msg: Vec<u8>,
        robot: &ParticipantId,
        token: LeaseToken,
    ) -> Result<LockstepResponse, LinkError> {
        Ok(self
            .keyring
            .open_lockstep_response(keyexpr, msg, robot, token)?)
    }

    /// Seals this participant's e-stop signal on `keyexpr`
    /// (`…/estop/{participant}`) under its generation's `k_estop`:
    /// kind 13, integrity only.
    ///
    /// # Errors
    /// [`LinkError::Key`] without it: this generation isn't attached (any
    /// more), so only the robot's own stop is left.
    pub fn estop_signal(
        &mut self,
        keyexpr: &str,
        signal: &EstopSignal,
    ) -> Result<Vec<u8>, LinkError> {
        Ok(self.keyring.seal_estop(keyexpr, signal)?)
    }

    /// Opens an e-stop signal received on `keyexpr` (`…/estop/{by}`): who
    /// sent it (from the key path, checked against its `k_estop`) and the
    /// signal. An edge latches it with `ESTOP_SIGNAL`.
    ///
    /// # Errors
    /// [`LinkError::Key`]: it is dropped (a non-member, a forgery, an ended
    /// generation, or a key not delivered yet).
    pub fn open_estop_signal(
        &mut self,
        keyexpr: &str,
        msg: Vec<u8>,
    ) -> Result<(ParticipantId, EstopSignal), LinkError> {
        let by = keyexpr
            .rsplit_once("/estop/")
            .and_then(|(_, id)| ParticipantId::new(id).ok())
            .ok_or(KeyError::Scope("not an e-stop key path"))?;
        let signal = self.keyring.open_estop(keyexpr, msg, &by)?;
        Ok((by, signal))
    }

    /// What the host issued: generation, device, keys' ids, instance.
    pub fn accept(&self) -> &HandshakeAccept {
        &self.accept
    }

    /// A verifier of the host's broadcasts with this link's floor, for a
    /// second subscriber (each keeps its own last `seq`).
    pub fn host_verifier(&self) -> SignedReceiver {
        self.host.clone()
    }

    /// Seals `command`, sent from `epoch`, as a new request on `keyexpr`
    /// (the session's `ctl`): its id and the bytes. The id stays
    /// unanswered until [`Self::reply`] takes its answer.
    ///
    /// # Errors
    /// [`LinkError::Seal`]; [`LinkError::Outstanding`] when an unanswered
    /// id would fall below the host's window.
    pub fn request(
        &mut self,
        keyexpr: &str,
        epoch: Option<Epoch>,
        command: &Command,
    ) -> Result<(u64, Vec<u8>), LinkError> {
        let id = self.last_request + 1;
        if let Some(&oldest) = self.unanswered.first()
            && id >= oldest + REQUEST_WINDOW
        {
            return Err(LinkError::Outstanding(oldest));
        }
        let sealed = self.resend(keyexpr, id, epoch, command)?;
        self.last_request = id;
        self.unanswered.insert(id);
        Ok((id, sealed))
    }

    /// Whether request `request_id` is still unanswered.
    pub fn unanswered(&self, request_id: u64) -> bool {
        self.unanswered.contains(&request_id)
    }

    /// Seals `command` again as request `request_id`, under a fresh `seq`:
    /// a retry, which the host answers from its reply cache.
    ///
    /// # Errors
    /// [`LinkError::Seal`].
    pub fn resend(
        &mut self,
        keyexpr: &str,
        request_id: u64,
        epoch: Option<Epoch>,
        command: &Command,
    ) -> Result<Vec<u8>, LinkError> {
        let request = Request {
            from: self.participant.clone(),
            epoch,
            request_id,
            command: command.clone(),
        };
        Ok(self
            .up
            .seal(Kind::CtlRequest, keyexpr, &codec::encode_request(&request))?)
    }

    /// Takes a reply on `keyexpr` to request `request_id`: opened under
    /// `k_down` in place, or, failing that, an unsealed refusal as a hint.
    /// An answer's keys are installed at `now_ns` (this participant's clock).
    ///
    /// # Errors
    /// [`LinkError`]: anything else, which is dropped.
    pub fn reply(
        &mut self,
        keyexpr: &str,
        msg: &mut [u8],
        request_id: u64,
        now_ns: i64,
    ) -> Result<ReplyOutcome, LinkError> {
        let plain = codec::decode_reply(msg);
        match self.down.open(keyexpr, msg) {
            Ok(opened) => {
                let reply = codec::decode_reply(opened.body)?;
                self.unanswered.remove(&reply.request_id);
                Ok(if reply.request_id == request_id {
                    for key in &reply.keys {
                        self.keyring.install(key, now_ns);
                    }
                    self.keyring.observe(&reply.state, now_ns);
                    ReplyOutcome::Answer(Box::new(reply))
                } else {
                    ReplyOutcome::Other(reply.request_id)
                })
            }
            Err(err) => match plain {
                Ok(CtlReply {
                    result: Err(refusal),
                    ..
                }) if refusal.status == CtlStatus::Unauthenticated => {
                    Ok(ReplyOutcome::Hint(refusal))
                }
                _ => Err(err.into()),
            },
        }
    }

    /// [`Self::reply`] for a robot that asked for its device view: the host answers it
    /// with its view and no authority
    /// state, and this path never calls the state's decoder, so a board
    /// that only takes views doesn't link it. The answer's keys are
    /// installed and its view observed at `now_ns`.
    ///
    /// # Errors
    /// [`LinkError`]: anything else, which is dropped.
    pub fn reply_view(
        &mut self,
        keyexpr: &str,
        msg: &mut [u8],
        request_id: u64,
        now_ns: i64,
    ) -> Result<ReplyOutcome<ViewReply>, LinkError> {
        let mut view = DeviceView::empty(self.participant.clone());
        Ok(
            match self.reply_view_into(keyexpr, msg, request_id, now_ns, &mut view, true)? {
                ReplyOutcome::Answer(answer) => {
                    let (mut reply, carried) = *answer;
                    if carried {
                        self.keyring.observe(&view, now_ns);
                    }
                    reply.view = carried.then_some(view);
                    ReplyOutcome::Answer(Box::new(reply))
                }
                ReplyOutcome::Other(id) => ReplyOutcome::Other(id),
                ReplyOutcome::Hint(refusal) => ReplyOutcome::Hint(refusal),
            },
        )
    }

    /// [`Self::reply_view`] with the reply's view decoded into `into`
    /// ([`codec::decode_view_reply_into`], its parts only when `parts`):
    /// the answer is the reply without it, and whether it carried one. The
    /// answer's keys are installed; its view is the caller's to observe
    /// ([`Self::observe`]), on a frame of its own: a board's deepest call
    /// is the keyring's.
    ///
    /// # Errors
    /// [`LinkError`]: anything else, which is dropped; `into` is then
    /// unspecified.
    pub fn reply_view_into(
        &mut self,
        keyexpr: &str,
        msg: &mut [u8],
        request_id: u64,
        now_ns: i64,
        into: &mut DeviceView,
        parts: bool,
    ) -> Result<ReplyOutcome<(ViewReply, bool)>, LinkError> {
        let refused = unauthenticated_refusal(msg);
        match self.down.open(keyexpr, msg) {
            Ok(opened) => {
                let (reply, carried) = codec::decode_view_reply_into(opened.body, into, parts)?;
                self.unanswered.remove(&reply.request_id);
                Ok(if reply.request_id == request_id {
                    for key in &reply.keys {
                        self.keyring.install(key, now_ns);
                    }
                    ReplyOutcome::Answer(Box::new((reply, carried)))
                } else {
                    ReplyOutcome::Other(reply.request_id)
                })
            }
            Err(err) => refused.map_or(Err(err.into()), |r| Ok(ReplyOutcome::Hint(r))),
        }
    }

    /// Seals a heartbeat on `keyexpr` (the participant's `hb`).
    ///
    /// # Errors
    /// [`LinkError::Seal`].
    pub fn heartbeat(
        &mut self,
        keyexpr: &str,
        heartbeat: &Heartbeat,
    ) -> Result<Vec<u8>, LinkError> {
        Ok(self.up.seal(
            Kind::Heartbeat,
            keyexpr,
            &codec::encode_heartbeat(heartbeat),
        )?)
    }

    /// Seals a probe of the host's datagram path on `keyexpr` (the
    /// participant's `udp`): only it and the host hold `k_up`, so the host
    /// sends its datagrams to wherever a probe that opened came from.
    ///
    /// # Errors
    /// [`LinkError::Seal`].
    pub fn datagram_probe(&mut self, keyexpr: &str) -> Result<Vec<u8>, LinkError> {
        Ok(self.up.seal(Kind::DatagramProbe, keyexpr, &[])?)
    }

    /// Seals an observer report on `keyexpr` (the participant's `report`)
    /// for the host, which relays it in its signed health.
    ///
    /// # Errors
    /// [`LinkError::Seal`].
    pub fn report(
        &mut self,
        keyexpr: &str,
        report: &crate::wire::ParticipantReport,
    ) -> Result<Vec<u8>, LinkError> {
        Ok(self.up.seal(
            Kind::ParticipantReport,
            keyexpr,
            &codec::encode_report(report),
        )?)
    }

    /// Opens a heartbeat echo on `keyexpr` (the participant's `hb/echo`) in
    /// place.
    ///
    /// # Errors
    /// [`LinkError`].
    pub fn echo(&mut self, keyexpr: &str, msg: &mut [u8]) -> Result<HeartbeatEcho, LinkError> {
        Ok(codec::decode_echo(self.down.open(keyexpr, msg)?.body)?)
    }

    /// Verifies a `host/state` broadcast on `keyexpr`.
    ///
    /// # Errors
    /// [`LinkError`].
    pub fn state(&mut self, keyexpr: &str, msg: &[u8]) -> Result<AuthoritySnapshot, LinkError> {
        open_state(&mut self.host, keyexpr, msg)
    }

    /// Verifies this robot's `host/view/{robot}` broadcast on `keyexpr`
    /// ([`open_view`]).
    ///
    /// # Errors
    /// [`LinkError`].
    pub fn open_view(&mut self, keyexpr: &str, msg: &[u8]) -> Result<DeviceView, LinkError> {
        open_view(&mut self.host, keyexpr, msg)
    }

    /// Opens this robot's device view sealed under `k_down` on `keyexpr`
    /// (`p/{robot}/view`) in place: what a host listing
    /// [`crate::wire::version::VIEW_DEVICE_SEALED`] sends a robot that asked.
    /// Only the host and this generation hold `k_down`, so it is as
    /// authentic as the signed view for this robot: the key expression (the
    /// session and the robot) is in the tag, the key ends with the
    /// generation, and a replayed seq is refused. The view must name this
    /// robot at this generation; that it is newer than the one held is the
    /// caller's to check, as for a signed view.
    ///
    /// # Errors
    /// [`LinkError`]: it is dropped.
    pub fn open_sealed_view(
        &mut self,
        keyexpr: &str,
        msg: &mut [u8],
    ) -> Result<DeviceView, LinkError> {
        let opened = self.down.open(keyexpr, msg)?;
        if opened.header.kind != Kind::DeviceView {
            return Err(OpenError::WrongKind(opened.header.kind).into());
        }
        let view = codec::decode_view(opened.body)?;
        self.own_view(&view)?;
        Ok(view)
    }

    /// [`Self::open_sealed_view`] into `into`, its parts skipped
    /// ([`codec::decode_view_into`]): nothing is allocated once `into` has
    /// held a view of the same members.
    ///
    /// # Errors
    /// [`LinkError`]: it is dropped, and `into` is unspecified.
    pub fn open_sealed_view_into(
        &mut self,
        keyexpr: &str,
        msg: &mut [u8],
        into: &mut DeviceView,
    ) -> Result<(), LinkError> {
        let opened = self.down.open(keyexpr, msg)?;
        if opened.header.kind != Kind::DeviceView {
            return Err(OpenError::WrongKind(opened.header.kind).into());
        }
        codec::decode_view_into(opened.body, into, false)?;
        self.own_view(into)
    }

    /// A sealed view must name this robot at this generation.
    fn own_view(&self, view: &DeviceView) -> Result<(), LinkError> {
        if view.robot != self.participant || view.generation != self.accept.generation {
            return Err(WireError::unread(
                "device view",
                Unread::Malformed("another robot's or generation's"),
            )
            .into());
        }
        Ok(())
    }

    /// A verifier of this link's broadcasts for another task
    /// ([`SignedChecker`]): a board checks them off its control loop.
    pub fn checker(&self) -> SignedChecker {
        self.host.checker()
    }

    /// [`Self::open_view`] for a view a [`Self::checker`] verified.
    ///
    /// # Errors
    /// [`LinkError`].
    pub fn take_view(&mut self, keyexpr: &str, checked: &Checked) -> Result<DeviceView, LinkError> {
        let verified = self.host.take(keyexpr, checked)?;
        if verified.header.kind != SignedKind::DeviceView {
            return Err(SignedError::UnknownKind(verified.header.kind.as_u8()).into());
        }
        decode_view_of(keyexpr, verified.body)
    }

    /// [`Self::take_view`] into `into`, its parts skipped, as
    /// [`Self::open_sealed_view_into`] does.
    ///
    /// # Errors
    /// [`LinkError`]; `into` is then unspecified.
    pub fn take_view_into(
        &mut self,
        keyexpr: &str,
        checked: &Checked,
        into: &mut DeviceView,
    ) -> Result<(), LinkError> {
        let verified = self.host.take(keyexpr, checked)?;
        if verified.header.kind != SignedKind::DeviceView {
            return Err(SignedError::UnknownKind(verified.header.kind.as_u8()).into());
        }
        decode_view_into_of(keyexpr, verified.body, into)
    }

    /// [`Self::open_view`] into `into`, its parts skipped, as
    /// [`Self::open_sealed_view_into`] does.
    ///
    /// # Errors
    /// [`LinkError`]; `into` is then unspecified.
    pub fn open_view_into(
        &mut self,
        keyexpr: &str,
        msg: &[u8],
        into: &mut DeviceView,
    ) -> Result<(), LinkError> {
        let verified = self.host.verify(keyexpr, msg)?;
        if verified.header.kind != SignedKind::DeviceView {
            return Err(SignedError::UnknownKind(verified.header.kind.as_u8()).into());
        }
        decode_view_into_of(keyexpr, verified.body, into)
    }

    /// Verifies a `host/tick` broadcast on `keyexpr`.
    ///
    /// # Errors
    /// [`LinkError`].
    pub fn tick(&mut self, keyexpr: &str, msg: &[u8]) -> Result<VirtualTick, LinkError> {
        open_tick(&mut self.host, keyexpr, msg)
    }

    /// Verifies a `host/tick` broadcast on `keyexpr`, keeping the host's
    /// sequence floor, without decoding it: for a robot that runs no
    /// lockstep.
    ///
    /// # Errors
    /// [`LinkError`].
    pub fn verify_tick(&mut self, keyexpr: &str, msg: &[u8]) -> Result<(), LinkError> {
        self.host.verify(keyexpr, msg)?;
        Ok(())
    }
}
