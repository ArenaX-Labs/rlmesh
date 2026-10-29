//! Machine pairing's exchanges (design §3), sans-IO: rlmeshd's
//! [`Pairings`] (its pending exchanges, their limits and the owner's
//! approval) and the laptop's [`Pairing`]. Both run
//! [`crate::crypto::pairing`] and the messages of [`crate::wire::pair`];
//! the caller moves the bytes on [`PAIR_KEY`] and passes the time.
//!
//! **Retries.** The laptop sends a request again byte for byte until it
//! has its answer, and rlmeshd answers a request equal to an exchange's
//! last one with the same reply before anything else, so a lost packet is
//! only a retry. A packet that fails its Noise or AEAD check is ignored and
//! changes nothing.

use alloc::collections::{BTreeMap, VecDeque};

use super::HostIdentity;
use crate::crypto::pairing::{
    DaemonCommit, LaptopCommit, PairInitiator, PairLink, PairResponder, PairingError, Sas,
};
use crate::crypto::{HandshakeError, Kind, Rng, RngError, StaticKeypair};
use crate::prelude::*;
use crate::wire::codec;
use crate::wire::pair::{PAIR_KEY, PairInfo, PairNonce, PairReply, PairRequest, PairStep};
use crate::wire::{CtlError, CtlStatus};

/// How long an exchange is kept, decided or not.
pub const EXPIRY_NS: i64 = 120_000_000_000;
/// The most exchanges pending at once.
pub const MAX_PENDING: usize = 3;
/// The most XX handshakes completed in a minute.
pub const HANDSHAKES_PER_MINUTE: usize = 10;
/// Wrong codes typed on the robot before every pending exchange is refused.
pub const WRONG_CODES: u32 = 5;
/// The kind a laptop pairs as.
pub const MACHINE: &str = "machine";
/// The kind a browser token pairs as, for the gateway that holds its key
/// (design §5).
pub const BROWSER: &str = "browser";
/// The longest name a laptop may give.
pub const MAX_NAME: usize = 64;

const MINUTE_NS: i64 = 60_000_000_000;

fn refusal(status: CtlStatus, message: &str) -> PairReply {
    PairReply::Error(CtlError::new(status, message))
}

/// The answer to a `PairWait` held past the exchange's expiry.
pub fn expired_reply() -> Vec<u8> {
    codec::encode_pair_reply(&refusal(
        CtlStatus::Expired,
        "the pairing expired before it was approved on the robot; start again",
    ))
}

/// What rlmeshd does with a pairing packet.
#[derive(Debug, PartialEq, Eq)]
pub enum Handled {
    /// Answer it with this `PairReply`.
    Reply(Vec<u8>),
    /// A `PairWait` with no result yet: hold the query, and answer it with
    /// the reply [`Pairings::take_results`] gives for this exchange, or
    /// [`expired_reply`] when [`Pairings::expire`] drops it.
    Wait([u8; 16]),
    /// Not a pairing packet, or one that failed its Noise or AEAD check:
    /// no answer, and nothing changed.
    Ignore,
}

/// An exchange showing a code, waiting for the owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pending {
    /// The exchange.
    pub exchange: [u8; 16],
    /// The name the machine gave.
    pub name: String,
    /// What it is.
    pub kind: String,
    /// How long ago it started.
    pub age_ns: i64,
}

/// The exchange an approval picked: what to enroll, before
/// [`Pairings::decide`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Approved {
    /// The exchange.
    pub exchange: [u8; 16],
    /// The machine's static X25519 key, which XX authenticated.
    pub public: [u8; 32],
    /// The name it gave.
    pub name: String,
    /// What it is.
    pub kind: String,
}

/// Why a typed code approved nothing. Nothing is enrolled.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ApproveError {
    /// No pending exchange shows the code.
    #[error("no pending pairing shows {0:06}")]
    NoMatch(u32),
    /// The last of [`WRONG_CODES`] wrong codes: every pending exchange
    /// was refused.
    #[error("{WRONG_CODES} wrong codes: every pending pairing was refused; start again")]
    Closed,
    /// Two pending exchanges show the code: both were refused.
    #[error("two pending pairings show this code, so both were refused; start again")]
    Ambiguous,
}

struct Shown {
    name: String,
    kind: String,
    sas: Sas,
}

struct Linked {
    link: PairLink,
    commit: DaemonCommit,
    shown: Option<Shown>,
    approving: bool,
    result: Option<Result<PairInfo, CtlError>>,
}

enum Stage {
    Responding(PairResponder),
    Linked(Box<Linked>),
    /// Aborted (a changed nonce): every packet gets this.
    Refused(CtlError),
}

struct Exchange {
    created_ns: i64,
    stage: Stage,
    /// The last request and its reply (`None`: a `PairWait` held).
    last: Option<(Vec<u8>, Option<Vec<u8>>)>,
}

impl Exchange {
    /// Pending: not decided or aborted.
    fn open(&self) -> bool {
        match &self.stage {
            Stage::Responding(_) => true,
            Stage::Linked(linked) => linked.result.is_none(),
            Stage::Refused(_) => false,
        }
    }
}

/// rlmeshd's pairing exchanges (design §3): at most [`MAX_PENDING`]
/// pending, [`HANDSHAKES_PER_MINUTE`] completed XX handshakes a minute,
/// each kept [`EXPIRY_NS`] (decided ones too, so a lost result can be
/// fetched again), and [`WRONG_CODES`] wrong codes close the window.
#[derive(Default)]
pub struct Pairings {
    exchanges: BTreeMap<[u8; 16], Exchange>,
    completed: VecDeque<i64>,
    wrong: u32,
    results: Vec<([u8; 16], Vec<u8>)>,
}

impl core::fmt::Debug for Pairings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Pairings")
            .field("exchanges", &self.exchanges.len())
            .field("wrong", &self.wrong)
            .finish_non_exhaustive()
    }
}

impl Pairings {
    /// No exchanges.
    pub fn new() -> Self {
        Self::default()
    }

    /// Answers one pairing packet `raw` at `now_ns`, as `host`.
    pub fn handle(
        &mut self,
        rng: &mut dyn Rng,
        host: &HostIdentity,
        now_ns: i64,
        raw: &[u8],
    ) -> Handled {
        let Ok(PairRequest { exchange, step }) = codec::decode_pair_request(raw) else {
            return Handled::Ignore;
        };
        if let Some(Exchange {
            last: Some((last, reply)),
            ..
        }) = self.exchanges.get(&exchange)
            && last == raw
        {
            return match reply {
                Some(reply) => Handled::Reply(reply.clone()),
                None => Handled::Wait(exchange),
            };
        }
        let reply = match step {
            PairStep::Xx1(xx1) => self.xx1(rng, &host.noise, now_ns, exchange, &xx1),
            PairStep::Xx3(xx3) => self.xx3(rng, now_ns, exchange, &xx3),
            PairStep::Sealed(msg) => self.sealed(exchange, msg),
        };
        match reply {
            Answer::Keep(reply) => {
                let reply = codec::encode_pair_reply(&reply);
                if let Some(ex) = self.exchanges.get_mut(&exchange) {
                    ex.last = Some((raw.to_vec(), Some(reply.clone())));
                }
                Handled::Reply(reply)
            }
            Answer::Once(reply) => Handled::Reply(codec::encode_pair_reply(&reply)),
            Answer::Hold => {
                if let Some(ex) = self.exchanges.get_mut(&exchange) {
                    ex.last = Some((raw.to_vec(), None));
                }
                Handled::Wait(exchange)
            }
            Answer::Ignore => Handled::Ignore,
        }
    }

    fn xx1(
        &mut self,
        rng: &mut dyn Rng,
        host: &StaticKeypair,
        now_ns: i64,
        exchange: [u8; 16],
        xx1: &[u8],
    ) -> Answer {
        if self.exchanges.contains_key(&exchange) {
            return Answer::Ignore;
        }
        if self.exchanges.values().filter(|e| e.open()).count() >= MAX_PENDING {
            return Answer::Once(refusal(
                CtlStatus::Unsupported,
                "rlmeshd has 3 pairings pending; try again within 2 minutes",
            ));
        }
        let Ok((responder, xx2)) = PairResponder::start(rng, host, &exchange, xx1) else {
            return Answer::Ignore;
        };
        self.exchanges.insert(
            exchange,
            Exchange {
                created_ns: now_ns,
                stage: Stage::Responding(responder),
                last: None,
            },
        );
        Answer::Keep(PairReply::Xx2(xx2))
    }

    fn xx3(&mut self, rng: &mut dyn Rng, now_ns: i64, exchange: [u8; 16], xx3: &[u8]) -> Answer {
        let Some(mut ex) = self.exchanges.remove(&exchange) else {
            return Answer::Once(unknown());
        };
        let answer = match ex.stage {
            Stage::Responding(responder) => {
                while self
                    .completed
                    .front()
                    .is_some_and(|&t| now_ns.saturating_sub(t) >= MINUTE_NS)
                {
                    self.completed.pop_front();
                }
                if self.completed.len() >= HANDSHAKES_PER_MINUTE {
                    ex.stage = Stage::Responding(responder);
                    Answer::Once(refusal(
                        CtlStatus::Unsupported,
                        "too many pairing attempts; try again in a minute",
                    ))
                } else {
                    match responder.finish(xx3) {
                        Err((responder, _)) => {
                            ex.stage = Stage::Responding(responder);
                            Answer::Ignore
                        }
                        Ok(link) => {
                            self.completed.push_back(now_ns);
                            let (stage, answer) = commit(rng, link);
                            ex.stage = stage;
                            answer
                        }
                    }
                }
            }
            stage => {
                ex.stage = stage;
                Answer::Ignore
            }
        };
        self.exchanges.insert(exchange, ex);
        answer
    }

    fn sealed(&mut self, exchange: [u8; 16], mut msg: Vec<u8>) -> Answer {
        let Some(ex) = self.exchanges.get_mut(&exchange) else {
            return Answer::Once(unknown());
        };
        let linked = match &mut ex.stage {
            Stage::Linked(linked) => linked,
            Stage::Refused(err) => return Answer::Once(PairReply::Error(err.clone())),
            Stage::Responding(_) => return Answer::Ignore,
        };
        let Ok(opened) = linked.link.receiver.open(PAIR_KEY, &mut msg) else {
            return Answer::Ignore;
        };
        match opened.header.kind {
            Kind::PairNonce => {
                let Ok(nonce) = codec::decode_pair_nonce(opened.body) else {
                    return Answer::Once(refusal(CtlStatus::InvalidRequest, "a malformed nonce"));
                };
                if ![MACHINE, BROWSER].contains(&nonce.kind.as_str())
                    || nonce.name.is_empty()
                    || nonce.name.len() > MAX_NAME
                {
                    return Answer::Once(refusal(
                        CtlStatus::InvalidRequest,
                        "a machine or browser pairs with a name of 1 to 64 bytes",
                    ));
                }
                match linked.commit.reveal(&nonce.ni) {
                    Ok((nr, sas)) => {
                        linked.shown = Some(Shown {
                            name: nonce.name,
                            kind: nonce.kind,
                            sas,
                        });
                        seal(
                            &mut linked.link,
                            Kind::PairReveal,
                            &codec::encode_pair_reveal(&nr),
                        )
                    }
                    Err(err) => {
                        let err = CtlError::new(CtlStatus::InvalidRequest, err);
                        ex.stage = Stage::Refused(err.clone());
                        Answer::Once(PairReply::Error(err))
                    }
                }
            }
            Kind::PairWait if linked.shown.is_none() => Answer::Once(refusal(
                CtlStatus::InvalidRequest,
                "a wait before the nonce",
            )),
            Kind::PairWait => match &linked.result {
                Some(result) => {
                    let body = codec::encode_pair_result(result);
                    seal(&mut linked.link, Kind::PairWait, &body)
                }
                None => Answer::Hold,
            },
            _ => Answer::Ignore,
        }
    }

    /// Drops every exchange older than [`EXPIRY_NS`] at `now_ns`: the
    /// dropped ones, whose held `PairWait`s get [`expired_reply`].
    pub fn expire(&mut self, now_ns: i64) -> Vec<[u8; 16]> {
        let old: Vec<[u8; 16]> = self
            .exchanges
            .iter()
            .filter(|(_, ex)| now_ns.saturating_sub(ex.created_ns) >= EXPIRY_NS)
            .map(|(id, _)| *id)
            .collect();
        for id in &old {
            self.exchanges.remove(id);
        }
        old
    }

    /// The exchanges showing a code and not yet decided, oldest first.
    pub fn pending(&self, now_ns: i64) -> Vec<Pending> {
        let mut out: Vec<Pending> = self
            .exchanges
            .iter()
            .filter_map(|(id, ex)| match &ex.stage {
                Stage::Linked(linked) if linked.result.is_none() && !linked.approving => {
                    linked.shown.as_ref().map(|shown| Pending {
                        exchange: *id,
                        name: shown.name.clone(),
                        kind: shown.kind.clone(),
                        age_ns: now_ns.saturating_sub(ex.created_ns),
                    })
                }
                _ => None,
            })
            .collect();
        out.sort_by_key(|p| core::cmp::Reverse(p.age_ns));
        out
    }

    /// The owner typed `code` on the robot: the one pending exchange
    /// showing it, to enroll and then [`decide`](Self::decide).
    ///
    /// # Errors
    /// [`ApproveError`]: nothing approved. Two exchanges showing the code
    /// are both refused, and so is every pending exchange at the
    /// [`WRONG_CODES`]th wrong code.
    pub fn approve(&mut self, code: u32) -> Result<Approved, ApproveError> {
        let showing: Vec<[u8; 16]> = self
            .exchanges
            .iter()
            .filter(|(_, ex)| match &ex.stage {
                Stage::Linked(linked) => {
                    linked.result.is_none()
                        && !linked.approving
                        && linked.shown.as_ref().is_some_and(|s| s.sas.value() == code)
                }
                _ => false,
            })
            .map(|(id, _)| *id)
            .collect();
        match showing.as_slice() {
            [] => {
                self.wrong += 1;
                if self.wrong < WRONG_CODES {
                    return Err(ApproveError::NoMatch(code));
                }
                self.wrong = 0;
                let open: Vec<[u8; 16]> = self
                    .exchanges
                    .iter()
                    .filter(|(_, ex)| ex.open())
                    .map(|(id, _)| *id)
                    .collect();
                for id in open {
                    self.refuse(id, "too many wrong codes were typed on the robot");
                }
                Err(ApproveError::Closed)
            }
            [one] => {
                self.wrong = 0;
                let Some(Exchange {
                    stage: Stage::Linked(linked),
                    ..
                }) = self.exchanges.get_mut(one)
                else {
                    unreachable!("filtered above");
                };
                linked.approving = true;
                let shown = linked.shown.as_ref().unwrap_or_else(|| unreachable!());
                Ok(Approved {
                    exchange: *one,
                    public: *linked.link.remote_static(),
                    name: shown.name.clone(),
                    kind: shown.kind.clone(),
                })
            }
            many => {
                for &id in many {
                    self.refuse(id, "another machine showed the same code");
                }
                Err(ApproveError::Ambiguous)
            }
        }
    }

    fn refuse(&mut self, exchange: [u8; 16], why: &str) {
        match self.exchanges.get(&exchange).map(|ex| &ex.stage) {
            Some(Stage::Linked(_)) => {
                self.decide(exchange, Err(CtlError::new(CtlStatus::NotAuthorized, why)));
            }
            Some(_) => {
                self.exchanges.remove(&exchange);
            }
            None => {}
        }
    }

    /// The outcome of an approved exchange (the device it was enrolled
    /// as, or why it wasn't): its `PairResult`, now for a held `PairWait`
    /// ([`take_results`](Self::take_results)) and for every later one.
    /// Whether the exchange was still there to take it.
    pub fn decide(&mut self, exchange: [u8; 16], result: Result<PairInfo, CtlError>) -> bool {
        let Some(Exchange {
            stage: Stage::Linked(linked),
            last,
            ..
        }) = self.exchanges.get_mut(&exchange)
        else {
            return false;
        };
        linked.approving = false;
        let body = codec::encode_pair_result(&result);
        linked.result = Some(result);
        if let Some((_, reply @ None)) = last
            && let Answer::Keep(sealed) = seal(&mut linked.link, Kind::PairWait, &body)
        {
            let sealed = codec::encode_pair_reply(&sealed);
            *reply = Some(sealed.clone());
            self.results.push((exchange, sealed));
        }
        true
    }

    /// The replies for held `PairWait`s that a decision answered.
    pub fn take_results(&mut self) -> Vec<([u8; 16], Vec<u8>)> {
        core::mem::take(&mut self.results)
    }
}

/// What a packet gets.
enum Answer {
    /// This reply, and again for the same request.
    Keep(PairReply),
    /// This reply, this time.
    Once(PairReply),
    /// Held until a decision.
    Hold,
    /// Nothing.
    Ignore,
}

fn unknown() -> PairReply {
    refusal(
        CtlStatus::Expired,
        "rlmeshd has no such pairing (it expired or restarted); start again",
    )
}

fn seal(link: &mut PairLink, kind: Kind, body: &[u8]) -> Answer {
    match link.sender.seal(kind, PAIR_KEY, body) {
        Ok(sealed) => Answer::Keep(PairReply::Sealed(sealed)),
        Err(_) => Answer::Ignore,
    }
}

/// XX done: draw `Nr` and send its commitment.
fn commit(rng: &mut dyn Rng, mut link: PairLink) -> (Stage, Answer) {
    let Ok(commit) = DaemonCommit::new(rng, link.handshake_hash()) else {
        let err = CtlError::new(CtlStatus::Unsupported, "rlmeshd's RNG failed");
        return (
            Stage::Refused(err.clone()),
            Answer::Once(PairReply::Error(err)),
        );
    };
    let body = commit
        .commitment()
        .map(codec::encode_pair_commit)
        .unwrap_or_default();
    let answer = seal(&mut link, Kind::PairCommit, &body);
    (
        Stage::Linked(Box::new(Linked {
            link,
            commit,
            shown: None,
            approving: false,
            result: None,
        })),
        answer,
    )
}

/// What a reply moved the laptop to.
#[derive(Debug, PartialEq, Eq)]
pub enum Progress {
    /// Send [`Pairing::request`] next.
    Next,
    /// Show this code, then send [`Pairing::request`] (a `PairWait`), again
    /// each time it times out, until the result.
    Show(Sas),
    /// Approved: the details to store, whose host key is the one XX
    /// authenticated.
    Paired(PairInfo),
    /// Not an answer to the current request (a forgery, a duplicate): send
    /// it again.
    Ignored,
}

/// Why the laptop's pairing failed.
#[derive(Debug, thiserror::Error)]
pub enum PairError {
    /// rlmeshd refused, or the owner did.
    #[error("{}", .0.message)]
    Refused(CtlError),
    /// The revealed nonce doesn't match its commitment: someone is in the
    /// middle.
    #[error(transparent)]
    Pairing(#[from] PairingError),
    /// XX failed.
    #[error(transparent)]
    Handshake(#[from] HandshakeError),
    /// The RNG failed.
    #[error(transparent)]
    Rng(#[from] RngError),
    /// The result names another host key than XX authenticated.
    #[error("rlmeshd's result names another host key than the one it paired with")]
    HostKey,
}

enum LaptopStage {
    Xx2(PairInitiator),
    Commit(Box<PairLink>),
    Reveal(Box<PairLink>, LaptopCommit),
    Wait(Box<PairLink>),
    Done,
}

/// The laptop's side of one exchange (`rlmesh pair`).
pub struct Pairing {
    exchange: [u8; 16],
    name: String,
    kind: &'static str,
    request: Vec<u8>,
    stage: LaptopStage,
}

impl core::fmt::Debug for Pairing {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Pairing")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl Pairing {
    /// Starts exchange `exchange` (16 random bytes) with `device`, the key
    /// this machine (or the browser token it holds) will attach with, under
    /// the name `name`, as `kind` ([`MACHINE`] or [`BROWSER`]).
    ///
    /// # Errors
    /// [`PairError::Handshake`].
    pub fn start(
        rng: &mut dyn Rng,
        device: &StaticKeypair,
        exchange: [u8; 16],
        name: &str,
        kind: &'static str,
    ) -> Result<Self, PairError> {
        let (initiator, xx1) = PairInitiator::start(rng, device, &exchange)?;
        Ok(Self {
            exchange,
            name: name.to_string(),
            kind,
            request: codec::encode_pair_request(&PairRequest {
                exchange,
                step: PairStep::Xx1(xx1),
            }),
            stage: LaptopStage::Xx2(initiator),
        })
    }

    /// The request to send (again, byte for byte, until it is answered).
    pub fn request(&self) -> &[u8] {
        &self.request
    }

    fn sealed(&mut self, msg: Vec<u8>) {
        self.request = codec::encode_pair_request(&PairRequest {
            exchange: self.exchange,
            step: PairStep::Sealed(msg),
        });
    }

    /// Takes rlmeshd's reply to [`request`](Self::request).
    ///
    /// # Errors
    /// [`PairError`]: the pairing is over.
    pub fn on_reply(&mut self, rng: &mut dyn Rng, reply: &[u8]) -> Result<Progress, PairError> {
        let reply = match codec::decode_pair_reply(reply) {
            Ok(PairReply::Error(err)) => return Err(PairError::Refused(err)),
            Ok(reply) => reply,
            Err(_) => return Ok(Progress::Ignored),
        };
        let stage = core::mem::replace(&mut self.stage, LaptopStage::Done);
        let (stage, progress) = self.advance(rng, stage, reply)?;
        self.stage = stage;
        Ok(progress)
    }

    fn advance(
        &mut self,
        rng: &mut dyn Rng,
        stage: LaptopStage,
        reply: PairReply,
    ) -> Result<(LaptopStage, Progress), PairError> {
        let open = |link: &mut PairLink, kind: Kind, mut msg: Vec<u8>| {
            let opened = link.receiver.open(PAIR_KEY, &mut msg).ok()?;
            (opened.header.kind == kind).then(|| opened.body.to_vec())
        };
        Ok(match (stage, reply) {
            (LaptopStage::Xx2(initiator), PairReply::Xx2(xx2)) => match initiator.finish(&xx2) {
                Err((initiator, _)) => (LaptopStage::Xx2(initiator), Progress::Ignored),
                Ok((link, xx3)) => {
                    self.request = codec::encode_pair_request(&PairRequest {
                        exchange: self.exchange,
                        step: PairStep::Xx3(xx3),
                    });
                    (LaptopStage::Commit(Box::new(link)), Progress::Next)
                }
            },
            (LaptopStage::Commit(mut link), PairReply::Sealed(msg)) => {
                let Some(c) = open(&mut link, Kind::PairCommit, msg)
                    .and_then(|body| codec::decode_pair_commit(&body).ok())
                else {
                    return Ok((LaptopStage::Commit(link), Progress::Ignored));
                };
                let laptop = LaptopCommit::new(rng, link.handshake_hash(), &c)?;
                let nonce = codec::encode_pair_nonce(&PairNonce {
                    ni: *laptop.nonce(),
                    name: self.name.clone(),
                    kind: self.kind.to_string(),
                });
                let sealed = link
                    .sender
                    .seal(Kind::PairNonce, PAIR_KEY, &nonce)
                    .map_err(|_| HandshakeError::Internal)?;
                self.sealed(sealed);
                (LaptopStage::Reveal(link, laptop), Progress::Next)
            }
            (LaptopStage::Reveal(mut link, laptop), PairReply::Sealed(msg)) => {
                let Some(nr) = open(&mut link, Kind::PairReveal, msg)
                    .and_then(|body| codec::decode_pair_reveal(&body).ok())
                else {
                    return Ok((LaptopStage::Reveal(link, laptop), Progress::Ignored));
                };
                let sas = laptop.check_reveal(&nr)?;
                let wait = link
                    .sender
                    .seal(Kind::PairWait, PAIR_KEY, &codec::encode_pair_wait())
                    .map_err(|_| HandshakeError::Internal)?;
                self.sealed(wait);
                (LaptopStage::Wait(link), Progress::Show(sas))
            }
            (LaptopStage::Wait(mut link), PairReply::Sealed(msg)) => {
                let Some(result) = open(&mut link, Kind::PairWait, msg)
                    .and_then(|body| codec::decode_pair_result(&body).ok())
                else {
                    return Ok((LaptopStage::Wait(link), Progress::Ignored));
                };
                let info = result.map_err(PairError::Refused)?;
                if &info.host_key != link.remote_static() {
                    return Err(PairError::HostKey);
                }
                (LaptopStage::Done, Progress::Paired(info))
            }
            (stage, _) => (stage, Progress::Ignored),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::test_rng::TestRng;

    fn info(public: [u8; 32]) -> PairInfo {
        PairInfo {
            device: "laptop".into(),
            host_key: public,
            host_id: "rlm-000000".into(),
            addresses: vec!["10.0.0.5:7447".into()],
        }
    }

    /// Runs `laptop` against `daemon` until it shows its code.
    fn to_code(
        daemon: &mut Pairings,
        host: &HostIdentity,
        laptop: &mut Pairing,
        rng: &mut TestRng,
    ) -> Sas {
        loop {
            let Handled::Reply(reply) = daemon.handle(rng, host, 0, laptop.request()) else {
                panic!("an answer");
            };
            // A lost reply is a retry: the same request gets the same reply.
            assert_eq!(
                daemon.handle(rng, host, 0, laptop.request()),
                Handled::Reply(reply.clone())
            );
            match laptop.on_reply(rng, &reply).expect("progress") {
                Progress::Next => {}
                Progress::Show(sas) => return sas,
                other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn only_the_shown_code_approves_and_the_held_wait_gets_the_result() {
        let mut rng = TestRng::new(1);
        let host = HostIdentity::from_secrets(&mut [6; 32], &mut [7; 32], &mut [8; 32]);
        let device = StaticKeypair::from_private(&mut [5; 32]);
        let mut daemon = Pairings::new();
        let mut laptop =
            Pairing::start(&mut rng, &device, [7; 16], "laptop-josh", MACHINE).unwrap();
        let sas = to_code(&mut daemon, &host, &mut laptop, &mut rng);
        assert_eq!(daemon.pending(5)[0].name, "laptop-josh");

        let exchange = [7; 16];
        assert_eq!(
            daemon.handle(&mut rng, &host, 0, laptop.request()),
            Handled::Wait(exchange)
        );
        let wrong = (sas.value() + 1) % 1_000_000;
        assert_eq!(daemon.approve(wrong), Err(ApproveError::NoMatch(wrong)));
        assert!(daemon.take_results().is_empty(), "nothing decided");
        assert_eq!(daemon.pending(5).len(), 1, "still pending");

        let approved = daemon.approve(sas.value()).expect("the shown code");
        assert_eq!(approved.public, *device.public());
        assert!(daemon.decide(exchange, Ok(info(*host.static_public()))));
        let [(id, reply)] = daemon.take_results().try_into().unwrap();
        assert_eq!(id, exchange);
        assert_eq!(
            laptop.on_reply(&mut rng, &reply).unwrap(),
            Progress::Paired(info(*host.static_public()))
        );
        // The same wait again gets the same result, until the expiry.
        assert_eq!(
            daemon.handle(&mut rng, &host, 0, laptop.request()),
            Handled::Reply(reply)
        );
        assert_eq!(daemon.expire(EXPIRY_NS), vec![exchange]);
    }

    #[test]
    fn wrong_codes_close_the_window() {
        let mut rng = TestRng::new(2);
        let host = HostIdentity::from_secrets(&mut [6; 32], &mut [7; 32], &mut [8; 32]);
        let device = StaticKeypair::from_private(&mut [5; 32]);
        let mut daemon = Pairings::new();
        let mut laptop = Pairing::start(&mut rng, &device, [1; 16], "a", MACHINE).unwrap();
        let sas = to_code(&mut daemon, &host, &mut laptop, &mut rng);
        let wrong = (sas.value() + 1) % 1_000_000;
        for _ in 1..WRONG_CODES {
            assert_eq!(daemon.approve(wrong), Err(ApproveError::NoMatch(wrong)));
        }
        assert_eq!(daemon.approve(wrong), Err(ApproveError::Closed));
        assert!(daemon.pending(0).is_empty());
        let Handled::Reply(reply) = daemon.handle(&mut rng, &host, 0, laptop.request()) else {
            panic!("the refusal");
        };
        assert!(matches!(
            laptop.on_reply(&mut rng, &reply),
            Err(PairError::Refused(_))
        ));
        assert_eq!(
            daemon.approve(sas.value()),
            Err(ApproveError::NoMatch(sas.value()))
        );
    }
}
