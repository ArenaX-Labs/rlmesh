//! A robot's side of the lobby (scene design §3.3, §3.5), sans-IO: what it
//! asks the home daemon while it waits, the one order it takes, and when it
//! leaves a run. An SDK robot process and a board run the same rules; the
//! caller does the networking.
//!
//! ```text
//!  lobby ──LobbyWait──▶ home ──Join──▶ detach, join the run, declare
//!    ▲                                         │
//!    └──── closing, lost, or the join failed ◀─┘   (superseded: stop)
//! ```
//!
//! - **The wait** is a long poll: a [`LobbyWait`] the host keeps pending
//!   until there is an order. The robot re-sends the same request id every
//!   [`WAIT_RETRY`]; the retry joins the pending operation (safety §8.6).
//! - **The order** is a [`Join`] naming this robot ([`Lobby::take`]); any
//!   other order is refused.
//! - **A failed join** (refused handshake or declare, host unreachable) is
//!   reported in the next wait ([`Lobby::failed`]), and the run shows it.
//! - **Leaving a run** ([`leave`]): the run is closing, this generation is
//!   gone from the view, or a newer one took over.

#[allow(unused_imports)]
use crate::prelude::*;

use core::time::Duration;

use crate::wire::{Assignment, CtlError, CtlStatus, Join, LobbyWait, ParticipantId, Reason};

/// The session every lobby participant attaches to on its home daemon.
pub const LOBBY_SESSION: &str = "_lobby";

/// How long a robot waits for one `LobbyWait` reply before it sends the
/// same request again.
pub const WAIT_RETRY: Duration = Duration::from_secs(10);

/// Why a robot leaves the run it joined.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Leave {
    /// The run is ending: detach and return to the lobby.
    Closing(Reason),
    /// Its generation is gone from the view (the host declared it lost).
    Lost,
    /// A newer generation under its name took over: stop, don't return.
    Superseded,
}

/// Whether a robot whose generation is `mine` leaves the run, given the
/// view's `closing` and its `current` generation there (`None`: not
/// present; an older one is a view from before its attach).
pub fn leave(closing: Option<&Reason>, current: Option<u32>, mine: u32) -> Option<Leave> {
    match current {
        Some(current) if current > mine => Some(Leave::Superseded),
        _ if closing.is_some() => closing.cloned().map(Leave::Closing),
        Some(_) => None,
        None => Some(Leave::Lost),
    }
}

/// A robot between runs: what its next wait reports, and the order it
/// takes.
#[derive(Clone, Debug)]
pub struct Lobby {
    robot: ParticipantId,
    failed: Option<(String, CtlError)>,
}

impl Lobby {
    /// Robot `robot`'s lobby side.
    pub fn new(robot: ParticipantId) -> Self {
        Self {
            robot,
            failed: None,
        }
    }

    /// The next wait: it reports the last run that failed for this robot.
    pub fn wait(&self) -> LobbyWait {
        match &self.failed {
            Some((run, error)) => LobbyWait {
                run: run.clone(),
                last: Some(error.clone()),
            },
            None => LobbyWait::default(),
        }
    }

    /// Takes the order a wait answered: a [`Join`] for this robot. The
    /// failure it reported is cleared.
    ///
    /// # Errors
    /// `INVALID_REQUEST` for a join naming another robot, or an order only
    /// a satellite takes.
    pub fn take(&mut self, assignment: Assignment) -> Result<Join, CtlError> {
        self.failed = None;
        match assignment {
            Assignment::Join(join) if join.robot == self.robot => Ok(join),
            Assignment::Join(join) => Err(CtlError::new(
                CtlStatus::InvalidRequest,
                format!("a join for {}, and this is {}", join.robot, self.robot),
            )),
            _ => Err(CtlError::new(
                CtlStatus::InvalidRequest,
                "an order for a satellite daemon, and this is a robot",
            )),
        }
    }

    /// Joining `run` failed with `error`: the next wait reports it.
    pub fn failed(&mut self, run: &str, error: CtlError) {
        self.failed = Some((run.to_string(), error));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{ReasonCode, SessionId};

    fn join(robot: &str) -> Assignment {
        Assignment::Join(Join {
            run: "r-1".into(),
            session: SessionId::named("run"),
            robot: ParticipantId::new(robot).expect("id"),
            addresses: Vec::new(),
            host_key: [0; 32],
            host_id: String::new(),
            tenant: "local".into(),
        })
    }

    #[test]
    fn a_robot_takes_its_own_join_reports_a_failed_one_and_leaves_on_closing_or_loss() {
        let mut lobby = Lobby::new(ParticipantId::new("arm").expect("id"));
        assert_eq!(lobby.wait(), LobbyWait::default());
        let refused = lobby.take(join("eye-l")).expect_err("another robot's");
        assert_eq!(refused.status, CtlStatus::InvalidRequest);
        let taken = lobby.take(join("arm")).expect("its own");
        let error = CtlError::new(CtlStatus::NotAuthorized, "pin");
        lobby.failed(&taken.run, error.clone());
        assert_eq!(lobby.wait().last, Some(error));
        lobby.take(join("arm")).expect("again");
        assert_eq!(lobby.wait(), LobbyWait::default(), "reported once");

        let closing = Reason::new(ReasonCode::SceneEnded);
        for (closing, current, leave_as) in [
            (None, Some(2), None),
            (
                Some(&closing),
                Some(2),
                Some(Leave::Closing(closing.clone())),
            ),
            (None, None, Some(Leave::Lost)),
            (None, Some(1), None),
            (Some(&closing), Some(3), Some(Leave::Superseded)),
        ] {
            assert_eq!(leave(closing, current, 2), leave_as, "{current:?}");
        }
    }
}
