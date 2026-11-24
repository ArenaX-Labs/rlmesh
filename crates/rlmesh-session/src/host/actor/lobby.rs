//! Lobby verbs (feature `fleet`): handing them to the lobby and answering them, now or later.

use crate::wire::{Command, ParticipantId};
use rlmesh_session_core::crypto::Kind;
use tracing::warn;

use crate::authority::Verb;
use crate::host::HostActor;
use crate::host::actor::LobbyDone;
use crate::host::actor::Outcome;
use crate::host::lock;
use crate::host::peer::Peer;
use crate::lobby::{LobbyAnswer, LobbyReply, LobbyResult, LobbyService};

impl HostActor {
    pub(super) fn answer_lobby(
        &self,
        to: &ParticipantId,
        result: LobbyResult,
        request_id: u64,
    ) -> Vec<u8> {
        let (result, lobby) = match result {
            Ok(LobbyReply::Assignment(assignment)) => (Ok(()), (Some(assignment), None)),
            Ok(LobbyReply::Run(run)) => (Ok(()), (None, Some(run))),
            Err(err) => (Err(err), (None, None)),
        };
        self.answer_with(
            to,
            result,
            (Vec::new(), Vec::new()),
            request_id,
            lobby,
            false,
        )
    }

    pub(super) fn lobby_verb(
        &self,
        peer: &Peer,
        lobby: &dyn LobbyService,
        command: Command,
    ) -> Outcome {
        if matches!(
            command,
            Command::RunScene { .. } | Command::StopScene { .. } | Command::ArmScene { .. }
        ) && let Err(err) = self
            .authority
            .config()
            .access
            .check_verb(&peer.identity.device, Verb::Arm)
        {
            return Outcome::refused(err);
        }
        match lobby.handle(&peer.participant, &peer.identity.device, command) {
            LobbyAnswer::Now(result) => Outcome::Lobby(result),
            LobbyAnswer::Later(answer) => Outcome::Later(answer),
        }
    }

    pub(super) async fn finish_lobby(&mut self, (key, result): LobbyDone) {
        let Some(waiting) = self.lobby.remove(&key) else {
            return;
        };
        let request_id = key.2;
        let encoded = self.answer_lobby(&waiting.peer.participant, result, request_id);
        lock(&waiting.peer.requests).complete(request_id, encoded.clone());
        if let Some(query) = waiting.query {
            let sealed = lock(&waiting.peer.down).seal(Kind::CtlReply, &self.keys.ctl(), &encoded);
            match sealed {
                Ok(sealed) => Self::reply(query, sealed),
                Err(err) => warn!(%err, "lobby reply not sealed"),
            }
        }
    }
}
