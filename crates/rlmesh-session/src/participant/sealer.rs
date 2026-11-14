//! The [`Sealer`]: sealing and opening under the keys the participant holds.

#[cfg(test)]
use std::sync::Arc;
use std::sync::Mutex;

use crate::wire::codec;
use crate::wire::{
    ActionChunk, ChannelId, HistoryQuery, LeaseToken, LockstepRequest, LockstepResponse,
    ParticipantId, Sample, SampleBatch,
};
use rlmesh_session_core::auth::{KeyError, StreamWindows};
use rlmesh_session_core::auth::{Link, LinkError};
use rlmesh_session_core::crypto::sealed;
use tokio::sync::mpsc;

use crate::error::SessionError;
use crate::participant::lock;
use crate::participant::shared::{KeyWork, SharedLink};

/// A participant's delivered keys, for its publishers, subscribers, edges
/// and lockstep links: seals and opens under what its link holds.
/// A message under a key it doesn't hold makes the participant recover its
/// keys (`GetKeys`). Cheap to clone.
#[derive(Clone)]
pub struct Sealer {
    pub(super) link: SharedLink,
    pub(super) work: mpsc::UnboundedSender<KeyWork>,
}

impl std::fmt::Debug for Sealer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sealer").finish_non_exhaustive()
    }
}

impl Sealer {
    /// A sealer with no link: everything is refused.
    #[cfg(test)]
    pub(crate) fn detached() -> Self {
        Self {
            link: Arc::default(),
            work: mpsc::unbounded_channel().0,
        }
    }

    fn with<R>(
        &self,
        f: impl FnOnce(&mut Link) -> Result<R, LinkError>,
    ) -> Result<R, SessionError> {
        let link = self.link.get().ok_or(SessionError::Closed)?;
        let result = f(&mut lock(link));
        result.map_err(|err| {
            if matches!(err, LinkError::Key(KeyError::Unknown(_))) {
                let _ = self.work.send(KeyWork::Missing);
            }
            SessionError::Key(err.to_string())
        })
    }

    pub(crate) fn seal_chunk(
        &self,
        keyexpr: &str,
        robot: &ParticipantId,
        chunk: &ActionChunk,
    ) -> Result<Vec<u8>, SessionError> {
        self.with(|link| link.seal_chunk(keyexpr, robot, chunk))
    }

    pub(crate) fn open_chunk(
        &self,
        keyexpr: &str,
        msg: Vec<u8>,
    ) -> Result<ActionChunk, SessionError> {
        self.with(|link| link.open_chunk(keyexpr, msg))
    }

    /// Seals one of this robot's samples (an `_applied` entry when
    /// `applied`) on its data key `keyexpr`.
    ///
    /// # Errors
    /// [`SessionError::Key`] before the stream's key arrived.
    pub fn seal_sample(
        &self,
        keyexpr: &str,
        sample: &Sample,
        applied: bool,
    ) -> Result<Vec<u8>, SessionError> {
        // Only the seq is taken under the link's lock: a camera frame is
        // sealed without holding up the process's chunks.
        let sealing =
            self.with(|link| Ok(link.keyring_mut().begin_sample(sample.channel, applied)?))?;
        let body = codec::encode_sample(sample).to_vec();
        let mut out = Vec::with_capacity(sealed::OVERHEAD + body.len());
        sealing
            .finish_to(keyexpr, &body, &mut out)
            .map_err(|err| SessionError::Key(err.to_string()))?;
        Ok(out)
    }

    /// Opens a sample of `robot`'s stream `channel` received on `keyexpr`
    /// for one subscription (its replay `windows`), verifying it outside
    /// the link's lock: the sample, its encoded body, and whether it is the
    /// first under a robot's newer stream key (its `seq`s restarted).
    pub(crate) fn open_stream(
        &self,
        keyexpr: &str,
        mut msg: Vec<u8>,
        robot: &ParticipantId,
        channel: ChannelId,
        applied: bool,
        windows: &Mutex<StreamWindows>,
    ) -> Result<(Sample, bytes::Bytes, bool), SessionError> {
        let (opening, version) = self.with(|link| {
            Ok(link
                .keyring()
                .begin_open_stream(&msg, robot, channel, applied, &lock(windows))?)
        })?;
        let key = |err: String| SessionError::Key(err);
        opening
            .finish(keyexpr, &mut msg)
            .map_err(|err| key(err.to_string()))?;
        let restarted = lock(windows)
            .commit(opening.header(), version)
            .map_err(|err| key(err.to_string()))?;
        self.with(|link| Ok(link.keyring_mut().opened(opening.header())?))?;
        let end = msg.len() - sealed::TAG_LEN;
        let body = bytes::Bytes::from(msg).slice(sealed::HEADER_LEN..end);
        let sample = codec::decode_sample(body.clone())?;
        if sample.channel != channel {
            return Err(key(format!("a sample of {} on {channel}", sample.channel)));
        }
        Ok((sample, body, restarted))
    }

    pub(crate) fn history_query(
        &self,
        keyexpr: &str,
        query: HistoryQuery,
    ) -> Result<(u64, Vec<u8>), SessionError> {
        self.with(|link| link.history_query(keyexpr, query))
    }

    pub(crate) fn history_page(
        &self,
        history_key: &str,
        data_key: &str,
        msg: Vec<u8>,
        request_id: u64,
        robot: &ParticipantId,
    ) -> Result<SampleBatch, SessionError> {
        self.with(|link| link.history_page(history_key, data_key, msg, request_id, robot))
    }

    /// Seals a lockstep request to `robot` under token `token` on
    /// `keyexpr` (`l/{robot}`).
    ///
    /// # Errors
    /// [`SessionError::Key`] without the lease's keys.
    pub fn seal_lockstep_request(
        &self,
        keyexpr: &str,
        robot: &ParticipantId,
        token: LeaseToken,
        request: &LockstepRequest,
    ) -> Result<Vec<u8>, SessionError> {
        self.with(|link| link.seal_lockstep_request(keyexpr, robot, token, request))
    }

    /// Opens a lockstep request to this robot: it and its key's token.
    ///
    /// # Errors
    /// [`SessionError::Key`]: it is dropped.
    pub fn open_lockstep_request(
        &self,
        keyexpr: &str,
        msg: Vec<u8>,
    ) -> Result<(LockstepRequest, LeaseToken), SessionError> {
        self.with(|link| link.open_lockstep_request(keyexpr, msg))
    }

    /// Seals this robot's response under token `token`'s `k_obs`.
    ///
    /// # Errors
    /// [`SessionError::Key`] without it.
    pub fn seal_lockstep_response(
        &self,
        keyexpr: &str,
        token: LeaseToken,
        response: &LockstepResponse,
    ) -> Result<Vec<u8>, SessionError> {
        self.with(|link| link.seal_lockstep_response(keyexpr, token, response))
    }

    /// Opens `robot`'s response under token `token`'s `k_obs`.
    ///
    /// # Errors
    /// [`SessionError::Key`]: it is dropped.
    pub fn open_lockstep_response(
        &self,
        keyexpr: &str,
        msg: Vec<u8>,
        robot: &ParticipantId,
        token: LeaseToken,
    ) -> Result<LockstepResponse, SessionError> {
        self.with(|link| link.open_lockstep_response(keyexpr, msg, robot, token))
    }
}
