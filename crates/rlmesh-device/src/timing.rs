//! What the session's crypto costs on a board, measured on the board's own
//! clock: a robot's attach handshake, the host-signed authority state it
//! verifies on every change, and the chunks it opens and samples it seals
//! on every control tick. The inputs are the sizes of a small robot's
//! messages ([`STATE_LEN`], [`CHUNK_LEN`], [`SAMPLE_LEN`]).

use alloc::vec::Vec;

use rlmesh_session_core::crypto::handshake::{self, IkInitiator, IkResponder};
use rlmesh_session_core::crypto::{
    Key, Kind, KindSet, Rng, SealReceiver, SealSender, Security, SignedKind, SignedReceiver,
    SignedSender, StaticKeypair,
};

/// An authority state's body: a few parts, leases and roster entries.
pub const STATE_LEN: usize = 1024;
/// A chunk's body: a few parts over a short horizon.
pub const CHUNK_LEN: usize = 256;
/// A sample's body: one part's state or `_applied` entry.
pub const SAMPLE_LEN: usize = 64;

const HANDSHAKES: u32 = 4;
const STATES: u32 = 8;
const MESSAGES: u32 = 100;

const STATE_KEY: &str = "rlmesh/v2/default/eyes/host/state";
const CHUNK_KEY: &str = "rlmesh/v2/default/eyes/robot/eye1/chunk";
const SAMPLE_KEY: &str = "rlmesh/v2/default/eyes/robot/eye1/sample/eye";

/// Mean microseconds per operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Timing {
    /// The robot's handshake msg 1: an ephemeral key and two key exchanges.
    pub handshake_start_us: u32,
    /// Reading msg 2 and splitting the keys: two key exchanges.
    pub handshake_finish_us: u32,
    /// Verifying a signed authority state ([`STATE_LEN`]).
    pub state_verify_us: u32,
    /// Opening a signed chunk ([`CHUNK_LEN`]), the default.
    pub chunk_open_sign_us: u32,
    /// Opening an encrypted chunk, for a part declared `seal`.
    pub chunk_open_seal_us: u32,
    /// Sealing a signed sample ([`SAMPLE_LEN`]), the default.
    pub sample_seal_sign_us: u32,
}

/// Why a run failed: the RNG, or the crypto disagreeing with itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingError {
    /// The caller's RNG failed.
    Rng,
    /// An operation that should succeed failed.
    Crypto,
}

/// Runs every measurement once; `now_us` is the board's microsecond clock.
/// Takes about a second on a classic ESP32, most of it signing the states
/// it then verifies (signing isn't a robot's job, so it isn't timed).
///
/// # Errors
/// [`TimingError`].
pub fn run(now_us: &mut dyn FnMut() -> u64, rng: &mut dyn Rng) -> Result<Timing, TimingError> {
    let mut out = Timing::default();
    handshakes(now_us, rng, &mut out)?;
    states(now_us, rng, &mut out)?;
    out.chunk_open_sign_us = open_chunks(now_us, rng, Security::Sign)?;
    out.chunk_open_seal_us = open_chunks(now_us, rng, Security::Seal)?;
    out.sample_seal_sign_us = seal_samples(now_us, rng)?;
    Ok(out)
}

fn handshakes(
    now_us: &mut dyn FnMut() -> u64,
    rng: &mut dyn Rng,
    out: &mut Timing,
) -> Result<(), TimingError> {
    let robot = StaticKeypair::generate(rng).map_err(|_| TimingError::Rng)?;
    let host = StaticKeypair::generate(rng).map_err(|_| TimingError::Rng)?;
    let prologue =
        handshake::prologue("default", "eyes", 1, "eye1").map_err(|_| TimingError::Crypto)?;
    let (mut start, mut finish) = (0, 0);
    for _ in 0..HANDSHAKES {
        let t = now_us();
        let (initiator, msg1) = IkInitiator::start(rng, &robot, host.public(), &prologue)
            .map_err(|_| TimingError::Crypto)?;
        start += now_us() - t;
        let responder = IkResponder::read_request(rng, &host, &prologue, &msg1)
            .map_err(|_| TimingError::Crypto)?;
        let (msg2, _) = responder
            .accept(b"accept")
            .map_err(|_| TimingError::Crypto)?;
        let t = now_us();
        let (accept, _) = initiator.finish(&msg2).map_err(|_| TimingError::Crypto)?;
        finish += now_us() - t;
        if accept != b"accept" {
            return Err(TimingError::Crypto);
        }
    }
    out.handshake_start_us = mean(start, HANDSHAKES);
    out.handshake_finish_us = mean(finish, HANDSHAKES);
    Ok(())
}

fn states(
    now_us: &mut dyn FnMut() -> u64,
    rng: &mut dyn Rng,
    out: &mut Timing,
) -> Result<(), TimingError> {
    let mut secret = [0u8; 32];
    rng.fill_bytes(&mut secret).map_err(|_| TimingError::Rng)?;
    let mut host = SignedSender::new(&secret, 1);
    let mut robot =
        SignedReceiver::new(&host.verifying_key(), 1).map_err(|_| TimingError::Crypto)?;
    let body = body(rng, STATE_LEN)?;
    let mut total = 0;
    for _ in 0..STATES {
        let msg = host
            .sign(SignedKind::AuthorityState, STATE_KEY, &body)
            .map_err(|_| TimingError::Crypto)?;
        let t = now_us();
        robot
            .verify(STATE_KEY, &msg)
            .map_err(|_| TimingError::Crypto)?;
        total += now_us() - t;
    }
    out.state_verify_us = mean(total, STATES);
    Ok(())
}

fn open_chunks(
    now_us: &mut dyn FnMut() -> u64,
    rng: &mut dyn Rng,
    security: Security,
) -> Result<u32, TimingError> {
    let (mut controller, mut robot) = pair(rng, Kind::ActionChunk, security)?;
    let body = body(rng, CHUNK_LEN)?;
    let mut msg = Vec::new();
    let mut total = 0;
    for _ in 0..MESSAGES {
        msg.clear();
        controller
            .seal_to(Kind::ActionChunk, CHUNK_KEY, &body, &mut msg)
            .map_err(|_| TimingError::Crypto)?;
        let t = now_us();
        let opened = robot
            .open(CHUNK_KEY, &mut msg)
            .map_err(|_| TimingError::Crypto)?;
        total += now_us() - t;
        // The level measured is the level asked for.
        if opened.encrypted() != (security == Security::Seal) {
            return Err(TimingError::Crypto);
        }
    }
    Ok(mean(total, MESSAGES))
}

fn seal_samples(now_us: &mut dyn FnMut() -> u64, rng: &mut dyn Rng) -> Result<u32, TimingError> {
    let (mut robot, mut reader) = pair(rng, Kind::Sample, Security::Sign)?;
    let body = body(rng, SAMPLE_LEN)?;
    let mut msg = Vec::new();
    let mut total = 0;
    for _ in 0..MESSAGES {
        msg.clear();
        let t = now_us();
        robot
            .seal_to(Kind::Sample, SAMPLE_KEY, &body, &mut msg)
            .map_err(|_| TimingError::Crypto)?;
        total += now_us() - t;
        reader
            .open(SAMPLE_KEY, &mut msg)
            .map_err(|_| TimingError::Crypto)?;
    }
    Ok(mean(total, MESSAGES))
}

/// A sender and receiver sharing a fresh key for `kind`.
fn pair(
    rng: &mut dyn Rng,
    kind: Kind,
    security: Security,
) -> Result<(SealSender, SealReceiver), TimingError> {
    let mut bytes = [0u8; 32];
    rng.fill_bytes(&mut bytes).map_err(|_| TimingError::Rng)?;
    let kinds = KindSet::of(&[kind]);
    let sender = SealSender::new(1, Key::from_bytes(bytes), kinds, security);
    let receiver = SealReceiver::new(1, Key::from_bytes(bytes), kinds, security);
    Ok((sender, receiver))
}

fn body(rng: &mut dyn Rng, len: usize) -> Result<Vec<u8>, TimingError> {
    let mut body = alloc::vec![0u8; len];
    rng.fill_bytes(&mut body).map_err(|_| TimingError::Rng)?;
    Ok(body)
}

fn mean(total: u64, n: u32) -> u32 {
    u32::try_from(total / u64::from(n)).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use rlmesh_session_core::crypto::RngError;

    use super::*;

    /// SplitMix64: a fixed stream that never repeats a key.
    struct SplitMix(u64);

    impl Rng for SplitMix {
        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), RngError> {
            for chunk in dest.chunks_mut(8) {
                self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                chunk.copy_from_slice(&z.to_le_bytes()[..chunk.len()]);
            }
            Ok(())
        }
    }

    struct Fails;

    impl Rng for Fails {
        fn fill_bytes(&mut self, _: &mut [u8]) -> Result<(), RngError> {
            Err(RngError)
        }
    }

    #[test]
    fn every_measurement_runs_and_is_the_mean() {
        // Each clock read is 10 µs after the last: every timed span is one.
        let mut clock = 0;
        let mut now = || {
            clock += 10;
            clock
        };
        let timing = run(&mut now, &mut SplitMix(0)).unwrap();
        assert_eq!(
            timing,
            Timing {
                handshake_start_us: 10,
                handshake_finish_us: 10,
                state_verify_us: 10,
                chunk_open_sign_us: 10,
                chunk_open_seal_us: 10,
                sample_seal_sign_us: 10,
            }
        );
    }

    #[test]
    fn a_failed_rng_is_reported() {
        assert_eq!(run(&mut || 0, &mut Fails), Err(TimingError::Rng));
    }
}
