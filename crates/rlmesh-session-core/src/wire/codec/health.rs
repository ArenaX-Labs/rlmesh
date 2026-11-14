//! Heartbeats, their echoes, participant reports and the host's health.

use super::convert::{mode_from_pb, mode_to_pb};
use super::pb;
use crate::prelude::*;

use prost::Message as _;

use crate::edge::Rejection;
use crate::wire::error::WireError;
use crate::wire::health::{DataPath, HostHealth, LinkHealth};
use crate::wire::heartbeat::{
    ChunkTiming, ClockRefusal, DatagramStats, Health, Heartbeat, HeartbeatEcho, LATE_BUCKETS,
    TimingSample,
};
use crate::wire::ids::{ChannelId, Epoch, ParticipantId};
use crate::wire::report::{ParticipantReport, PublisherCounters, RunCounters};

/// Encodes a heartbeat.
pub fn encode_heartbeat(hb: &Heartbeat) -> Vec<u8> {
    pb::Heartbeat {
        participant: hb.participant.to_string(),
        epoch: hb.epoch.0,
        mode: mode_to_pb(hb.mode) as i32,
        seq: hb.seq,
        t1_ns: hb.t1_ns,
        previous: hb.previous.map(timing_to_pb),
        last_seq_seen: hb
            .last_seq_seen
            .iter()
            .map(|(channel, seq)| pb::ChannelSeq {
                channel: channel.0,
                seq: *seq,
            })
            .collect(),
        last_chunk_accepted: hb.last_chunk_accepted,
        last_chunk_applied: hb.last_chunk_applied,
        queue_depth: hb.queue_depth,
        health: health_to_pb(hb.health) as i32,
        rejected: rejected_to_pb(&hb.rejected),
        clock_refused: clock_refused_to_pb(&hb.clock_refused),
        chunk_timing: timings_to_pb(&hb.chunk_timing),
        chunks_accepted: hb.chunks_accepted,
        datagram: hb.datagram.map(datagram_to_pb),
    }
    .encode_to_vec()
}

fn health_to_pb(health: Health) -> pb::Health {
    match health {
        Health::Unknown => pb::Health::Unspecified,
        Health::Ok => pb::Health::Ok,
        Health::Degraded => pb::Health::Degraded,
        Health::Unhealthy => pb::Health::Unhealthy,
    }
}

/// An unknown (newer) health value reads as unknown, not an error.
fn health_from_pb(raw: i32) -> Health {
    match pb::Health::try_from(raw) {
        Ok(pb::Health::Ok) => Health::Ok,
        Ok(pb::Health::Degraded) => Health::Degraded,
        Ok(pb::Health::Unhealthy) => Health::Unhealthy,
        _ => Health::Unknown,
    }
}

/// Rejection counts on the wire: the reason's stats index plus one.
fn clock_refused_to_pb(refused: &[ClockRefusal]) -> Vec<pb::ClockRefusal> {
    refused
        .iter()
        .map(|r| pb::ClockRefusal {
            channel: r.channel.0,
            participant: r.participant.to_string(),
            error_ns: r.error_ns,
            bound_ns: r.bound_ns,
        })
        .collect()
}

fn clock_refused_from_pb(refused: Vec<pb::ClockRefusal>) -> Result<Vec<ClockRefusal>, WireError> {
    refused
        .into_iter()
        .map(|r| {
            Ok(ClockRefusal {
                channel: ChannelId(r.channel),
                participant: r.participant.parse()?,
                error_ns: r.error_ns,
                bound_ns: r.bound_ns,
            })
        })
        .collect()
}

fn rejected_to_pb(rejected: &[(Rejection, u64)]) -> Vec<pb::RejectionCount> {
    rejected
        .iter()
        .filter(|(_, n)| *n > 0)
        .map(|(reason, count)| pb::RejectionCount {
            reason: i32::try_from(reason.index() + 1).unwrap_or(0),
            count: *count,
        })
        .collect()
}

/// An unknown (newer) reason is skipped, not an error: counters are
/// telemetry.
fn rejected_from_pb(rejected: Vec<pb::RejectionCount>) -> Vec<(Rejection, u64)> {
    rejected
        .into_iter()
        .filter_map(|r| {
            let index = usize::try_from(r.reason).ok()?.checked_sub(1)?;
            Some((*Rejection::ALL.get(index)?, r.count))
        })
        .collect()
}

/// Decodes a heartbeat.
pub fn decode_heartbeat(buf: &[u8]) -> Result<Heartbeat, WireError> {
    let hb = pb::Heartbeat::decode(buf)?;
    Ok(Heartbeat {
        participant: ParticipantId::new(hb.participant)?,
        epoch: Epoch(hb.epoch),
        mode: mode_from_pb(hb.mode)?,
        seq: hb.seq,
        t1_ns: hb.t1_ns,
        previous: hb.previous.map(timing_from_pb),
        last_seq_seen: hb
            .last_seq_seen
            .into_iter()
            .map(|c| (ChannelId(c.channel), c.seq))
            .collect(),
        last_chunk_accepted: hb.last_chunk_accepted,
        last_chunk_applied: hb.last_chunk_applied,
        queue_depth: hb.queue_depth,
        health: health_from_pb(hb.health),
        rejected: rejected_from_pb(hb.rejected),
        clock_refused: clock_refused_from_pb(hb.clock_refused)?,
        chunk_timing: timings_from_pb(hb.chunk_timing),
        chunks_accepted: hb.chunks_accepted,
        datagram: hb.datagram.map(datagram_from_pb),
    })
}

fn timings_to_pb(timings: &[ChunkTiming]) -> Vec<pb::ChunkTiming> {
    timings
        .iter()
        .map(|t| pb::ChunkTiming {
            channel: t.channel.0,
            received: t.received,
            accepted: t.accepted,
            lost: t.lost,
            late_ms: t.late_ms.to_vec(),
            late_max_us: t.late_max_us,
            gap_max_us: t.gap_max_us,
        })
        .collect()
}

/// Buckets past [`LATE_BUCKETS`] (a newer sender's) count in the last.
fn timings_from_pb(timings: Vec<pb::ChunkTiming>) -> Vec<ChunkTiming> {
    timings
        .into_iter()
        .map(|t| {
            let mut late_ms = [0_u32; LATE_BUCKETS];
            for (i, n) in t.late_ms.into_iter().enumerate() {
                let slot = &mut late_ms[i.min(LATE_BUCKETS - 1)];
                *slot = slot.saturating_add(n);
            }
            ChunkTiming {
                channel: ChannelId(t.channel),
                received: t.received,
                accepted: t.accepted,
                lost: t.lost,
                late_ms,
                late_max_us: t.late_max_us,
                gap_max_us: t.gap_max_us,
            }
        })
        .collect()
}

fn datagram_to_pb(d: DatagramStats) -> pb::DatagramStats {
    pb::DatagramStats {
        active: d.active,
        sent: d.sent,
        received: d.received,
        lost: d.lost,
        reordered: d.reordered,
        unopened: d.unopened,
        fallbacks: d.fallbacks,
    }
}

fn datagram_from_pb(d: pb::DatagramStats) -> DatagramStats {
    DatagramStats {
        active: d.active,
        sent: d.sent,
        received: d.received,
        lost: d.lost,
        reordered: d.reordered,
        unopened: d.unopened,
        fallbacks: d.fallbacks,
    }
}

fn data_path_to_pb(path: DataPath) -> pb::DataPath {
    match path {
        DataPath::Unknown => pb::DataPath::Unspecified,
        DataPath::Session => pb::DataPath::Session,
        DataPath::Datagram => pb::DataPath::Datagram,
    }
}

fn data_path_from_pb(raw: i32) -> DataPath {
    match pb::DataPath::try_from(raw) {
        Ok(pb::DataPath::Session) => DataPath::Session,
        Ok(pb::DataPath::Datagram) => DataPath::Datagram,
        _ => DataPath::Unknown,
    }
}

/// Encodes a heartbeat echo.
pub fn encode_echo(echo: &HeartbeatEcho) -> Vec<u8> {
    pb::HeartbeatEcho {
        seq: echo.seq,
        t1_ns: echo.t1_ns,
        t2_ns: echo.t2_ns,
        t3_ns: echo.t3_ns,
        epoch: echo.epoch.0,
    }
    .encode_to_vec()
}

/// Decodes a heartbeat echo.
pub fn decode_echo(buf: &[u8]) -> Result<HeartbeatEcho, WireError> {
    let echo = pb::HeartbeatEcho::decode(buf)?;
    Ok(HeartbeatEcho {
        seq: echo.seq,
        t1_ns: echo.t1_ns,
        t2_ns: echo.t2_ns,
        t3_ns: echo.t3_ns,
        epoch: Epoch(echo.epoch),
    })
}

/// Encodes a participant report.
pub fn encode_report(report: &ParticipantReport) -> Vec<u8> {
    report_to_pb(report).encode_to_vec()
}

fn report_to_pb(report: &ParticipantReport) -> pb::ParticipantReport {
    pb::ParticipantReport {
        participant: report.participant.to_string(),
        epoch: report.epoch.0,
        sent_ns: report.sent_ns,
        streams: report
            .streams
            .iter()
            .map(|s| pb::PublisherCounters {
                channel: s.channel.0,
                published: s.published,
                coalesced: s.coalesced,
            })
            .collect(),
        run: report.run.map(|r| pb::RunCounters {
            starts: r.starts,
            completions: r.completions,
            busy_skips: r.busy_skips,
            waiting_for: r.waiting_for,
            stale_input: r.stale_input,
            window_gap: r.window_gap,
            decode: r.decode,
            no_lease: r.no_lease,
            chunks_sent: r.chunks_sent,
            pending_ns: r.pending_ns,
            latency_p50_ns: r.latency_p50_ns,
            latency_p99_ns: r.latency_p99_ns,
            drifting: r.drifting,
        }),
    }
}

/// Decodes a participant report.
pub fn decode_report(buf: &[u8]) -> Result<ParticipantReport, WireError> {
    report_from_pb(pb::ParticipantReport::decode(buf)?)
}

fn report_from_pb(report: pb::ParticipantReport) -> Result<ParticipantReport, WireError> {
    Ok(ParticipantReport {
        participant: ParticipantId::new(report.participant)?,
        epoch: Epoch(report.epoch),
        sent_ns: report.sent_ns,
        streams: report
            .streams
            .into_iter()
            .map(|s| PublisherCounters {
                channel: ChannelId(s.channel),
                published: s.published,
                coalesced: s.coalesced,
            })
            .collect(),
        run: report.run.map(|r| RunCounters {
            starts: r.starts,
            completions: r.completions,
            busy_skips: r.busy_skips,
            waiting_for: r.waiting_for,
            stale_input: r.stale_input,
            window_gap: r.window_gap,
            decode: r.decode,
            no_lease: r.no_lease,
            chunks_sent: r.chunks_sent,
            pending_ns: r.pending_ns,
            latency_p50_ns: r.latency_p50_ns,
            latency_p99_ns: r.latency_p99_ns,
            drifting: r.drifting,
        }),
    })
}

/// Encodes the host's health message.
pub fn encode_health(health: &HostHealth) -> Vec<u8> {
    pb::HostHealth {
        epoch: health.epoch.0,
        host_ns: health.host_ns,
        links: health
            .links
            .iter()
            .map(|l| pb::LinkHealth {
                participant: l
                    .participant
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_default(),
                heartbeats: l.heartbeats,
                rtt_ns: l.rtt_ns,
                jitter_ns: l.jitter_ns,
                heartbeat_gap_max_ns: l.heartbeat_gap_max_ns,
                last_heartbeat_age_ns: l.last_heartbeat_age_ns,
                mapped: l.offset.is_some(),
                offset_ns: l.offset.map_or(0, |o| o.0),
                offset_error_ns: l.offset.map_or(0, |o| o.1),
                last_chunk_accepted: l.last_chunk_accepted,
                last_chunk_applied: l.last_chunk_applied,
                queue_depth: l.queue_depth,
                health: health_to_pb(l.health) as i32,
                rejected: rejected_to_pb(&l.rejected),
                clock_refused: clock_refused_to_pb(&l.clock_refused),
                report: l.report.as_ref().map(report_to_pb),
                last_seq_seen: l
                    .last_seq_seen
                    .iter()
                    .map(|&(channel, seq)| pb::ChannelSeq {
                        channel: channel.0,
                        seq,
                    })
                    .collect(),
                chunk_timing: timings_to_pb(&l.chunk_timing),
                chunks_accepted: l.chunks_accepted,
                data_path: data_path_to_pb(l.data_path) as i32,
                datagram_board: l.datagram_board.map(datagram_to_pb),
                datagram_host: l.datagram_host.map(datagram_to_pb),
            })
            .collect(),
    }
    .encode_to_vec()
}

/// Decodes the host's health message.
pub fn decode_health(buf: &[u8]) -> Result<HostHealth, WireError> {
    let health = pb::HostHealth::decode(buf)?;
    Ok(HostHealth {
        epoch: Epoch(health.epoch),
        host_ns: health.host_ns,
        links: health
            .links
            .into_iter()
            .map(|l| {
                Ok(LinkHealth {
                    participant: Some(ParticipantId::new(l.participant)?),
                    heartbeats: l.heartbeats,
                    rtt_ns: l.rtt_ns,
                    jitter_ns: l.jitter_ns,
                    heartbeat_gap_max_ns: l.heartbeat_gap_max_ns,
                    last_heartbeat_age_ns: l.last_heartbeat_age_ns,
                    offset: l.mapped.then_some((l.offset_ns, l.offset_error_ns)),
                    last_chunk_accepted: l.last_chunk_accepted,
                    last_chunk_applied: l.last_chunk_applied,
                    queue_depth: l.queue_depth,
                    health: health_from_pb(l.health),
                    rejected: rejected_from_pb(l.rejected),
                    clock_refused: clock_refused_from_pb(l.clock_refused)?,
                    report: l.report.map(report_from_pb).transpose()?,
                    last_seq_seen: l
                        .last_seq_seen
                        .into_iter()
                        .map(|c| (ChannelId(c.channel), c.seq))
                        .collect(),
                    chunk_timing: timings_from_pb(l.chunk_timing),
                    chunks_accepted: l.chunks_accepted,
                    data_path: data_path_from_pb(l.data_path),
                    datagram_board: l.datagram_board.map(datagram_from_pb),
                    datagram_host: l.datagram_host.map(datagram_from_pb),
                })
            })
            .collect::<Result<_, WireError>>()?,
    })
}

fn timing_to_pb(s: TimingSample) -> pb::TimingSample {
    pb::TimingSample {
        seq: s.seq,
        t1_ns: s.t1_ns,
        t2_ns: s.t2_ns,
        t3_ns: s.t3_ns,
        t4_ns: s.t4_ns,
    }
}

fn timing_from_pb(s: pb::TimingSample) -> TimingSample {
    TimingSample {
        seq: s.seq,
        t1_ns: s.t1_ns,
        t2_ns: s.t2_ns,
        t3_ns: s.t3_ns,
        t4_ns: s.t4_ns,
    }
}
