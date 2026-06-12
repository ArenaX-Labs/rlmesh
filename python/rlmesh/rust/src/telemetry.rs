use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, fmt, registry};

pub fn init_tracing(process_role: &'static str) {
    static INIT: OnceLock<()> = OnceLock::new();

    INIT.get_or_init(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let fmt_layer = fmt::layer().with_writer(std::io::stderr).with_target(false);
        let _ = registry().with(filter).with(fmt_layer).try_init();
    });

    if profiling_enabled() {
        tracing::info!(process_role, "RLMesh profiling enabled");
    }
}

pub fn profiling_enabled() -> bool {
    env_flag("RLMESH_PROFILE")
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            matches!(
                value.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
}

#[derive(Debug, Clone, Default)]
struct PhaseSummary {
    count: u64,
    total: Duration,
    min: Option<Duration>,
    max: Duration,
    total_bytes: u64,
}

impl PhaseSummary {
    fn record(&mut self, duration: Duration, bytes: u64) {
        self.count += 1;
        self.total += duration;
        self.min = Some(match self.min {
            Some(min) => min.min(duration),
            None => duration,
        });
        self.max = self.max.max(duration);
        self.total_bytes += bytes;
    }

    fn avg_ms(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total.as_secs_f64() * 1000.0 / self.count as f64
        }
    }

    fn total_ms(&self) -> f64 {
        self.total.as_secs_f64() * 1000.0
    }

    fn min_ms(&self) -> f64 {
        self.min
            .map(|duration| duration.as_secs_f64() * 1000.0)
            .unwrap_or(0.0)
    }

    fn max_ms(&self) -> f64 {
        self.max.as_secs_f64() * 1000.0
    }

    fn avg_bytes(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_bytes as f64 / self.count as f64
        }
    }
}

pub struct ProfileCollector {
    role: &'static str,
    enabled: bool,
    phases: Mutex<BTreeMap<&'static str, PhaseSummary>>,
    logged: AtomicBool,
}

impl ProfileCollector {
    pub fn new(role: &'static str) -> Arc<Self> {
        Arc::new(Self {
            role,
            enabled: profiling_enabled(),
            phases: Mutex::new(BTreeMap::new()),
            logged: AtomicBool::new(false),
        })
    }

    pub fn record(&self, phase: &'static str, duration: Duration, bytes: usize) {
        if !self.enabled {
            return;
        }

        if let Ok(mut phases) = self.phases.lock() {
            phases
                .entry(phase)
                .or_default()
                .record(duration, bytes as u64);
        }
    }

    /// Whether profiling is active.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn start(self: &Arc<Self>, phase: &'static str) -> PhaseGuard {
        if !self.enabled {
            return PhaseGuard {
                collector: Arc::clone(self),
                phase,
                start: None,
                bytes: 0,
                recorded: true,
            };
        }
        PhaseGuard {
            collector: Arc::clone(self),
            phase,
            start: Some(Instant::now()),
            bytes: 0,
            recorded: false,
        }
    }

    pub fn log_summary_once(&self) {
        if !self.enabled || self.logged.swap(true, Ordering::SeqCst) {
            return;
        }

        let Ok(phases) = self.phases.lock() else {
            return;
        };

        for (phase, summary) in phases.iter() {
            tracing::info!(
                process_role = self.role,
                phase = *phase,
                count = summary.count,
                total_ms = summary.total_ms(),
                avg_ms = summary.avg_ms(),
                min_ms = summary.min_ms(),
                max_ms = summary.max_ms(),
                total_bytes = summary.total_bytes,
                avg_bytes = summary.avg_bytes(),
                "profiling summary"
            );
        }
    }
}

pub struct PhaseGuard {
    collector: Arc<ProfileCollector>,
    phase: &'static str,
    start: Option<Instant>,
    bytes: usize,
    recorded: bool,
}

impl PhaseGuard {
    pub fn finish(mut self, bytes: usize) -> Duration {
        let Some(start) = self.start else {
            // Profiling disabled: nothing to record.
            self.recorded = true;
            return Duration::ZERO;
        };
        self.bytes = bytes;
        self.recorded = true;
        let duration = start.elapsed();
        self.collector.record(self.phase, duration, self.bytes);
        duration
    }
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        if self.recorded {
            return;
        }

        if let Some(start) = self.start {
            self.collector
                .record(self.phase, start.elapsed(), self.bytes);
        }
    }
}
