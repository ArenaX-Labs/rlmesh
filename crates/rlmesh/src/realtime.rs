//! A real-time policy on either clock (Experimental).
//!
//! A policy is written once, as a [`Predictor`] against its named inputs
//! and actions, and the robot it binds to decides the clock, by what it
//! declared when it attached (`rlmesh_session::wire::RobotClock`). The
//! policy, its options and the predictor are `rlmesh_session`'s, as for a
//! [`Controller`]:
//!
//! - **Wall**: the robot keeps its own time and the policy keeps up. [`bind`]
//!   returns [`BoundPolicy::Wall`], the session's [`Controller`]: starts on
//!   the replan grid, timed chunks, leases and the edge.
//! - **Lockstep**: the robot advances one step per action and waits for the
//!   policy. [`bind`] returns [`BoundPolicy::Lockstep`], a
//!   [`LockstepPolicy`]: one `predict` per chunk, a chunk of H actions is H
//!   steps with their rewards summed, and the binder's conversions and
//!   history windows run as on the wall clock. The policy's action rate sets
//!   the session time each step stands for, so input ages and windows keep
//!   their meaning; nothing waits on real time. No input is ever stale or
//!   missing: each start sees what the robot published, its age in session
//!   time. Arming and the lease are taken care of.
//!
//! [`BoundPolicy::run`] runs either; [`RunSettings`] says how, and refuses
//! at once a setting the robot's clock gives no meaning to: `duration`,
//! `mode: Shadow` and `resume` are the wall clock's, `episodes` and `seed`
//! lockstep's. A lockstep run resets the robot at the start of each episode,
//! each episode's seed derived from `seed` as [`crate::Model::run`] derives
//! it, so one seed draws the same episodes through either.
//!
//! One lockstep robot per run: a policy that binds several robots, any of
//! them in lockstep, is refused.
//!
//! ```no_run
//! use std::time::Duration;
//! use rlmesh::realtime::{self, RunSettings};
//! use rlmesh_session::Predictor;
//! use rlmesh_session::spec::{Policy, RunOptions};
//! use rlmesh_session::wire::{PriorityClass, RobotClock};
//!
//! # async fn demo(session: rlmesh_session::SessionAddress, policy: Policy)
//! #     -> Result<(), Box<dyn std::error::Error>> {
//! let options = RunOptions::named("reach").robot("sim")?;
//! let wait = Duration::from_secs(10);
//! let bound = realtime::bind(&session, "reach", &policy, &options, PriorityClass::Model, wait)
//!     .await?;
//! assert_eq!(bound.clock(), RobotClock::Lockstep);
//! # Ok(())
//! # }
//! # fn run(bound: &realtime::BoundPolicy, predict: &mut impl Predictor) {
//! // Blocks: call it from a plain thread, not from inside the async runtime.
//! let settings = RunSettings { episodes: Some(20), seed: Some(7), ..Default::default() };
//! let outcome = bound.run(predict, settings);
//! # let _ = outcome;
//! # }
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use rlmesh_session::cache::{InputSnapshot, Observation, Snapshot};
use rlmesh_session::controller::{
    Controller, ControllerError, Predictor, Resume, RunConfig, RunError, RunMode, Step,
};
use rlmesh_session::sdk::{self, BindPolicyError};
use rlmesh_session::spec::bind::Binding;
use rlmesh_session::spec::{Policy, RunOptions};
use rlmesh_session::wire::RobotClock as Clock;
use rlmesh_session::wire::{ParticipantId, PriorityClass};
use rlmesh_session::{RunStats, SessionAddress};

use crate::lockstep::view::ViewStep;
use crate::lockstep::{ConnectOptions, GymError, GymView};

/// A policy bound to its robot, on the robot's clock.
pub enum BoundPolicy {
    /// A wall-clock robot (or several), run by the session's controller.
    Wall(Controller),
    /// One lockstep robot.
    Lockstep(LockstepPolicy),
}

/// Attaches as policy `name` in the session at `session`, waits for the
/// robots `options` names, binds `policy` and follows the robot's clock: a
/// wall-clock robot is driven by a [`Controller`], a lockstep robot through
/// its lockstep exchange (the policy then attaches again, as its driver).
/// Refuses what the binder refuses, and a policy binding several robots
/// where any runs in lockstep.
pub async fn bind(
    session: &SessionAddress,
    name: &str,
    policy: &Policy,
    options: &RunOptions,
    priority: PriorityClass,
    timeout: Duration,
) -> Result<BoundPolicy, BindError> {
    match sdk::bind_built_policy(session, name, policy, None, options, priority, timeout).await {
        Ok((controller, _)) => Ok(BoundPolicy::Wall(controller)),
        Err(BindPolicyError::Controller(ControllerError::Lockstep(robot))) => {
            let address = session.participant_invite(robot.as_str());
            let bound = LockstepPolicy::bind(&address, name, policy, options, timeout).await?;
            Ok(BoundPolicy::Lockstep(bound))
        }
        Err(err) => Err(err.into()),
    }
}

/// Why [`bind`] failed.
#[derive(Debug, thiserror::Error)]
pub enum BindError {
    /// Joining or binding failed, or the binder refused.
    #[error(transparent)]
    Policy(#[from] BindPolicyError),
    /// Driving the lockstep robot failed: the binder refused its parts
    /// ([`GymError::Map`]), it did not serve in time, or the session failed.
    #[error(transparent)]
    Lockstep(#[from] GymError),
}

impl BoundPolicy {
    /// The robot's clock.
    pub fn clock(&self) -> Clock {
        match self {
            Self::Wall(_) => Clock::Wall,
            Self::Lockstep(_) => Clock::Lockstep,
        }
    }

    /// The binding in force.
    pub fn binding(&self) -> Arc<Binding> {
        match self {
            Self::Wall(controller) => controller.binding(),
            Self::Lockstep(policy) => policy.binding(),
        }
    }

    /// Asks a run in progress (on another thread) to return.
    pub fn stop(&self) {
        match self {
            Self::Wall(controller) => controller.stop(),
            Self::Lockstep(policy) => policy.stop(),
        }
    }

    /// Runs the policy with `predictor` on the robot's clock (see the
    /// module docs). Blocks: call it from a plain thread, not from inside
    /// the async runtime.
    pub fn run<P: Predictor>(
        &self,
        predictor: &mut P,
        settings: RunSettings,
    ) -> Result<RunOutcome, PolicyRunError<P::Error>> {
        settings
            .check(self.clock())
            .map_err(PolicyRunError::Settings)?;
        match self {
            Self::Wall(controller) => {
                controller
                    .run(predictor, settings.wall())
                    .map_err(PolicyRunError::Wall)?;
                Ok(RunOutcome::Wall(controller.stats()))
            }
            Self::Lockstep(policy) => policy
                .run(predictor, &settings)
                .map(RunOutcome::Lockstep)
                .map_err(PolicyRunError::Lockstep),
        }
    }

    /// Gives the leases back and detaches.
    pub async fn close(self) {
        match self {
            Self::Wall(controller) => {
                let _ = sdk::close_policy(Arc::new(controller)).await;
            }
            Self::Lockstep(policy) => policy.close().await,
        }
    }
}

/// How [`BoundPolicy::run`] runs. Each setting belongs to one clock and is
/// refused on the other ([`RunSettings::check`]); the defaults suit both.
#[derive(Clone, Debug, Default)]
pub struct RunSettings {
    /// Wall clock: stop after this long (`None`: until stopped).
    pub duration: Option<Duration>,
    /// Wall clock: what a start that cannot command does. Lockstep
    /// refuses [`RunMode::Shadow`]: it always commands.
    pub mode: RunMode,
    /// Wall clock: what a part a higher class took does once it is free
    /// (`None`: [`Resume::Manual`]).
    pub resume: Option<Resume>,
    /// Lockstep: episodes to run (`None`: one).
    pub episodes: Option<u64>,
    /// Lockstep: the base seed each episode's reset seed derives from
    /// (`None`: unseeded resets).
    pub seed: Option<i64>,
}

impl RunSettings {
    /// Why these settings cannot run on `clock`: a setting the other clock
    /// owns, or no episodes.
    pub fn check(&self, clock: Clock) -> Result<(), String> {
        let refused = |setting: &str, other: &str| {
            format!(
                "{setting} is for a {other} robot, and this policy's robot runs on the \
                 {} clock",
                match clock {
                    Clock::Wall => "wall",
                    Clock::Lockstep => "lockstep",
                }
            )
        };
        match clock {
            Clock::Wall if self.episodes.is_some() => Err(format!(
                "{}: a wall-clock run goes on until duration= or stop()",
                refused("episodes=", "lockstep")
            )),
            Clock::Wall if self.seed.is_some() => Err(format!(
                "{}: the world seeds itself",
                refused("seed=", "lockstep")
            )),
            Clock::Lockstep if self.duration.is_some() => Err(format!(
                "{}: a lockstep run counts episodes=, not time",
                refused("duration=", "wall-clock")
            )),
            Clock::Lockstep if self.mode == RunMode::Shadow => Err(format!(
                "{}: a lockstep robot waits for every action, so a run always commands",
                refused("mode=shadow", "wall-clock")
            )),
            Clock::Lockstep if self.resume.is_some() => Err(format!(
                "{}: no other class takes a lockstep robot's lease",
                refused("resume=", "wall-clock")
            )),
            Clock::Lockstep if self.episodes == Some(0) => {
                Err("episodes=0: a lockstep run runs at least one episode".to_string())
            }
            _ => Ok(()),
        }
    }

    /// The controller's settings (the wall clock's).
    pub fn wall(&self) -> RunConfig {
        let mut config = RunConfig::default()
            .mode(self.mode)
            .resume(self.resume.unwrap_or(Resume::Manual));
        config.duration = self.duration;
        config
    }
}

/// What [`BoundPolicy::run`] counted.
#[derive(Clone, Debug)]
pub enum RunOutcome {
    /// The controller's counters.
    Wall(RunStats),
    /// The episodes.
    Lockstep(LockstepStats),
}

/// Why [`BoundPolicy::run`] returned early.
#[derive(Debug, thiserror::Error)]
pub enum PolicyRunError<E> {
    /// A setting the robot's clock refuses ([`RunSettings::check`]).
    #[error("{0}")]
    Settings(String),
    /// The wall-clock run failed.
    #[error(transparent)]
    Wall(RunError<E>),
    /// The lockstep run failed.
    #[error(transparent)]
    Lockstep(LockstepRunError<E>),
}

/// Why a lockstep run returned early.
#[derive(Debug, thiserror::Error)]
pub enum LockstepRunError<E> {
    /// A setting lockstep refuses ([`RunSettings::check`]).
    #[error("{0}")]
    Settings(String),
    /// `predict`, the idle hook or the step hook failed.
    #[error("predict failed")]
    Predict(E),
    /// The robot refused a step or reset (an action that does not fit the
    /// binding, a robot that failed), or the session did.
    #[error(transparent)]
    Robot(GymError),
}

/// One episode of a lockstep run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Episode {
    /// Its return: every step's reward, summed.
    pub reward: f64,
    /// Lockstep steps taken.
    pub steps: u64,
    /// It reached a terminal state.
    pub terminated: bool,
    /// It was cut short.
    pub truncated: bool,
}

/// What a lockstep run counted.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LockstepStats {
    /// Calls of `predict`.
    pub starts: u64,
    /// Chunks played (one per start).
    pub chunks_sent: u64,
    /// Every episode, in order; the last may be unfinished when the run was
    /// stopped.
    pub episodes: Vec<Episode>,
}

/// A policy bound to one lockstep robot (see the module docs).
pub struct LockstepPolicy {
    view: GymView,
    robot: ParticipantId,
    binding: Arc<Binding>,
    seed_key: String,
    stop: AtomicBool,
    stats: Mutex<LockstepStats>,
}

impl LockstepPolicy {
    /// Attaches as `name` to the lockstep robot at the participant address
    /// `address` (its session's key included), binds `policy` to it with
    /// `options` and takes its lease, waiting up to `timeout` for the robot
    /// to declare its parts and serve.
    pub async fn bind(
        address: &str,
        name: &str,
        policy: &Policy,
        options: &RunOptions,
        timeout: Duration,
    ) -> Result<Self, GymError> {
        let connect = ConnectOptions {
            id: Some(name.to_string()),
            timeout,
            ..ConnectOptions::default()
        };
        let (view, seed_key) = crate::lockstep::session::connect_policy_view(
            address,
            policy,
            options.clone(),
            connect,
        )
        .await?;
        let binding = view
            .binding()
            .cloned()
            .map(Arc::new)
            .ok_or_else(|| GymError::Map("the view has no binding".into()))?;
        let robot = binding.robot.clone();
        Ok(Self {
            view,
            robot,
            binding,
            seed_key,
            stop: AtomicBool::new(false),
            stats: Mutex::default(),
        })
    }

    /// The robot it drives.
    pub fn robot(&self) -> &ParticipantId {
        &self.robot
    }

    /// The binding in force.
    pub fn binding(&self) -> Arc<Binding> {
        self.binding.clone()
    }

    /// Session time per step, in ns: one action period.
    pub fn tick_ns(&self) -> u64 {
        self.view.tick_ns()
    }

    /// The reset seed of episode `episode` (0-based) of a run seeded
    /// `seed`, as [`crate::Model::run`] derives it for this robot.
    pub fn episode_seed(&self, seed: i64, episode: u64) -> i64 {
        crate::run::runtime::episode_seed(seed, &self.seed_key, episode)
    }

    /// Asks a run in progress (on another thread) to return after its
    /// current step.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// The latest run's counters (so far, while it runs).
    pub fn stats(&self) -> LockstepStats {
        self.stats
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Runs `settings.episodes` episodes (one by default), each from a
    /// reset, until each ends or [`Self::stop`]. Blocks: call it from a
    /// plain thread, not from inside the async runtime.
    pub fn run<P: Predictor>(
        &self,
        predictor: &mut P,
        settings: &RunSettings,
    ) -> Result<LockstepStats, LockstepRunError<P::Error>> {
        settings
            .check(Clock::Lockstep)
            .map_err(LockstepRunError::Settings)?;
        self.stop.store(false, Ordering::Relaxed);
        *self.stats.lock().unwrap_or_else(PoisonError::into_inner) = LockstepStats::default();
        let runtime = crate::run::drive::runtime();
        for episode in 0..settings.episodes.unwrap_or(1) {
            if self.stop.load(Ordering::Relaxed) {
                break;
            }
            let seed = settings
                .seed
                .map(|base| self.episode_seed(base, episode) as u64);
            let mut last = runtime
                .block_on(self.view.reset(seed, ""))
                .map_err(LockstepRunError::Robot)?;
            self.update(|stats| stats.episodes.push(Episode::default()));
            loop {
                predictor.idle().map_err(LockstepRunError::Predict)?;
                if self.stop.load(Ordering::Relaxed) {
                    break;
                }
                let t0_ns = self.view.time_ns();
                let actions = predictor
                    .predict(observation(last), t0_ns)
                    .map_err(LockstepRunError::Predict)?;
                self.update(|stats| stats.starts += 1);
                let step = runtime
                    .block_on(self.view.step_chunks(&actions))
                    .map_err(LockstepRunError::Robot)?;
                let played = Step {
                    shadow: false,
                    snapshot_ns: t0_ns,
                    actions: self.view.played(&step, t0_ns),
                };
                let done = step.terminated || step.truncated;
                self.update(|stats| {
                    stats.chunks_sent += 1;
                    if let Some(current) = stats.episodes.last_mut() {
                        current.reward += step.reward;
                        current.steps += u64::from(step.steps);
                        current.terminated = step.terminated;
                        current.truncated = step.truncated;
                    }
                });
                predictor.step(&played).map_err(LockstepRunError::Predict)?;
                if done {
                    break;
                }
                last = step;
            }
        }
        Ok(self.stats())
    }

    fn update(&self, change: impl FnOnce(&mut LockstepStats)) {
        change(&mut self.stats.lock().unwrap_or_else(PoisonError::into_inner));
    }

    /// Releases the lease and detaches; a later run fails.
    pub async fn close(&self) {
        self.view.close().await;
    }
}

/// A view step as a policy start sees it: every input as the robot last
/// published it, its age in session time, never stale.
fn observation(step: ViewStep) -> Observation {
    let ages: Vec<i64> = step
        .ages_ns
        .iter()
        .filter_map(|(_, age)| age.and_then(|a| i64::try_from(a).ok()))
        .collect();
    let skew_ns = match (ages.iter().max(), ages.iter().min()) {
        (Some(max), Some(min)) => max - min,
        _ => 0,
    };
    let inputs = step
        .ages_ns
        .iter()
        .map(|(entry, age)| InputSnapshot {
            entry: entry.clone(),
            samples: Vec::new(),
            age_ns: age.and_then(|a| i64::try_from(a).ok()),
            stale: false,
            future: false,
            padded: step
                .padded
                .iter()
                .find(|(name, _)| name == entry)
                .map_or(0, |(_, n)| *n),
            blank: step.blank.contains(entry),
        })
        .collect();
    Observation {
        values: step.observation,
        meta: Snapshot {
            inputs,
            skew_ns,
            based_on: Vec::new(),
        },
    }
}
