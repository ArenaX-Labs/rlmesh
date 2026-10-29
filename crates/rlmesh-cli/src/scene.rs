//! `rlmesh scene check scene.toml` (scene design §2, §4.1 stage 1): a scene
//! file checked offline against this machine's rlmeshd, its device table
//! and robot pins ([`crate::devices`]), as `RunScene` will check it.
//!
//! On top of the file's own checks ([`rlmesh_session::scene::Scene::parse`]):
//! - a robot's `device` assertion matches its pin, its pinned device is
//!   still in the table, an unpinned robot is one its asserted device may
//!   pin at first embodiment, and a run hosted away from home has every
//!   robot pinned;
//! - a controller's device (or the machine serving it) is `local` or in
//!   the table, and its `max_priority` is within that device's cap, never
//!   silently clamped;
//! - `host` is `home` or a paired machine.
//!
//! Whether the machines are attached to the lobby is known only at run.
//!
//! `rlmesh run`, `rlmesh lobby`, `rlmesh stop` and `rlmesh arm` are clients
//! of the home daemon's lobby ([`crate::lobby`]).

use std::io::Write;

use anyhow::{Context, Result};
use rlmesh_session::auth::LOCAL_DEVICE;
use rlmesh_session::scene::{Problem, Refusal, Role, Scene, Source};
use rlmesh_session::wire::{CtlStatus, PriorityClass};

use std::time::Duration;

use rlmesh_session::lobby::{LobbyClient, find_home};
use rlmesh_session::wire::{MemberState, RunStage, RunState};

use crate::cli::{HomeArgs, RunArgs, RunIdArgs, SceneArgs, SceneCommand};
use crate::devices::{DEVICES, Table};

fn class_name(class: Option<PriorityClass>) -> &'static str {
    match class {
        None => "observer",
        Some(PriorityClass::Scripted) => "scripted",
        Some(PriorityClass::Model) => "model",
        Some(PriorityClass::Operator) => "operator",
        Some(PriorityClass::Safety) => "safety",
    }
}

/// The problems `scene` has against `table`, in file order.
pub fn check(scene: &Scene, table: &Table) -> Vec<Problem> {
    let mut problems = Vec::new();
    let home = scene.home.as_deref().unwrap_or("the home machine");
    let mut refuse = |section: &str, key: &str, what: String, fix: String| {
        problems.push(Problem {
            status: CtlStatus::NotAuthorized,
            line: scene.line(section, key),
            key: if key.is_empty() {
                section.to_string()
            } else {
                format!("{section}.{key}")
            },
            what,
            fix,
        });
    };
    let paired = |device: &str| device == LOCAL_DEVICE || table.devices.contains_key(device);
    let away = scene.host.as_deref().is_some_and(|h| h != "home");
    for member in &scene.members {
        let name = member.name.as_str();
        let section = format!("members.{name}");
        match &member.role {
            Role::Robot { device, .. } => match (table.robots.get(name), device) {
                (Some(pin), Some(asserted)) if &pin.device != asserted => refuse(
                    &section,
                    "device",
                    format!("{name} is pinned to {}, not {asserted}", pin.device),
                    format!(
                        "if it moved, on {home}: rlmesh devices move {name} --to {asserted}; \
                         a scene never moves a pin"
                    ),
                ),
                (Some(pin), _) if !paired(&pin.device) => refuse(
                    &section,
                    "",
                    format!(
                        "{name} is pinned to {}, which is not in {DEVICES} (revoked?)",
                        pin.device
                    ),
                    format!("pair it again, or on {home}: rlmesh devices move {name} --to DEVICE"),
                ),
                // TODO: a tethered robot hosted away from home also needs its
                // board_id pinned; check it once the bridge says which robots
                // it serves (scene step 10).
                (Some(_), _) => {}
                (None, _) if away => refuse(
                    &section,
                    "",
                    format!(
                        "{name} is not pinned, and a run hosted on {} needs every robot pinned: \
                         only the home daemon pins at first embodiment",
                        scene.host.as_deref().unwrap_or_default()
                    ),
                    format!("run {name} once in a scene hosted at home, or host this one at home"),
                ),
                (None, Some(asserted)) if !table.may_pin(name, asserted) => refuse(
                    &section,
                    "device",
                    if paired(asserted) {
                        format!("{name} is not pinned, and {asserted} may not pin it")
                    } else {
                        format!("{asserted} is not in {DEVICES}")
                    },
                    format!(
                        "provision the board as {name}, pair {asserted}, or on {home}: rlmesh \
                         devices allow {asserted} --verbs arm"
                    ),
                ),
                (None, _) => {}
            },
            Role::Controller {
                source,
                max_priority,
                ..
            } => {
                let (device, key) = match source {
                    Source::Device(device) => (device.as_str(), "device"),
                    Source::Serve { on, .. } if on == "home" => (LOCAL_DEVICE, "on"),
                    Source::Serve { on, .. } => (on.as_str(), "on"),
                };
                let Some(entry) = table.devices.get(device) else {
                    if device != LOCAL_DEVICE {
                        refuse(
                            &section,
                            key,
                            format!("{device} is not in {DEVICES} (never paired, or revoked)"),
                            format!("on {device}: rlmesh pair {home}"),
                        );
                    }
                    continue;
                };
                let cap = crate::devices::class(&entry.max_priority).ok().flatten();
                if cap.is_none_or(|cap| *max_priority > cap) {
                    let asked = class_name(Some(*max_priority));
                    refuse(
                        &section,
                        "max_priority",
                        format!(
                            "{device} may lease at {} at most (asked {asked})",
                            class_name(cap)
                        ),
                        format!("on {home}: rlmesh devices allow {device} --max {asked}"),
                    );
                }
            }
        }
    }
    if let Some(host) = scene.host.as_deref().filter(|h| *h != "home" && !paired(h)) {
        refuse(
            "scene",
            "host",
            format!("{host} is not a paired machine"),
            format!("on {host}: rlmesh pair {home}, or host = \"home\""),
        );
    }
    problems.sort_by_key(|p| p.line.unwrap_or(usize::MAX));
    problems
}

/// `rlmesh scene`: 0 for a scene that would pass stage 1, 1 with every
/// problem printed otherwise.
pub fn scene(args: &SceneArgs, stdout: &mut impl Write) -> Result<i32> {
    match &args.command {
        SceneCommand::Check { file } => {
            let text = std::fs::read_to_string(file)
                .with_context(|| format!("reading {}", file.display()))?;
            let dir = crate::daemon::state_dir(args.state_dir.as_deref())?;
            let table = Table::read(&dir)?;
            let refusal = match Scene::parse(&text) {
                Ok(scene) => {
                    let problems = check(&scene, &table);
                    if problems.is_empty() {
                        let robots = scene
                            .members
                            .iter()
                            .filter(|m| matches!(m.role, Role::Robot { .. }))
                            .count();
                        writeln!(
                            stdout,
                            "scene {}: {} members ({robots} robots, {} placed), fingerprint {}",
                            scene.name,
                            scene.members.len(),
                            scene.places().len(),
                            scene.fingerprint()
                        )?;
                        return Ok(0);
                    }
                    Refusal {
                        scene: scene.name,
                        problems,
                    }
                }
                Err(refusal) => refusal,
            };
            write!(stdout, "{refusal}")?;
            Ok(1)
        }
    }
}

/// How long a client waits for the home daemon.
const HOME_TIMEOUT: Duration = Duration::from_secs(10);

async fn connect(home: &HomeArgs) -> Result<LobbyClient> {
    let lobby = find_home(home.home.as_deref()).await?;
    LobbyClient::connect(&lobby, HOME_TIMEOUT)
        .await
        .with_context(|| format!("the lobby at {}", lobby.base))
}

use rlmesh_session::sdk::{member_stage_name, run_stage_name, status_name};

fn ended(stage: RunStage) -> bool {
    matches!(stage, RunStage::Ended | RunStage::Failed)
}

/// Prints what changed from `old` to `new`: the stage, each member's
/// stage and error, and the run's error.
fn print_change(stdout: &mut impl Write, old: Option<&RunState>, new: &RunState) -> Result<()> {
    if old.is_none_or(|o| o.stage != new.stage) {
        writeln!(stdout, "{}: {}", new.run, run_stage_name(new.stage))?;
    }
    for member in &new.members {
        let before: Option<&MemberState> =
            old.and_then(|o| o.members.iter().find(|m| m.name == member.name));
        if before != Some(member) {
            write!(
                stdout,
                "  {}: {}",
                member.name,
                member_stage_name(member.stage)
            )?;
            if let Some(error) = &member.error {
                write!(stdout, " ({})", error.message)?;
            }
            writeln!(stdout)?;
        }
    }
    if let Some(error) = new
        .error
        .as_ref()
        .filter(|_| old.is_none_or(|o| o.error.is_none()))
    {
        writeln!(
            stdout,
            "{}: [{}] {}",
            new.run,
            status_name(error.status),
            error.message
        )?;
    }
    Ok(())
}

/// Follows run `state` until it ends (Ctrl-C stops it on the first, and
/// leaves on the second): 0 when it ended, 1 when it failed.
async fn follow(lobby: &LobbyClient, mut state: RunState, stdout: &mut impl Write) -> Result<i32> {
    let mut stopping = false;
    let mut interrupt = std::pin::pin!(crate::host::shutdown_signal());
    while !ended(state.stage) {
        let next = tokio::select! {
            next = lobby.status(&state.run, state.version, None) => next?,
            signal = &mut interrupt => {
                signal?;
                if stopping {
                    return Ok(130);
                }
                stopping = true;
                interrupt.set(crate::host::shutdown_signal());
                lobby.stop(&state.run).await?
            }
        };
        print_change(stdout, Some(&state), &next)?;
        stdout.flush()?;
        state = next;
    }
    Ok(i32::from(state.stage == RunStage::Failed))
}

/// `rlmesh run`: starts the scene and follows it until it ends.
pub async fn run(args: &RunArgs, stdout: &mut impl Write) -> Result<i32> {
    let manual = match args.arm.as_str() {
        "auto" => false,
        "manual" => true,
        other => anyhow::bail!("--arm {other}: auto or manual"),
    };
    let text = std::fs::read_to_string(&args.file)
        .with_context(|| format!("reading {}", args.file.display()))?;
    let lobby = connect(&args.home).await?;
    let state = match lobby.run_scene(&text, manual).await {
        Ok(state) => state,
        Err(rlmesh_session::SessionError::Rejected(refusal)) => {
            writeln!(
                stdout,
                "[{}] {}",
                status_name(refusal.status),
                refusal.message
            )?;
            return Ok(1);
        }
        Err(err) => return Err(err.into()),
    };
    print_change(stdout, None, &state)?;
    stdout.flush()?;
    let code = follow(&lobby, state, stdout).await;
    let _ = lobby.close().await;
    code
}

/// `rlmesh lobby`: the robots waiting.
pub async fn lobby(args: &HomeArgs, stdout: &mut impl Write) -> Result<()> {
    let lobby = connect(args).await?;
    let waiting = lobby.fetch_waiting().await?;
    if waiting.is_empty() {
        writeln!(stdout, "no robots waiting at {}", lobby.lobby().base)?;
    }
    for robot in waiting {
        writeln!(stdout, "{}  (from {})", robot.id, robot.device)?;
    }
    let _ = lobby.close().await;
    Ok(())
}

/// `rlmesh stop`: stops a run and waits until it ended.
pub async fn stop(args: &RunIdArgs, stdout: &mut impl Write) -> Result<i32> {
    let lobby = connect(&args.home).await?;
    let state = lobby.stop(&args.run).await?;
    print_change(stdout, None, &state)?;
    let ended = lobby
        .wait_until(&args.run, RunStage::Ended, HOME_TIMEOUT)
        .await;
    let _ = lobby.close().await;
    match ended {
        Ok(state) => writeln!(stdout, "{}: {}", state.run, run_stage_name(state.stage))?,
        Err(rlmesh_session::SessionError::Rejected(err)) => {
            writeln!(
                stdout,
                "{}: [{}] {}",
                args.run,
                status_name(err.status),
                err.message
            )?;
            return Ok(1);
        }
        Err(err) => return Err(err.into()),
    }
    Ok(0)
}

/// `rlmesh arm`: arms a run waiting for it.
pub async fn arm(args: &RunIdArgs, stdout: &mut impl Write) -> Result<()> {
    let lobby = connect(&args.home).await?;
    let state = lobby.arm(&args.run).await?;
    writeln!(stdout, "{}: {}", state.run, run_stage_name(state.stage))?;
    let _ = lobby.close().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlmesh_session::auth::address::base32;

    /// The scene design's §2 example, as written there.
    const EYEBALL_RIG: &str = include_str!("../tests/fixtures/eyeball-rig.toml");

    fn devices(skull_board: &str) -> Table {
        let key = |n: u8| base32(&[n; 32]);
        Table::parse(&format!(
            r#"
[devices.laptop-josh]
public_key = "{}"
max_priority = "model"

[devices.arm-pc]
public_key = "{}"
max_priority = "model"

[devices.gpu-desktop]
public_key = "{}"
max_priority = "operator"

[devices.eye-l]
public_key = "{}"
kind = "board"
name = "eye-l"
max_priority = "model"

[robots.eye-l]
device = "eye-l"

[robots.skull]
device = "local"
board_id = "{skull_board}"

[robots.arm]
device = "arm-pc"

[robots.cam-1]
device = "local"
"#,
            key(1),
            key(2),
            key(3),
            key(4),
        ))
        .expect("devices.toml")
    }

    #[test]
    fn scene_check_accepts_the_designs_rig_and_refuses_a_bad_pin_and_an_unknown_device() {
        let scene = Scene::parse(EYEBALL_RIG).unwrap_or_else(|refusal| panic!("{refusal}"));
        assert_eq!(scene.places().len(), 4);
        let mut table = devices("0011223344556677");
        let problems = check(&scene, &table);
        assert!(problems.is_empty(), "the rig passes: {problems:?}");

        // The arm moved to this machine's pin; the tracker's laptop is gone.
        table.robots.get_mut("arm").expect("arm").device = LOCAL_DEVICE.into();
        table.devices.remove("laptop-josh");
        let refusal = Refusal {
            scene: scene.name.clone(),
            problems: check(&scene, &table),
        };
        let text = refusal.to_string();
        assert_eq!(
            refusal
                .problems
                .iter()
                .map(|p| (p.status, p.key.as_str()))
                .collect::<Vec<_>>(),
            [
                (CtlStatus::NotAuthorized, "members.arm.device"),
                (CtlStatus::NotAuthorized, "members.tracker.device"),
            ],
            "{text}"
        );
        assert!(
            text.contains(
                "  [NOT_AUTHORIZED] line 23, members.arm.device: arm is pinned to local, not arm-pc\n    fix: if it moved, on rig-pi.local: rlmesh devices move arm --to arm-pc; a scene never moves a pin"
            ),
            "{text}"
        );
    }
}
