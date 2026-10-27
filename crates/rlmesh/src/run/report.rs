//! Evaluation results over saved `rlmesh.result.v1` bundles: per-task,
//! per-suite and overall success with Wilson 95% intervals, episode length
//! and wall time, and the run telemetry merged across workloads.
//!
//! The definitions match the managed platform. Success is the share of
//! counted episodes (`includedInMetrics`) whose env reported a success; an
//! episode without the signal counts as a failure and is reported as
//! `unknown`. Telemetry rows merge as the platform rollup merges windows: per
//! `(op, component, metric)`, counts add and `avg` and each percentile are
//! count-weighted means (`weighted` in `rollup.go`). A bundle carries each
//! run's session rows, nearest-rank percentiles over the whole run, so no
//! warm-up window is discarded (the rollup's session-row fallback).

use std::path::{Path, PathBuf};

use crate::recorder::EpisodeRecord;
use crate::{EpisodeResult, Error, Result, RunResult, TelemetryRow};

/// The `config` key naming a workload's suite (a benchmark's task family,
/// such as `libero_spatial`).
pub const SUITE_CONFIG_KEY: &str = "suite";

/// Two-sided 95% normal quantile.
const Z95: f64 = 1.959_963_984_540_054;

/// One workload read from a bundle.
#[derive(Clone, Debug, PartialEq)]
pub struct BundleWorkload {
    /// The model label.
    pub model: String,
    /// The env label.
    pub env: String,
    /// The task label (the env label when the recorder was given none).
    pub task: String,
    /// The suite, from the workload's `config["suite"]`.
    pub suite: Option<String>,
    /// The workload's episodes, in bundle order.
    pub episodes: Vec<EpisodeRecord>,
    /// The telemetry rows of the runs recorded into the workload.
    pub telemetry: Vec<TelemetryRow>,
}

/// Every workload the bundles at `paths` hold, in order. A path is a bundle
/// (a folder holding `result.json`, or a zip), or a folder of bundles, read
/// in name order.
pub fn read_bundles<P: AsRef<Path>>(
    paths: impl IntoIterator<Item = P>,
) -> Result<Vec<BundleWorkload>> {
    let mut workloads = Vec::new();
    for path in paths {
        let path = path.as_ref();
        if path.is_dir() && !path.join("result.json").is_file() {
            let mut bundles = std::fs::read_dir(path)
                .map_err(|err| Error::Os(format!("{}: {err}", path.display())))?
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|p| {
                    p.join("result.json").is_file()
                        || (p.is_file()
                            && p.extension().is_some_and(|s| s.eq_ignore_ascii_case("zip")))
                })
                .collect::<Vec<PathBuf>>();
            if bundles.is_empty() {
                return Err(Error::invalid_argument(format!(
                    "{} holds no rlmesh.result.v1 bundle",
                    path.display()
                )));
            }
            bundles.sort();
            for bundle in bundles {
                workloads.extend(read_bundle(&bundle)?);
            }
        } else {
            workloads.extend(read_bundle(path)?);
        }
    }
    Ok(workloads)
}

fn read_bundle(bundle: &Path) -> Result<Vec<BundleWorkload>> {
    use std::io::Read;
    let os = |err: std::io::Error| Error::Os(format!("{}: {err}", bundle.display()));
    let invalid = |err: &dyn std::fmt::Display| {
        Error::invalid_argument(format!("{}: {err}", bundle.display()))
    };
    let text = if bundle.is_dir() {
        std::fs::read_to_string(bundle.join("result.json")).map_err(os)?
    } else {
        let file = std::fs::File::open(bundle).map_err(os)?;
        let mut archive = zip::ZipArchive::new(file).map_err(|err| invalid(&err))?;
        let mut entry = archive
            .by_name("result.json")
            .map_err(|err| invalid(&err))?;
        let mut text = String::new();
        entry.read_to_string(&mut text).map_err(os)?;
        text
    };
    let document: serde_json::Value = serde_json::from_str(&text).map_err(|err| invalid(&err))?;
    let malformed = |key: &str| invalid(&format!("an episode has no valid {key:?}"));
    let text_of = |value: &serde_json::Value| value.as_str().unwrap_or_default().to_string();
    let list = |value: &serde_json::Value| value.as_array().cloned().unwrap_or_default();
    list(&document["workloads"])
        .iter()
        .map(|w| {
            let env = text_of(&w["env"]);
            let episodes = list(&w["episodes"])
                .iter()
                .map(|e| {
                    Ok(EpisodeRecord {
                        index: e["index"].as_i64().ok_or_else(|| malformed("index"))?,
                        seed: e["seed"].as_i64(),
                        steps: e["steps"].as_i64().ok_or_else(|| malformed("steps"))?,
                        reward: e["reward"].as_f64().ok_or_else(|| malformed("reward"))?,
                        terminated: e["terminated"]
                            .as_bool()
                            .ok_or_else(|| malformed("terminated"))?,
                        truncated: e["truncated"]
                            .as_bool()
                            .ok_or_else(|| malformed("truncated"))?,
                        success: e["success"].as_bool(),
                        duration_s: e["durationS"].as_f64().unwrap_or(0.0),
                        included_in_metrics: e["includedInMetrics"].as_bool().unwrap_or(true),
                        media: Vec::new(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let telemetry = list(&w["telemetry"])
                .iter()
                .map(|row| TelemetryRow {
                    op: text_of(&row["op"]),
                    component: text_of(&row["component"]),
                    metric: text_of(&row["metric"]),
                    unit: text_of(&row["unit"]),
                    count: row["count"].as_i64().unwrap_or(0),
                    avg: row["avg"].as_f64().unwrap_or(0.0),
                    p50: row["p50"].as_f64().unwrap_or(0.0),
                    p95: row["p95"].as_f64().unwrap_or(0.0),
                    p99: row["p99"].as_f64().unwrap_or(0.0),
                })
                .collect();
            Ok(BundleWorkload {
                model: text_of(&w["model"]),
                task: w["task"]
                    .as_str()
                    .map_or_else(|| env.clone(), str::to_string),
                env,
                suite: w["config"][SUITE_CONFIG_KEY].as_str().map(str::to_string),
                episodes,
                telemetry,
            })
        })
        .collect()
}

/// The workloads as one [`RunResult`]: their counted episodes in order,
/// indexed afresh from 0, and their telemetry merged. An episode left out of
/// the metrics (`includedInMetrics: false`) is dropped, so the result's
/// aggregates read as the workload's and the report's. Workloads of several
/// models are refused: their outcomes are not one model's result.
pub fn merge(workloads: &[BundleWorkload]) -> Result<RunResult> {
    let models = distinct(workloads.iter().map(|w| w.model.as_str()));
    if models.len() > 1 {
        return Err(Error::invalid_argument(format!(
            "the bundles hold workloads of {} models ({}); load each model's bundles \
             separately, or report them with EvalReport",
            models.len(),
            models.join(", ")
        )));
    }
    let episodes = workloads
        .iter()
        .flat_map(|w| &w.episodes)
        .filter(|e| e.included_in_metrics)
        .enumerate()
        .map(|(index, e)| EpisodeResult {
            index: index as i64,
            seed: e.seed,
            steps: e.steps,
            reward: e.reward,
            terminated: e.terminated,
            truncated: e.truncated,
            success: e.success,
            duration_s: e.duration_s,
            ..Default::default()
        })
        .collect();
    Ok(RunResult {
        episodes,
        telemetry: merge_telemetry(workloads.iter().flat_map(|w| &w.telemetry)),
        ..Default::default()
    })
}

/// Rows of the same `(op, component, metric, unit)` merged as the platform
/// rollup merges windows: counts add, `avg` and percentiles are
/// count-weighted means. First-seen order.
pub fn merge_telemetry<'a>(rows: impl IntoIterator<Item = &'a TelemetryRow>) -> Vec<TelemetryRow> {
    let mut merged: Vec<TelemetryRow> = Vec::new();
    for row in rows.into_iter().filter(|row| row.count > 0) {
        let same = |m: &&mut TelemetryRow| {
            (&m.op, &m.component, &m.metric, &m.unit)
                == (&row.op, &row.component, &row.metric, &row.unit)
        };
        match merged.iter_mut().find(same) {
            Some(m) => {
                let total = (m.count + row.count) as f64;
                let mix = |a: f64, b: f64| (a * m.count as f64 + b * row.count as f64) / total;
                (m.avg, m.p50, m.p95, m.p99) = (
                    mix(m.avg, row.avg),
                    mix(m.p50, row.p50),
                    mix(m.p95, row.p95),
                    mix(m.p99, row.p99),
                );
                m.count += row.count;
            }
            None => merged.push(row.clone()),
        }
    }
    merged
}

/// The Wilson score interval for `successes` of `trials` at 95%, or `None`
/// with no trials.
pub fn wilson95(successes: u64, trials: u64) -> Option<(f64, f64)> {
    if trials == 0 {
        return None;
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = Z95 * Z95;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = Z95 * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denominator;
    Some(((center - half).max(0.0), (center + half).min(1.0)))
}

/// Outcomes of a group of counted episodes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outcomes {
    /// Counted episodes.
    pub episodes: u64,
    /// Episodes the env reported a success.
    pub successes: u64,
    /// Episodes that reported no outcome (counted as failures).
    pub unknown: u64,
    /// Env steps over the episodes.
    pub steps: i64,
    /// Episode wall time over the episodes, in seconds.
    pub duration_s: f64,
    /// Total reward over the episodes.
    pub reward: f64,
}

impl Outcomes {
    fn add(&mut self, episode: &EpisodeRecord) {
        self.episodes += 1;
        self.successes += u64::from(episode.success == Some(true));
        self.unknown += u64::from(episode.success.is_none());
        self.steps += episode.steps;
        self.duration_s += episode.duration_s;
        self.reward += episode.reward;
    }

    /// The success share by [`RunResult::success_rate`]'s rule, or `None`
    /// when no episode reported an outcome.
    pub fn success_rate(&self) -> Option<f64> {
        crate::run::success_rate(self.successes, self.unknown, self.episodes)
    }

    /// The Wilson 95% interval of [`Self::success_rate`].
    pub fn ci95(&self) -> Option<(f64, f64)> {
        self.success_rate()?;
        wilson95(self.successes, self.episodes)
    }

    fn mean(&self, total: f64) -> Option<f64> {
        (self.episodes > 0).then(|| total / self.episodes as f64)
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "episodes": self.episodes,
            "successes": self.successes,
            "unknown": self.unknown,
            "successRate": self.success_rate(),
            "ci95": self.ci95().map(|(lo, hi)| [lo, hi]),
            "steps": self.steps,
            "meanSteps": self.mean(self.steps as f64),
            "durationS": self.duration_s,
            "meanDurationS": self.mean(self.duration_s),
            "meanReward": self.mean(self.reward),
        })
    }
}

/// One model's results.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelReport {
    /// The model label.
    pub model: String,
    /// Workloads read for the model.
    pub workloads: usize,
    /// All counted episodes.
    pub overall: Outcomes,
    /// Per suite, sorted by name; empty when no workload names one.
    pub suites: Vec<(String, Outcomes)>,
    /// Per `(suite, task)`, sorted.
    pub tasks: Vec<((Option<String>, String), Outcomes)>,
    /// The workloads' telemetry, merged.
    pub telemetry: Vec<TelemetryRow>,
}

/// Results per model, in the order the models first appear.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EvalReport {
    /// One entry per model.
    pub models: Vec<ModelReport>,
}

impl EvalReport {
    /// The report of the bundles at `paths` (see [`read_bundles`]).
    pub fn load<P: AsRef<Path>>(paths: impl IntoIterator<Item = P>) -> Result<Self> {
        Ok(Self::from_workloads(&read_bundles(paths)?))
    }

    /// The report of `workloads`.
    pub fn from_workloads(workloads: &[BundleWorkload]) -> Self {
        let models = distinct(workloads.iter().map(|w| w.model.as_str()))
            .into_iter()
            .map(|model| {
                let mine: Vec<_> = workloads.iter().filter(|w| w.model == model).collect();
                let mut report = ModelReport {
                    model: model.to_string(),
                    workloads: mine.len(),
                    telemetry: merge_telemetry(mine.iter().flat_map(|w| &w.telemetry)),
                    ..Default::default()
                };
                let mut suites = std::collections::BTreeMap::<String, Outcomes>::new();
                let mut tasks = std::collections::BTreeMap::new();
                for w in &mine {
                    let key = (w.suite.clone(), w.task.clone());
                    for e in w.episodes.iter().filter(|e| e.included_in_metrics) {
                        report.overall.add(e);
                        tasks
                            .entry(key.clone())
                            .or_insert_with(Outcomes::default)
                            .add(e);
                        if let Some(suite) = &w.suite {
                            suites.entry(suite.clone()).or_default().add(e);
                        }
                    }
                }
                report.suites = suites.into_iter().collect();
                report.tasks = tasks.into_iter().collect();
                report
            })
            .collect();
        Self { models }
    }

    /// The report as a JSON document.
    pub fn to_json(&self) -> serde_json::Value {
        let rows = |rows: &[TelemetryRow]| {
            rows.iter()
                .map(|r| {
                    serde_json::json!({
                        "op": r.op, "component": r.component, "metric": r.metric,
                        "unit": r.unit, "count": r.count, "avg": r.avg,
                        "p50": r.p50, "p95": r.p95, "p99": r.p99,
                    })
                })
                .collect::<Vec<_>>()
        };
        let models = self
            .models
            .iter()
            .map(|m| {
                let suites = m
                    .suites
                    .iter()
                    .map(|(suite, o)| with(o.to_json(), [("suite", suite.as_str().into())]))
                    .collect::<Vec<_>>();
                let tasks = m
                    .tasks
                    .iter()
                    .map(|((suite, task), o)| {
                        with(
                            o.to_json(),
                            [
                                ("suite", suite.as_deref().into()),
                                ("task", task.as_str().into()),
                            ],
                        )
                    })
                    .collect::<Vec<_>>();
                serde_json::json!({
                    "model": m.model,
                    "workloads": m.workloads,
                    "overall": m.overall.to_json(),
                    "suites": suites,
                    "tasks": tasks,
                    "telemetry": rows(&m.telemetry),
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({ "models": models })
    }

    /// The report as aligned text tables: one row per task, per suite and
    /// overall, then the merged telemetry, for each model.
    pub fn format_table(&self) -> String {
        if self.models.is_empty() {
            return "(no workloads)".to_string();
        }
        let header = [
            "suite",
            "task",
            "episodes",
            "success",
            "95% CI",
            "mean steps",
            "mean s",
        ];
        let percent = |v: f64| format!("{:.1}%", v * 100.0);
        let row = |suite: &str, task: &str, o: &Outcomes| {
            [
                suite.to_string(),
                task.to_string(),
                o.episodes.to_string(),
                o.success_rate().map_or_else(|| "-".to_string(), percent),
                o.ci95().map_or_else(
                    || "-".to_string(),
                    |(lo, hi)| format!("{}-{}", percent(lo), percent(hi)),
                ),
                o.mean(o.steps as f64)
                    .map_or_else(|| "-".to_string(), |v| format!("{v:.1}")),
                o.mean(o.duration_s)
                    .map_or_else(|| "-".to_string(), |v| format!("{v:.2}")),
            ]
        };
        self.models
            .iter()
            .map(|m| {
                let mut rows = Vec::new();
                for ((suite, task), o) in &m.tasks {
                    rows.push(row(suite.as_deref().unwrap_or("-"), task, o));
                }
                for (suite, o) in &m.suites {
                    rows.push(row(suite, "(all)", o));
                }
                rows.push(row("(all)", "(all)", &m.overall));
                let mut out = format!(
                    "model {}: {} workload(s), {} episode(s)\n{}",
                    m.model,
                    m.workloads,
                    m.overall.episodes,
                    align(&header, &rows, 2)
                );
                if m.overall.unknown > 0 {
                    out.push_str(&format!(
                        "\n{} episode(s) reported no success signal and count as failures",
                        m.overall.unknown
                    ));
                }
                let telemetry = RunResult {
                    telemetry: m.telemetry.clone(),
                    ..Default::default()
                };
                format!("{out}\n\n{}", telemetry.format_telemetry())
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// Text columns left-aligned up to `left`, the rest right-aligned.
fn align<const N: usize>(header: &[&str; N], rows: &[[String; N]], left: usize) -> String {
    let widths: Vec<usize> = (0..N)
        .map(|i| {
            rows.iter()
                .map(|row| row[i].chars().count())
                .chain([header[i].chars().count()])
                .max()
                .unwrap_or(0)
        })
        .collect();
    let line = |cells: Vec<&str>| {
        cells
            .iter()
            .enumerate()
            .map(|(i, cell)| {
                let pad = " ".repeat(widths[i] - cell.chars().count());
                if i < left {
                    format!("{cell}{pad}")
                } else {
                    format!("{pad}{cell}")
                }
            })
            .collect::<Vec<_>>()
            .join("  ")
            .trim_end()
            .to_string()
    };
    std::iter::once(line(header.to_vec()))
        .chain(
            rows.iter()
                .map(|row| line(row.iter().map(String::as_str).collect())),
        )
        .collect::<Vec<_>>()
        .join("\n")
}

fn with<const N: usize>(
    mut value: serde_json::Value,
    keys: [(&str, serde_json::Value); N],
) -> serde_json::Value {
    let object = value.as_object_mut().expect("outcomes are an object");
    for (key, item) in keys {
        object.insert(key.to_string(), item);
    }
    value
}

fn distinct<'a>(items: impl Iterator<Item = &'a str>) -> Vec<&'a str> {
    let mut seen = Vec::new();
    for item in items {
        if !seen.contains(&item) {
            seen.push(item);
        }
    }
    seen
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recorder::{Recorder, RecorderAddOptions};

    #[test]
    fn wilson95_matches_reference_intervals() {
        // Newcombe (1998), method 3; also statsmodels `proportion_confint(method="wilson")`.
        for (successes, trials, expected) in [
            (0, 10, Some((0.0, 0.277533))),
            (5, 10, Some((0.236593, 0.763407))),
            (10, 10, Some((0.722467, 1.0))),
            (80, 100, Some((0.711171, 0.866633))),
            (43, 50, Some((0.738138, 0.930492))),
            (0, 0, None),
        ] {
            let got = wilson95(successes, trials);
            assert_eq!(got.is_some(), expected.is_some(), "{successes}/{trials}");
            if let (Some((lo, hi)), Some((want_lo, want_hi))) = (got, expected) {
                assert!(
                    (lo - want_lo).abs() < 1e-6 && (hi - want_hi).abs() < 1e-6,
                    "{successes}/{trials}: ({lo}, {hi})"
                );
            }
        }
    }

    fn run(outcomes: &[Option<bool>], telemetry: (i64, f64)) -> RunResult {
        RunResult {
            episodes: outcomes
                .iter()
                .map(|&success| EpisodeResult {
                    steps: 10,
                    duration_s: 0.5,
                    success,
                    ..Default::default()
                })
                .collect(),
            telemetry: vec![TelemetryRow {
                op: "env.step".into(),
                component: "env".into(),
                metric: "rpc.total".into(),
                unit: "ms".into(),
                count: telemetry.0,
                avg: telemetry.1,
                p50: telemetry.1,
                p95: telemetry.1 * 2.0,
                p99: telemetry.1 * 3.0,
            }],
            ..Default::default()
        }
    }

    fn add(
        recorder: &mut Recorder,
        model: &str,
        suite: Option<&str>,
        task: &str,
        result: &RunResult,
    ) {
        let config = suite.map(|s| {
            [(SUITE_CONFIG_KEY, crate::Value::from(s))]
                .into_iter()
                .collect()
        });
        let options = RecorderAddOptions {
            task: Some(task.into()),
            config,
            ..Default::default()
        };
        recorder.add(result, model, "env", options).unwrap();
    }

    #[test]
    fn bundles_load_and_report_per_task_suite_and_model() {
        let dir = tempfile::tempdir().unwrap();
        let export = |recorder: Recorder, name: &str| {
            recorder
                .export_(dir.path().join(name), Default::default())
                .unwrap()
        };
        // A folder holding a two-workload bundle and a zip of one workload.
        let mut two = Recorder::new(Default::default()).unwrap();
        add(
            &mut two,
            "a",
            Some("s1"),
            "t1",
            &run(&[Some(true), Some(false)], (1, 10.0)),
        );
        add(
            &mut two,
            "a",
            Some("s1"),
            "t2",
            &run(&[Some(true)], (3, 30.0)),
        );
        export(two, "1-two");
        let mut one = Recorder::new(Default::default()).unwrap();
        add(
            &mut one,
            "a",
            Some("s2"),
            "t1",
            &run(&[None, Some(true)], (0, 99.0)),
        );
        // An errored episode the platform does not count.
        let excluded = RecorderAddOptions {
            task: Some("t1".into()),
            included_in_metrics: false,
            ..Default::default()
        };
        one.add(&run(&[None], (0, 0.0)), "a", "env", excluded)
            .unwrap();
        export(one, "2-one.zip");

        let loaded = RunResult::load(dir.path()).unwrap();
        assert_eq!(
            loaded
                .episodes
                .iter()
                .map(|e| (e.index, e.success))
                .collect::<Vec<_>>(),
            [
                (0, Some(true)),
                (1, Some(false)),
                (2, Some(true)),
                (3, None),
                (4, Some(true))
            ]
        );
        assert_eq!(loaded.success_rate(), Some(0.6));
        // Counts add; avg and percentiles are count-weighted; empty rows drop.
        let row = &loaded.telemetry[..];
        assert_eq!(row.len(), 1);
        assert_eq!((row[0].count, row[0].avg, row[0].p95), (4, 25.0, 50.0));

        let report = EvalReport::load([dir.path()]).unwrap();
        let [model] = &report.models[..] else {
            panic!("one model")
        };
        let rates = |groups: Vec<(String, &Outcomes)>| {
            groups
                .into_iter()
                .map(|(key, o)| (key, o.episodes, o.successes, o.unknown, o.success_rate()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            rates(
                model
                    .tasks
                    .iter()
                    .map(|((s, t), o)| (format!("{s:?}/{t}"), o))
                    .collect()
            ),
            [
                ("Some(\"s1\")/t1".into(), 2, 1, 0, Some(0.5)),
                ("Some(\"s1\")/t2".into(), 1, 1, 0, Some(1.0)),
                ("Some(\"s2\")/t1".into(), 2, 1, 1, Some(0.5)),
            ]
        );
        assert_eq!(
            rates(model.suites.iter().map(|(s, o)| (s.clone(), o)).collect()),
            [
                ("s1".into(), 3, 2, 0, Some(2.0 / 3.0)),
                ("s2".into(), 2, 1, 1, Some(0.5))
            ]
        );
        assert_eq!(
            (
                model.overall.episodes,
                model.overall.steps,
                model.overall.duration_s
            ),
            (5, 50, 2.5)
        );
        assert_eq!(model.overall.ci95(), wilson95(3, 5));
        assert_eq!(model.overall.success_rate(), loaded.success_rate());

        // A second model is reported beside the first, and refused by load.
        let mut other = Recorder::new(Default::default()).unwrap();
        add(&mut other, "b", None, "t1", &run(&[Some(false)], (1, 1.0)));
        export(other, "3-other");
        let report = EvalReport::load([dir.path()]).unwrap();
        assert_eq!(
            report
                .models
                .iter()
                .map(|m| m.model.as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert!(report.models[1].suites.is_empty());
        let error = RunResult::load(dir.path()).unwrap_err();
        assert_eq!(error.class(), crate::ErrorClass::ValueError);
    }
}
