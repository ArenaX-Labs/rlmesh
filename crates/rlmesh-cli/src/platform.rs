use crate::cli::{EvalIdArgs, EvalListArgs, EvalSubmitArgs, ProfileArgs, TokenArgs};
use crate::render::Style;

use anyhow::{Context, Result, bail};
use rlmesh::binding::managed::{self as store, ProfileStore, get_json, http_client};
use rlmesh::managed::Client;
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const WAIT_POLL: Duration = Duration::from_secs(10);
const TERMINAL: [&str; 3] = ["completed", "failed", "cancelled"];

/// The platform client for `--profile` (or `RLMESH_API_KEY`), on its own copy
/// of the store: the session it refreshes is written back to the same store.
async fn connect(profiles: &ProfileStore, args: &ProfileArgs) -> Result<Client> {
    Ok(store::connect(profiles.clone(), args.profile.as_deref()).await?)
}

async fn dashboard_url(platform: &Client) -> Option<String> {
    // Discovery is unauthenticated; never send the bearer where the
    // contract does not ask for it.
    let info: Value = get_json(
        &http_client().ok()?,
        &format!("{}/v1/info", platform.url()),
        None,
        "fetching platform info",
    )
    .await
    .ok()?;
    info["urls"]["dashboard"].as_str().map(str::to_owned)
}

pub async fn token(
    profiles: &ProfileStore,
    args: &TokenArgs,
    stdout: &mut impl Write,
) -> Result<()> {
    let platform = connect(profiles, &args.profile).await?;
    if args.json {
        writeln!(
            stdout,
            "{}",
            json!({
                "platform": platform.url(),
                "token": store::access_token(&platform),
                "expiresAt": store::expires_at(&platform).map(rfc3339_utc),
            })
        )?;
    } else {
        writeln!(stdout, "{}", store::access_token(&platform))?;
    }
    Ok(())
}

pub async fn submit(
    profiles: &ProfileStore,
    args: &EvalSubmitArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<i32> {
    let request = read_request(&args.request)?;
    let mut platform = connect(profiles, &args.profile).await?;

    if args.preview {
        let preview = platform.preview(&request).await?;
        if args.json {
            writeln!(stdout, "{}", serde_json::to_string_pretty(&preview)?)?;
        } else {
            let workloads = preview["workloads"].as_array().map_or(0, Vec::len);
            writeln!(stdout, "{}", style.bold(&format!("{workloads} workloads")))?;
            writeln!(
                stdout,
                "{}",
                serde_json::to_string_pretty(&preview["estimates"])?
            )?;
            write_warnings(stdout, style, &preview)?;
        }
        return Ok(0);
    }

    let command = platform.submit(&request).await?;
    let id = command["id"]
        .as_str()
        .context("submit response has no id")?;
    if args.json {
        writeln!(stdout, "{}", serde_json::to_string_pretty(&command)?)?;
    } else {
        writeln!(
            stdout,
            "{}",
            style.success(&format!("Submitted {}", style.bold(id)))
        )?;
        if let Some(dashboard) = dashboard_url(&platform).await {
            writeln!(stdout, "  {dashboard}/evaluations/{id}")?;
        }
        write_warnings(stdout, style, &command)?;
    }
    if args.wait {
        return wait_for(&mut platform, id, stdout, style).await;
    }
    Ok(0)
}

pub async fn list(
    profiles: &ProfileStore,
    args: &EvalListArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<()> {
    let mut platform = connect(profiles, &args.profile).await?;
    let mut query = vec![("limit", args.limit.to_string())];
    if let Some(status) = &args.status {
        query.push(("status", status.clone()));
    }
    if let Some(q) = &args.q {
        query.push(("q", q.clone()));
    }
    for tag in &args.tags {
        if !tag.contains(':') {
            bail!("--tag takes key:value, got {tag:?}");
        }
        query.push(("tag", tag.clone()));
    }
    let query: Vec<(&str, &str)> = query
        .iter()
        .map(|(key, value)| (*key, value.as_str()))
        .collect();
    let page = platform.get("/v1/evaluations", &query).await?;
    if args.json {
        writeln!(stdout, "{}", serde_json::to_string_pretty(&page)?)?;
        return Ok(());
    }
    let items = page["items"].as_array().cloned().unwrap_or_default();
    if items.is_empty() {
        writeln!(stdout, "{}", style.muted("no evaluations"))?;
        return Ok(());
    }
    for item in &items {
        let progress = &item["progress"];
        let status = format!("{:<12}", item["status"].as_str().unwrap_or("?"));
        let status = match item["status"].as_str() {
            Some("completed") => style.green(&status),
            Some("failed" | "cancelled") => style.yellow(&status),
            _ => status,
        };
        writeln!(
            stdout,
            "{}  {status}  {:>6}/{:<6}  {}",
            item["id"].as_str().unwrap_or("?"),
            progress["completedEpisodes"],
            progress["totalEpisodes"],
            item["name"]
                .as_str()
                .or_else(|| item["metadata"]["evaluationLabel"].as_str())
                .unwrap_or(""),
        )?;
    }
    if !page["nextCursor"].is_null() {
        writeln!(
            stdout,
            "{}",
            style.muted("more: raise --limit or narrow the filter")
        )?;
    }
    Ok(())
}

pub async fn get(
    profiles: &ProfileStore,
    args: &EvalIdArgs,
    stdout: &mut impl Write,
) -> Result<()> {
    let mut platform = connect(profiles, &args.profile).await?;
    let evaluation = platform.evaluation(&args.id).await?;
    writeln!(stdout, "{}", serde_json::to_string_pretty(&evaluation)?)?;
    Ok(())
}

pub async fn wait(
    profiles: &ProfileStore,
    args: &EvalIdArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<i32> {
    let mut platform = connect(profiles, &args.profile).await?;
    wait_for(&mut platform, &args.id, stdout, style).await
}

pub async fn cancel(
    profiles: &ProfileStore,
    args: &EvalIdArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<()> {
    let mut platform = connect(profiles, &args.profile).await?;
    let command = platform.cancel(&args.id).await?;
    writeln!(
        stdout,
        "{}",
        style.success(&format!(
            "{} is {}",
            args.id,
            command["status"].as_str().unwrap_or("cancelling")
        ))
    )?;
    Ok(())
}

async fn wait_for(
    platform: &mut Client,
    id: &str,
    stdout: &mut impl Write,
    style: Style,
) -> Result<i32> {
    let mut last = String::new();
    loop {
        let evaluation = platform.evaluation(id).await?;
        let status = evaluation["status"].as_str().unwrap_or("?").to_owned();
        let progress = &evaluation["progress"];
        let line = format!(
            "{status}  {}/{} episodes",
            progress["completedEpisodes"], progress["totalEpisodes"]
        );
        if line != last {
            writeln!(stdout, "{}", style.muted(&line))?;
            stdout.flush()?;
            last = line;
        }
        if TERMINAL.contains(&status.as_str()) {
            return Ok(i32::from(status != "completed"));
        }
        tokio::time::sleep(WAIT_POLL).await;
    }
}

/// RFC 3339 in UTC with second precision, e.g. `2026-09-22T14:03:00Z`.
fn rfc3339_utc(time: SystemTime) -> String {
    let secs = time
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (hour, minute, second) = (rem / 3_600, rem % 3_600 / 60, rem % 60);

    // Civil date from days since the epoch (proleptic Gregorian).
    let z = i64::try_from(days).unwrap_or(i64::MAX / 2) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

pub(crate) fn read_request(path: &str) -> Result<Value> {
    let text = if path == "-" {
        let mut text = String::new();
        std::io::stdin()
            .read_to_string(&mut text)
            .context("reading request from stdin")?;
        text
    } else {
        std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?
    };
    serde_json::from_str(&text).with_context(|| format!("parsing {path} as JSON"))
}

fn write_warnings(stdout: &mut impl Write, style: Style, response: &Value) -> Result<()> {
    for warning in response["warnings"].as_array().into_iter().flatten() {
        if let Some(warning) = warning.as_str() {
            writeln!(stdout, "  {} {warning}", style.yellow("warning:"))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stdin_marker_and_files_parse_as_json() {
        let dir = std::env::temp_dir().join(format!("rlmesh-cli-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("request.json");
        std::fs::write(&file, r#"{"models":[{"model":"prebuilt/xvla/libero"}]}"#).unwrap();
        let request = read_request(file.to_str().unwrap()).unwrap();
        assert_eq!(request["models"][0]["model"], "prebuilt/xvla/libero");

        std::fs::write(&file, "not json").unwrap();
        assert!(read_request(file.to_str().unwrap()).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn formats_rfc3339_utc() {
        assert_eq!(rfc3339_utc(UNIX_EPOCH), "1970-01-01T00:00:00Z");
        assert_eq!(
            rfc3339_utc(UNIX_EPOCH + Duration::from_secs(1_700_000_000)),
            "2023-11-14T22:13:20Z"
        );
        assert_eq!(
            rfc3339_utc(UNIX_EPOCH + Duration::from_secs(951_782_400)),
            "2000-02-29T00:00:00Z"
        );
    }
}
