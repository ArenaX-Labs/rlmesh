use crate::auth::refresh_session;
use crate::cli::{EvalIdArgs, EvalListArgs, EvalSubmitArgs, ProfileArgs, TokenArgs};
use crate::config::ProfileStore;
use crate::helpers::{expect_json, http_client};
use crate::render::Style;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::time::Duration;

const WAIT_POLL: Duration = Duration::from_secs(10);
const TERMINAL: [&str; 3] = ["completed", "failed", "cancelled"];

/// A signed-in platform: the profile's base URL plus a freshly refreshed
/// access token.
pub(crate) struct Platform {
    client: reqwest::Client,
    url: String,
    token: String,
}

impl Platform {
    pub(crate) async fn connect(profiles: &mut ProfileStore, args: &ProfileArgs) -> Result<Self> {
        let profile = profiles.resolve(args.profile.as_deref());
        let client = http_client()?;
        let session = refresh_session(&client, profiles, &profile).await?;
        Ok(Self {
            client,
            url: profile.platform_url.unwrap_or_default(),
            token: session.credentials.access_token,
        })
    }

    async fn get(&self, path: &str, query: &[(&str, String)]) -> Result<Value> {
        let response = self
            .client
            .get(format!("{}{path}", self.url))
            .query(query)
            .bearer_auth(&self.token)
            .send()
            .await
            .with_context(|| format!("GET {path}: could not reach {}", self.url))?;
        expect_json(response, &format!("GET {path}")).await
    }

    async fn post(&self, path: &str, body: &Value) -> Result<Value> {
        let response = self
            .client
            .post(format!("{}{path}", self.url))
            .bearer_auth(&self.token)
            .json(body)
            .send()
            .await
            .with_context(|| format!("POST {path}: could not reach {}", self.url))?;
        expect_json(response, &format!("POST {path}")).await
    }

    async fn dashboard_url(&self) -> Option<String> {
        let info: Value = self.get("/v1/info", &[]).await.ok()?;
        info["urls"]["dashboard"].as_str().map(str::to_owned)
    }
}

pub async fn token(
    profiles: &mut ProfileStore,
    args: &TokenArgs,
    stdout: &mut impl Write,
) -> Result<()> {
    let platform = Platform::connect(profiles, &args.profile).await?;
    if args.json {
        writeln!(
            stdout,
            "{}",
            json!({"platform": platform.url, "token": platform.token})
        )?;
    } else {
        writeln!(stdout, "{}", platform.token)?;
    }
    Ok(())
}

pub async fn submit(
    profiles: &mut ProfileStore,
    args: &EvalSubmitArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<i32> {
    let request = read_request(&args.request)?;
    let platform = Platform::connect(profiles, &args.profile).await?;

    if args.preview {
        let preview = platform.post("/v1/evaluation-previews", &request).await?;
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

    let command = platform.post("/v1/evaluations", &request).await?;
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
        if let Some(dashboard) = platform.dashboard_url().await {
            writeln!(stdout, "  {dashboard}/evaluations/{id}")?;
        }
        write_warnings(stdout, style, &command)?;
    }
    if args.wait {
        return wait_for(&platform, id, stdout, style).await;
    }
    Ok(0)
}

pub async fn list(
    profiles: &mut ProfileStore,
    args: &EvalListArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<()> {
    let platform = Platform::connect(profiles, &args.profile).await?;
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
    profiles: &mut ProfileStore,
    args: &EvalIdArgs,
    stdout: &mut impl Write,
) -> Result<()> {
    let platform = Platform::connect(profiles, &args.profile).await?;
    let evaluation = platform
        .get(&format!("/v1/evaluations/{}", args.id), &[])
        .await?;
    writeln!(stdout, "{}", serde_json::to_string_pretty(&evaluation)?)?;
    Ok(())
}

pub async fn wait(
    profiles: &mut ProfileStore,
    args: &EvalIdArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<i32> {
    let platform = Platform::connect(profiles, &args.profile).await?;
    wait_for(&platform, &args.id, stdout, style).await
}

pub async fn cancel(
    profiles: &mut ProfileStore,
    args: &EvalIdArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<()> {
    let platform = Platform::connect(profiles, &args.profile).await?;
    let command = platform
        .post(
            &format!("/v1/evaluations/{}/cancellations", args.id),
            &json!({}),
        )
        .await?;
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
    platform: &Platform,
    id: &str,
    stdout: &mut impl Write,
    style: Style,
) -> Result<i32> {
    let mut last = String::new();
    loop {
        let evaluation = platform.get(&format!("/v1/evaluations/{id}"), &[]).await?;
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

fn read_request(path: &str) -> Result<Value> {
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
}
