//! Submit and read evaluations on a managed RLMesh platform (Experimental,
//! feature `managed`).
//!
//! [`Client`] is a thin client over the platform's public `/v1` HTTP API,
//! the same client as Python's `rlmesh.managed.Client`. It signs in the way
//! the `rlmesh` CLI does and shares the CLI's profiles: after `rlmesh login`,
//! [`Client::connect`] uses the default profile's session and refreshes it
//! the way the CLI does, so the two never race over the single-use refresh
//! token. `RLMESH_API_KEY` (with `RLMESH_PLATFORM_URL`) bypasses profiles.
//!
//! Requests and responses are the platform's own JSON shapes, as
//! [`serde_json::Value`]; `EvaluationRequest` in the platform's OpenAPI
//! contract lists every field a request accepts.
//!
//! ```no_run
//! use rlmesh::managed::{Client, ClientOptions, WaitOptions};
//! use serde_json::json;
//!
//! # async fn run() -> rlmesh::managed::Result<()> {
//! let mut client = Client::connect(ClientOptions::default()).await?;
//! let submitted = client
//!     .submit(&json!({
//!         "models": [{"model": "prebuilt/xvla/libero"}],
//!         "tasks": [{"env": "prebuilt/libero", "taskId": "libero_spatial/0", "episodes": 10}],
//!     }))
//!     .await?;
//! let id = submitted["id"].as_str().unwrap_or_default();
//! let evaluation = client.wait(id, WaitOptions::default()).await?;
//! println!("{} {:#}", evaluation["status"], client.scores(id, None).await?);
//! # Ok(())
//! # }
//! ```

pub(crate) mod http;
pub(crate) mod session;
pub(crate) mod settings;
pub(crate) mod sign_in;
pub(crate) mod store;

use self::http::{expect_json, http_client, normalize_base_url};
use self::session::{Refresh, ensure_fresh_session};
use self::settings::Settings;
use self::store::{DEFAULT_PLATFORM_URL, ProfileStore, ResolvedProfile};

use anyhow::Context;
use futures::Stream;
use reqwest::{Method, StatusCode};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::time::{Duration, SystemTime};

const TERMINAL: [&str; 3] = ["completed", "failed", "cancelled"];
const NO_QUERY: &[(&str, &str)] = &[];

/// The result of a platform call.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A failed platform call: an HTTP error status with the server's message,
/// or a failure before any response (no signed-in profile, a refresh the
/// identity provider refused, an unreachable platform).
///
/// `Display` is the outermost description and [`source`] walks the causes,
/// so `{:#}` through `anyhow` prints the whole chain as the CLI does.
///
/// [`source`]: std::error::Error::source
pub struct Error(anyhow::Error);

impl Error {
    /// The HTTP status the platform answered with, when it answered.
    pub fn status(&self) -> Option<u16> {
        self.http().map(|error| error.status.as_u16())
    }

    /// The server's error message for an HTTP error, else the description.
    pub fn message(&self) -> String {
        self.http()
            .map_or_else(|| self.0.to_string(), |error| error.message.clone())
    }

    fn http(&self) -> Option<&http::HttpError> {
        self.0.chain().find_map(|cause| cause.downcast_ref())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.chain().nth(1)
    }
}

/// How [`Client::connect`] authenticates (Python's `Client(...)` keywords).
///
/// A `token` (else `RLMESH_API_KEY`) is used as the bearer credential on
/// `url` (else `RLMESH_PLATFORM_URL`, else `https://api.rlmesh.dev`).
/// Without one, the CLI profile `profile` (else the CLI's default profile)
/// is used on its own platform; `url` then is an error, so a profile's
/// session is never sent to another host.
#[derive(Clone, Default)]
pub struct ClientOptions {
    /// The CLI profile to use; default: the CLI's default profile.
    pub profile: Option<String>,
    /// The platform base URL for `token`.
    pub url: Option<String>,
    /// A static bearer credential (an API key), bypassing profiles.
    pub token: Option<String>,
}

impl fmt::Debug for ClientOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClientOptions")
            .field("profile", &self.profile)
            .field("url", &self.url)
            .field("token", &self.token.as_ref().map(|_| "[REDACTED]"))
            .finish()
    }
}

/// Filters for [`Client::evaluations`].
#[derive(Clone, Debug)]
pub struct EvaluationsOptions {
    /// Only evaluations in this status.
    pub status: Option<String>,
    /// Only evaluations carrying every one of these tags.
    pub tags: BTreeMap<String, String>,
    /// A free-text search.
    pub q: Option<String>,
    /// The page size. Default 100.
    pub limit: u64,
}

impl Default for EvaluationsOptions {
    fn default() -> Self {
        Self {
            status: None,
            tags: BTreeMap::new(),
            q: None,
            limit: 100,
        }
    }
}

/// Polling for [`Client::wait`].
#[derive(Clone, Debug)]
pub struct WaitOptions {
    /// The time between polls. Default 15 s.
    pub poll: Duration,
    /// Give up after this long. Default `None`: wait indefinitely.
    pub timeout: Option<Duration>,
    /// Print a status line to stderr whenever it changes. Default true.
    pub progress: bool,
}

impl Default for WaitOptions {
    fn default() -> Self {
        Self {
            poll: Duration::from_secs(15),
            timeout: None,
            progress: true,
        }
    }
}

/// A client of the platform's `/v1` API, authenticated by a CLI profile or a
/// static token. A request the platform rejects as unauthorized is retried
/// once with a forcibly refreshed profile session.
pub struct Client {
    http: reqwest::Client,
    url: String,
    token: String,
    expires_at: Option<SystemTime>,
    /// The profile the token came from; `None` for a static token.
    session: Option<(ProfileStore, ResolvedProfile)>,
}

/// Where [`Client::connect`] gets its credential from.
#[derive(Debug, PartialEq, Eq)]
enum Source {
    Token { url: String, token: String },
    Profile(Option<String>),
}

fn source(options: ClientOptions, settings: &Settings) -> anyhow::Result<Source> {
    let url = options.url.as_deref().map(normalize_base_url);
    if let Some(token) = options.token.or_else(|| settings.api_key.clone()) {
        let url = url
            .or_else(|| settings.platform_url.clone())
            .unwrap_or_else(|| DEFAULT_PLATFORM_URL.to_owned());
        return Ok(Source::Token { url, token });
    }
    if url.is_some() {
        anyhow::bail!(
            "ClientOptions::url needs a token; a CLI profile is used on its own platform"
        );
    }
    Ok(Source::Profile(options.profile))
}

impl Client {
    /// Connects as `options` says, refreshing a profile session that is
    /// about to expire. Reads the `RLMESH_*` environment the CLI reads.
    pub async fn connect(options: ClientOptions) -> Result<Self> {
        let settings = Settings::from_env();
        match source(options, &settings).map_err(Error)? {
            Source::Token { url, token } => Self::with_token(url, token),
            Source::Profile(profile) => {
                let store = ProfileStore::load(&settings).map_err(Error)?;
                let profile = store.resolve(profile.as_deref());
                Self::signed_in(store, profile).await
            }
        }
    }

    pub(crate) fn with_token(url: String, token: String) -> Result<Self> {
        Ok(Self {
            http: http_client().map_err(Error)?,
            url,
            token,
            expires_at: None,
            session: None,
        })
    }

    /// The CLI's own path: `RLMESH_API_KEY` from the store's settings first,
    /// then the named (or default) profile.
    pub(crate) async fn from_store(store: ProfileStore, profile: Option<&str>) -> Result<Self> {
        if let Some((token, url)) = store.api_key() {
            return Self::with_token(url.to_owned(), token.to_owned());
        }
        let profile = store.resolve(profile);
        Self::signed_in(store, profile).await
    }

    pub(crate) async fn signed_in(
        mut store: ProfileStore,
        profile: ResolvedProfile,
    ) -> Result<Self> {
        let http = http_client().map_err(Error)?;
        let session = ensure_fresh_session(&http, &mut store, &profile, Refresh::IfStale)
            .await
            .map_err(Error)?;
        Ok(Self {
            http,
            url: profile.platform_url.clone().unwrap_or_default(),
            token: session.credentials.access_token,
            expires_at: session.expires_at,
            session: Some((store, profile)),
        })
    }

    pub(crate) fn access_token(&self) -> &str {
        &self.token
    }

    pub(crate) fn expires_at(&self) -> Option<SystemTime> {
        self.expires_at
    }

    /// The platform base URL.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// `GET path` (`/v1/...`); `query` keys may repeat.
    pub async fn get(&mut self, path: &str, query: &[(&str, &str)]) -> Result<Value> {
        self.request(Method::GET, path, query, None).await
    }

    /// `POST path` with a JSON body.
    pub async fn post(&mut self, path: &str, body: &Value) -> Result<Value> {
        self.request(Method::POST, path, NO_QUERY, Some(body)).await
    }

    /// `PATCH path` with a JSON body.
    pub async fn patch(&mut self, path: &str, body: &Value) -> Result<Value> {
        self.request(Method::PATCH, path, NO_QUERY, Some(body))
            .await
    }

    /// Validates and sizes a request without launching it.
    pub async fn preview(&mut self, request: &Value) -> Result<Value> {
        self.post("/v1/evaluation-previews", request).await
    }

    /// Launches an evaluation; returns `{"id", "status", "warnings", ...}`.
    pub async fn submit(&mut self, request: &Value) -> Result<Value> {
        self.post("/v1/evaluations", request).await
    }

    /// One evaluation: status, progress, request, and effective config.
    pub async fn evaluation(&mut self, eval_id: &str) -> Result<Value> {
        self.get(&format!("/v1/evaluations/{eval_id}"), &[]).await
    }

    /// Evaluations newest first, following pagination as the stream is read.
    pub fn evaluations(
        &mut self,
        options: EvaluationsOptions,
    ) -> impl Stream<Item = Result<Value>> + '_ {
        let mut query = vec![("limit".to_owned(), options.limit.to_string())];
        query.extend(options.status.map(|status| ("status".to_owned(), status)));
        query.extend(options.q.map(|q| ("q".to_owned(), q)));
        query.extend(
            options
                .tags
                .into_iter()
                .map(|(key, value)| ("tag".to_owned(), format!("{key}:{value}"))),
        );
        let state = (self, query, VecDeque::new(), false);
        futures::stream::try_unfold(
            state,
            |(client, mut query, mut items, mut done)| async move {
                loop {
                    if let Some(item) = items.pop_front() {
                        return Ok(Some((item, (client, query, items, done))));
                    }
                    if done {
                        return Ok(None);
                    }
                    let page = client
                        .request(Method::GET, "/v1/evaluations", &query, None)
                        .await?;
                    items.extend(page["items"].as_array().cloned().unwrap_or_default());
                    match page["nextCursor"]
                        .as_str()
                        .filter(|cursor| !cursor.is_empty())
                    {
                        Some(cursor) => {
                            query.retain(|(key, _)| key != "cursor");
                            query.push(("cursor".to_owned(), cursor.to_owned()));
                        }
                        None => done = true,
                    }
                }
            },
        )
    }

    /// Polls until the evaluation is `completed`, `failed` or `cancelled`
    /// and returns it.
    #[allow(clippy::print_stderr)]
    pub async fn wait(&mut self, eval_id: &str, options: WaitOptions) -> Result<Value> {
        let deadline = options
            .timeout
            .map(|timeout| tokio::time::Instant::now() + timeout);
        let mut last = String::new();
        loop {
            let evaluation = self.evaluation(eval_id).await?;
            let status = evaluation["status"].as_str().unwrap_or("?").to_owned();
            let progress = &evaluation["progress"];
            let line = format!(
                "{status}  {}/{} episodes",
                progress["completedEpisodes"], progress["totalEpisodes"]
            );
            if options.progress && line != last {
                eprintln!("{line}");
                last = line;
            }
            if TERMINAL.contains(&status.as_str()) {
                return Ok(evaluation);
            }
            if deadline.is_some_and(|deadline| tokio::time::Instant::now() >= deadline) {
                return Err(Error(anyhow::anyhow!("{eval_id} still {status}")));
            }
            tokio::time::sleep(options.poll).await;
        }
    }

    /// Cancels a running evaluation.
    pub async fn cancel(&mut self, eval_id: &str) -> Result<Value> {
        self.post(
            &format!("/v1/evaluations/{eval_id}/cancellations"),
            &json!({}),
        )
        .await
    }

    /// Outcome rollups grouped by `task` (`None`), `scenario`, `benchmark`,
    /// `perturbation` or `model`.
    pub async fn scores(&mut self, eval_id: &str, group_by: Option<&str>) -> Result<Value> {
        let group_by = group_by.unwrap_or("task");
        self.get(
            &format!("/v1/evaluations/{eval_id}/scores"),
            &[("groupBy", group_by)],
        )
        .await
    }

    /// Throughput and timing rollups over every task.
    pub async fn metrics(&mut self, eval_id: &str) -> Result<Value> {
        self.get(&format!("/v1/evaluations/{eval_id}/metrics"), &[])
            .await
    }

    async fn request<Q: Serialize + ?Sized>(
        &mut self,
        method: Method,
        path: &str,
        query: &Q,
        body: Option<&Value>,
    ) -> Result<Value> {
        let mut response = self.send(method.clone(), path, query, body).await?;
        if response.status() == StatusCode::UNAUTHORIZED && self.refresh_forced().await? {
            response = self.send(method.clone(), path, query, body).await?;
        }
        expect_json(response, &format!("{method} {path}"))
            .await
            .map_err(Error)
    }

    async fn send<Q: Serialize + ?Sized>(
        &self,
        method: Method,
        path: &str,
        query: &Q,
        body: Option<&Value>,
    ) -> Result<reqwest::Response> {
        let mut request = self
            .http
            .request(method.clone(), format!("{}{path}", self.url))
            .query(query)
            .bearer_auth(&self.token);
        if let Some(body) = body {
            request = request.json(body);
        }
        request
            .send()
            .await
            .with_context(|| format!("{method} {path}: could not reach {}", self.url))
            .map_err(Error)
    }

    /// Rotates the session after the platform rejected the current token;
    /// false when the credential cannot be refreshed at all.
    async fn refresh_forced(&mut self) -> Result<bool> {
        let Some((store, profile)) = &mut self.session else {
            return Ok(false);
        };
        let session = ensure_fresh_session(&self.http, store, profile, Refresh::Force)
            .await
            .map_err(Error)?;
        self.token = session.credentials.access_token;
        self.expires_at = session.expires_at;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_token_wins_over_profiles_and_a_url_needs_one() {
        let bare = Settings::isolated("/c", "/d");
        let keyed = Settings::isolated("/c", "/d").with_api_key("env-key", Some("env.example.com"));
        let options =
            |profile: Option<&str>, url: Option<&str>, token: Option<&str>| ClientOptions {
                profile: profile.map(str::to_owned),
                url: url.map(str::to_owned),
                token: token.map(str::to_owned),
            };
        let token = |url: &str, token: &str| Source::Token {
            url: url.to_owned(),
            token: token.to_owned(),
        };

        let cases = [
            (
                options(None, None, None),
                &bare,
                Some(Source::Profile(None)),
            ),
            (
                options(Some("staging"), None, None),
                &bare,
                Some(Source::Profile(Some("staging".to_owned()))),
            ),
            // The environment's key beats a named profile, as in the CLI.
            (
                options(Some("staging"), None, None),
                &keyed,
                Some(token("https://env.example.com", "env-key")),
            ),
            (
                options(None, Some("api.example.com/"), Some("mine")),
                &keyed,
                Some(token("https://api.example.com", "mine")),
            ),
            (
                options(None, None, Some("mine")),
                &bare,
                Some(token(DEFAULT_PLATFORM_URL, "mine")),
            ),
            (options(None, Some("api.example.com"), None), &bare, None),
        ];
        for (options, settings, expected) in cases {
            let described = format!("{options:?}");
            assert_eq!(source(options, settings).ok(), expected, "{described}");
        }
    }

    #[test]
    fn errors_carry_the_status_and_the_server_message() {
        let error = Error(
            anyhow::Error::new(http::HttpError {
                what: "GET /v1/evaluations/eval_1".to_owned(),
                status: StatusCode::NOT_FOUND,
                message: "no such evaluation".to_owned(),
            })
            .context("outer"),
        );
        assert_eq!(error.status(), Some(404));
        assert_eq!(error.message(), "no such evaluation");

        let error = Error(anyhow::anyhow!("profile \"x\" is not signed in"));
        assert_eq!(error.status(), None);
        assert_eq!(error.message(), "profile \"x\" is not signed in");
    }
}
