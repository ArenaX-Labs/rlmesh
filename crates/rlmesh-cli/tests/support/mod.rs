//! A scriptable mock of the managed platform plus the identity provider it
//! advertises, and a harness that confines the CLI to a temp directory.

// Test scaffolding: panicking on an unexpected state is the point.
#![allow(dead_code, clippy::unwrap_used)]

use axum::Router;
use axum::extract::{Form, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use base64::Engine;
use rlmesh_cli::Settings;
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tempfile::TempDir;

pub const CLIENT_ID: &str = "client_cli";
pub const USER_ID: &str = "user_1";

#[derive(Clone, Copy, Debug)]
pub enum TokenReply {
    Pending,
    SlowDown,
    AccessDenied,
    ExpiredToken,
    Grant,
}

#[derive(Clone, Copy, Debug)]
pub enum RevokeMode {
    NoContent,
    NotFound,
    MethodNotAllowed,
}

/// What the mock says next. Mutable from tests through `Mock::script`.
pub struct Script {
    /// Replies to successive device-code polls; an empty queue grants.
    pub token_replies: VecDeque<TokenReply>,
    /// The one access token the platform currently accepts.
    pub access_token: String,
    /// The one refresh token the provider currently accepts (single use).
    pub refresh_token: String,
    /// Lifetime of newly minted access tokens, in seconds.
    pub access_ttl: i64,
    /// The session's active organization (provider id).
    pub org_id: String,
    /// (provider id, name) of every organization the account belongs to.
    pub orgs: Vec<(String, String)>,
    pub me_email: Option<String>,
    /// Fail the next /v1/me with a 500 (a platform that cannot confirm).
    pub me_fails_once: bool,
    pub revoke: RevokeMode,
    /// Reject the next /v1/evaluations call with a 401.
    pub evaluations_401_once: bool,
    pub api_key: Option<String>,
    pub advertise_endpoints: bool,
    pub registry_host: String,
    pub minted: u32,
}

impl Default for Script {
    fn default() -> Self {
        Self {
            token_replies: VecDeque::new(),
            access_token: String::new(),
            refresh_token: String::new(),
            access_ttl: 3_600,
            org_id: "org_01H".to_owned(),
            orgs: vec![("org_01H".to_owned(), "Acme".to_owned())],
            me_email: Some("dev@example.com".to_owned()),
            me_fails_once: false,
            revoke: RevokeMode::NoContent,
            evaluations_401_once: false,
            api_key: None,
            advertise_endpoints: true,
            registry_host: "registry.test".to_owned(),
            minted: 0,
        }
    }
}

impl Script {
    fn mint(&mut self) -> (String, String) {
        self.mint_with_ttl(self.access_ttl)
    }

    fn mint_with_ttl(&mut self, access_ttl: i64) -> (String, String) {
        self.minted += 1;
        let exp = now_secs() + access_ttl;
        self.access_token = mint_jwt(USER_ID, &self.org_id, exp);
        self.refresh_token = format!("refresh_{}", self.minted);
        (self.access_token.clone(), self.refresh_token.clone())
    }

    fn accepts(&self, bearer: Option<&str>) -> bool {
        let Some(bearer) = bearer else { return false };
        (!self.access_token.is_empty() && bearer == self.access_token)
            || self.api_key.as_deref() == Some(bearer)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Seen {
    Info,
    DeviceAuth,
    DeviceGrant,
    RefreshGrant {
        refresh_token: String,
        organization_id: Option<String>,
    },
    Me,
    Organizations,
    Revoke,
    Evaluations,
    RegistryInfo,
}

pub fn now_secs() -> i64 {
    i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    )
    .unwrap()
}

/// An unsigned JWT with the claims the CLI reads.
pub fn mint_jwt(sub: &str, org_id: &str, exp: i64) -> String {
    let encode =
        |value: &Value| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(value.to_string());
    format!(
        "{}.{}.signature",
        encode(&json!({"alg": "none", "typ": "JWT"})),
        encode(&json!({"sub": sub, "org_id": org_id, "exp": exp, "sid": "session_1"}))
    )
}

struct Inner {
    url: String,
    script: Mutex<Script>,
    seen: Mutex<Vec<Seen>>,
}

type Shared = Arc<Inner>;

pub struct Mock {
    pub url: String,
    inner: Shared,
}

impl Mock {
    pub async fn start(script: Script) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let inner = Arc::new(Inner {
            url: url.clone(),
            script: Mutex::new(script),
            seen: Mutex::new(Vec::new()),
        });
        let router = Router::new()
            .route("/v1/info", get(info))
            .route("/oauth/device", post(device))
            .route("/oauth/token", post(token))
            .route("/v1/me", get(me))
            .route("/v1/me/organizations", get(organizations))
            .route("/v1/me/session", delete(revoke))
            .route("/v1/evaluations", get(evaluations))
            .route("/v1/registry/info", get(registry_info))
            .with_state(inner.clone());
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        Self { url, inner }
    }

    pub fn script(&self) -> std::sync::MutexGuard<'_, Script> {
        self.inner.script.lock().unwrap()
    }

    pub fn seen(&self) -> Vec<Seen> {
        self.inner.seen.lock().unwrap().clone()
    }

    pub fn refresh_grants(&self) -> Vec<Seen> {
        self.seen()
            .into_iter()
            .filter(|seen| matches!(seen, Seen::RefreshGrant { .. }))
            .collect()
    }

    /// Mints a session the platform accepts, with an access token that
    /// expires `access_ttl` seconds from now (negative: already expired).
    pub fn mint_session(&self, access_ttl: i64) -> (String, String) {
        self.script().mint_with_ttl(access_ttl)
    }
}

fn record(state: &Shared, seen: Seen) {
    state.seen.lock().unwrap().push(seen);
}

fn bearer(headers: &HeaderMap) -> Option<String> {
    headers
        .get("authorization")?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
        .map(str::to_owned)
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        axum::Json(json!({"error": {"message": "missing or invalid credential"}})),
    )
        .into_response()
}

fn oauth_error(error: &str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        axum::Json(json!({"error": error, "error_description": format!("mock: {error}")})),
    )
        .into_response()
}

async fn info(State(state): State<Shared>) -> Response {
    record(&state, Seen::Info);
    let advertise = state.script.lock().unwrap().advertise_endpoints;
    let mut auth = json!({
        "cli": {"clientId": CLIENT_ID},
        "dashboard": {"clientId": "client_dashboard"},
        "issuer": state.url,
    });
    if advertise {
        auth["deviceAuthorizationEndpoint"] = json!(format!("{}/oauth/device", state.url));
        auth["tokenEndpoint"] = json!(format!("{}/oauth/token", state.url));
    }
    axum::Json(json!({
        "auth": auth,
        "urls": {"api": state.url, "dashboard": format!("{}/dash", state.url), "docs": state.url},
    }))
    .into_response()
}

async fn device(
    State(state): State<Shared>,
    Form(form): Form<HashMap<String, String>>,
) -> Response {
    record(&state, Seen::DeviceAuth);
    if form.get("client_id").map(String::as_str) != Some(CLIENT_ID) {
        return oauth_error("invalid_client");
    }
    axum::Json(json!({
        "device_code": "device_1",
        "user_code": "ABCD-1234",
        "verification_uri": format!("{}/verify", state.url),
        "verification_uri_complete": format!("{}/verify?code=ABCD-1234", state.url),
        "expires_in": 60,
        "interval": 1,
    }))
    .into_response()
}

async fn token(State(state): State<Shared>, Form(form): Form<HashMap<String, String>>) -> Response {
    if form.get("client_id").map(String::as_str) != Some(CLIENT_ID) {
        return oauth_error("invalid_client");
    }
    let mut script = state.script.lock().unwrap();
    match form.get("grant_type").map(String::as_str) {
        Some("urn:ietf:params:oauth:grant-type:device_code") => {
            record(&state, Seen::DeviceGrant);
            match script
                .token_replies
                .pop_front()
                .unwrap_or(TokenReply::Grant)
            {
                TokenReply::Pending => oauth_error("authorization_pending"),
                TokenReply::SlowDown => oauth_error("slow_down"),
                TokenReply::AccessDenied => oauth_error("access_denied"),
                TokenReply::ExpiredToken => oauth_error("expired_token"),
                TokenReply::Grant => grant(&mut script),
            }
        }
        Some("refresh_token") => {
            let presented = form.get("refresh_token").cloned().unwrap_or_default();
            let organization_id = form.get("organization_id").cloned();
            record(
                &state,
                Seen::RefreshGrant {
                    refresh_token: presented.clone(),
                    organization_id: organization_id.clone(),
                },
            );
            if presented.is_empty() || presented != script.refresh_token {
                return oauth_error("invalid_grant");
            }
            if let Some(organization_id) = organization_id
                && script.orgs.iter().any(|(id, _)| *id == organization_id)
            {
                script.org_id = organization_id;
            }
            grant(&mut script)
        }
        _ => oauth_error("unsupported_grant_type"),
    }
}

fn grant(script: &mut Script) -> Response {
    let (access_token, refresh_token) = script.mint();
    axum::Json(json!({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expires_in": script.access_ttl,
        // Provider-specific extras the CLI must ignore.
        "user": {"id": USER_ID, "email": "ignored@example.com"},
        "organization_id": script.org_id,
    }))
    .into_response()
}

async fn me(State(state): State<Shared>, headers: HeaderMap) -> Response {
    record(&state, Seen::Me);
    let mut script = state.script.lock().unwrap();
    if !script.accepts(bearer(&headers).as_deref()) {
        return unauthorized();
    }
    if script.me_fails_once {
        script.me_fails_once = false;
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({"error": {"message": "identity backend unavailable"}})),
        )
            .into_response();
    }
    let is_key = script.api_key.is_some() && bearer(&headers) == script.api_key;
    let mut subject = if is_key {
        json!({"type": "organizationApiKey", "id": "key_1"})
    } else {
        json!({"type": "userSession", "id": USER_ID, "displayName": "Dev User"})
    };
    if let Some(email) = script.me_email.as_ref().filter(|_| !is_key) {
        subject["email"] = json!(email);
    }
    let name = script
        .orgs
        .iter()
        .find(|(id, _)| *id == script.org_id)
        .map(|(_, name)| name.clone())
        .unwrap_or_default();
    axum::Json(json!({
        "subject": subject,
        "organization": {
            "id": "org_9f3c2a",
            "name": name,
            "providerId": script.org_id,
            "linked": true,
        },
        "access": {"roles": [], "permissions": [], "featureFlags": [], "fullAccess": true, "demoPublisher": false},
    }))
    .into_response()
}

async fn organizations(State(state): State<Shared>, headers: HeaderMap) -> Response {
    record(&state, Seen::Organizations);
    let script = state.script.lock().unwrap();
    if !script.accepts(bearer(&headers).as_deref()) {
        return unauthorized();
    }
    let organizations: Vec<Value> = script
        .orgs
        .iter()
        .map(|(id, name)| {
            json!({
                "name": name,
                "providerId": id,
                "id": null,
                "registryNamespace": "acme",
                "active": *id == script.org_id,
            })
        })
        .collect();
    axum::Json(json!({"organizations": organizations})).into_response()
}

async fn revoke(State(state): State<Shared>, headers: HeaderMap) -> Response {
    record(&state, Seen::Revoke);
    let script = state.script.lock().unwrap();
    if !script.accepts(bearer(&headers).as_deref()) {
        return unauthorized();
    }
    match script.revoke {
        RevokeMode::NoContent => StatusCode::NO_CONTENT.into_response(),
        RevokeMode::NotFound => (
            StatusCode::NOT_FOUND,
            axum::Json(json!({"error": {"message": "not found"}})),
        )
            .into_response(),
        RevokeMode::MethodNotAllowed => StatusCode::METHOD_NOT_ALLOWED.into_response(),
    }
}

async fn evaluations(State(state): State<Shared>, headers: HeaderMap) -> Response {
    record(&state, Seen::Evaluations);
    let mut script = state.script.lock().unwrap();
    if script.evaluations_401_once {
        script.evaluations_401_once = false;
        return unauthorized();
    }
    if !script.accepts(bearer(&headers).as_deref()) {
        return unauthorized();
    }
    axum::Json(json!({
        "items": [{
            "id": "eval_1",
            "status": "completed",
            "progress": {"completedEpisodes": 10, "totalEpisodes": 10},
            "name": "smoke",
        }],
        "nextCursor": null,
    }))
    .into_response()
}

async fn registry_info(State(state): State<Shared>, headers: HeaderMap) -> Response {
    record(&state, Seen::RegistryInfo);
    let script = state.script.lock().unwrap();
    if !script.accepts(bearer(&headers).as_deref()) {
        return unauthorized();
    }
    axum::Json(json!({"host": script.registry_host, "namespaces": ["acme"]})).into_response()
}

/// The CLI confined to a temp directory (no keychain, no real config) and
/// pointed at a mock platform.
pub struct Harness {
    pub dir: TempDir,
    pub settings: Settings,
    pub mock: Mock,
}

impl Harness {
    pub async fn new(script: Script) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let settings = Settings::isolated(dir.path().join("config"), dir.path().join("data"));
        let mock = Mock::start(script).await;
        Self {
            dir,
            settings,
            mock,
        }
    }

    pub async fn run(&self, args: &[&str]) -> (i32, String, String) {
        self.run_with(self.settings.clone(), args).await
    }

    pub async fn run_with(&self, settings: Settings, args: &[&str]) -> (i32, String, String) {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let code = rlmesh_cli::run_cli_in(
            settings,
            args.iter().map(OsString::from).collect(),
            &mut stdout,
            &mut stderr,
            false,
            false,
        )
        .await
        .unwrap();
        (
            code,
            String::from_utf8(stdout).unwrap(),
            String::from_utf8(stderr).unwrap(),
        )
    }

    /// A subprocess invocation of one of the crate's binaries, confined to
    /// the same directories and insulated from the developer's environment.
    pub fn command(&self, bin: &str) -> Command {
        let mut command = Command::new(bin);
        command
            .env("RLMESH_CONFIG_DIR", self.dir.path().join("config"))
            .env("RLMESH_DATA_DIR", self.dir.path().join("data"))
            .env("RLMESH_KEYCHAIN", "off")
            .env("DOCKER_CONFIG", self.dir.path().join("docker"))
            .env_remove("RLMESH_PROFILE")
            .env_remove("RLMESH_API_KEY")
            .env_remove("RLMESH_PLATFORM_URL");
        command
    }

    pub fn config_path(&self) -> PathBuf {
        self.dir.path().join("config").join("config.toml")
    }

    pub fn credentials_path(&self) -> PathBuf {
        self.dir.path().join("data").join("credentials.json")
    }

    /// Writes a config.toml with one profile pointed at the mock, pinned to
    /// the mock's token endpoint unless `token_endpoint` overrides it.
    pub fn seed_profile(
        &self,
        name: &str,
        registry_host: Option<&str>,
        token_endpoint: Option<&str>,
    ) {
        let token_endpoint =
            token_endpoint.map_or_else(|| format!("{}/oauth/token", self.mock.url), str::to_owned);
        self.write_profile(name, registry_host, Some(&token_endpoint));
    }

    /// A profile as a CLI from before endpoint pinning would have written it.
    pub fn seed_unpinned_profile(&self, name: &str) {
        self.write_profile(name, None, None);
    }

    fn write_profile(&self, name: &str, registry_host: Option<&str>, token_endpoint: Option<&str>) {
        let registry = registry_host.map_or(String::new(), |host| {
            format!("registry_host = \"{host}\"\n")
        });
        let pin = token_endpoint.map_or(String::new(), |endpoint| {
            format!("token_endpoint = \"{endpoint}\"\n")
        });
        let text = format!(
            "default_profile = \"{name}\"\n\n[profiles.{name}]\nplatform_url = \"{}\"\n{registry}{pin}\n[profiles.{name}.identity]\nuser_id = \"{USER_ID}\"\nemail = \"dev@example.com\"\norganization_id = \"org_01H\"\norganization_name = \"Acme\"\n",
            self.mock.url
        );
        std::fs::create_dir_all(self.config_path().parent().unwrap()).unwrap();
        std::fs::write(self.config_path(), text).unwrap();
    }

    /// Mints a session the mock accepts and stores it for `name`.
    pub fn seed_session(&self, name: &str, access_ttl: i64) -> (String, String) {
        let (access, refresh) = self.mock.mint_session(access_ttl);
        let mut all: serde_json::Map<String, Value> = std::fs::read(self.credentials_path())
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default();
        all.insert(
            name.to_owned(),
            json!({"access_token": access, "refresh_token": refresh}),
        );
        std::fs::create_dir_all(self.credentials_path().parent().unwrap()).unwrap();
        std::fs::write(
            self.credentials_path(),
            serde_json::to_vec_pretty(&Value::Object(all)).unwrap(),
        )
        .unwrap();
        (access, refresh)
    }

    pub fn stored_credentials(&self, name: &str) -> Option<(String, String)> {
        let all: Value =
            serde_json::from_slice(&std::fs::read(self.credentials_path()).ok()?).ok()?;
        let entry = all.get(name)?;
        Some((
            entry["access_token"].as_str()?.to_owned(),
            entry["refresh_token"].as_str()?.to_owned(),
        ))
    }

    pub fn config_text(&self) -> String {
        std::fs::read_to_string(self.config_path()).unwrap_or_default()
    }
}
