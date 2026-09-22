//! End-to-end auth flows against a mock platform: sign-in, token lifecycle,
//! locking, org switching, logout, API keys, and the docker helper.
//!
//! Every invocation passes `--profile`/`--platform` explicitly so a
//! developer's `RLMESH_PROFILE`/`RLMESH_PLATFORM_URL` cannot leak in through
//! clap's env bindings.

// Tests panic on unexpected state by design.
#![allow(clippy::unwrap_used)]

mod support;

use std::io::Write;
use std::process::Stdio;
use std::time::{Duration, Instant};

use serde_json::Value;
use support::{Harness, RevokeMode, Script, Seen, TokenReply, USER_ID, now_secs};

const RLMESH: &str = env!("CARGO_BIN_EXE_rlmesh");
const HELPER: &str = env!("CARGO_BIN_EXE_docker-credential-rlmesh");

fn login_args(url: &str) -> Vec<&str> {
    vec!["login", "--profile", "default", "--platform", url]
}

#[tokio::test]
async fn login_stores_rotated_credentials_and_identity() {
    let harness = Harness::new(Script {
        token_replies: [TokenReply::Pending, TokenReply::Grant].into(),
        ..Script::default()
    })
    .await;

    let (code, stdout, stderr) = harness.run(&login_args(&harness.mock.url)).await;

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("ABCD-1234"), "{stdout}");
    assert!(stdout.contains("Signed in as profile"), "{stdout}");
    let (access, refresh) = harness.stored_credentials("default").unwrap();
    let script = harness.mock.script();
    assert_eq!(access, script.access_token);
    assert_eq!(refresh, script.refresh_token);
    drop(script);

    let config = harness.config_text();
    assert!(config.contains("email = \"dev@example.com\""), "{config}");
    assert!(config.contains("display_name = \"Dev User\""), "{config}");
    assert!(config.contains("organization_id = \"org_01H\""), "{config}");
    assert!(config.contains("organization_name = \"Acme\""), "{config}");
    assert!(
        config.contains(&format!(
            "token_endpoint = \"{}/oauth/token\"",
            harness.mock.url
        )),
        "{config}"
    );
    let seen = harness.mock.seen();
    assert_eq!(
        seen.iter().filter(|s| **s == Seen::DeviceGrant).count(),
        2,
        "{seen:?}"
    );
    assert!(seen.contains(&Seen::Me), "{seen:?}");
}

#[tokio::test]
async fn login_honors_slow_down() {
    let harness = Harness::new(Script {
        token_replies: [TokenReply::SlowDown, TokenReply::Grant].into(),
        ..Script::default()
    })
    .await;

    let started = Instant::now();
    let (code, stdout, stderr) = harness.run(&login_args(&harness.mock.url)).await;

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    // interval 1s, then slow_down adds 5s before the next poll.
    assert!(
        started.elapsed() >= Duration::from_secs(6),
        "{:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn login_reports_denied_and_expired_codes() {
    for (reply, message) in [
        (TokenReply::AccessDenied, "declined"),
        (TokenReply::ExpiredToken, "expired"),
    ] {
        let harness = Harness::new(Script {
            token_replies: [reply].into(),
            ..Script::default()
        })
        .await;
        let (code, _stdout, stderr) = harness.run(&login_args(&harness.mock.url)).await;
        assert_eq!(code, 1);
        assert!(stderr.contains(message), "{reply:?}: {stderr}");
        assert!(harness.stored_credentials("default").is_none());
    }
}

#[tokio::test]
async fn login_without_advertised_endpoints_fails_clearly() {
    let harness = Harness::new(Script {
        advertise_endpoints: false,
        ..Script::default()
    })
    .await;

    let (code, _stdout, stderr) = harness.run(&login_args(&harness.mock.url)).await;

    assert_eq!(code, 1);
    assert!(
        stderr.contains("does not advertise sign-in endpoints"),
        "{stderr}"
    );
    assert!(!harness.mock.seen().contains(&Seen::DeviceAuth));
}

#[tokio::test]
async fn login_refuses_a_plaintext_platform() {
    let harness = Harness::new(Script::default()).await;

    let (code, _stdout, stderr) = harness
        .run(&[
            "login",
            "--profile",
            "default",
            "--platform",
            "http://platform.example.com",
        ])
        .await;

    assert_eq!(code, 1);
    assert!(stderr.contains("is not https"), "{stderr}");
    assert!(harness.mock.seen().is_empty());
}

#[tokio::test]
async fn login_warns_when_the_platform_cannot_confirm_identity() {
    let harness = Harness::new(Script {
        me_fails_once: true,
        ..Script::default()
    })
    .await;

    let (code, stdout, stderr) = harness.run(&login_args(&harness.mock.url)).await;

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("did not confirm your identity"), "{stdout}");
    assert!(harness.stored_credentials("default").is_some());
    assert!(
        !harness
            .config_text()
            .contains("[profiles.default.identity]")
    );
}

#[tokio::test]
async fn fresh_token_needs_no_network() {
    let harness = Harness::new(Script::default()).await;
    harness.seed_profile("default", None, None);
    let (access, _) = harness.seed_session("default", 3_600);

    let (code, stdout, stderr) = harness
        .run(&["token", "--json", "--profile", "default"])
        .await;

    assert_eq!(code, 0, "stderr={stderr}");
    let report: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(report["token"], access);
    assert_eq!(report["platform"], harness.mock.url);
    let expires_at = report["expiresAt"].as_str().unwrap();
    assert!(
        expires_at.ends_with('Z') && expires_at.len() == 20,
        "{expires_at}"
    );
    assert!(harness.mock.seen().is_empty(), "{:?}", harness.mock.seen());
}

#[tokio::test]
async fn expired_token_refreshes_once_and_persists_the_rotation() {
    let harness = Harness::new(Script::default()).await;
    harness.seed_profile("default", None, None);
    let (old_access, old_refresh) = harness.seed_session("default", -10);

    let (code, stdout, stderr) = harness.run(&["token", "--profile", "default"]).await;

    assert_eq!(code, 0, "stderr={stderr}");
    let (access, refresh) = harness.stored_credentials("default").unwrap();
    assert_ne!(access, old_access);
    assert_ne!(refresh, old_refresh);
    assert_eq!(stdout.trim(), access);
    assert_eq!(
        harness.mock.refresh_grants(),
        vec![Seen::RefreshGrant {
            refresh_token: old_refresh,
            organization_id: None
        }]
    );

    // The rotated token is fresh, so the next call is local.
    let seen_before = harness.mock.seen().len();
    let (code, _, _) = harness.run(&["token", "--profile", "default"]).await;
    assert_eq!(code, 0);
    assert_eq!(harness.mock.seen().len(), seen_before);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_refresh_rotates_the_session_once() {
    let harness = Harness::new(Script::default()).await;
    harness.seed_profile("default", None, None);
    harness.seed_session("default", -10);

    let spawn = || {
        harness
            .command(RLMESH)
            .args(["token", "--profile", "default"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap()
    };
    let (first, second) = (spawn(), spawn());
    let outputs = tokio::task::spawn_blocking(move || {
        (
            first.wait_with_output().unwrap(),
            second.wait_with_output().unwrap(),
        )
    })
    .await
    .unwrap();

    for output in [&outputs.0, &outputs.1] {
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    assert_eq!(outputs.0.stdout, outputs.1.stdout);
    assert_eq!(
        harness.mock.refresh_grants().len(),
        1,
        "{:?}",
        harness.mock.seen()
    );
    let (access, _) = harness.stored_credentials("default").unwrap();
    assert_eq!(String::from_utf8_lossy(&outputs.0.stdout).trim(), access);
}

#[tokio::test]
async fn a_rejected_request_is_retried_once_after_a_forced_refresh() {
    let harness = Harness::new(Script {
        evaluations_401_once: true,
        ..Script::default()
    })
    .await;
    harness.seed_profile("default", None, None);
    harness.seed_session("default", 3_600);

    let (code, stdout, stderr) = harness.run(&["eval", "list", "--profile", "default"]).await;

    assert_eq!(code, 0, "stderr={stderr}");
    assert!(stdout.contains("eval_1"), "{stdout}");
    let relevant: Vec<Seen> = harness
        .mock
        .seen()
        .into_iter()
        .filter(|seen| matches!(seen, Seen::Evaluations | Seen::RefreshGrant { .. }))
        .map(|seen| match seen {
            Seen::RefreshGrant { .. } => Seen::RefreshGrant {
                refresh_token: String::new(),
                organization_id: None,
            },
            other => other,
        })
        .collect();
    assert_eq!(
        relevant,
        vec![
            Seen::Evaluations,
            Seen::RefreshGrant {
                refresh_token: String::new(),
                organization_id: None
            },
            Seen::Evaluations
        ]
    );
}

#[tokio::test]
async fn pinned_token_endpoint_host_blocks_a_moved_provider() {
    let harness = Harness::new(Script::default()).await;
    harness.seed_profile("default", None, Some("https://id.example.com/oauth/token"));
    harness.seed_session("default", -10);

    let (code, _stdout, stderr) = harness.run(&["token", "--profile", "default"]).await;

    assert_eq!(code, 1);
    assert!(
        stderr.contains("refusing to send the stored session"),
        "{stderr}"
    );
    assert!(stderr.contains("rlmesh login"), "{stderr}");
    assert!(harness.mock.refresh_grants().is_empty());
}

#[tokio::test]
async fn an_unpinned_profile_must_sign_in_again_before_refreshing() {
    let harness = Harness::new(Script::default()).await;
    harness.seed_unpinned_profile("default");
    harness.seed_session("default", -10);

    let (code, _stdout, stderr) = harness.run(&["token", "--profile", "default"]).await;

    assert_eq!(code, 1);
    assert!(
        stderr.contains("before the CLI pinned sign-in endpoints"),
        "{stderr}"
    );
    assert!(stderr.contains("rlmesh login"), "{stderr}");
    assert!(harness.mock.refresh_grants().is_empty());

    // A fresh token still works without any pin: no refresh is attempted.
    harness.seed_session("default", 3_600);
    let (code, _, stderr) = harness.run(&["token", "--profile", "default"]).await;
    assert_eq!(code, 0, "{stderr}");
}

#[tokio::test]
async fn org_switch_uses_the_provider_id() {
    let harness = Harness::new(Script {
        orgs: vec![
            ("org_01H".to_owned(), "Acme".to_owned()),
            ("org_02H".to_owned(), "Beta".to_owned()),
        ],
        ..Script::default()
    })
    .await;
    harness.seed_profile("default", None, None);
    harness.seed_session("default", 3_600);

    let (code, stdout, stderr) = harness
        .run(&["org", "switch", "org_02H", "--profile", "default"])
        .await;

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("now uses Beta (org_02H)"), "{stdout}");
    assert!(matches!(
        harness.mock.refresh_grants().as_slice(),
        [Seen::RefreshGrant { organization_id: Some(id), .. }] if id == "org_02H"
    ));
    let config = harness.config_text();
    assert!(config.contains("organization_id = \"org_02H\""), "{config}");
    assert!(config.contains("organization_name = \"Beta\""), "{config}");

    // An organization the account does not belong to is kept out.
    let (code, _stdout, stderr) = harness
        .run(&["org", "switch", "org_nope", "--profile", "default"])
        .await;
    assert_eq!(code, 1);
    assert!(stderr.contains("kept \"org_02H\" active"), "{stderr}");

    let (code, stdout, _) = harness
        .run(&["org", "list", "--json", "--profile", "default"])
        .await;
    assert_eq!(code, 0);
    let organizations: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(organizations[1]["providerId"], "org_02H");
    assert_eq!(organizations[1]["active"], true);
}

#[tokio::test]
async fn logout_revokes_the_platform_session_best_effort() {
    for (mode, expected) in [
        (RevokeMode::NoContent, "Session revoked on the platform"),
        (RevokeMode::NotFound, "endpoint not available"),
        (RevokeMode::MethodNotAllowed, "endpoint not available"),
    ] {
        let harness = Harness::new(Script {
            revoke: mode,
            ..Script::default()
        })
        .await;
        harness.seed_profile("default", None, None);
        harness.seed_session("default", 3_600);

        let (code, stdout, stderr) = harness.run(&["logout", "--profile", "default"]).await;

        assert_eq!(code, 0, "{mode:?}: {stderr}");
        assert!(stdout.contains(expected), "{mode:?}: {stdout}");
        assert!(
            stdout.contains("Signed out of profile"),
            "{mode:?}: {stdout}"
        );
        assert!(harness.mock.seen().contains(&Seen::Revoke), "{mode:?}");
        assert!(harness.stored_credentials("default").is_none(), "{mode:?}");
    }
}

#[tokio::test]
async fn api_key_drives_token_eval_and_whoami() {
    let harness = Harness::new(Script {
        api_key: Some("sk_test_key".to_owned()),
        ..Script::default()
    })
    .await;
    let settings = harness
        .settings
        .clone()
        .with_api_key("sk_test_key", Some(&harness.mock.url));

    let (code, stdout, _) = harness
        .run_with(settings.clone(), &["token", "--json"])
        .await;
    assert_eq!(code, 0);
    let report: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(report["token"], "sk_test_key");
    assert!(report["expiresAt"].is_null());

    let (code, stdout, stderr) = harness.run_with(settings.clone(), &["eval", "list"]).await;
    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.contains("eval_1"), "{stdout}");

    let (code, stdout, _) = harness.run_with(settings, &["whoami", "--json"]).await;
    assert_eq!(code, 0, "{stdout}");
    let report: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(report["status"], "api_key");
    assert_eq!(report["verified"], true);
    assert!(report["profile"].is_null());
    assert_eq!(report["identity"]["userId"], "key_1");
    assert!(harness.mock.refresh_grants().is_empty());
}

#[tokio::test]
async fn whoami_and_profile_list_report_json_shapes() {
    let harness = Harness::new(Script::default()).await;

    let (code, stdout, _) = harness
        .run(&["whoami", "--json", "--profile", "nobody"])
        .await;
    assert_eq!(code, 1);
    let report: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(report["status"], "signed_out");
    assert_eq!(report["verified"], false);
    assert!(report["identity"].is_null());

    harness.seed_profile("default", None, None);
    harness.seed_session("default", 3_600);
    let (code, stdout, _) = harness
        .run(&["whoami", "--json", "--profile", "default"])
        .await;
    assert_eq!(code, 0, "{stdout}");
    let report: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(report["status"], "signed_in");
    assert_eq!(report["verified"], true);
    assert_eq!(report["identity"]["email"], "dev@example.com");
    assert_eq!(report["identity"]["organizationId"], "org_01H");

    let (code, stdout, _) = harness.run(&["profile", "list", "--json"]).await;
    assert_eq!(code, 0);
    let profiles: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(profiles[0]["name"], "default");
    assert_eq!(profiles[0]["status"], "signed_in");
    assert_eq!(profiles[0]["default"], true);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn registry_login_and_credential_helper_hand_docker_a_fresh_token() {
    let harness = Harness::new(Script::default()).await;
    harness.seed_profile("default", None, None);
    let (access, _) = harness.seed_session("default", 3_600);

    let mut login = harness.command(RLMESH);
    login.args(["registry", "login", "--profile", "default"]);
    let output = tokio::task::spawn_blocking(move || login.output().unwrap())
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("registry.test/acme"), "{stdout}");
    let docker: Value = serde_json::from_slice(
        &std::fs::read(harness.dir.path().join("docker").join("config.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(docker["credHelpers"]["registry.test"], "rlmesh");

    let mut get = harness.command(HELPER);
    get.arg("get").stdin(Stdio::piped()).stdout(Stdio::piped());
    let output = tokio::task::spawn_blocking(move || {
        let mut child = get.spawn().unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(b"registry.test\n")
            .unwrap();
        child.wait_with_output().unwrap()
    })
    .await
    .unwrap();
    assert!(output.status.success());
    let credential: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(credential["ServerURL"], "registry.test");
    assert_eq!(credential["Username"], USER_ID);
    assert_eq!(credential["Secret"], access);

    let mut list = harness.command(HELPER);
    list.arg("list").stdin(Stdio::null()).stdout(Stdio::piped());
    let output = tokio::task::spawn_blocking(move || list.output().unwrap())
        .await
        .unwrap();
    let hosts: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(hosts["registry.test"], USER_ID);

    // An unknown host is docker's "not found" sentinel, not an error.
    let mut get = harness.command(HELPER);
    get.arg("get").stdin(Stdio::piped()).stdout(Stdio::piped());
    let output = tokio::task::spawn_blocking(move || {
        let mut child = get.spawn().unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(b"other.test\n")
            .unwrap();
        child.wait_with_output().unwrap()
    })
    .await
    .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("credentials not found"),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let _ = now_secs();
}
