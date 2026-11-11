//! Tests of the image checks.

use super::*;
use rlmesh_proto::CURRENT_WORKFLOW_EDITION;

/// What a 0.2 image exposes: its session's base port and the next three.
const PORTS: &[&str] = &["7447/tcp", "7448/tcp", "7449/tcp", "7450/tcp"];

fn image(entrypoint: &[&str], cmd: &[&str], env: &[&str], ports: &[&str]) -> ImageConfig {
    ImageConfig {
        os: "linux".to_owned(),
        architecture: "amd64".to_owned(),
        entrypoint: entrypoint.iter().map(|s| (*s).to_owned()).collect(),
        cmd: cmd.iter().map(|s| (*s).to_owned()).collect(),
        env: env.iter().map(|s| (*s).to_owned()).collect(),
        exposed_ports: ports.iter().map(|s| (*s).to_owned()).collect(),
        labels: BTreeMap::new(),
    }
}

#[test]
fn serve_command_table() {
    struct Case {
        name: &'static str,
        entrypoint: &'static [&'static str],
        cmd: &'static [&'static str],
        env: &'static [&'static str],
        ports: &'static [&'static str],
        kind: Option<Kind>,
        failed: &'static [&'static str],
        warnings: &'static [&'static str],
        not_checked: &'static [&'static str],
    }
    let cases = [
        Case {
            name: "model image, clean",
            entrypoint: &["python", "-m", "rlmesh.serve", "pkg:Policy"],
            cmd: &[],
            env: &[],
            ports: PORTS,
            kind: Some(Kind::Model),
            failed: &[],
            warnings: &[],
            not_checked: &[],
        },
        Case {
            name: "env image via sh -c, no EXPOSE, baked env address",
            entrypoint: &["sh", "-c", "python -m rlmesh.serve --env pkg:Env"],
            cmd: &[],
            env: &["RLMESH_ADDRESS=0.0.0.0:6000"],
            ports: &[],
            kind: Some(Kind::Env),
            failed: &[],
            warnings: &[
                "env: image-level ENV",
                "ports: EXPOSE 6000-6003 is incomplete",
            ],
            not_checked: &[],
        },
        Case {
            name: "--address baked in",
            entrypoint: &["python"],
            cmd: &[
                "-m",
                "rlmesh.serve",
                "pkg:Policy",
                "--address",
                "0.0.0.0:7000",
            ],
            env: &[],
            ports: &["7000/tcp", "7001/tcp", "7002/tcp", "7003/tcp"],
            kind: None,
            failed: &[],
            warnings: &["entrypoint: --address is baked"],
            not_checked: &[],
        },
        Case {
            name: "--env on a model label",
            entrypoint: &["python", "-m", "rlmesh.serve", "--env=pkg:Env"],
            cmd: &[],
            env: &[],
            ports: PORTS,
            kind: Some(Kind::Model),
            failed: &["--env present=true but the describe label says kind model"],
            warnings: &[],
            not_checked: &[],
        },
        Case {
            name: "no target",
            entrypoint: &[
                "python",
                "-m",
                "rlmesh.serve",
                "--workflow-edition",
                "2026.06",
            ],
            cmd: &[],
            env: &[],
            ports: PORTS,
            kind: None,
            failed: &["names no module:Class"],
            warnings: &[],
            not_checked: &[],
        },
        Case {
            name: "custom entrypoint",
            entrypoint: &["python", "serve.py"],
            cmd: &[],
            env: &[],
            ports: PORTS,
            kind: None,
            failed: &[],
            warnings: &[],
            not_checked: &["custom entrypoint"],
        },
        Case {
            name: "no command at all",
            entrypoint: &[],
            cmd: &[],
            env: &[],
            ports: &[],
            kind: None,
            failed: &[],
            warnings: &["no Entrypoint/Cmd"],
            not_checked: &[],
        },
        Case {
            name: "a 0.1 image's port only",
            entrypoint: &["python", "-m", "rlmesh.serve", "pkg:Policy"],
            cmd: &[],
            env: &[],
            ports: &["50051/tcp", "7447/tcp"],
            kind: None,
            failed: &[],
            warnings: &["ports: EXPOSE 7447-7450 is incomplete (missing 7448, 7449, 7450)"],
            not_checked: &[],
        },
    ];
    for case in cases {
        let config = image(case.entrypoint, case.cmd, case.env, case.ports);
        let report = check_serve_command(&config, case.kind);
        for (bucket, expected) in [
            (&report.failed, case.failed),
            (&report.warnings, case.warnings),
            (&report.not_checked, case.not_checked),
        ] {
            assert_eq!(bucket.len(), expected.len(), "{}: {report:?}", case.name);
            for (message, needle) in bucket.iter().zip(expected) {
                assert!(message.contains(needle), "{}: {message}", case.name);
            }
        }
    }
}

#[test]
fn the_port_check_follows_the_session_address() {
    let command = ["python", "-m", "rlmesh.serve", "pkg:Policy"];
    let report = check_serve_command(&image(&command, &[], &[], PORTS), None);
    assert!(
        report
            .passed
            .contains(&"ports: EXPOSE 7447-7450".to_owned()),
        "{report:?}"
    );
    let missing = check_serve_command(&image(&command, &[], &[], &[]), None);
    let warning = &missing.warnings[0];
    assert!(warning.contains("HOST:7447#KEY/NAME"), "{warning}");
    assert!(!warning.contains("50051"), "{warning}");
    assert_eq!(SERVE_PORT, 7447);
    assert_eq!(SESSION_PORTS, 4);
}

#[test]
fn exec_form_keeps_docker_argument_boundaries() {
    // A JSON value with spaces is one word; the first bare word is the target.
    let config = image(
        &["python", "-m", "rlmesh.serve"],
        &["pkg:Policy", "--kwargs-json", "{\"a\": 1}"],
        &[],
        &[],
    );
    let command = parse_serve_command(&config);
    assert_eq!(command.serve, Some(2));
    assert_eq!(command.target.as_deref(), Some("pkg:Policy"));
    assert_eq!(command.argv.len(), 6);
    // A whitespace-bearing exec-form element is NOT a shell script: it stays one word.
    let config = image(&["python -m rlmesh.serve pkg:Policy"], &[], &[], &[]);
    assert_eq!(parse_serve_command(&config).serve, None);
}

#[test]
fn shell_form_splits_the_script_and_appends_the_rest() {
    let config = image(
        &[
            "/bin/sh",
            "-c",
            "uv run python -m rlmesh.serve --env 'pkg:Env' --framework torch",
        ],
        &[],
        &[],
        &[],
    );
    let command = parse_serve_command(&config);
    assert_eq!(command.serve, Some(4));
    assert!(command.env);
    assert_eq!(command.target.as_deref(), Some("pkg:Env"));
    assert_eq!(command.kind(), Some(Kind::Env));
    assert_eq!(command.address, None);
    // `bash -c 'exec python "$@"' --` hands Cmd through unchanged.
    let config = image(
        &[
            "bash",
            "-c",
            "source /sim/setup.sh && exec /sim/bin/python3 \"$@\"",
            "--",
        ],
        &["-m", "rlmesh.serve", "--env", "franka_reach:FrankaReach"],
        &[],
        &[],
    );
    let command = parse_serve_command(&config);
    assert_eq!(command.target.as_deref(), Some("franka_reach:FrankaReach"));
    assert!(command.env);
}

#[test]
fn shell_words_honor_posix_quoting() {
    assert_eq!(
        shell_words(
            r#"python -m rlmesh.serve pkg:Policy --kwargs-json '{"a": 1}' --x "a b" c\ d "$@""#
        ),
        [
            "python",
            "-m",
            "rlmesh.serve",
            "pkg:Policy",
            "--kwargs-json",
            r#"{"a": 1}"#,
            "--x",
            "a b",
            "c d",
            "$@"
        ]
    );
    assert_eq!(
        shell_words(r#""esc \" \\ \$x" 'lit \n'"#),
        ["esc \" \\ $x", "lit \\n"]
    );
    assert_eq!(shell_words("  "), Vec::<String>::new());
    assert_eq!(shell_words("''"), [""]);
}

#[test]
fn platform_must_be_linux_amd64() {
    let mut config = image(&["python"], &[], &[], &[]);
    assert!(check_platform(&config).ok());
    config.architecture = "arm64".to_owned();
    let report = check_platform(&config);
    assert!(report.failed[0].contains("linux/arm64"), "{report:?}");
    config.architecture.clear();
    assert_eq!(check_platform(&config).not_checked.len(), 1);
}

#[test]
fn docker_inspect_parses_nulls_and_ports() {
    let json = r#"[{"Os":"linux","Architecture":"amd64","Config":{"Entrypoint":null,
        "Cmd":["python","-m","rlmesh.serve","pkg:Policy"],"Env":["PATH=/usr/bin"],
        "ExposedPorts":{"7447/tcp":{},"7448/udp":{}},"Labels":null}}]"#;
    let config = ImageConfig::from_docker_inspect(json).unwrap();
    assert_eq!(config.cmd.len(), 4);
    assert!(config.entrypoint.is_empty());
    assert!(config.exposes_port(7447));
    assert!(
        !config.exposes_port(7448),
        "a UDP port is not a session port"
    );
    assert_eq!(config.env_value("PATH"), Some("/usr/bin"));
    assert!(ImageConfig::from_docker_inspect("[]").is_err());
}

#[test]
fn oci_config_parses_like_docker_inspect() {
    // What `docker buildx imagetools inspect REF --format '{{json .Image}}'` prints.
    let json = r#"{"architecture":"amd64","os":"linux","config":{"ExposedPorts":
        {"7447/tcp":{}},"Env":["CUDA_VERSION=12.4.1"],"Cmd":["python","-m",
        "rlmesh.serve","pkg:Policy"],"Labels":{"dev.rlmesh.package":"{}"}},
        "rootfs":{"type":"layers","diff_ids":[]}}"#;
    let config = ImageConfig::from_oci_config(json).unwrap();
    assert_eq!(
        (config.os.as_str(), config.architecture.as_str()),
        ("linux", "amd64")
    );
    assert_eq!(config.env_value("CUDA_VERSION"), Some("12.4.1"));
    assert!(config.exposes_port(SERVE_PORT));
    assert_eq!(config.labels[PACKAGE_LABEL], "{}");
    assert_eq!(parse_serve_command(&config).kind(), Some(Kind::Model));
    assert!(ImageConfig::from_oci_config("\"linux\"").is_err());
}

#[test]
fn check_image_validates_the_variant_blocks() {
    let mut config = image(
        &["python", "-m", "rlmesh.serve", "pkg:Policy"],
        &[],
        &["CUDA_VERSION=12.4.1", "NVIDIA_REQUIRE_CUDA=cuda>=12.4"],
        PORTS,
    );
    config.labels.insert(
        PACKAGE_LABEL.to_owned(),
        serde_json::json!({
            "schemaVersion": 1,
            "variant": {"key": "torch-cuda12", "requires": {"accel.vendor": "nvidia",
                "accel.cuda": ">=12.2", "accel.vram": ">=24Gi"}},
            "profiles": [{"key": "osmesa", "default": true}, {"key": "egl", "default": true}],
        })
        .to_string(),
    );
    let report = check_image(&config, None);
    let variant: Vec<&String> = report
        .failed
        .iter()
        .filter(|m| m.starts_with("variant:") || m.starts_with("profiles:"))
        .collect();
    assert_eq!(variant.len(), 2, "{report:#?}");
    assert!(
        variant[0].contains("accel.vram \">=24Gi\" is not a quantity like \"16Gi\""),
        "{variant:?}"
    );
    assert!(variant[1].contains("at most one may be"));
    assert!(
        report
            .warnings
            .iter()
            .any(|m| m.contains("admits drivers older than the image's CUDA 12.4")),
        "{report:#?}"
    );
    // An image without a variant block reports what inference supplies.
    config.labels.clear();
    let report = check_image(&config, None);
    assert!(
        report
            .passed
            .iter()
            .any(|m| m.starts_with("variant: none declared; inferred from CUDA_VERSION=12.4.1")),
        "{report:#?}"
    );
    assert!(
        report.passed.iter().any(|m| m == "rows: default"),
        "{report:#?}"
    );
}

fn label(kind: &str, runtime: serde_json::Value) -> String {
    serde_json::json!({
        "schema_version": 1,
        "kind": kind,
        "target": {"entrypoint": "pkg:Policy", "qualname": "pkg:Policy"},
        "runtime": runtime,
    })
    .to_string()
}

#[test]
fn describe_label_wrapper_is_read() {
    let parsed = parse_describe_label(&label(
        "model",
        serde_json::json!({"supported_workflow_editions": ["2026.06"]}),
    ))
    .unwrap();
    assert_eq!(parsed.kind, Some(Kind::Model));
    assert_eq!(parsed.target.as_deref(), Some("pkg:Policy"));
    assert_eq!(
        parsed.runtime.session_offer(),
        Some(SessionOffer::new(&["2026.06"]))
    );
    assert!(parse_describe_label("{nope").is_err());
    assert!(parse_describe_label(r#"{"schema_version":2,"kind":"model"}"#).is_err());
    assert!(parse_describe_label(r#"{"schema_version":1,"kind":"thing"}"#).is_err());
    assert!(
        parse_describe_label(r#"{"schema_version":1,"kind":"env"}"#)
            .unwrap()
            .runtime
            .session_offer()
            .is_none()
    );
}

#[test]
fn missing_describe_label_is_a_note_not_a_failure() {
    let config = image(
        &["python", "-m", "rlmesh.serve", "pkg:Policy"],
        &[],
        &[],
        &[],
    );
    let (report, label) = check_labels(&config);
    assert!(label.is_none());
    assert!(report.ok());
    assert!(report.not_checked[0].contains("handshake"), "{report:?}");
}

#[test]
fn missing_describe_label_on_a_custom_entrypoint_points_at_the_native_describe() {
    let config = image(&["/usr/local/bin/chrono_env"], &["0.0.0.0:50051"], &[], &[]);
    let (report, label) = check_labels(&config);
    assert!(label.is_none());
    assert!(report.ok());
    let note = &report.not_checked[0];
    assert!(!note.contains("rlmesh.serve"), "{note}");
    assert!(note.contains("C ABI"), "{note}");
    assert!(note.contains("rlmesh_env_describe_json"), "{note}");
}

#[test]
fn edition_table() {
    let platform = platform_offer(&["2026.06-0.1.0-rc.13".to_owned(), "2026.06".to_owned()]);
    let offer =
        |generation: Option<&str>, editions: &[&str], preferred: Option<&str>| RuntimeOffer {
            protocol_generation: generation.map(str::to_owned),
            supported_workflow_editions: editions.iter().map(|e| (*e).to_owned()).collect(),
            preferred_workflow_edition: preferred.map(str::to_owned),
            ..RuntimeOffer::default()
        };
    // Sealed base in common: a newer or older prerelease still meets the platform.
    let report = check_editions(
        &offer(
            Some(PROTOCOL_GENERATION),
            &["2026.06-0.1.0-rc.12", "2026.06"],
            Some("2026.06"),
        ),
        Some(Kind::Model),
        &platform,
    );
    assert!(report.ok(), "{report:?}");
    assert!(report.passed.iter().any(|m| m.contains("run at 2026.06")));
    // Nothing in common: refused, naming both sides.
    let report = check_editions(
        &offer(Some(PROTOCOL_GENERATION), &["2031.01"], None),
        Some(Kind::Env),
        &platform,
    );
    assert_eq!(report.failed.len(), 1, "{report:?}");
    assert!(report.failed[0].contains("share no workflow edition"));
    // Wrong generation is a failure even when editions agree.
    let report = check_editions(
        &offer(Some("rlmesh-wire-v2"), &["2026.06"], None),
        None,
        &platform,
    );
    assert!(
        report.failed[0].contains("protocol generation"),
        "{report:?}"
    );
    let legacy = check_editions(
        &offer(Some("rlmesh-wire-v1"), &["2026.06"], None),
        Some(Kind::Env),
        &platform,
    );
    assert!(
        legacy.failed.iter().any(|m| m.contains("0.1 runner")),
        "{legacy:?}"
    );
    // No fields at all: not checked, twice (generation and editions).
    let report = check_editions(&offer(None, &[], None), None, &platform);
    assert!(report.ok());
    assert_eq!(report.not_checked.len(), 2, "{report:?}");
    // A declaration the image's own rlmesh refused: fails without Python.
    let mut impossible = offer(Some(PROTOCOL_GENERATION), &["2026.06"], Some("2026.06"));
    impossible.workflow_edition_error = Some("cannot drive 1999.01".to_owned());
    let report = check_editions(&impossible, None, &platform);
    assert!(
        report.failed.iter().any(|m| m.contains("refuse to start")),
        "{report:?}"
    );
}

#[test]
fn platform_offer_for_version_knows_only_this_build() {
    let own = platform_offer_for_version(env!("CARGO_PKG_VERSION")).unwrap();
    assert_eq!(own.editions, supported_workflow_editions());
    // PyPI's spelling of the same build.
    let pep440 = env!("CARGO_PKG_VERSION").replace("-rc.", "rc");
    assert_eq!(platform_offer_for_version(&pep440), Some(own));
    // Another release's retained list is not this build's to guess: rc.12
    // offered only its cohort, so adding this build's sealed base would
    // pass an image the real handshake refuses.
    assert_eq!(platform_offer_for_version("0.1.0-rc.12"), None);
    assert_eq!(platform_offer_for_version("0.0.1"), None);
    assert_eq!(platform_offer_for_version("99.0.0"), None);
}

#[test]
fn version_spellings_normalize() {
    for (raw, expected) in [
        ("0.1.0-rc.13", "0.1.0-rc.13"),
        ("0.1.0rc13", "0.1.0-rc.13"),
        ("0.1.0-rc13", "0.1.0-rc.13"),
        ("0.1.0.rc.13", "0.1.0-rc.13"),
        ("v0.1.0RC13", "0.1.0-rc.13"),
        ("0.1.0", "0.1.0"),
        ("0.1.0a1", "0.1.0-a.1"),
        ("0.1.0-alpha.1", "0.1.0-a.1"),
        ("0.1.0-beta.3", "0.1.0-b.3"),
        ("0.1.0-dev.abc123", "0.1.0-dev.abc123"),
    ] {
        assert_eq!(normalize_version(raw), expected, "{raw}");
    }
}

#[test]
fn host_baked_label_is_refused() {
    let runtime = |os: &str, arch: &str| RuntimeOffer {
        os: Some(os.to_owned()),
        arch: Some(arch.to_owned()),
        ..RuntimeOffer::default()
    };
    let report = check_label_runtime(&runtime("macos", "arm64"));
    assert!(
        report.failed[0].contains("generated on macos"),
        "{report:?}"
    );
    assert!(report.warnings[0].contains("arm64"), "{report:?}");
    assert!(check_label_runtime(&runtime("linux", "x86_64")).ok());
    assert!(
        check_label_runtime(&runtime("linux", "amd64"))
            .warnings
            .is_empty()
    );
    // An older envelope without the fields is not judged.
    assert_eq!(
        check_label_runtime(&RuntimeOffer::default()),
        CheckReport::default()
    );
    // And it is wired into check_image off the label's runtime block.
    let mut config = image(
        &["python", "-m", "rlmesh.serve", "pkg:Policy"],
        &[],
        &[],
        &[],
    );
    config.labels.insert(
        DESCRIBE_LABEL.to_owned(),
        label("model", serde_json::json!({"os": "macos", "arch": "arm64"})),
    );
    let report = check_image(&config, None);
    assert!(
        report
            .failed
            .iter()
            .any(|m| m.contains("generated on macos")),
        "{report:?}"
    );
}

#[test]
fn image_declared_edition_overrides_the_label() {
    let mut config = image(
        &[
            "python",
            "-m",
            "rlmesh.serve",
            "pkg:Policy",
            "--workflow-edition",
            "2020.01",
        ],
        &[],
        &[],
        PORTS,
    );
    config.labels.insert(
        DESCRIBE_LABEL.to_owned(),
        label(
            "model",
            serde_json::json!({
                "protocol_generation": PROTOCOL_GENERATION,
                "supported_workflow_editions": ["2026.06-0.1.0-rc.13", "2026.06"],
                "preferred_workflow_edition": "2026.06",
            }),
        ),
    );
    let platform = platform_offer(&["2026.06".to_owned()]);
    // The flag is a ceiling below the one edition both can drive: refused,
    // even though the label alone would have passed.
    assert_eq!(
        declared_workflow_edition(&config),
        Some(Some("2020.01".to_owned()))
    );
    let report = check_image(&config, Some(&platform));
    assert!(
        report
            .failed
            .iter()
            .any(|m| m.contains("share no workflow edition")),
        "{report:?}"
    );
    // An empty ENV declares none: the peer floats to its newest, and a
    // sealed base in common lets the session run.
    config.env.push(format!("{WORKFLOW_EDITION_ENV}="));
    assert_eq!(declared_workflow_edition(&config), Some(None));
    let report = check_image(&config, Some(&platform));
    assert!(report.ok(), "{report:?}");
    assert!(
        report
            .passed
            .iter()
            .any(|m| m.contains("floating to its newest")),
        "{report:?}"
    );
    // A non-empty ENV wins over the flag.
    config.env.pop();
    config.env.push(format!("{WORKFLOW_EDITION_ENV}=2026.06"));
    assert_eq!(
        declared_workflow_edition(&config),
        Some(Some("2026.06".to_owned()))
    );
    assert!(check_image(&config, Some(&platform)).ok());
}

#[test]
fn check_image_ties_the_label_to_the_command() {
    let mut config = image(
        &["python", "-m", "rlmesh.serve", "pkg:Other"],
        &[],
        &[],
        PORTS,
    );
    config.labels.insert(
        DESCRIBE_LABEL.to_owned(),
        label(
            "model",
            serde_json::json!({
                "protocol_generation": PROTOCOL_GENERATION,
                "supported_workflow_editions": supported_workflow_editions(),
                "preferred_workflow_edition": CURRENT_WORKFLOW_EDITION,
            }),
        ),
    );
    let report = check_image(&config, Some(&SessionOffer::this_build(None)));
    assert_eq!(report.failed.len(), 1, "{report:?}");
    assert!(report.failed[0].contains("serves \"pkg:Other\""));
    config.cmd.clear();
    config.entrypoint[3] = "pkg:Policy".to_owned();
    let report = check_image(&config, Some(&SessionOffer::this_build(None)));
    assert!(report.ok(), "{report:?}");
    assert!(report.not_checked.is_empty(), "{report:?}");
}
