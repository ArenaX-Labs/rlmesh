//! Tests of the image checks.

use super::*;
use rlmesh_proto::CURRENT_WORKFLOW_EDITION;

const PORTS: &[&str] = &["7447/tcp"];

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
            warnings: &["env: image-level ENV", "ports: EXPOSE 6000 is missing"],
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
            ports: &["7000/tcp"],
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
            entrypoint: &["python", "-m", "rlmesh.serve"],
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
            ports: &["50051/tcp"],
            kind: None,
            failed: &[],
            warnings: &["ports: EXPOSE 7447 is missing"],
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
        report.passed.contains(&"ports: EXPOSE 7447".to_owned()),
        "{report:?}"
    );
    let missing = check_serve_command(&image(&command, &[], &[], &[]), None);
    let warning = &missing.warnings[0];
    assert!(warning.contains("HOST:7447#KEY/NAME"), "{warning}");
    assert!(!warning.contains("50051"), "{warning}");
    assert_eq!(SERVE_PORT, 7447);
    let older = ["7447/tcp", "7448/tcp", "7449/tcp", "7450/tcp"];
    let report = check_serve_command(&image(&command, &[], &[], &older), None);
    assert!(report.warnings.is_empty(), "{report:?}");
}

#[test]
fn exec_form_keeps_docker_argument_boundaries() {
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
        serde_json::json!({"workflow_edition": "2026.10"}),
    ))
    .unwrap();
    assert_eq!(parsed.kind, Some(Kind::Model));
    assert_eq!(parsed.target.as_deref(), Some("pkg:Policy"));
    assert_eq!(parsed.runtime.workflow_edition.as_deref(), Some("2026.10"));
    assert!(parse_describe_label("{nope").is_err());
    assert!(parse_describe_label(r#"{"schema_version":2,"kind":"model"}"#).is_err());
    assert!(parse_describe_label(r#"{"schema_version":1,"kind":"thing"}"#).is_err());
    assert!(
        parse_describe_label(r#"{"schema_version":1,"kind":"env"}"#)
            .unwrap()
            .runtime
            .workflow_edition
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
    let platform = EditionTarget::platform(platform_editions(&[
        " 2026.06".to_owned(),
        "2026.10".to_owned(),
        " ".to_owned(),
    ]));
    assert_eq!(platform.editions, ["2026.06", "2026.10"]);
    let runtime = |generation: Option<&str>, edition: Option<&str>| DescribeRuntime {
        protocol_generation: generation.map(str::to_owned),
        session_wire_revision: Some(WIRE_REVISION),
        workflow_edition: edition.map(str::to_owned),
        ..DescribeRuntime::default()
    };
    let report = check_editions(
        &runtime(Some(PROTOCOL_GENERATION), Some("2026.06")),
        &platform,
    );
    assert!(report.ok() && report.warnings.is_empty(), "{report:?}");
    assert!(
        report.passed.contains(
            &"editions: workflow edition 2026.06, which the target platform knows".to_owned()
        ),
        "{report:?}"
    );
    let report = check_editions(
        &runtime(Some(PROTOCOL_GENERATION), Some("2031.01")),
        &platform,
    );
    assert_eq!(
        report.failed,
        [
            "editions: the image declares workflow edition 2031.01, which the target platform \
          does not know (it knows [\"2026.06\", \"2026.10\"]); run it on an rlmesh release \
          that knows 2031.01, or rebuild the image on a release whose edition the target \
          platform knows"
        ]
    );
    let report = check_editions(&runtime(Some(PROTOCOL_GENERATION), None), &platform);
    assert!(report.ok(), "{report:?}");
    assert!(
        report
            .not_checked
            .iter()
            .any(|m| m.contains("no workflow_edition")),
        "{report:?}"
    );
    let report = check_editions(&runtime(Some("rlmesh-wire-v2"), Some("2026.10")), &platform);
    assert!(
        report.failed[0].contains("protocol generation"),
        "{report:?}"
    );
    let legacy = check_editions(&runtime(Some("rlmesh-wire-v1"), Some("2026.10")), &platform);
    assert!(
        legacy.failed.iter().any(|m| m.contains("0.1 runner")),
        "{legacy:?}"
    );
    let on = |revision| DescribeRuntime {
        session_wire_revision: revision,
        ..runtime(Some(PROTOCOL_GENERATION), Some("2026.10"))
    };
    let unchecked = check_editions(&on(Some(WIRE_REVISION + 1)), &platform);
    assert!(unchecked.ok(), "{unchecked:?}");
    assert!(
        unchecked
            .not_checked
            .iter()
            .any(|m| m.contains("the target platform names no wire revision")),
        "{unchecked:?}"
    );
    let this_build = EditionTarget::this_build();
    let report = check_editions(&on(Some(WIRE_REVISION + 1)), &this_build);
    assert!(
        report.failed.iter().any(|m| m.contains(&format!(
            "the image speaks session wire revision {} (a later pre-release)",
            WIRE_REVISION + 1
        ))),
        "{report:?}"
    );
    let report = check_editions(&on(Some(1)), &this_build);
    assert!(
        report
            .failed
            .iter()
            .any(|m| m.contains("session wire revision 1 (negotiated editions")),
        "{report:?}"
    );
    let report = check_editions(&on(None), &this_build);
    assert!(report.ok(), "{report:?}");
    assert!(
        report
            .warnings
            .iter()
            .any(|m| m.contains("no session_wire_revision")),
        "{report:?}"
    );
}

#[test]
fn platform_editions_for_version_knows_only_this_build() {
    let own = platform_editions_for_version(env!("CARGO_PKG_VERSION")).unwrap();
    assert_eq!(own, [CURRENT_WORKFLOW_EDITION]);
    let pep440 = env!("CARGO_PKG_VERSION").replace("-rc.", "rc");
    assert_eq!(platform_editions_for_version(&pep440), Some(own));
    assert_eq!(platform_editions_for_version("0.1.0-rc.12"), None);
    assert_eq!(platform_editions_for_version("0.0.1"), None);
    assert_eq!(platform_editions_for_version("99.0.0"), None);
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
    let runtime = |os: &str, arch: &str| DescribeRuntime {
        os: Some(os.to_owned()),
        arch: Some(arch.to_owned()),
        ..DescribeRuntime::default()
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
    assert_eq!(
        check_label_runtime(&DescribeRuntime::default()),
        CheckReport::default()
    );
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
        &["python", "-m", "rlmesh.serve", "pkg:Policy"],
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
                "session_wire_revision": WIRE_REVISION,
                "workflow_edition": "2026.06",
            }),
        ),
    );
    let platform = EditionTarget::platform(vec!["2026.06".to_owned()]);
    assert_eq!(declared_workflow_edition(&config), None);
    assert!(check_image(&config, Some(&platform)).ok());
    config.env.push(format!("{WORKFLOW_EDITION_ENV}= "));
    assert_eq!(declared_workflow_edition(&config), None);
    assert!(check_image(&config, Some(&platform)).ok());
    config.env.pop();
    config.env.push(format!("{WORKFLOW_EDITION_ENV}=2020.01"));
    assert_eq!(
        declared_workflow_edition(&config),
        Some("2020.01".to_owned())
    );
    let report = check_image(&config, Some(&platform));
    assert!(
        report
            .failed
            .iter()
            .any(|m| m.contains("declares workflow edition 2020.01")),
        "{report:?}"
    );
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
                "session_wire_revision": WIRE_REVISION,
                "workflow_edition": CURRENT_WORKFLOW_EDITION,
            }),
        ),
    );
    let report = check_image(&config, Some(&EditionTarget::this_build()));
    assert_eq!(report.failed.len(), 1, "{report:?}");
    assert!(report.failed[0].contains("serves \"pkg:Other\""));
    config.cmd.clear();
    config.entrypoint[3] = "pkg:Policy".to_owned();
    let report = check_image(&config, Some(&EditionTarget::this_build()));
    assert!(report.ok(), "{report:?}");
    assert!(report.not_checked.is_empty(), "{report:?}");
}

#[test]
fn a_local_check_names_this_build_and_no_platform() {
    let mut config = image(
        &["python", "-m", "rlmesh.serve", "pkg:Policy"],
        &[],
        &[],
        &[],
    );
    config.architecture = "arm64".to_owned();
    config.labels.insert(
        DESCRIBE_LABEL.to_owned(),
        label(
            "model",
            serde_json::json!({
                "protocol_generation": PROTOCOL_GENERATION,
                "session_wire_revision": WIRE_REVISION,
                "workflow_edition": "2099.01",
                "os": "macos",
                "arch": "arm64",
            }),
        ),
    );
    config.labels.insert(
        PACKAGE_LABEL.to_owned(),
        r#"{"schemaVersion": 2}"#.to_owned(),
    );
    let this_build = EditionTarget::this_build();
    let report = check_image(&config, Some(&this_build));
    let refused = report
        .failed
        .iter()
        .find(|m| m.starts_with("editions:"))
        .unwrap_or_else(|| panic!("{report:?}"));
    assert_eq!(
        refused,
        &format!(
            "editions: the image declares workflow edition 2099.01, which this rlmesh build \
             does not know (it knows [{CURRENT_WORKFLOW_EDITION:?}]); run it on an rlmesh \
             release that knows 2099.01, or rebuild the image on a release whose edition this \
             rlmesh build knows"
        )
    );
    let every = report
        .failed
        .iter()
        .chain(&report.warnings)
        .chain(&report.not_checked)
        .chain(&report.passed);
    for message in every {
        assert!(!message.contains("the platform"), "{message}");
        assert!(!message.contains("managed platform"), "{message}");
    }
    config.labels.clear();
    let report = check_image(&config, Some(&this_build));
    assert!(
        report
            .not_checked
            .iter()
            .any(|m| m.contains("the session handshake checks them when the image runs")),
        "{report:?}"
    );
}
