//! Fixtures the variant tests share.

use serde_json::{Map, Value, json};

use crate::image_check::{CheckReport, DESCRIBE_LABEL, ImageConfig, PACKAGE_LABEL};

pub(super) fn object(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        other => panic!("not an object: {other}"),
    }
}

pub(super) fn image(env: &[&str], package: Option<Value>) -> ImageConfig {
    let mut config = ImageConfig {
        os: "linux".to_owned(),
        architecture: "amd64".to_owned(),
        env: env.iter().map(|e| (*e).to_owned()).collect(),
        ..ImageConfig::default()
    };
    if let Some(package) = package {
        config
            .labels
            .insert(PACKAGE_LABEL.to_owned(), package.to_string());
    }
    config
}

pub(super) fn with_torch(mut config: ImageConfig, versions: Value) -> ImageConfig {
    config.labels.insert(
        DESCRIBE_LABEL.to_owned(),
        json!({"schema_version": 1, "runtime": {"framework_versions": versions}}).to_string(),
    );
    config
}

/// Assert each bucket holds exactly one message per needle, in order.
pub(super) fn assert_buckets(report: &CheckReport, failed: &[&str], warnings: &[&str]) {
    for (bucket, expected) in [(&report.failed, failed), (&report.warnings, warnings)] {
        assert_eq!(bucket.len(), expected.len(), "{report:#?}");
        for (message, needle) in bucket.iter().zip(expected) {
            assert!(message.contains(needle), "{needle:?} not in {message:?}");
        }
    }
}
