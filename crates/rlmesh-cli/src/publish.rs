//! `rlmesh registry publish`: assemble pushed per-variant images into one OCI
//! image index and push it as a version. The managed platform reads the index
//! as the version and its non-attestation children as compute variants, each
//! declared by the `variant` block of its `dev.rlmesh.package` label (see
//! [`crate::variant`]).
//!
//! Registry access goes through `docker buildx imagetools` (inspect to read
//! each source, create to assemble and push), so docker's own credential
//! helpers, including `docker-credential-rlmesh`, authenticate it. Sources are
//! pinned by digest before the index is created, so a tag moving underneath a
//! publish cannot swap a child. A version tag (TARGET, `--tag`) is pushed
//! only over a tag the registry says is absent, or one already holding the
//! same index (which is left alone); a different index, or a tag that cannot
//! be read, needs `--force`. The channel tag moves. The index is pushed to
//! TARGET alone and checked as the registry stores it; only then are the
//! other tags pointed at TARGET's digest, so every tag is one manifest.

mod imagetools;
mod source;
mod table;
mod tags;
#[cfg(test)]
mod testing;
mod validate;

use std::io::Write;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::cli::PublishArgs;
use crate::image_check::PACKAGE_LABEL;
use crate::render::Style;
use imagetools::{DockerImagetools, Imagetools, existing_manifest};
use source::{attestation_errors, docker_list_error, resolve_source};
use table::write_table;
use tags::{Existing, TagState, create_args, report_tags, same_index, tag_states, target_tags};
use validate::{
    VariantImage, annotation_dropped, check_variants, index_disagreements, index_package,
};

pub(crate) fn publish(args: &PublishArgs, stdout: &mut impl Write, style: Style) -> Result<i32> {
    publish_with(&DockerImagetools, args, stdout, style)
}

pub(crate) fn publish_with(
    tools: &impl Imagetools,
    args: &PublishArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<i32> {
    let (target, tags) = target_tags(args)?;
    let mut warnings = Vec::new();
    let mut annotation = match &args.index_package {
        Some(path) => {
            let raw = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            let (annotation, index_warnings) = index_package(&raw, &path.display().to_string())?;
            warnings.extend(index_warnings);
            Some(annotation)
        }
        None => None,
    };

    let mut children = Vec::new();
    let mut errors = Vec::new();
    for source in &args.sources {
        match resolve_source(tools, source) {
            Ok(child) => children.push(child),
            Err(error) => errors.push(format!("{error:#}")),
        }
    }
    errors.extend(attestation_errors(&children));
    if annotation.is_some()
        && children.len() == args.sources.len()
        && let Some(error) = docker_list_error(&children)
    {
        errors.push(error);
    }
    // A lone index source is republished as is, its own annotation included,
    // so that annotation is the version's and is checked like --index-package.
    // Anywhere else, docker assembles a new index without it.
    let inherits = annotation.is_none() && args.sources.len() == 1;
    for child in &children {
        let Some(raw) = &child.index_annotation else {
            continue;
        };
        if !inherits {
            warnings.push(format!(
                "{}: the {PACKAGE_LABEL} annotation on its index is not carried into the version \
                 (only --index-package sets the version's)",
                child.source
            ));
            continue;
        }
        let origin = format!(
            "{}: the {PACKAGE_LABEL} annotation on its index",
            child.source
        );
        match index_package(raw, &origin) {
            Ok((inherited, index_warnings)) => {
                warnings.push(format!(
                    "{origin} is kept on the published version (a single index source is \
                     republished as is); pass --index-package to replace it"
                ));
                warnings.extend(index_warnings);
                annotation = Some(inherited);
            }
            Err(error) => errors.push(format!(
                "{error:#}; it would be kept on the published version, so fix it or pass \
                 --index-package to replace it"
            )),
        }
    }
    let (variants, check_warnings, check_errors) = check_variants(children);
    warnings.extend(check_warnings);
    if let Some(annotation) = &annotation {
        warnings.extend(index_disagreements(annotation, &variants));
    }
    errors.extend(check_errors);

    writeln!(
        stdout,
        "{}",
        style.bold(&format!(
            "{} {} ({} variant{})",
            if args.dry_run {
                "Dry run: would publish"
            } else {
                "Publishing"
            },
            args.target,
            variants.len(),
            if variants.len() == 1 { "" } else { "s" }
        ))
    )?;
    if !variants.is_empty() {
        writeln!(stdout)?;
        write_table(stdout, style, &variants)?;
    }
    if !warnings.is_empty() {
        writeln!(stdout)?;
        for warning in &warnings {
            writeln!(stdout, "{}  {warning}", style.yellow("warn"))?;
        }
    }
    let cannot_publish = |errors: &[String]| {
        anyhow::anyhow!(
            "cannot publish {}:\n{}",
            args.target,
            errors
                .iter()
                .map(|error| format!("  - {error}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };
    if !errors.is_empty() {
        return Err(cannot_publish(&errors));
    }

    // The index as docker will assemble it: printed under --dry-run, checked
    // to carry the annotation, and compared with what each tag holds. Only
    // --index-package is passed as an annotation; an inherited one rides
    // along with its index.
    let explicit = args.index_package.as_ref().and(annotation.as_deref());
    let preview = tools.create(&create_args(&tags, explicit, &variants, true))?;
    if args.dry_run {
        writeln!(stdout)?;
        writeln!(
            stdout,
            "{}",
            style.muted(&format!("Index for {} (nothing pushed):", tags.join(", ")))
        )?;
        writeln!(stdout, "{}", preview.trim())?;
    }
    let index: Value = serde_json::from_str(&preview)
        .context("parsing the index docker buildx imagetools create --dry-run printed")?;
    if explicit.is_some()
        && let Some(error) = annotation_dropped(&index)
    {
        return Err(cannot_publish(&[format!("docker would assemble {error}")]));
    }
    let existing: Vec<Existing> = tags
        .iter()
        .map(|tag| existing_manifest(tools, tag))
        .collect();
    let states = tag_states(&existing, &index);
    let channel = args
        .channel
        .as_deref()
        .map(|channel| format!("{}:{}", target.repository, channel.trim()))
        .filter(|channel| {
            *channel != tags[0]
                && !args
                    .tags
                    .iter()
                    .any(|tag| *channel == format!("{}:{}", target.repository, tag.trim()))
        });
    let refusals = report_tags(
        stdout,
        style,
        &tags,
        &states,
        channel.as_deref(),
        args.force,
    )?;
    if !refusals.is_empty() {
        return Err(cannot_publish(&refusals));
    }
    if args.dry_run {
        return Ok(0);
    }

    // The version is TARGET's digest: the one it holds when it already holds
    // this index (never pushed again, since docker would re-serialize it),
    // else that of the index pushed to TARGET alone and checked as stored.
    // Only then do the other tags move, each pointed at that digest, which
    // docker copies byte for byte, so every alias is the same manifest and a
    // failed check leaves them where they were.
    let repository = &target.repository;
    let version = match &states[0] {
        TagState::Unchanged(digest) => digest.clone(),
        before => push_target(
            tools, repository, &tags[0], before, explicit, &variants, &index,
        )?,
    };
    let aliases: Vec<&String> = tags[1..]
        .iter()
        .zip(&states[1..])
        .filter(|(_, state)| **state != TagState::Unchanged(version.clone()))
        .map(|(tag, _)| tag)
        .collect();
    writeln!(stdout)?;
    if aliases.is_empty() && states[0] == TagState::Unchanged(version.clone()) {
        writeln!(
            stdout,
            "{}",
            style.success(&format!(
                "Already published {repository}@{version}; every tag holds this index, nothing \
                 pushed"
            ))
        )?;
        return Ok(0);
    }
    if !aliases.is_empty() {
        let mut args = Vec::new();
        for tag in &aliases {
            args.push("--tag".to_owned());
            args.push((*tag).clone());
        }
        args.push(format!("{repository}@{version}"));
        tools.create(&args).with_context(|| {
            format!(
                "{} holds the version at {version}, but pointing {} at it failed; run the same \
                 publish again to finish (TARGET is left as is and the other tags are pointed \
                 at it)",
                tags[0],
                aliases
                    .iter()
                    .map(|tag| tag.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
    }
    writeln!(
        stdout,
        "{}",
        style.success(&format!("Published {repository}@{version}"))
    )?;
    for (tag, state) in tags.iter().zip(&states) {
        match state {
            TagState::Unchanged(digest) if *digest == version => writeln!(
                stdout,
                "  {} {tag} (already {digest})",
                style.muted("unchanged")
            )?,
            TagState::Unchanged(digest) => writeln!(
                stdout,
                "  {} {tag} (same index, from {digest})",
                style.muted("re-pointed")
            )?,
            _ => writeln!(stdout, "  {} {tag}", style.muted("tagged"))?,
        }
    }
    Ok(0)
}

/// Push the index to TARGET alone and check it as the registry stores it:
/// the index assembled (the same manifests, annotations and media type),
/// and, with `--index-package`, an OCI image index carrying that annotation.
/// Returns its digest. On a failed check only TARGET has moved, and the error
/// says how to recover from what it held `before`.
fn push_target(
    tools: &impl Imagetools,
    repository: &str,
    target: &str,
    before: &TagState,
    explicit: Option<&str>,
    variants: &[VariantImage],
    index: &Value,
) -> Result<String> {
    tools.create(&create_args(
        &[target.to_owned()],
        explicit,
        variants,
        false,
    ))?;
    let recovery = match before {
        TagState::Moves(previous) => format!(
            "{target} pointed at {previous} before this run; restore it with `docker buildx \
             imagetools create --tag {target} {repository}@{previous}`, or publish again with \
             --force once the cause is fixed"
        ),
        TagState::Unreadable(_) => format!(
            "{target} could not be read before this run, so what it held is unknown; publish \
             again with --force once the cause is fixed"
        ),
        TagState::New | TagState::Unchanged(_) => format!(
            "{target} did not exist before this run, so it holds only this push: publish again \
             with --force once the cause is fixed, or delete {target} from the registry first"
        ),
    };
    let stored = || -> Result<(String, Value)> {
        let view: Value = serde_json::from_str(&tools.manifest(target)?)
            .context("docker printed a manifest that is not JSON")?;
        let digest = view
            .get("digest")
            .and_then(Value::as_str)
            .context("docker printed a manifest without a digest")?
            .to_owned();
        let raw = serde_json::from_str(&tools.raw(&format!("{repository}@{digest}"))?)
            .context("its raw manifest is not JSON")?;
        Ok((digest, raw))
    };
    let (digest, raw) = stored().map_err(|error| {
        anyhow::anyhow!(
            "pushed {target}, but could not read it back to check it ({error:#}); no other tag \
             was moved. {recovery}"
        )
    })?;
    let problem = explicit.and_then(|_| annotation_dropped(&raw)).or_else(|| {
        (!same_index(&raw, index)).then(|| {
            "an index other than the one assembled (its manifests, annotations or media \
                 type differ)"
                .to_owned()
        })
    });
    if let Some(problem) = problem {
        bail!(
            "pushed {target} ({digest}), but the registry holds {problem}; no other tag was \
             moved. {recovery}"
        );
    }
    Ok(digest)
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;
    use serde_json::json;

    #[test]
    fn dry_run_prints_the_summary_and_the_index_without_pushing() {
        let registry = three_variants();
        let dir = tempfile::tempdir().unwrap();
        let package = dir.path().join("version.json");
        std::fs::write(
            &package,
            r#"{"checkpoints":[{"name":"base","uri":"hf://org/pi0","default":true}]}"#,
        )
        .unwrap();
        let mut args = args("reg.example/ns/pi0:v3", SOURCES);
        args.dry_run = true;
        args.channel = Some("latest".to_owned());
        args.tags = vec!["v3.0".to_owned()];
        args.index_package = Some(package);
        let (result, out) = run(&registry, &args);
        assert_eq!(result.unwrap(), 0, "{out}");

        let created = registry.created.borrow();
        assert_eq!(created.len(), 1);
        assert_eq!(
            created[0],
            [
                "--dry-run",
                "--tag",
                "reg.example/ns/pi0:v3",
                "--tag",
                "reg.example/ns/pi0:v3.0",
                "--tag",
                "reg.example/ns/pi0:latest",
                "--annotation",
                r#"index:dev.rlmesh.package={"checkpoints":[{"default":true,"name":"base","uri":"hf://org/pi0"}],"schemaVersion":1}"#,
                // Indexes stay whole (attestations ride along); bare manifests as is.
                &format!("reg.example/ns/pi0@{}", digest('1')),
                &format!("reg.example/ns/pi0@{}", digest('2')),
                &format!("reg.example/ns/pi0@{}", digest('j')),
            ]
        );
        assert!(
            out.starts_with("Dry run: would publish reg.example/ns/pi0:v3 (3 variants)"),
            "{out}"
        );
        let header = out.lines().find(|l| l.starts_with("VARIANT")).unwrap();
        for column in [
            "KIND", "PLATFORM", "PRIORITY", "FACETS", "REQUIRES", "ROWS", "IMAGE",
        ] {
            assert!(header.contains(column), "{header}");
        }
        let row = out.lines().find(|l| l.starts_with("torch-cuda12")).unwrap();
        for cell in [
            "model",
            "linux/amd64",
            "10",
            "accel=cuda, framework=torch",
            "accel.cuda>=12.4",
            // The version's default row: the first linux/amd64 child.
            "torch-cuda12*",
        ] {
            assert!(row.contains(cell), "{cell:?} not in {row:?}");
        }
        assert!(
            row.ends_with("reg.example/ns/pi0:v3-cuda12 (cccccccccccc)"),
            "{row}"
        );
        let row = out.lines().find(|l| l.starts_with("torch-rocm6")).unwrap();
        assert!(row.contains("accel.gfx in [gfx942,gfx90a]"), "{row}");
        assert!(row.contains("accel=rocm"), "inferred facet: {row}");
        let row = out.lines().find(|l| l.starts_with("jax")).unwrap();
        // No requires declared and no markers: a CPU image, rows per profile.
        assert!(row.contains("accel=cpu, framework=jax"), "{row}");
        assert!(row.contains("jax-osmesa, jax-egl"), "{row}");
        assert!(out.contains("* the version's default row"), "{out}");
        assert!(out.contains("(nothing pushed)"), "{out}");
        assert!(
            out.contains("application/vnd.oci.image.index.v1+json"),
            "{out}"
        );
        assert!(!out.contains("warn"), "{out}");
    }

    #[test]
    fn publish_pushes_and_reports_the_index_digest() {
        let registry = three_variants();
        let (result, out) = run(&registry, &args("reg.example/ns/pi0:v3", SOURCES));
        assert_eq!(result.unwrap(), 0, "{out}");
        let created = registry.created.borrow();
        // The index is assembled first (to check it), then pushed.
        assert_eq!(created.len(), 2);
        assert!(created[0].contains(&"--dry-run".to_owned()));
        assert!(!created[1].contains(&"--dry-run".to_owned()));
        assert!(!created[1].contains(&"--annotation".to_owned()));
        let published = registry.digest_of("reg.example/ns/pi0:v3");
        assert_eq!(
            published,
            sha256(&registry.raw("reg.example/ns/pi0:v3").unwrap())
        );
        assert!(
            out.contains(&format!("Published reg.example/ns/pi0@{published}")),
            "{out}"
        );
        assert!(out.contains("tagged reg.example/ns/pi0:v3"), "{out}");
    }

    #[test]
    fn a_lone_index_source_passes_its_annotation_through_checked() {
        let mut registry = FakeRegistry::default();
        registry.add(
            "r/x:a",
            Some('i'),
            'a',
            oci_config(
                &[],
                MODEL,
                Some(json!({"schemaVersion": 1, "name": "pi0", "variant": {"key": "a"}})),
            ),
        );
        let annotate = |registry: &FakeRegistry, package: Value| {
            let mut index = buildkit_index(&digest('a'), ("linux", "amd64"));
            index["digest"] = Value::from(digest('i'));
            index["annotations"] = json!({PACKAGE_LABEL: package.to_string()});
            registry.insert("r/x:a", &index);
        };
        let mut publish = args("r/x:v1", &["r/x:a"]);
        publish.dry_run = true;

        annotate(
            &registry,
            json!({"schemaVersion": 2, "checkpoints": [{"name": "stale"}]}),
        );
        let (result, _) = run(&registry, &publish);
        let error = format!("{:#}", result.unwrap_err());
        assert!(
            error.contains(
                "r/x:a: the dev.rlmesh.package annotation on its index schemaVersion 2 is not \
                 the supported version 1; it would be kept on the published version"
            ),
            "{error}"
        );
        assert!(registry.created.borrow().is_empty());

        // Valid, but carrying child-level blocks and a stale name: warned
        // about exactly as --index-package would be.
        annotate(
            &registry,
            json!({"schemaVersion": 1, "name": "old", "variant": {"key": "a"}, "profiles": []}),
        );
        let (result, out) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out}");
        for needle in [
            "r/x:a: the dev.rlmesh.package annotation on its index is kept on the published \
             version",
            "index annotation keys ignored (they belong on the child images): profiles, variant",
            "r/x:a: sets name differently from the index annotation, which wins",
        ] {
            assert!(out.contains(needle), "{needle:?} not in:\n{out}");
        }
        // Docker keeps it; it is not passed again.
        assert!(
            !registry.created.borrow()[0].contains(&"--annotation".to_owned()),
            "{:?}",
            registry.created.borrow()
        );

        // With a second source, docker assembles a new index without it.
        registry.add("r/x:b", None, 'b', keyed("b"));
        publish.sources.push("r/x:b".to_owned());
        let (result, out) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert!(
            out.contains(
                "r/x:a: the dev.rlmesh.package annotation on its index is not carried into the \
                 version"
            ),
            "{out}"
        );
        assert!(!out.contains("is kept"), "{out}");
    }

    #[test]
    fn a_failed_check_after_the_push_moves_only_target_and_says_how_to_recover() {
        let registry = three_variants();
        let dir = tempfile::tempdir().unwrap();
        let (target, extra, channel) = (
            "reg.example/ns/pi0:v3",
            "reg.example/ns/pi0:v3.0",
            "reg.example/ns/pi0:latest",
        );
        let mut publish = args(target, SOURCES);
        publish.tags = vec!["v3.0".to_owned()];
        publish.channel = Some("latest".to_owned());
        publish.index_package = Some(version_json(&dir, r#"{"checkpoints":[]}"#));

        // The registry loses the annotation on the way in.
        registry.mangle.set(Some(|index| {
            index.as_object_mut().unwrap().remove("annotations");
        }));
        let (result, _) = run(&registry, &publish);
        let error = format!("{:#}", result.unwrap_err());
        let stored = registry.digest_of(target);
        for needle in [
            format!(
                "pushed {target} ({stored}), but the registry holds an index without the \
                 dev.rlmesh.package annotation --index-package sets; no other tag was moved"
            ),
            format!(
                "{target} did not exist before this run, so it holds only this push: publish \
                 again with --force once the cause is fixed"
            ),
        ] {
            assert!(error.contains(&needle), "{needle:?} not in:\n{error}");
        }
        assert_eq!(registry.pushes(), 1, "only TARGET was pushed");
        for tag in [extra, channel] {
            assert!(registry.manifest(tag).is_err(), "{tag} did not move");
        }

        // A retry without --force is refused at TARGET, as the error said;
        // with it, the version is published and every tag is its digest.
        registry.mangle.set(None);
        let error = format!("{:#}", run(&registry, &publish).0.unwrap_err());
        assert!(
            error.contains(&format!("{target} already points at {stored}")),
            "{error}"
        );
        assert_eq!(registry.pushes(), 1);
        publish.force = true;
        let (result, out) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out}");
        let version = registry.digest_of(target);
        assert_ne!(version, stored);
        for tag in [extra, channel] {
            assert_eq!(registry.digest_of(tag), version, "{tag}");
        }
        let pushed: Value = serde_json::from_str(&registry.raw(target).unwrap()).unwrap();
        assert!(pushed["annotations"][PACKAGE_LABEL].is_string(), "{pushed}");

        // Forcing over a published version: the error names the digest it
        // held and how to put it back, and the channel stays on it.
        registry.mangle.set(Some(|index| {
            index["manifests"].as_array_mut().unwrap().reverse();
        }));
        let mut over = args(target, &SOURCES[..2]);
        over.channel = Some("latest".to_owned());
        over.force = true;
        let error = format!("{:#}", run(&registry, &over).0.unwrap_err());
        for needle in [
            "but the registry holds an index other than the one assembled".to_owned(),
            format!(
                "{target} pointed at {version} before this run; restore it with `docker buildx \
                 imagetools create --tag {target} reg.example/ns/pi0@{version}`"
            ),
        ] {
            assert!(error.contains(&needle), "{needle:?} not in:\n{error}");
        }
        assert_eq!(registry.digest_of(channel), version);
    }
}
