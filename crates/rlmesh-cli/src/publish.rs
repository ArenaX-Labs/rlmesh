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

mod build;
mod imagetools;
mod media;
mod source;
mod table;
mod tags;
#[cfg(test)]
mod testing;
mod validate;

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::cli::PublishArgs;
use crate::image_check::PACKAGE_LABEL;
use crate::manifest;
use crate::render::Style;
use build::{Builder, DockerBuilder, build_args};
use imagetools::{DockerImagetools, Imagetools, existing_manifest};
use source::{attestation_errors, docker_list_error, resolve_source};
use table::{write_build_table, write_table};
use tags::{
    Existing, TagState, check_tag, create_args, report_tags, same_index, tag_states, target_tags,
};
use validate::{
    VariantImage, annotation_dropped, check_variants, index_disagreements, index_package,
};

pub(crate) fn publish(args: &PublishArgs, stdout: &mut impl Write, style: Style) -> Result<i32> {
    publish_with(&DockerImagetools, &DockerBuilder, args, stdout, style)
}

pub(crate) fn publish_with(
    tools: &impl Imagetools,
    builder: &impl Builder,
    args: &PublishArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<i32> {
    if args.sources.is_empty() {
        return publish_manifest(tools, builder, args, stdout, style);
    }
    let mut warnings = Vec::new();
    let index = match &args.index_package {
        Some(path) => {
            let raw = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            let (annotation, index_warnings) = index_package(&raw, &path.display().to_string())?;
            warnings.extend(index_warnings);
            Some(annotation)
        }
        None => None,
    };
    assemble(tools, args, &args.sources, index, warnings, stdout, style)
}

/// Build each variant of `rlmesh.toml` with its rendered label, push it to
/// TARGET's repository as `<TARGET tag>-<variant key>`, then assemble those,
/// pinned by digest, with the manifest's version-level keys as the index
/// annotation. Under `--dry-run` nothing is built: the plan is printed
/// (`--json`: the builds and labels).
fn publish_manifest(
    tools: &impl Imagetools,
    builder: &impl Builder,
    args: &PublishArgs,
    stdout: &mut impl Write,
    style: Style,
) -> Result<i32> {
    let path = args
        .config
        .clone()
        .unwrap_or_else(|| PathBuf::from(manifest::FILE_NAME));
    if args.config.is_none() && !path.exists() {
        bail!(
            "nothing to publish: pass the pushed SOURCE images, or write an {} declaring the \
             variants to build",
            manifest::FILE_NAME
        );
    }
    let (manifest, mut warnings) = manifest::load(&path)?;
    let origin = if args.config.as_deref() == Some(Path::new("-")) {
        "stdin".to_owned()
    } else {
        path.display().to_string()
    };
    let index = if manifest.version.is_empty() {
        None
    } else {
        let (annotation, index_warnings) = index_package(
            &Value::Object(manifest.version.clone()).to_string(),
            &origin,
        )?;
        warnings.extend(index_warnings);
        Some(annotation)
    };
    let (target, tags) = target_tags(args)?;
    let repository = target.repository;
    let mut names = Vec::new();
    for variant in &manifest.variants {
        let tag = format!(
            "{}-{}",
            target.tag.as_deref().unwrap_or_default(),
            variant.key
        );
        check_tag(&tag).with_context(|| format!("the build tag of variant {}", variant.key))?;
        let name = format!("{repository}:{tag}");
        if tags.contains(&name) {
            bail!(
                "{name} is where variant {} is built, so it cannot also tag the version",
                variant.key
            );
        }
        names.push(name);
    }
    let metadata = |key: &str| {
        std::env::temp_dir().join(format!("rlmesh-build-{}-{key}.json", std::process::id()))
    };

    if args.json {
        let variants: Vec<Value> = manifest
            .variants
            .iter()
            .enumerate()
            .map(|(index, variant)| {
                json!({
                    "key": variant.key,
                    "tag": names[index],
                    "dockerfile": variant.dockerfile,
                    "context": variant.context,
                    "buildArgs": variant.build_args,
                    "label": variant.label,
                })
            })
            .collect();
        let index: Option<Value> = index.as_deref().map(serde_json::from_str).transpose()?;
        let plan = json!({"index": index, "variants": variants, "warnings": warnings});
        writeln!(stdout, "{}", serde_json::to_string_pretty(&plan)?)?;
        return Ok(0);
    }
    let count = manifest.variants.len();
    writeln!(
        stdout,
        "{}",
        style.bold(&format!(
            "{} {} ({count} variant{} from {origin})",
            if args.dry_run {
                "Dry run: would build and publish"
            } else {
                "Building and publishing"
            },
            args.target,
            if count == 1 { "" } else { "s" },
        ))
    )?;
    writeln!(stdout)?;
    write_build_table(stdout, style, &manifest.variants)?;
    if args.dry_run {
        if !warnings.is_empty() {
            writeln!(stdout)?;
            for warning in &warnings {
                writeln!(stdout, "{}  {warning}", style.yellow("warn"))?;
            }
        }
        writeln!(stdout)?;
        writeln!(
            stdout,
            "{}",
            style.muted("Builds (nothing built or pushed; --json prints each label):")
        )?;
        for (variant, name) in manifest.variants.iter().zip(&names) {
            let mut argv = build_args(variant, name, &metadata(&variant.key));
            for arg in &mut argv {
                if arg.starts_with(&format!("{PACKAGE_LABEL}=")) {
                    *arg = format!("{PACKAGE_LABEL}=<{} label>", variant.key);
                }
            }
            writeln!(stdout, "  docker buildx {}", argv.join(" "))?;
        }
        return Ok(0);
    }

    // A fresh build differs from any index already published, so a version
    // tag in use (or unreadable) would only be refused after building; say
    // so first. The channel is not a version tag.
    if !args.force {
        let protected = std::iter::once(tags[0].clone()).chain(
            args.tags
                .iter()
                .map(|tag| format!("{repository}:{}", tag.trim())),
        );
        for tag in protected {
            match existing_manifest(tools, &tag) {
                Existing::Absent => {}
                Existing::Found { digest, .. } => bail!(
                    "{tag} already holds a version ({digest}), and a new build would replace \
                     it: publish under a new tag, or pass --force"
                ),
                Existing::Unreadable(reason) => bail!(
                    "{tag} could not be read ({reason}), so nothing says it is free: fix \
                     access to it, or pass --force"
                ),
            }
        }
    }

    writeln!(stdout)?;
    let mut sources = Vec::new();
    for (variant, name) in manifest.variants.iter().zip(&names) {
        let digest = builder
            .build(&build_args(variant, name, &metadata(&variant.key)))
            .with_context(|| format!("building variant {} of {origin}", variant.key))?;
        writeln!(stdout, "  {} {name} ({digest})", style.muted("built"))?;
        sources.push(format!("{repository}@{digest}"));
    }
    writeln!(stdout)?;
    assemble(tools, args, &sources, index, warnings, stdout, style)
}

/// Assemble `sources` into the version and push it under every tag, with
/// `index` (when set) as its annotation.
fn assemble(
    tools: &impl Imagetools,
    args: &PublishArgs,
    sources: &[String],
    index: Option<String>,
    mut warnings: Vec<String>,
    stdout: &mut impl Write,
    style: Style,
) -> Result<i32> {
    let (target, tags) = target_tags(args)?;
    let mut annotation = index.clone();

    let mut children = Vec::new();
    let mut errors = Vec::new();
    for source in sources {
        match resolve_source(tools, source) {
            Ok(child) => children.push(child),
            Err(error) => errors.push(format!("{error:#}")),
        }
    }
    errors.extend(attestation_errors(&children));
    if annotation.is_some()
        && children.len() == sources.len()
        && let Some(error) = docker_list_error(&children)
    {
        errors.push(error);
    }
    // A lone index source is republished as is, its own annotation included,
    // so that annotation is the version's and is checked like --index-package.
    // Anywhere else, docker assembles a new index without it.
    let inherits = annotation.is_none() && sources.len() == 1;
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
    let explicit = index.as_deref();
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

    const MANIFEST: &str = r#"
name = "pi0"

[[checkpoints]]
name = "base"
uri = "hf://org/pi0"
default = true

[variant.cuda12]
build_args.TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
accel.vendor = "nvidia"
accel.cuda = ">=12.4"

[variant.cpu]
dockerfile = "Dockerfile.cpu"
"#;

    /// A directory holding `MANIFEST`, and a registry and builder whose
    /// builds of it land as `c` (cuda12) and `p` (cpu).
    fn manifest_setup() -> (tempfile::TempDir, FakeRegistry, FakeBuilder, PublishArgs) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rlmesh.toml");
        std::fs::write(&path, MANIFEST).unwrap();
        let (manifest, _) = manifest::load(&path).unwrap();
        let mut registry = FakeRegistry::default();
        for (variant, seed, env) in [
            (&manifest.variants[0], 'c', &["CUDA_VERSION=12.4.1"][..]),
            (&manifest.variants[1], 'p', &[][..]),
        ] {
            registry.add(
                &format!("reg.example/ns/pi0:built-{}", variant.key),
                None,
                seed,
                oci_config(env, MODEL, Some(Value::Object(variant.label.clone()))),
            );
        }
        let builder = FakeBuilder {
            digests: [("cuda12".to_owned(), 'c'), ("cpu".to_owned(), 'p')].into(),
            ..FakeBuilder::default()
        };
        let mut args = args("reg.example/ns/pi0:v3", &[]);
        args.config = Some(path);
        (dir, registry, builder, args)
    }

    #[test]
    fn a_manifest_builds_each_variant_then_publishes_them_as_one_version() {
        let (dir, registry, builder, args) = manifest_setup();
        let (result, out) = run_with(&registry, &builder, &args);
        assert_eq!(result.unwrap(), 0, "{out}");

        let builds = builder.builds.borrow();
        assert_eq!(builds.len(), 2);
        let first = &builds[0];
        let after =
            |flag: &str| first[first.iter().position(|arg| arg == flag).unwrap() + 1].clone();
        assert_eq!(
            after("--file"),
            dir.path().join("Dockerfile").display().to_string()
        );
        assert_eq!(after("--platform"), "linux/amd64");
        assert_eq!(
            after("--build-arg"),
            "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
        );
        assert_eq!(
            after("--output"),
            "type=image,name=reg.example/ns/pi0:v3-cuda12,push=true,oci-mediatypes=true"
        );
        let label: Value = serde_json::from_str(
            after("--label")
                .strip_prefix("dev.rlmesh.package=")
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            label["variant"],
            json!({"key": "cuda12", "requires": {"accel.vendor": "nvidia", "accel.cuda": ">=12.4"}})
        );
        assert_eq!(first.last().unwrap(), &dir.path().display().to_string());
        assert!(builds[1].contains(&dir.path().join("Dockerfile.cpu").display().to_string()));

        // Assembled from the builds, in the manifest's order, with its
        // version-level keys as the index annotation.
        let pushed = registry.created.borrow();
        let create = pushed
            .iter()
            .find(|args| !args.contains(&"--dry-run".to_owned()))
            .unwrap();
        assert_eq!(
            create[create.len() - 3..],
            [
                r#"index:dev.rlmesh.package={"checkpoints":[{"default":true,"name":"base","uri":"hf://org/pi0"}],"name":"pi0","schemaVersion":1}"#.to_owned(),
                format!("reg.example/ns/pi0@{}", digest('c')),
                format!("reg.example/ns/pi0@{}", digest('p')),
            ]
        );
        assert!(
            out.contains(&format!(
                "built reg.example/ns/pi0:v3-cuda12 ({})",
                digest('c')
            )),
            "{out}"
        );
        assert!(out.contains("Published reg.example/ns/pi0@"), "{out}");
    }

    #[test]
    fn a_manifest_dry_run_builds_nothing_and_shows_the_plan() {
        let (_dir, registry, builder, mut args) = manifest_setup();
        args.dry_run = true;
        let (result, out) = run_with(&registry, &builder, &args);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert!(builder.builds.borrow().is_empty());
        assert!(registry.created.borrow().is_empty());
        assert!(
            out.starts_with("Dry run: would build and publish reg.example/ns/pi0:v3 (2 variants"),
            "{out}"
        );
        let row = out.lines().find(|l| l.starts_with("cuda12")).unwrap();
        assert!(row.contains("accel.cuda>=12.4"), "{row}");
        assert!(row.contains("cuda12*"), "{row}");
        let row = out.lines().find(|l| l.starts_with("cpu")).unwrap();
        assert!(row.contains("(inferred from the image)"), "{row}");
        assert!(
            out.contains("--label dev.rlmesh.package=<cuda12 label>"),
            "{out}"
        );
    }

    #[test]
    fn a_manifest_dry_run_prints_its_builds_and_labels_as_json() {
        let (dir, registry, builder, mut args) = manifest_setup();
        args.dry_run = true;
        args.json = true;
        let (result, out) = run_with(&registry, &builder, &args);
        assert_eq!(result.unwrap(), 0, "{out}");
        let plan: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(
            plan["index"],
            json!({"schemaVersion": 1, "name": "pi0",
                   "checkpoints": [{"name": "base", "uri": "hf://org/pi0", "default": true}]})
        );
        let keys: Vec<&str> = plan["variants"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v["key"].as_str().unwrap())
            .collect();
        assert_eq!(keys, ["cuda12", "cpu"]);
        assert_eq!(
            plan["variants"][1]["dockerfile"],
            json!(dir.path().join("Dockerfile.cpu"))
        );
        assert_eq!(
            plan["variants"][1]["label"]["variant"],
            json!({"key": "cpu"})
        );
        assert_eq!(plan["variants"][1]["tag"], "reg.example/ns/pi0:v3-cpu");
        assert_eq!(plan["warnings"], json!([]));
    }

    #[test]
    fn a_failed_build_publishes_nothing() {
        let (_dir, registry, mut builder, args) = manifest_setup();
        builder.digests.remove("cpu");
        let (result, out) = run_with(&registry, &builder, &args);
        let error = format!("{:#}", result.unwrap_err());
        assert!(error.contains("building variant cpu of"), "{error}");
        assert_eq!(registry.pushes(), 0, "{out}");
    }

    #[test]
    fn a_version_tag_in_use_is_refused_before_anything_is_built() {
        let (_dir, registry, builder, mut args) = manifest_setup();
        registry.insert(
            "reg.example/ns/pi0:v3",
            &json!({"mediaType": "application/vnd.oci.image.index.v1+json",
                    "digest": digest('x')}),
        );
        let (result, _) = run_with(&registry, &builder, &args);
        let error = format!("{:#}", result.unwrap_err());
        assert!(
            error.contains("reg.example/ns/pi0:v3 already holds a version"),
            "{error}"
        );
        assert!(builder.builds.borrow().is_empty());

        args.force = true;
        let (result, out) = run_with(&registry, &builder, &args);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert_eq!(builder.builds.borrow().len(), 2);
    }
}
