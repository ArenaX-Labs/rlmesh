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
//! only over a tag the registry said was absent when checked, or one already
//! holding the same index (which is left alone); a different index, or a tag
//! that cannot be read, needs `--force`. (The check and the push are separate
//! registry calls, so a tag created between them is not caught.) The channel
//! tag moves. The index is pushed to
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
use crate::manifest::{self, VariantBuild};
use crate::render::Style;
use build::{Builder, DockerBuilder, build_args};
use imagetools::{DockerImagetools, Imagetools, existing_manifest};
use source::{Child, attestation_errors, docker_list_error, resolve_source};
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
    assemble(
        tools,
        args,
        &args.sources,
        index,
        warnings,
        false,
        stdout,
        style,
    )
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
                    "target": variant.target,
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

    // What the version and build tags hold decides, before anything is built
    // (and under --dry-run too), whether this publish builds, finishes an
    // earlier one from its builds, or is refused.
    if !args.force {
        let plan = manifest_plan(
            tools,
            args,
            &repository,
            &tags[0],
            &manifest.variants,
            &names,
            index.as_deref(),
            &origin,
        )
        .map_err(|refusals| cannot_publish(&args.target, &refusals))?;
        if let ManifestPlan::Finish(sources) = plan {
            writeln!(stdout)?;
            writeln!(
                stdout,
                "{}",
                style.muted(&format!(
                    "{} already holds the index of the builds at {}, whose labels match {origin}: \
                     {} it from them and build nothing (to publish a new build, use a new tag or \
                     pass --force)",
                    tags[0],
                    names.join(", "),
                    if args.dry_run {
                        "would finish"
                    } else {
                        "finishing"
                    },
                ))
            )?;
            writeln!(stdout)?;
            return assemble(tools, args, &sources, index, warnings, true, stdout, style);
        }
    }
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
    assemble(tools, args, &sources, index, warnings, true, stdout, style)
}

/// What a publish from `rlmesh.toml` does, absent `--force`.
#[derive(Debug)]
enum ManifestPlan {
    /// Build every variant, then publish.
    Build,
    /// TARGET already holds the index assembled from the builds at the build
    /// tags (each labelled as the file renders it), as an earlier publish
    /// that stopped before pointing every tag leaves it: publish from those
    /// builds, pinned here, without building.
    Finish(Vec<String>),
}

/// Read the version tags (TARGET, each `--tag`) and the build tags before
/// building. A fresh build differs from any index already published, and
/// overwrites its build tag, so each refusal is known now, not after
/// building: a version tag holding any manifest, a build tag holding an image
/// other than a build of its variant from this file (a separately published
/// `1.0-cpu`, say), or either unreadable. A build tag holding an earlier
/// build of the same variant (the same rendered label) is rebuilt over. When
/// TARGET already holds exactly the index of the builds at the build tags,
/// the publish is finished from them instead (the channel and `--tag` tags are
/// then checked as in a SOURCE publish). Returns every refusal at once.
#[allow(clippy::too_many_arguments)]
fn manifest_plan(
    tools: &impl Imagetools,
    args: &PublishArgs,
    repository: &str,
    target: &str,
    variants: &[VariantBuild],
    names: &[String],
    index: Option<&str>,
    origin: &str,
) -> std::result::Result<ManifestPlan, Vec<String>> {
    enum Build {
        Absent,
        Unreadable(String),
        /// An earlier build of this variant, from the same label.
        Ours(Box<Child>),
        /// Something else, at this digest.
        Other(String),
    }
    let builds: Vec<Build> = variants
        .iter()
        .zip(names)
        .map(|(variant, name)| match existing_manifest(tools, name) {
            Existing::Absent => Build::Absent,
            Existing::Unreadable(reason) => Build::Unreadable(reason),
            Existing::Found { digest, .. } => {
                match resolve_source(tools, &format!("{repository}@{digest}")) {
                    Ok(child) if built_from(&child, variant) => Build::Ours(Box::new(child)),
                    _ => Build::Other(digest),
                }
            }
        })
        .collect();
    let target_state = existing_manifest(tools, target);

    if let Existing::Found { raw, .. } = &target_state {
        let children: Option<Vec<Child>> = builds
            .iter()
            .map(|build| match build {
                Build::Ours(child) => Some((**child).clone()),
                _ => None,
            })
            .collect();
        if let Some(children) = children {
            let sources: Vec<String> = children.iter().map(|c| c.pinned.clone()).collect();
            let (checked, _, errors) = check_variants(children);
            let assembled = errors
                .is_empty()
                .then(|| tools.create(&create_args(&[target.to_owned()], index, &checked, true)))
                .and_then(Result::ok)
                .and_then(|preview| serde_json::from_str::<Value>(&preview).ok());
            if assembled.is_some_and(|assembled| same_index(raw, &assembled)) {
                return Ok(ManifestPlan::Finish(sources));
            }
        }
    }

    let mut refusals = Vec::new();
    let others = args
        .tags
        .iter()
        .map(|tag| format!("{repository}:{}", tag.trim()))
        .map(|tag| {
            let state = existing_manifest(tools, &tag);
            (tag, state)
        });
    for (tag, state) in std::iter::once((target.to_owned(), target_state)).chain(others) {
        match state {
            Existing::Absent => {}
            Existing::Found { digest, .. } => refusals.push(format!(
                "{tag} already holds a version ({digest}), and a new build would replace it: \
                 publish under a new tag, or pass --force"
            )),
            Existing::Unreadable(reason) => refusals.push(format!(
                "{tag} could not be read ({reason}), so nothing says it is free: fix access to \
                 it, or pass --force"
            )),
        }
    }
    for ((variant, name), build) in variants.iter().zip(names).zip(builds) {
        match build {
            Build::Absent | Build::Ours(_) => {}
            Build::Other(digest) => refusals.push(format!(
                "{name} already holds {digest}, which is not a build of variant {} from {origin} \
                 (its {PACKAGE_LABEL} label differs), and building the variant would overwrite \
                 it: publish under a new tag, or pass --force",
                variant.key
            )),
            Build::Unreadable(reason) => refusals.push(format!(
                "{name} could not be read ({reason}), so nothing says building variant {} would \
                 not overwrite another image there: fix access to it, or pass --force",
                variant.key
            )),
        }
    }
    if refusals.is_empty() {
        Ok(ManifestPlan::Build)
    } else {
        Err(refusals)
    }
}

/// Whether `child` is a build of `variant`: its package label is the one the
/// manifest renders for it.
fn built_from(child: &Child, variant: &VariantBuild) -> bool {
    child
        .config
        .labels
        .get(PACKAGE_LABEL)
        .and_then(|label| serde_json::from_str::<Value>(label).ok())
        .is_some_and(|label| label == Value::Object(variant.label.clone()))
}

/// The error for a publish refused before pushing, listing every reason.
fn cannot_publish(target: &str, errors: &[String]) -> anyhow::Error {
    anyhow::anyhow!(
        "cannot publish {target}:\n{}",
        errors
            .iter()
            .map(|error| format!("  - {error}"))
            .collect::<Vec<_>>()
            .join("\n")
    )
}

/// Assemble `sources` into the version and push it under every tag, with
/// `index` (when set) as its annotation. `from_manifest` says the sources are
/// the builds of `rlmesh.toml`, which changes how a rerun recovers.
#[allow(clippy::too_many_arguments)]
fn assemble(
    tools: &impl Imagetools,
    args: &PublishArgs,
    sources: &[String],
    index: Option<String>,
    mut warnings: Vec<String>,
    from_manifest: bool,
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
    let cannot_publish = |errors: &[String]| cannot_publish(&args.target, errors);
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
        let mut create = Vec::new();
        for tag in &aliases {
            create.push("--tag".to_owned());
            create.push((*tag).clone());
        }
        create.push(format!("{repository}@{version}"));
        tools.create(&create).with_context(|| {
            let names = |tags: &[&String]| {
                tags.iter()
                    .map(|tag| tag.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            // The version tags this run moved only because of --force.
            let forced: Vec<&String> = tags[1..]
                .iter()
                .zip(&states[1..])
                .filter(|(tag, state)| {
                    channel.as_deref() != Some(tag.as_str())
                        && matches!(state, TagState::Moves(_) | TagState::Unreadable(_))
                })
                .map(|(tag, _)| tag)
                .collect();
            let rerun = if !from_manifest {
                ", or run the same publish again: TARGET already holds this index, so it is not \
                 pushed again and only the other tags are pointed at it"
                    .to_owned()
            } else if !args.force {
                ", or run the same publish again: TARGET already holds the index of the builds \
                 just pushed, so nothing is rebuilt and only the other tags are pointed at it"
                    .to_owned()
            } else if forced.is_empty() {
                ", or run the same publish again without --force (with it, every variant is \
                 rebuilt): TARGET already holds the index of the builds just pushed, so nothing \
                 is rebuilt and only the other tags are pointed at it"
                    .to_owned()
            } else {
                format!(
                    "; rerunning the publish would not finish it (without --force it refuses to \
                     move {}, and with it every variant is rebuilt)",
                    names(&forced)
                )
            };
            format!(
                "{} holds the version at {version}, but pointing {} at it failed. To finish, \
                 point them yourself with `docker buildx imagetools create {}`{rerun}",
                tags[0],
                names(&aliases),
                create.join(" ")
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
mod tests;
