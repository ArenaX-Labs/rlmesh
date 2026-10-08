//! The tags a publish writes, and what each holds before it: what moves, what
//! is left alone, and what is refused.

use std::io::Write;

use anyhow::{Result, bail};
use serde_json::Value;

use super::source::Reference;
use super::validate::VariantImage;
use crate::cli::PublishArgs;
use crate::image_check::PACKAGE_LABEL;
use crate::render::Style;

pub(super) fn check_tag(tag: &str) -> Result<()> {
    let mut chars = tag.chars();
    let valid = tag.len() <= 128
        && chars
            .next()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'));
    if !valid {
        bail!("{tag:?} is not a valid image tag");
    }
    Ok(())
}

/// The full target references: TARGET, each `--tag` in its repository, then
/// the channel, without repeats.
pub(crate) fn target_tags(args: &PublishArgs) -> Result<(Reference, Vec<String>)> {
    let target = Reference::parse(&args.target)?;
    if target.digest.is_some() || target.tag.is_none() {
        bail!(
            "target {:?} must be REPOSITORY:TAG (an index is published under a tag, not a digest)",
            args.target
        );
    }
    let mut tags = vec![format!(
        "{}:{}",
        target.repository,
        target.tag.as_deref().unwrap_or_default()
    )];
    for tag in args.tags.iter().chain(&args.channel) {
        let tag = tag.trim();
        check_tag(tag)?;
        let full = format!("{}:{tag}", target.repository);
        if !tags.contains(&full) {
            tags.push(full);
        }
    }
    Ok((target, tags))
}

/// The `imagetools create` arguments: every tag, the index annotation, and the
/// sources pinned by digest, in command-line order.
pub(crate) fn create_args(
    tags: &[String],
    annotation: Option<&str>,
    variants: &[VariantImage],
    dry_run: bool,
) -> Vec<String> {
    let mut args = Vec::new();
    if dry_run {
        args.push("--dry-run".to_owned());
    }
    for tag in tags {
        args.push("--tag".to_owned());
        args.push(tag.clone());
    }
    if let Some(annotation) = annotation {
        args.push("--annotation".to_owned());
        args.push(format!("index:{PACKAGE_LABEL}={annotation}"));
    }
    args.extend(variants.iter().map(|variant| variant.child.pinned.clone()));
    args
}

/// What a tag holds before the publish.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Existing {
    /// The registry says there is no such manifest.
    Absent,
    /// It could not be read (auth, transport, a registry error, output that
    /// does not parse), so nothing is known about it; the reason.
    Unreadable(String),
    /// A manifest at `digest`, as stored (`raw`).
    Found { digest: String, raw: Value },
}

/// Where a tag points before the publish, against the index being published.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TagState {
    /// Nothing there yet.
    New,
    /// Not readable; the reason.
    Unreadable(String),
    /// Already this index, at this digest: left as is.
    Unchanged(String),
    /// Another manifest, at this digest.
    Moves(String),
}

/// Whether an existing manifest is the index about to be published. Both are
/// compared as stored, field by field: the same media type and artifactType,
/// the same manifests in the same order (digest, size, platform, annotations,
/// every descriptor field), the same annotations and subject. Only the
/// serialization (whitespace, key order) may differ.
pub(crate) fn same_index(existing: &Value, index: &Value) -> bool {
    existing == index
}

/// Each tag's state against `index`, the index `imagetools create --dry-run`
/// assembles.
pub(crate) fn tag_states(existing: &[Existing], index: &Value) -> Vec<TagState> {
    existing
        .iter()
        .map(|existing| match existing {
            Existing::Absent => TagState::New,
            Existing::Unreadable(reason) => TagState::Unreadable(reason.clone()),
            Existing::Found { digest, raw } if same_index(raw, index) => {
                TagState::Unchanged(digest.clone())
            }
            Existing::Found { digest, .. } => TagState::Moves(digest.clone()),
        })
        .collect()
}

/// Print each tag's state and return the refusals: TARGET and `--tag` tags
/// name a version and do not move, or get pushed over a state that could not
/// be read, without `--force`; the channel tag (its full reference, unless it
/// is also a version tag) moves, which is its purpose, and says from where.
/// Every tag ends at TARGET's digest, so another tag holding the same index
/// at another digest (another serialization) is re-pointed.
pub(super) fn report_tags(
    stdout: &mut impl Write,
    style: Style,
    tags: &[String],
    states: &[TagState],
    channel: Option<&str>,
    force: bool,
) -> Result<Vec<String>> {
    let mut refusals = Vec::new();
    let width = tags
        .iter()
        .map(|tag| tag.chars().count())
        .max()
        .unwrap_or(0);
    let version = match &states[0] {
        TagState::Unchanged(digest) => Some(digest.as_str()),
        _ => None,
    };
    writeln!(stdout)?;
    writeln!(stdout, "{}", style.bold("Tags:"))?;
    for (position, (tag, state)) in tags.iter().zip(states).enumerate() {
        let is_channel = channel == Some(tag.as_str());
        let note = match state {
            TagState::New => style.muted("new"),
            TagState::Unchanged(digest) if position == 0 || version == Some(digest.as_str()) => {
                style.muted(&format!("unchanged ({digest}), not pushed again"))
            }
            TagState::Unchanged(digest) => style.muted(&match version {
                Some(version) => {
                    format!("same index at {digest}, re-pointed at {version} (TARGET's digest)")
                }
                None => format!(
                    "same index at {digest}, re-pointed at TARGET's new digest if that differs"
                ),
            }),
            TagState::Moves(digest) if is_channel => format!("moves from {digest} (channel)"),
            TagState::Moves(digest) if force => {
                style.yellow(&format!("moves from {digest} (--force)"))
            }
            TagState::Moves(digest) => {
                refusals.push(format!(
                    "{tag} already points at {digest}, a different index; a version tag does \
                     not move without --force"
                ));
                style.red_bold(&format!("refused: points at {digest}"))
            }
            TagState::Unreadable(reason) if is_channel => {
                style.yellow(&format!("unreadable, overwritten (channel): {reason}"))
            }
            TagState::Unreadable(reason) if force => {
                style.yellow(&format!("unreadable, overwritten (--force): {reason}"))
            }
            TagState::Unreadable(reason) => {
                refusals.push(format!(
                    "{tag} could not be read ({reason}), so it may already hold another \
                     version; a version tag is pushed only where the registry says it is \
                     absent, so fix access to it or pass --force"
                ));
                style.red_bold("refused: could not be read")
            }
        };
        writeln!(stdout, "  {tag:<width$}  {note}")?;
    }
    Ok(refusals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::publish::imagetools::Imagetools;
    use crate::publish::testing::*;
    use crate::publish::validate::OCI_INDEX;
    use serde_json::json;

    /// The `Tags:` line of `tag` in a publish's output.
    fn tag_line<'a>(out: &'a str, tag: &str) -> &'a str {
        out.lines()
            .find(|line| line.trim_start().starts_with(&format!("{tag} ")))
            .unwrap_or_else(|| panic!("no tag line for {tag} in:\n{out}"))
    }

    #[test]
    fn version_tags_do_not_move_without_force_and_channels_say_where_from() {
        let registry = three_variants();
        let mut publish = args("reg.example/ns/pi0:v3", SOURCES);
        publish.tags = vec!["v3.0".to_owned()];
        publish.channel = Some("latest".to_owned());
        let (result, out) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert!(
            tag_line(&out, "reg.example/ns/pi0:v3").ends_with(" new"),
            "{out}"
        );
        // TARGET is pushed alone, then the other tags point at its digest.
        let v3 = registry.digest_of("reg.example/ns/pi0:v3");
        for tag in ["reg.example/ns/pi0:v3.0", "reg.example/ns/pi0:latest"] {
            assert_eq!(registry.digest_of(tag), v3, "{tag}");
        }
        assert_eq!(
            registry.created.borrow()[2],
            [
                "--tag",
                "reg.example/ns/pi0:v3.0",
                "--tag",
                "reg.example/ns/pi0:latest",
                &format!("reg.example/ns/pi0@{v3}"),
            ]
        );
        assert_eq!(registry.pushes(), 2);
        // Publishing the same version again changes nothing and is allowed;
        // no tag is pushed again.
        let (result, out) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert!(
            tag_line(&out, "reg.example/ns/pi0:v3")
                .ends_with(&format!(" unchanged ({v3}), not pushed again")),
            "{out}"
        );
        assert!(
            out.contains(&format!(
                "Already published reg.example/ns/pi0@{v3}; every tag holds this index"
            )),
            "{out}"
        );
        assert_eq!(registry.pushes(), 2);

        // Other children under the same version tag: the dry run says what
        // would be refused, and the real run refuses it.
        let mut other = args("reg.example/ns/pi0:v3", &SOURCES[..2]);
        other.channel = Some("latest".to_owned());
        other.dry_run = true;
        let (result, out) = run(&registry, &other);
        let error = format!("{:#}", result.unwrap_err());
        assert!(
            error.contains(&format!(
                "reg.example/ns/pi0:v3 already points at {v3}, a different index; a version tag \
                 does not move without --force"
            )),
            "{error}"
        );
        assert!(
            out.contains("(nothing pushed)"),
            "the index is shown: {out}"
        );
        assert!(
            tag_line(&out, "reg.example/ns/pi0:v3").ends_with(&format!("refused: points at {v3}")),
            "{out}"
        );
        // The channel is not refused: moving is what it is for.
        assert!(
            tag_line(&out, "reg.example/ns/pi0:latest")
                .ends_with(&format!("moves from {v3} (channel)")),
            "{out}"
        );
        other.dry_run = false;
        assert!(run(&registry, &other).0.is_err());
        assert_eq!(registry.pushes(), 2, "nothing is pushed");
        other.force = true;
        let (result, out) = run(&registry, &other);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert!(
            tag_line(&out, "reg.example/ns/pi0:v3")
                .ends_with(&format!("moves from {v3} (--force)")),
            "{out}"
        );
        assert_eq!(registry.pushes(), 4);
        assert_eq!(
            registry.digest_of("reg.example/ns/pi0:latest"),
            registry.digest_of("reg.example/ns/pi0:v3")
        );

        // A new version moves only the channel, without --force.
        let mut next = args("reg.example/ns/pi0:v4", SOURCES);
        next.channel = Some("latest".to_owned());
        let (result, out) = run(&registry, &next);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert!(
            tag_line(&out, "reg.example/ns/pi0:v4").ends_with(" new"),
            "{out}"
        );
        assert!(
            tag_line(&out, "reg.example/ns/pi0:latest").contains("moves from"),
            "{out}"
        );

        // A channel that is also the version tag is protected like one.
        let mut same = args("reg.example/ns/pi0:latest", &SOURCES[..1]);
        same.channel = Some("latest".to_owned());
        let error = format!("{:#}", run(&registry, &same).0.unwrap_err());
        assert!(error.contains("does not move without --force"), "{error}");
    }

    #[test]
    fn an_unreadable_version_tag_is_refused_without_force() {
        let registry = three_variants();
        let tag = "reg.example/ns/pi0:v5";
        for (canned, reason) in [
            (
                "ERROR: unexpected status from HEAD request to \
                 https://reg.example/v2/ns/pi0/manifests/v5: 401 Unauthorized",
                "401 Unauthorized",
            ),
            (
                "ERROR: failed to do request: Head \"https://reg.example/v2/ns/pi0/manifests/v5\": \
                 dial tcp: i/o timeout",
                "i/o timeout",
            ),
            (
                "ERROR: failed to authorize: failed to fetch oauth token: unexpected status from \
                 GET request to https://reg.example/token: 404 Not Found",
                "token: 404 Not Found",
            ),
            ("<html>bad gateway</html>", "is not JSON"),
        ] {
            registry.insert_raw(tag, canned);
            let mut publish = args(tag, SOURCES);
            publish.dry_run = true;
            let (result, out) = run(&registry, &publish);
            let error = format!("{:#}", result.unwrap_err());
            assert!(
                error.contains(&format!("{tag} could not be read (")) && error.contains(reason),
                "{reason:?} not in:\n{error}"
            );
            assert!(error.contains("pass --force"), "{error}");
            assert!(
                tag_line(&out, tag).ends_with("refused: could not be read"),
                "{out}"
            );
            publish.dry_run = false;
            assert!(run(&registry, &publish).0.is_err());
            assert_eq!(registry.pushes(), 0, "nothing is pushed");
        }
        // A real absence is new.
        registry.manifests.borrow_mut().remove(tag);
        let mut publish = args(tag, SOURCES);
        publish.dry_run = true;
        let (result, out) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert!(tag_line(&out, tag).ends_with(" new"), "{out}");

        // --force pushes over it, and the channel is overwritten regardless;
        // both say why.
        registry.insert_raw(tag, "ERROR: unexpected status: 401 Unauthorized");
        let channel = "reg.example/ns/pi0:latest";
        registry.insert_raw(channel, "ERROR: unexpected status: 503 Service Unavailable");
        publish.dry_run = false;
        publish.channel = Some("latest".to_owned());
        let (result, out) = run(&registry, &publish);
        let error = format!("{:#}", result.unwrap_err());
        assert!(
            error.contains(&format!("{tag} could not be read")),
            "{error}"
        );
        assert!(
            !error.contains(channel),
            "the channel is not refused: {error}"
        );
        publish.force = true;
        let (result, out2) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out2}");
        assert!(
            tag_line(&out2, tag).ends_with(
                "unreadable, overwritten (--force): ERROR: unexpected status: 401 Unauthorized"
            ),
            "{out2}"
        );
        for out in [&out, &out2] {
            assert!(
                tag_line(out, channel).ends_with(
                    "unreadable, overwritten (channel): ERROR: unexpected status: 503 Service \
                     Unavailable"
                ),
                "{out}"
            );
        }
        assert_eq!(registry.pushes(), 2);
        assert_eq!(registry.digest_of(channel), registry.digest_of(tag));
    }

    #[test]
    fn an_existing_index_is_unchanged_only_when_every_field_matches() {
        let registry = three_variants();
        let target = "reg.example/ns/pi0:v3";
        let (result, out) = run(&registry, &args(target, SOURCES));
        assert_eq!(result.unwrap(), 0, "{out}");
        let stored: Value = serde_json::from_str(&registry.raw(target).unwrap()).unwrap();

        let assembled = registry.digest_of(target);

        // The same index, serialized differently (compact, other key order),
        // so at another digest: unchanged, and not pushed again, so its
        // digest stays put.
        let reordered = format!(
            "{{\"manifests\":{},\"schemaVersion\":2,\"mediaType\":\"{OCI_INDEX}\"}}",
            stored["manifests"]
        );
        let version = sha256(&reordered);
        assert_ne!(version, assembled);
        // The channel already holds the same index, as docker assembles it.
        registry.store("reg.example/ns/pi0:latest", &registry.raw(target).unwrap());
        registry.store(target, &reordered);
        let mut publish = args(target, SOURCES);
        publish.tags = vec!["v3.0".to_owned()];
        publish.channel = Some("latest".to_owned());
        let (result, out) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert!(tag_line(&out, target).contains("unchanged"), "{out}");
        assert!(
            tag_line(&out, "reg.example/ns/pi0:latest").ends_with(&format!(
                "same index at {assembled}, re-pointed at {version} (TARGET's digest)"
            )),
            "{out}"
        );
        // Only the other tags are pushed, pointed at TARGET's stored digest
        // rather than assembled again, so every tag is the same manifest.
        let created = registry.created.borrow().last().unwrap().clone();
        assert_eq!(
            created,
            [
                "--tag",
                "reg.example/ns/pi0:v3.0",
                "--tag",
                "reg.example/ns/pi0:latest",
                &format!("reg.example/ns/pi0@{version}"),
            ]
        );
        for tag in [
            target,
            "reg.example/ns/pi0:v3.0",
            "reg.example/ns/pi0:latest",
        ] {
            assert_eq!(registry.digest_of(tag), version, "{tag}");
            assert_eq!(registry.raw(tag).unwrap(), reordered, "{tag}");
        }
        assert!(
            out.contains(&format!("Published reg.example/ns/pi0@{version}")),
            "{out}"
        );
        assert!(
            out.contains(&format!("unchanged {target} (already {version})")),
            "{out}"
        );
        assert!(out.contains("tagged reg.example/ns/pi0:v3.0"), "{out}");
        assert!(
            out.contains("re-pointed reg.example/ns/pi0:latest"),
            "{out}"
        );
        // Now every tag holds the version's digest: nothing to push.
        let pushes = registry.pushes();
        let (result, out) = run(&registry, &publish);
        assert_eq!(result.unwrap(), 0, "{out}");
        assert!(out.contains("Already published"), "{out}");
        assert_eq!(registry.pushes(), pushes);

        // An index carrying a subject or an artifactType the replacement would
        // drop is a different index, though docker's .Manifest view hides both.
        for (field, value) in [
            (
                "subject",
                json!({"mediaType": "application/vnd.oci.image.manifest.v1+json",
                       "digest": digest('s'), "size": 7}),
            ),
            ("artifactType", json!("application/vnd.example.sbom")),
        ] {
            let mut carrying = stored.clone();
            carrying[field] = value;
            registry
                .raws
                .borrow_mut()
                .insert(target.to_owned(), carrying.to_string());
            let error = format!(
                "{:#}",
                run(&registry, &args(target, SOURCES)).0.unwrap_err()
            );
            assert!(
                error.contains(&format!(
                    "{target} already points at {version}, a different index"
                )),
                "{field}: {error}"
            );
        }
        // So is one whose manifests differ only in a descriptor's size.
        let mut resized = stored.clone();
        resized["manifests"][0]["size"] = Value::from(1);
        registry
            .raws
            .borrow_mut()
            .insert(target.to_owned(), resized.to_string());
        assert!(run(&registry, &args(target, SOURCES)).0.is_err());
    }

    #[test]
    fn target_must_be_a_tag_and_extra_tags_join_its_repository() {
        let mut publish = args("localhost:5000/ns/pi0:v3", &["s"]);
        publish.tags = vec!["v3".to_owned(), "v3.0".to_owned()];
        publish.channel = Some("latest".to_owned());
        let (_, tags) = target_tags(&publish).unwrap();
        assert_eq!(
            tags,
            [
                "localhost:5000/ns/pi0:v3",
                "localhost:5000/ns/pi0:v3.0",
                "localhost:5000/ns/pi0:latest"
            ]
        );
        assert!(target_tags(&args("ns/pi0", &["s"])).is_err());
        assert!(target_tags(&args(&format!("ns/pi0@{}", digest('a')), &["s"])).is_err());
        publish.channel = Some("not/a/tag".to_owned());
        assert!(target_tags(&publish).is_err());
    }
}
