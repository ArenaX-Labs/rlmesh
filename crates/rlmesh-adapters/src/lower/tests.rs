//! The binding of v1 declarations, end to end: lowered, bound and applied.

mod unknown_kind_tests {
    use crate::error::ErrorCode;
    use crate::lower::resolve;
    use crate::space_view::SpaceView;
    use crate::spec::{EnvTags, ModelSpec};

    fn space(json: &str) -> SpaceView {
        serde_json::from_str(json).expect("parse space")
    }

    fn do_resolve(
        env_tags: &str,
        obs_space: &str,
        action_space: &str,
        model_spec: &str,
    ) -> Result<crate::plans::Binding, crate::error::AdapterResolutionError> {
        let tags: EnvTags = serde_json::from_str(env_tags).expect("parse env tags");
        let spec: ModelSpec = serde_json::from_str(model_spec).expect("parse model spec");
        resolve(&tags, &space(obs_space), &space(action_space), &spec, false)
    }

    // A one-actuator action shared by the cases below.
    const ACTION_SPACE: &str = r#"{"kind":"box","shape":[1],"dtype":"float32"}"#;
    const ACTION_OUT: &str = r#"{"components":[{"role":"a","dim":1}]}"#;
    const ACTION_TAGS: &str = r#"{"components":[{"role":"a","dim":1}]}"#;

    #[test]
    fn unreferenced_unknown_obs_kind_resolves_with_advisory() {
        // The env declares an `audio` observation an old core can't build. The
        // model references only the camera, so resolution succeeds and the
        // unknown leaf is ignored with one deterministic advisory.
        let env_tags = format!(
            r#"{{"observation":{{"cam":{{"type":"image","role":"image/primary"}},
                "mic":{{"type":"audio","role":"audio/mic","sample_rate":16000}}}},
                "action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["cam","mic"],"children":[
            {"kind":"box","shape":[4,4,3],"dtype":"uint8"},
            {"kind":"box","shape":[16],"dtype":"float32"}]}"#;
        let model = format!(
            r#"{{"input":{{"pixels":{{"type":"image","role":"image/primary"}}}},"output":{ACTION_OUT}}}"#
        );
        let adapter = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect("resolves");
        let advisories = adapter.advisories();
        assert!(
            advisories
                .iter()
                .any(|a| a.message.contains("mic") && a.message.contains("audio")),
            "expected an unknown-kind advisory, got: {advisories:?}"
        );
        // The dropped modality is also surfaced in the human summary (it produces
        // no obs plan, so without this it would be invisible to describe()).
        let described = adapter.describe();
        assert!(
            described.contains("dropped:") && described.contains("mic"),
            "expected a dropped-modality note in describe(), got:\n{described}"
        );
    }

    #[test]
    fn referenced_unknown_obs_kind_is_unsupported_kind() {
        // The model needs role "weird", which the env offers only as an
        // unrecognized kind: a localized UnsupportedKind ("upgrade the runtime"),
        // not a misdirecting MissingRole.
        let env_tags = format!(
            r#"{{"observation":{{"sensor":{{"type":"lidar","role":"weird"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["sensor"],"children":[
            {"kind":"box","shape":[4],"dtype":"float32"}]}"#;
        let model = format!(
            r#"{{"input":{{"s":{{"type":"state","components":["weird"]}}}},"output":{ACTION_OUT}}}"#
        );
        let err = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect_err("unsupported");
        assert_eq!(err.code, ErrorCode::UnsupportedKind);
        assert!(
            err.message.contains("upgrade the runtime"),
            "got: {}",
            err.message
        );
    }

    #[test]
    fn optional_state_referencing_unknown_kind_is_unsupported_kind() {
        // An *optional* component still fails loud when the env provides its role
        // under an unrecognized kind: the data exists but is unreadable here, so
        // the operator is told to upgrade rather than being silently fed zeros.
        let env_tags = format!(
            r#"{{"observation":{{"sensor":{{"type":"lidar","role":"weird"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["sensor"],"children":[
            {"kind":"box","shape":[4],"dtype":"float32"}]}"#;
        let model = format!(
            r#"{{"input":{{"s":{{"type":"state","components":[{{"role":"weird","dim":4,"optional":true}}]}}}},"output":{ACTION_OUT}}}"#
        );
        let err = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect_err("unsupported");
        assert_eq!(err.code, ErrorCode::UnsupportedKind);
        assert!(
            err.message.contains("upgrade the runtime"),
            "got: {}",
            err.message
        );
    }

    #[test]
    fn filled_text_referencing_unknown_kind_is_unsupported_kind() {
        // A text input with a fill still fails loud when the env provides its
        // role under an unrecognized kind -- the fill must not mask present-
        // but-unreadable data.
        let env_tags = format!(
            r#"{{"observation":{{"note":{{"type":"richtext","role":"instruction"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["note"],"children":[
            {"kind":"box","shape":[4],"dtype":"float32"}]}"#;
        let model = format!(
            r#"{{"input":{{"t":{{"type":"text","role":"instruction","fill":"hi"}}}},"output":{ACTION_OUT}}}"#
        );
        let err = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect_err("unsupported");
        assert_eq!(err.code, ErrorCode::UnsupportedKind);
        assert!(
            err.message.contains("upgrade the runtime"),
            "got: {}",
            err.message
        );
    }

    #[test]
    fn lone_camera_fallback_does_not_bind_an_unknown_kind_role() {
        // With exactly one real camera, the lone-camera fallback papers over
        // role-name mismatches -- but not when the requested role is the env's
        // *unknown-kind* leaf: that fails loud (upgrade), it does not silently
        // bind the wrong camera.
        let env_tags = format!(
            r#"{{"observation":{{"cam":{{"type":"image","role":"image/primary"}},
                "extra":{{"type":"lidar","role":"image/overhead"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["cam","extra"],"children":[
            {"kind":"box","shape":[4,4,3],"dtype":"uint8"},
            {"kind":"box","shape":[4],"dtype":"float32"}]}"#;
        let model = format!(
            r#"{{"input":{{"pixels":{{"type":"image","role":"image/overhead"}}}},"output":{ACTION_OUT}}}"#
        );
        let err = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect_err("unsupported");
        assert_eq!(err.code, ErrorCode::UnsupportedKind);
        assert!(
            err.message.contains("upgrade the runtime"),
            "got: {}",
            err.message
        );
    }

    #[test]
    fn model_input_of_unknown_kind_is_unsupported_kind() {
        // A model input of an unrecognized kind has no apply path on an old core;
        // it fails at resolve regardless of what the env offers.
        let env_tags = format!(
            r#"{{"observation":{{"cam":{{"type":"image","role":"image/primary"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["cam"],"children":[
            {"kind":"box","shape":[4,4,3],"dtype":"uint8"}]}"#;
        let model = format!(
            r#"{{"input":{{"x":{{"type":"audio","role":"image/primary"}}}},"output":{ACTION_OUT}}}"#
        );
        let err = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect_err("unsupported");
        assert_eq!(err.code, ErrorCode::UnsupportedKind);
        assert!(
            err.message.contains("unrecognized kind"),
            "got: {}",
            err.message
        );
    }

    #[test]
    fn bare_unknown_field_on_known_kind_taints_at_resolve() {
        // §8 central asymmetry: a bare additive field on a recognized kind is
        // must-understand — fail closed at resolve even though the leaf is
        // referenced and otherwise valid.
        let env_tags = format!(
            r#"{{"observation":{{"cam":{{"type":"image","role":"image/primary","normalize":false}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["cam"],"children":[
            {"kind":"box","shape":[4,4,3],"dtype":"uint8"}]}"#;
        let model = format!(
            r#"{{"input":{{"pixels":{{"type":"image","role":"image/primary"}}}},"output":{ACTION_OUT}}}"#
        );
        let err = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect_err("tainted");
        assert_eq!(err.code, ErrorCode::UnsupportedKind);
        assert!(err.message.contains("normalize"), "got: {}", err.message);
    }

    /// Deliberate invariant: join advisories (env-declaration hints like a
    /// mislabeled image layout) surface through `advisories()` but stay OUT of
    /// `describe()` — they are not dropped env modalities, and the conformance
    /// vectors pin describe()'s exact text including its `dropped:` section.
    #[test]
    fn image_layout_advisory_surfaces_in_advisories_but_not_describe() {
        let env_tags = format!(
            r#"{{"observation":{{"cam":{{"type":"image","role":"image/primary","layout":"hwc"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["cam"],"children":[
            {"kind":"box","shape":[3,224,224],"dtype":"uint8"}]}"#;
        let model = format!(
            r#"{{"input":{{"pixels":{{"type":"image","role":"image/primary"}}}},"output":{ACTION_OUT}}}"#
        );
        let adapter = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect("resolves");
        let advisories = adapter.advisories();
        assert!(
            advisories
                .iter()
                .any(|a| a.message.contains("layout=hwc") && a.message.contains("looks like chw")),
            "expected the join layout hint in advisories(), got: {advisories:?}"
        );
        let described = adapter.describe();
        assert!(
            !described.contains("looks like chw") && !described.contains("dropped:"),
            "join advisories must stay out of describe(), got:\n{described}"
        );
    }

    #[test]
    fn x_prefixed_field_does_not_taint_at_resolve() {
        // The producer's `x-` opt-out: a marked-ignorable field resolves cleanly.
        let env_tags = format!(
            r#"{{"observation":{{"cam":{{"type":"image","role":"image/primary","x-note":"hi"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["cam"],"children":[
            {"kind":"box","shape":[4,4,3],"dtype":"uint8"}]}"#;
        let model = format!(
            r#"{{"input":{{"pixels":{{"type":"image","role":"image/primary"}}}},"output":{ACTION_OUT}}}"#
        );
        do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect("x- field tolerated");
    }

    #[test]
    fn ad_hoc_role_the_env_lacks_draws_one_info_advisory() {
        // The optional part zero-fills instead of hard-erroring, so without the
        // advisory the author never learns their private role matched nothing.
        let env_tags = format!(
            r#"{{"observation":{{"cam":{{"type":"image","role":"image/primary"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["cam"],"children":[
            {"kind":"box","shape":[4,4,3],"dtype":"uint8"}]}"#;
        let model = format!(
            r#"{{"input":{{"s":{{"type":"state","components":[
                {{"role":"proprio/made_up","dim":3,"optional":true}}]}}}},"output":{ACTION_OUT}}}"#
        );
        let adapter = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect("resolves");
        let advisories = adapter.advisories();
        let hits: Vec<&rlmesh_session_core::spec::role::Advisory> = advisories
            .iter()
            .filter(|a| a.message.contains("ad-hoc role"))
            .collect();
        assert_eq!(hits.len(), 1, "one advisory per role, got: {hits:?}");
        assert!(hits[0].message.contains("proprio/made_up"), "{hits:?}");
        // A registered role the env also lacks is a contract, not a typo: silent.
        let model = format!(
            r#"{{"input":{{"s":{{"type":"state","components":[
                {{"role":"proprio/eef_pos","dim":3,"optional":true}}]}}}},"output":{ACTION_OUT}}}"#
        );
        let adapter = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect("resolves");
        assert!(
            !adapter
                .advisories()
                .iter()
                .any(|a| a.message.contains("ad-hoc role")),
            "{:?}",
            adapter.advisories()
        );
    }

    #[test]
    fn a_camera_role_rebinds_to_the_only_camera_only_when_allowed_and_unparted() {
        let env_tags = format!(
            r#"{{"observation":{{"cam":{{"type":"image","role":"image/primary"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs_space = r#"{"kind":"dict","dtype":"unspecified","keys":["cam"],"children":[
            {"kind":"box","shape":[4,4,3],"dtype":"uint8"}]}"#;
        let model = format!(
            r#"{{"input":{{"pixels":{{"type":"image","role":"image/wrist","part":"right_arm"}}}},"output":{ACTION_OUT}}}"#
        );
        let err = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect_err("no rebind");
        assert_eq!(err.code, ErrorCode::MissingRole);
        // An unparted wrist rebinds only under the named projection: a lone
        // third-person camera is not a wrist view. A named part never does,
        // since a lone camera cannot be one arm of a pair.
        let model = format!(
            r#"{{"input":{{"pixels":{{"type":"image","role":"image/wrist"}}}},"output":{ACTION_OUT}}}"#
        );
        let err = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect_err("refused");
        assert_eq!(err.code, ErrorCode::MissingRole);
        assert!(err.message.contains("rebind_camera"), "{}", err.message);
        let model = format!(
            r#"{{"input":{{"pixels":{{"type":"image","role":"image/wrist"}}}},"output":{ACTION_OUT},"allow":["rebind_camera"]}}"#
        );
        let adapter = do_resolve(&env_tags, obs_space, ACTION_SPACE, &model).expect("rebinds");
        assert!(
            adapter
                .advisories()
                .iter()
                .any(|a| a.message.contains("only camera") && !a.message.contains("  ")),
            "{:?}",
            adapter.advisories()
        );
    }
}

/// The C14 geometry rules, exercised directly on [`check_geometry`] so each of
/// the five cases is pinned once rather than through a full resolve.
/// The `part` identity rules (section 3 of the parts design), end to end.
mod part_tests {
    use crate::error::ErrorCode;
    use crate::lower::resolve;
    use crate::space_view::SpaceView;
    use crate::spec::{EnvTags, ModelSpec};
    use rlmesh_session_core::spec::role::Severity as AdvisorySeverity;

    fn space(json: &str) -> SpaceView {
        serde_json::from_str(json).expect("parse space")
    }

    fn do_resolve(
        env_tags: &str,
        obs_space: &str,
        action_space: &str,
        model_spec: &str,
    ) -> Result<crate::plans::Binding, crate::error::AdapterResolutionError> {
        let tags: EnvTags = serde_json::from_str(env_tags).expect("parse env tags");
        let spec: ModelSpec = serde_json::from_str(model_spec).expect("parse model spec");
        resolve(&tags, &space(obs_space), &space(action_space), &spec, false)
    }

    const ACTION_SPACE: &str = r#"{"kind":"box","shape":[1],"dtype":"float32"}"#;
    const ACTION_OUT: &str = r#"{"components":[{"role":"action/gripper","dim":1}]}"#;
    const ACTION_TAGS: &str = r#"{"components":[{"role":"action/gripper","dim":1}]}"#;
    const TWO_POS: &str = r#"{"kind":"dict","dtype":"unspecified","keys":["l","r"],"children":[
        {"kind":"box","shape":[3],"dtype":"float32"},
        {"kind":"box","shape":[3],"dtype":"float32"}]}"#;
    const ONE_POS: &str = r#"{"kind":"dict","dtype":"unspecified","keys":["l"],"children":[
        {"kind":"box","shape":[3],"dtype":"float32"}]}"#;

    fn env_two_arms() -> String {
        format!(
            r#"{{"observation":{{
                "l":{{"type":"state","role":"proprio/eef_pos","part":"left_arm"}},
                "r":{{"type":"state","role":"proprio/eef_pos","part":"right_arm"}}}},
                "action":{ACTION_TAGS}}}"#
        )
    }

    fn env_one_arm(part: &str) -> String {
        format!(
            r#"{{"observation":{{"l":{{"type":"state","role":"proprio/eef_pos","part":{part:?}}}}},
                "action":{ACTION_TAGS}}}"#
        )
    }

    fn model_part(part: Option<&str>) -> String {
        let part = part.map_or(String::new(), |part| format!(r#","part":{part:?}"#));
        format!(
            r#"{{"input":{{"s":{{"type":"state","components":[{{"role":"proprio/eef_pos","dim":3{part}}}]}}}},"output":{ACTION_OUT}}}"#
        )
    }

    #[test]
    fn a_named_part_binds_only_its_leaf_and_prints_it() {
        let adapter = do_resolve(
            &env_two_arms(),
            TWO_POS,
            ACTION_SPACE,
            &model_part(Some("right_arm")),
        )
        .expect("resolves");
        let described = adapter.describe();
        assert!(
            described.contains(r##""s" <- concat(r[:3]#right_arm)"##),
            "got:\n{described}"
        );
        assert!(
            adapter.advisories().is_empty(),
            "{:?}",
            adapter.advisories()
        );
    }

    #[test]
    fn no_part_against_several_parts_is_missing_role_naming_them() {
        let err = do_resolve(&env_two_arms(), TWO_POS, ACTION_SPACE, &model_part(None))
            .expect_err("ambiguous");
        assert_eq!(err.code, ErrorCode::MissingRole);
        assert!(
            err.message
                .contains(r#"under parts ["left_arm", "right_arm"]; declare part="#),
            "got: {}",
            err.message
        );
        // `optional` does not turn ambiguity into a fill: the env has the data.
        let optional = format!(
            r#"{{"input":{{"s":{{"type":"state","components":[{{"role":"proprio/eef_pos","dim":3,"optional":true}}]}}}},"output":{ACTION_OUT}}}"#
        );
        let err = do_resolve(&env_two_arms(), TWO_POS, ACTION_SPACE, &optional).expect_err("err");
        assert_eq!(err.code, ErrorCode::MissingRole);
    }

    #[test]
    fn no_part_against_one_part_rebinds_with_an_info() {
        let adapter = do_resolve(
            &env_one_arm("left_arm"),
            ONE_POS,
            ACTION_SPACE,
            &model_part(None),
        )
        .expect("resolves");
        let notes = adapter.advisories();
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert_eq!(notes[0].severity, AdvisorySeverity::Info);
        assert!(
            notes[0]
                .message
                .contains(r#"declares it only under part "left_arm"; bound to that leaf"#),
            "{}",
            notes[0].message
        );
        assert!(
            adapter.describe().contains("l[:3]#left_arm"),
            "got:\n{}",
            adapter.describe()
        );
        assert!(!adapter.describe().contains("dropped:"));
    }

    #[test]
    fn a_named_part_never_rebinds() {
        // The env has the role under another part: a named part is missing,
        // not rebound -- fill if optional, else MissingRole naming what exists.
        let err = do_resolve(
            &env_one_arm("left_arm"),
            ONE_POS,
            ACTION_SPACE,
            &model_part(Some("torso")),
        )
        .expect_err("missing");
        assert_eq!(err.code, ErrorCode::MissingRole);
        assert!(
            err.message.contains(r#"(part "torso")"#)
                && err.message.contains(r#"["proprio/eef_pos#left_arm"]"#),
            "got: {}",
            err.message
        );
        let optional = format!(
            r#"{{"input":{{"s":{{"type":"state","components":[{{"role":"proprio/eef_pos","dim":3,"part":"torso","optional":true}}]}}}},"output":{ACTION_OUT}}}"#
        );
        let adapter =
            do_resolve(&env_one_arm("left_arm"), ONE_POS, ACTION_SPACE, &optional).expect("fills");
        assert!(
            adapter.describe().contains("zeros(3)#torso"),
            "got:\n{}",
            adapter.describe()
        );
    }

    #[test]
    fn an_unknown_kind_prefix_parses_and_fails_at_resolve() {
        // The kind set is closed; a newer peer's kind relays and dies here,
        // named. `x/` stays the whole-role escape.
        let model = format!(
            r#"{{"input":{{"s":{{"type":"state","components":[{{"role":"audio/mic","dim":3}}]}}}},"output":{ACTION_OUT}}}"#
        );
        let err =
            do_resolve(&env_one_arm("left_arm"), ONE_POS, ACTION_SPACE, &model).expect_err("err");
        assert_eq!(err.code, ErrorCode::UnsupportedKind);
        assert!(
            err.message.contains(r#"model input "s""#)
                && err.message.contains("kind this core does not define")
                && err.message.contains("upgrade the runtime"),
            "{}",
            err.message
        );
        let env = format!(
            r#"{{"observation":{{"l":{{"type":"state","role":"audio/mic"}}}},"action":{ACTION_TAGS}}}"#
        );
        let err = do_resolve(&env, ONE_POS, ACTION_SPACE, &model_part(None)).expect_err("err");
        assert_eq!(err.code, ErrorCode::InvalidTag);
        let escaped = format!(
            r#"{{"observation":{{"l":{{"type":"state","role":"x/mic"}}}},"action":{ACTION_TAGS}}}"#
        );
        let model = format!(
            r#"{{"input":{{"s":{{"type":"state","components":[{{"role":"x/mic","dim":3}}]}}}},"output":{ACTION_OUT}}}"#
        );
        do_resolve(&escaped, ONE_POS, ACTION_SPACE, &model).expect("x/ escapes");
    }

    #[test]
    fn an_action_role_split_by_part_binds_per_arm() {
        let env = r#"{"observation":{"l":{"type":"state","role":"proprio/eef_pos","part":"left_arm"}},
            "action":{"components":[
                {"role":"action/joint_pos","dim":2,"part":"left_arm"},
                {"role":"action/joint_pos","dim":2,"part":"right_arm"}]}}"#;
        let model = r#"{"input":{"s":{"type":"state","components":[{"role":"proprio/eef_pos","dim":3,"part":"left_arm"}]}},
            "output":{"components":[
                {"role":"action/joint_pos","dim":2,"part":"right_arm"},
                {"role":"action/joint_pos","dim":2,"part":"left_arm"}]}}"#;
        let action_space = r#"{"kind":"box","shape":[4],"dtype":"float32"}"#;
        let adapter = do_resolve(env, ONE_POS, action_space, model).expect("resolves");
        let described = adapter.describe();
        assert!(
            described.contains(r##""action/joint_pos" <- model[2:4]#left_arm"##)
                && described.contains(r##""action/joint_pos" <- model[0:2]#right_arm"##),
            "got:\n{described}"
        );
        // A part-less env actuator binds the model's only output of the role
        // under any part, with the same info; two candidates are an error.
        let env_bare = r#"{"observation":{"l":{"type":"state","role":"proprio/eef_pos","part":"left_arm"}},
            "action":{"components":[{"role":"action/joint_pos","dim":2}]}}"#;
        let model_one = r#"{"input":{"s":{"type":"state","components":[{"role":"proprio/eef_pos","dim":3,"part":"left_arm"}]}},
            "output":{"components":[{"role":"action/joint_pos","dim":2,"part":"left_arm"}]}}"#;
        let adapter = do_resolve(
            env_bare,
            ONE_POS,
            r#"{"kind":"box","shape":[2],"dtype":"float32"}"#,
            model_one,
        )
        .expect("resolves");
        assert!(
            adapter.advisories().iter().any(|note| note
                .message
                .contains(r#"env action: role "action/joint_pos" declares no part"#)),
            "{:?}",
            adapter.advisories()
        );
        let err = do_resolve(
            env_bare,
            ONE_POS,
            r#"{"kind":"box","shape":[2],"dtype":"float32"}"#,
            model,
        )
        .expect_err("ambiguous");
        assert_eq!(err.code, ErrorCode::MissingRole);
        assert!(err.message.contains("declare part="), "{}", err.message);
    }

    #[test]
    fn a_custom_part_binds_on_exact_agreement_with_no_advisory() {
        for part in ["franka", "x/franka"] {
            let adapter = do_resolve(
                &env_one_arm(part),
                ONE_POS,
                ACTION_SPACE,
                &model_part(Some(part)),
            )
            .expect("resolves on exact agreement");
            assert!(
                adapter.advisories().is_empty(),
                "{:?}",
                adapter.advisories()
            );
            assert!(!adapter.describe().contains("dropped:"));
        }
    }

    #[test]
    fn a_parted_camera_follows_the_same_rules() {
        let env = format!(
            r#"{{"observation":{{"cam":{{"type":"image","role":"image/wrist","part":"left_arm"}}}},"action":{ACTION_TAGS}}}"#
        );
        let obs = r#"{"kind":"dict","dtype":"unspecified","keys":["cam"],"children":[
            {"kind":"box","shape":[4,4,3],"dtype":"uint8"}]}"#;
        let model = format!(
            r#"{{"input":{{"pixels":{{"type":"image","role":"image/wrist"}}}},"output":{ACTION_OUT}}}"#
        );
        let adapter = do_resolve(&env, obs, ACTION_SPACE, &model).expect("rebinds by part");
        assert!(
            adapter
                .describe()
                .contains(r##"<- image "cam"#left_arm ("##),
            "{}",
            adapter.describe()
        );
        assert!(
            adapter
                .advisories()
                .iter()
                .all(|note| note.severity == AdvisorySeverity::Info),
            "{:?}",
            adapter.advisories()
        );
        // A named part never falls through to the lone-camera fallback.
        let model = format!(
            r#"{{"input":{{"pixels":{{"type":"image","role":"image/primary","part":"head"}}}},"output":{ACTION_OUT}}}"#
        );
        let err = do_resolve(&env, obs, ACTION_SPACE, &model).expect_err("no rebind");
        assert_eq!(err.code, ErrorCode::MissingRole);
    }
}

/// The `labels` layout rules (section 3 of the design), end to end, on the
/// Go2: SDK motor order on the env, Isaac order on a model.
mod labels_tests {
    use std::collections::BTreeMap;

    use crate::apply::{NoCustoms, Value};
    use crate::error::ErrorCode;
    use crate::lower::resolve;
    use crate::space_view::SpaceView;
    use crate::spec::{EnvTags, ModelSpec};

    /// Go2 joints in SDK motor order: FR, FL, RR, RL x hip, thigh, calf.
    const GO2_JOINTS: [&str; 12] = [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ];

    /// SDK order → Isaac order: FL,FR,RL,RR x hip,thigh,calf.
    const ISAAC: [usize; 12] = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8];

    fn labels(order: &[usize]) -> String {
        let names: Vec<String> = order
            .iter()
            .map(|&i| format!("{:?}", GO2_JOINTS[i]))
            .collect();
        format!("[{}]", names.join(","))
    }

    fn sdk() -> String {
        labels(&(0..12).collect::<Vec<_>>())
    }

    fn space(json: &str) -> SpaceView {
        serde_json::from_str(json).expect("parse space")
    }

    fn go2_env(actuator_extra: &str) -> String {
        let sdk = sdk();
        format!(
            r#"{{"observation":{{
                "joint_pos":{{"type":"state","role":"proprio/joint_pos","labels":{sdk}}},
                "joint_vel":{{"type":"state","role":"proprio/joint_vel","labels":{sdk}}}}},
                "action":{{"components":[{{"role":"action/joint_pos","dim":12,"labels":{sdk}{actuator_extra}}}]}}}}"#
        )
    }

    const GO2_OBS: &str = r#"{"kind":"dict","dtype":"unspecified","keys":["joint_pos","joint_vel"],"children":[
        {"kind":"box","shape":[12],"dtype":"float32"},
        {"kind":"box","shape":[12],"dtype":"float32"}]}"#;
    const GO2_ACT: &str = r#"{"kind":"box","shape":[12],"dtype":"float32"}"#;

    fn go2_model(model_labels: Option<&str>) -> String {
        let labels = model_labels.map_or(String::new(), |labels| format!(r#","labels":{labels}"#));
        let dim = if model_labels.is_some() {
            ""
        } else {
            r#","dim":12"#
        };
        format!(
            r#"{{"input":{{"obs":{{"type":"state","components":[
                {{"role":"proprio/joint_pos"{dim}{labels},"axis_offset":[0.0,-0.8,1.5,0.0,-0.8,1.5,0.0,-0.8,1.5,0.0,-0.8,1.5]}},
                {{"role":"proprio/joint_vel"{dim}{labels},"scale":0.05}}]}}}},
                "output":{{"components":[{{"role":"action/joint_pos","dim":12{labels},
                    "axis_scale":[0.125,0.25,0.25,0.125,0.25,0.25,0.125,0.25,0.25,0.125,0.25,0.25],
                    "axis_offset":[0.0,0.8,-1.5,0.0,0.8,-1.5,0.0,0.8,-1.5,0.0,0.8,-1.5]}}]}}}}"#
        )
    }

    fn do_resolve(
        env: &str,
        model: &str,
    ) -> Result<crate::plans::Binding, crate::error::AdapterResolutionError> {
        let tags: EnvTags = serde_json::from_str(env).expect("parse env tags");
        let spec: ModelSpec = serde_json::from_str(model).expect("parse model spec");
        resolve(&tags, &space(GO2_OBS), &space(GO2_ACT), &spec, false)
    }

    fn tensor(values: &[f32]) -> Value {
        Value::Tensor(crate::apply::value::tensor_from_f32(
            vec![values.len() as i64],
            values,
        ))
    }

    #[test]
    fn an_isaac_order_model_resolves_to_the_permutation_on_both_sides() {
        let adapter =
            do_resolve(&go2_env(""), &go2_model(Some(&labels(&ISAAC)))).expect("resolves");
        let described = adapter.describe();
        assert!(
            described.contains("joint_pos perm[3,4,5,0,1,2,9,10,11,6,7,8] (+[FL_hip_joint:0.0,FL_thigh_joint:-0.8,FL_calf_joint:1.5,")
                && described.contains("joint_vel perm[3,4,5,0,1,2,9,10,11,6,7,8] (*0.05)")
                && described.contains(
                    "\"action/joint_pos\" <- model[0:12] (model *[FL_hip_joint:0.125,FL_thigh_joint:0.25,FL_calf_joint:0.25,"
                )
                && described.ends_with("]) perm[3,4,5,0,1,2,9,10,11,6,7,8]"),
            "got:\n{described}"
        );
        // The permutation is its own inverse on the Go2, so the action-side
        // scatter lists the same indices; the values prove the direction.
        assert!(
            adapter.advisories().is_empty(),
            "{:?}",
            adapter.advisories()
        );
        let mut raw: BTreeMap<String, Value> = BTreeMap::new();
        let sdk_values: Vec<f32> = (0..12).map(|i| i as f32).collect();
        raw.insert("joint_pos".to_owned(), tensor(&sdk_values));
        raw.insert("joint_vel".to_owned(), tensor(&sdk_values));
        let Value::Map(payload) =
            crate::apply::transform_obs(&adapter, &raw, &NoCustoms).expect("apply")
        else {
            panic!("expected a map");
        };
        let Value::Tensor(obs) = &payload["obs"] else {
            panic!("expected a tensor");
        };
        let obs = crate::apply::value::to_f32_vec(obs);
        // joint_pos: SDK value at ISAAC[i], minus the stand pose in Isaac order.
        let expected_pos: Vec<f32> = ISAAC
            .iter()
            .enumerate()
            .map(|(i, &j)| sdk_values[j] + [0.0, -0.8, 1.5][i % 3])
            .collect();
        for (got, want) in obs[..12].iter().zip(&expected_pos) {
            assert!((got - want).abs() < 1e-6, "{obs:?}");
        }
        assert!((obs[12] - 3.0 * 0.05).abs() < 1e-6, "{obs:?}");
        // Action: the model emits ones in Isaac order; each env axis reads its
        // own joint's scale and pose back in SDK order.
        let action = crate::apply::transform_action(&adapter, &tensor(&[1.0; 12])).expect("apply");
        let action = crate::apply::value::to_f32_vec(&action);
        let expected_act: Vec<f32> = (0..12)
            .map(|j| [0.125, 0.25, 0.25][j % 3] + [0.0, 0.8, -1.5][j % 3])
            .collect();
        for (got, want) in action.iter().zip(&expected_act) {
            assert!((got - want).abs() < 1e-6, "{action:?}");
        }
    }

    #[test]
    fn an_sdk_order_model_is_the_identity() {
        let adapter = do_resolve(&go2_env(""), &go2_model(Some(&sdk()))).expect("resolves");
        let described = adapter.describe();
        assert!(
            described.contains("joint_pos[:12] (+[FR_hip_joint:0.0,FR_thigh_joint:-0.8,")
                && !described.contains("perm")
                && !described.contains("select"),
            "got:\n{described}"
        );
        assert!(
            adapter.advisories().is_empty(),
            "{:?}",
            adapter.advisories()
        );
    }

    #[test]
    fn model_only_labels_are_a_mismatch_and_env_only_labels_are_silent() {
        let unlabeled_env = r#"{"observation":{
                "joint_pos":{"type":"state","role":"proprio/joint_pos"},
                "joint_vel":{"type":"state","role":"proprio/joint_vel"}},
                "action":{"components":[{"role":"action/joint_pos","dim":12}]}}"#;
        let err = do_resolve(unlabeled_env, &go2_model(Some(&sdk()))).expect_err("model-only");
        assert_eq!(err.code, ErrorCode::LabelMismatch);
        assert!(
            err.message.contains("the env leaf declares none"),
            "{}",
            err.message
        );
        // Env-only: silent, and the env's names annotate a per-axis vector the
        // model declared positionally.
        let adapter = do_resolve(&go2_env(""), &go2_model(None)).expect("env-only labels resolve");
        let described = adapter.describe();
        assert!(
            described.contains("joint_pos[:12] (+[FR_hip_joint:0.0,FR_thigh_joint:-0.8,")
                && described.contains("(model *[FR_hip_joint:0.125,"),
            "got:\n{described}"
        );
        assert!(
            adapter.advisories().is_empty(),
            "{:?}",
            adapter.advisories()
        );
    }

    #[test]
    fn a_subset_of_the_env_labels_is_a_mismatch_on_either_side() {
        // A front-legs-only checkpoint: six of the twelve labels.
        let front = labels(&[3, 4, 5, 0, 1, 2]);
        let model = format!(
            r#"{{"input":{{"obs":{{"type":"state","components":[{{"role":"proprio/joint_pos","labels":{front}}}]}}}},
                "output":{{"components":[{{"role":"action/joint_pos","dim":12}}]}}}}"#
        );
        let err = do_resolve(&go2_env(""), &model).expect_err("observation subset");
        assert_eq!(err.code, ErrorCode::LabelMismatch);
        assert!(
            err.message
                .contains(r#"the model input lacks ["RR_hip_joint""#)
                && !err.message.contains("the env leaf lacks"),
            "{}",
            err.message
        );
        let model = format!(
            r#"{{"input":{{"obs":{{"type":"state","components":[{{"role":"proprio/joint_pos","dim":12}}]}}}},
                "output":{{"components":[{{"role":"action/joint_pos","dim":6,"labels":{front}}}]}}}}"#
        );
        let env = go2_env(r#","optional":true,"fill":0.5"#);
        let err = do_resolve(&env, &model).expect_err("action subset, even optional");
        assert_eq!(err.code, ErrorCode::LabelMismatch);
        assert!(
            err.message.contains(r#"the model lacks ["RR_hip_joint""#),
            "{}",
            err.message
        );
    }

    #[test]
    fn a_label_the_env_lacks_is_a_mismatch_even_when_the_part_is_optional() {
        let mut model_labels: Vec<String> = GO2_JOINTS
            .iter()
            .map(|label| format!("{label:?}"))
            .collect();
        model_labels[1] = r#""FR_shin""#.to_owned();
        let model_labels = format!("[{}]", model_labels.join(","));
        for optional in ["", r#","optional":true"#] {
            let model = format!(
                r#"{{"input":{{"obs":{{"type":"state","components":[{{"role":"proprio/joint_pos","labels":{model_labels}{optional}}}]}}}},
                    "output":{{"components":[{{"role":"action/joint_pos","dim":12}}]}}}}"#
            );
            let err = do_resolve(&go2_env(""), &model).expect_err("mismatched label");
            assert_eq!(err.code, ErrorCode::LabelMismatch);
            assert!(
                err.message.contains(r#"the env leaf lacks ["FR_shin"]"#)
                    && err
                        .message
                        .contains(r#"the model input lacks ["FR_thigh_joint"]"#),
                "{}",
                err.message
            );
        }
    }

    #[test]
    fn a_three_cycle_of_custom_labels_permutes_each_direction_the_right_way() {
        // Not its own inverse, so a gather/scatter direction mistake shows;
        // the names are no robot's, and no registry is consulted.
        let env = r#"{"observation":{"q":{"type":"state","role":"proprio/joint_pos","labels":["tail_a","tail_b","tail_c"]}},
            "action":{"components":[{"role":"action/joint_pos","dim":3,"labels":["tail_a","tail_b","tail_c"]}]}}"#;
        let model = r#"{"input":{"q":{"type":"state","components":[{"role":"proprio/joint_pos","labels":["tail_b","tail_c","tail_a"]}]}},
            "output":{"components":[{"role":"action/joint_pos","dim":3,"labels":["tail_b","tail_c","tail_a"]}]}}"#;
        let tags: EnvTags = serde_json::from_str(env).expect("parse env tags");
        let spec: ModelSpec = serde_json::from_str(model).expect("parse model spec");
        let obs = space(
            r#"{"kind":"dict","dtype":"unspecified","keys":["q"],"children":[{"kind":"box","shape":[3],"dtype":"float32"}]}"#,
        );
        let act = space(r#"{"kind":"box","shape":[3],"dtype":"float32"}"#);
        let adapter = resolve(&tags, &obs, &act, &spec, false).expect("resolves");
        assert!(
            adapter.advisories().is_empty(),
            "{:?}",
            adapter.advisories()
        );
        let described = adapter.describe();
        assert!(
            described.contains("q perm[1,2,0]") && described.ends_with("perm[2,0,1]"),
            "got:\n{described}"
        );
        let mut raw: BTreeMap<String, Value> = BTreeMap::new();
        raw.insert("q".to_owned(), tensor(&[10.0, 20.0, 30.0]));
        let Value::Map(payload) =
            crate::apply::transform_obs(&adapter, &raw, &NoCustoms).expect("apply")
        else {
            panic!("expected a map");
        };
        let Value::Tensor(q) = &payload["q"] else {
            panic!("expected a tensor");
        };
        assert_eq!(crate::apply::value::to_f32_vec(q), vec![20.0, 30.0, 10.0]);
        let action =
            crate::apply::transform_action(&adapter, &tensor(&[20.0, 30.0, 10.0])).expect("apply");
        assert_eq!(
            crate::apply::value::to_f32_vec(&action),
            vec![10.0, 20.0, 30.0]
        );
    }

    #[test]
    fn a_per_axis_vector_must_match_the_resolved_width() {
        let model = r#"{"input":{"obs":{"type":"state","components":[{"role":"proprio/joint_pos","dim":12,"axis_scale":[1.0,2.0]}]}},
            "output":{"components":[{"role":"action/joint_pos","dim":12}]}}"#;
        let err = do_resolve(&go2_env(""), model).expect_err("short vector");
        assert_eq!(err.code, ErrorCode::DimMismatch);
        assert!(
            err.message
                .contains("axis_scale has 2 values but the resolved width is 12"),
            "{}",
            err.message
        );
        let model = r#"{"input":{"obs":{"type":"state","components":[{"role":"proprio/joint_pos","dim":12}]}},
            "output":{"components":[{"role":"action/joint_pos","dim":12,"axis_offset":[1.0]}]}}"#;
        let err = do_resolve(&go2_env(""), model).expect_err("short vector");
        assert_eq!(err.code, ErrorCode::DimMismatch);
        assert!(
            err.message.contains("the model's axis_offset has 1 values"),
            "{}",
            err.message
        );
    }

    #[test]
    fn unfamiliar_env_labels_resolve_silently() {
        let env = r#"{"observation":{"j":{"type":"state","role":"proprio/joint_pos","labels":["j0","j1"]}},
            "action":{"components":[{"role":"action/joint_pos","dim":12}]}}"#;
        let obs = r#"{"kind":"dict","dtype":"unspecified","keys":["j"],"children":[{"kind":"box","shape":[2],"dtype":"float32"}]}"#;
        let model = r#"{"input":{"obs":{"type":"state","components":[{"role":"proprio/joint_pos","dim":2}]}},
            "output":{"components":[{"role":"action/joint_pos","dim":12}]}}"#;
        let tags: EnvTags = serde_json::from_str(env).unwrap();
        let spec: ModelSpec = serde_json::from_str(model).unwrap();
        let adapter = resolve(&tags, &space(obs), &space(GO2_ACT), &spec, false).expect("resolves");
        assert!(
            adapter.advisories().is_empty(),
            "{:?}",
            adapter.advisories()
        );
    }

    /// The Go2 body: gyro, IMU orientation and the velocity command beside the
    /// joints, with the model reading the orientation as projected gravity and
    /// clamping the assembled vector (rl_sar `robot_lab`, 45 - 12 for the
    /// previous action, which is not part of this pairing).
    fn go2_body_env(base_quat: &str) -> String {
        let sdk = sdk();
        format!(
            r#"{{"observation":{{
                "ang_vel":{{"type":"state","role":"proprio/base_ang_vel","frame":"robot_base","provenance":"sensed","range":[-20.0,20.0]}},
                {base_quat},
                "command":{{"type":"state","role":"command/base_vel","frame":"robot_base","range":[-3.0,3.0]}},
                "joint_pos":{{"type":"state","role":"proprio/joint_pos","labels":{sdk},"provenance":"sensed"}},
                "joint_vel":{{"type":"state","role":"proprio/joint_vel","labels":{sdk},"provenance":"sensed","range":[-30.0,30.0]}}}},
                "action":{{"components":[{{"role":"action/joint_pos","dim":12,"labels":{sdk}}}]}}}}"#
        )
    }

    const GO2_BODY_QUAT: &str = r#""base_quat":{"type":"state","role":"proprio/base_rot","encoding":"quat_wxyz","frame":"world","provenance":"sensed"}"#;

    fn go2_body_obs(keys: &[&str]) -> String {
        let children: Vec<&str> = keys
            .iter()
            .map(|key| match *key {
                "joint_pos" | "joint_vel" => r#"{"kind":"box","shape":[12],"dtype":"float32"}"#,
                "base_quat" | "base_quat_est" => r#"{"kind":"box","shape":[4],"dtype":"float32"}"#,
                _ => r#"{"kind":"box","shape":[3],"dtype":"float32"}"#,
            })
            .collect();
        let keys: Vec<String> = keys.iter().map(|key| format!("{key:?}")).collect();
        format!(
            r#"{{"kind":"dict","dtype":"unspecified","keys":[{}],"children":[{}]}}"#,
            keys.join(","),
            children.join(",")
        )
    }

    fn go2_body_model(provenance: &str) -> String {
        let sdk = sdk();
        format!(
            r#"{{"input":{{"obs":{{"type":"state","clip":[-100.0,100.0],"components":[
                {{"role":"proprio/base_ang_vel","frame":"robot_base","scale":0.25}},
                {{"role":"proprio/base_rot","encoding":"gravity_xyz","frame":"world"{provenance}}},
                {{"role":"command/base_vel","frame":"robot_base"}},
                {{"role":"proprio/joint_pos","labels":{sdk},"axis_offset":[0.0,-0.8,1.5,0.0,-0.8,1.5,0.0,-0.8,1.5,0.0,-0.8,1.5]}},
                {{"role":"proprio/joint_vel","labels":{sdk},"scale":0.05}}]}}}},
                "output":{{"components":[{{"role":"action/joint_pos","dim":12,"labels":{sdk},
                    "axis_scale":[0.125,0.25,0.25,0.125,0.25,0.25,0.125,0.25,0.25,0.125,0.25,0.25],
                    "axis_offset":[0.0,0.8,-1.5,0.0,0.8,-1.5,0.0,0.8,-1.5,0.0,0.8,-1.5]}}]}}}}"#
        )
    }

    #[test]
    fn the_go2_body_resolves_to_a_33_wide_clamped_observation_reading_gravity() {
        let tags: EnvTags = serde_json::from_str(&go2_body_env(GO2_BODY_QUAT)).unwrap();
        let spec: ModelSpec = serde_json::from_str(&go2_body_model("")).unwrap();
        let obs = go2_body_obs(&["ang_vel", "base_quat", "command", "joint_pos", "joint_vel"]);
        let adapter =
            resolve(&tags, &space(&obs), &space(GO2_ACT), &spec, false).expect("resolves");
        let described = adapter.describe();
        assert!(
            described.contains(
                "concat(ang_vel (*0.25)@robot_base#sensed, base_quat (quat_wxyz->gravity_xyz)@world#sensed, command@robot_base, joint_pos[:12] (+[FR_hip_joint:0.0,"
            ) && described.contains("#sensed, joint_vel[:12] (*0.05)#sensed) clip[-100.0,100.0]\n"),
            "got:\n{described}"
        );
        assert!(
            adapter.advisories().is_empty(),
            "{:?}",
            adapter.advisories()
        );
        let crate::plans::ObsPlan::State(plan) = &adapter.obs_plans[0] else {
            panic!("expected a state plan");
        };
        assert_eq!(plan.native_width, Some(33));
        let mut raw: BTreeMap<String, Value> = BTreeMap::new();
        raw.insert("ang_vel".to_owned(), tensor(&[4.0, 0.0, 0.0]));
        // Identity orientation, wxyz: gravity is straight down in the base.
        raw.insert("base_quat".to_owned(), tensor(&[1.0, 0.0, 0.0, 0.0]));
        raw.insert("command".to_owned(), tensor(&[0.5, 0.0, 0.0]));
        let mut joints = vec![0.0; 12];
        joints[0] = 1.0e6;
        raw.insert("joint_pos".to_owned(), tensor(&joints));
        raw.insert("joint_vel".to_owned(), tensor(&[0.0; 12]));
        let Value::Map(payload) =
            crate::apply::transform_obs(&adapter, &raw, &NoCustoms).expect("apply")
        else {
            panic!("expected a map");
        };
        let Value::Tensor(obs) = &payload["obs"] else {
            panic!("expected a tensor");
        };
        let obs = crate::apply::value::to_f32_vec(obs);
        assert_eq!(obs.len(), 33);
        assert_eq!(&obs[..3], &[1.0, 0.0, 0.0]);
        assert_eq!(&obs[3..6], &[0.0, 0.0, -1.0]);
        assert_eq!(&obs[6..9], &[0.5, 0.0, 0.0]);
        // The container clamp caught the runaway encoder reading.
        assert_eq!(obs[9], 100.0);
        assert!((obs[10] + 0.8).abs() < 1e-6, "{obs:?}");
    }

    /// The design's Go2 pairing complete: the body model plus a part reading
    /// the model's own previous action, 45 wide.
    fn go2_full_model(previous: &str) -> String {
        go2_body_model("").replacen(
            r#""scale":0.05}]}"#,
            &format!(r#""scale":0.05}},{previous}]}}"#),
            1,
        )
    }

    const PREVIOUS: &str = r#"{"role":"action/joint_pos","source":"action"}"#;

    fn go2_full_resolve(
        env: &str,
        previous: &str,
    ) -> Result<crate::plans::Binding, crate::error::AdapterResolutionError> {
        let tags: EnvTags = serde_json::from_str(env).unwrap();
        let spec: ModelSpec = serde_json::from_str(&go2_full_model(previous)).unwrap();
        let obs = go2_body_obs(&["ang_vel", "base_quat", "command", "joint_pos", "joint_vel"]);
        resolve(&tags, &space(&obs), &space(GO2_ACT), &spec, false)
    }

    fn go2_body_raw() -> BTreeMap<String, Value> {
        let mut raw: BTreeMap<String, Value> = BTreeMap::new();
        raw.insert("ang_vel".to_owned(), tensor(&[4.0, 0.0, 0.0]));
        raw.insert("base_quat".to_owned(), tensor(&[1.0, 0.0, 0.0, 0.0]));
        raw.insert("command".to_owned(), tensor(&[0.5, 0.0, 0.0]));
        raw.insert("joint_pos".to_owned(), tensor(&[0.0; 12]));
        raw.insert("joint_vel".to_owned(), tensor(&[0.0; 12]));
        raw
    }

    /// The assembled `obs` vector at `step`, through the stateful seam.
    fn assembled(
        adapter: &crate::plans::Binding,
        buffers: &mut crate::stateful::FrameBuffers,
        step: i64,
    ) -> Vec<f32> {
        let payload = crate::stateful::assemble_obs(
            adapter,
            &go2_body_raw(),
            "ep",
            step,
            buffers,
            &NoCustoms,
            &crate::stateful::NoEncodings,
        )
        .expect("assemble");
        let Value::Map(payload) = payload else {
            panic!("expected a map");
        };
        let Value::Tensor(obs) = &payload["obs"] else {
            panic!("expected a tensor");
        };
        crate::apply::value::to_f32_vec(obs)
    }

    #[test]
    fn the_previous_action_completes_the_go2_observation_at_45_wide() {
        let adapter = go2_full_resolve(&go2_body_env(GO2_BODY_QUAT), PREVIOUS).expect("resolves");
        let described = adapter.describe();
        assert!(
            described.contains(
                "joint_vel[:12] (*0.05)#sensed, previous action/joint_pos (fill 0.0)) clip[-100.0,100.0]\n"
            ),
            "got:\n{described}"
        );
        // Answered by the model's own actuator: no ad-hoc-role nudge, and the
        // route holds history the way a stacked one does.
        assert!(
            adapter.advisories().is_empty(),
            "{:?}",
            adapter.advisories()
        );
        assert_eq!(adapter.history_keys(), vec!["obs".to_owned()]);
        assert!(adapter.reads_previous_action() && adapter.holds_history());
        assert!(adapter.history_windows().is_empty());
        let crate::plans::ObsPlan::State(plan) = &adapter.obs_plans[0] else {
            panic!("expected a state plan");
        };
        assert_eq!(plan.native_width, Some(45));
        // The stateless transform reads the fill; the stateful seam reads the
        // raw output executed at the previous step, before the actuator's
        // affine, and the fill at the episode's first step.
        let Value::Map(payload) =
            crate::apply::transform_obs(&adapter, &go2_body_raw(), &NoCustoms).expect("apply")
        else {
            panic!("expected a map");
        };
        let Value::Tensor(obs) = &payload["obs"] else {
            panic!("expected a tensor");
        };
        assert_eq!(&crate::apply::value::to_f32_vec(obs)[33..], &[0.0; 12]);
        let mut buffers = crate::stateful::FrameBuffers::new();
        assert_eq!(&assembled(&adapter, &mut buffers, 0)[33..], &[0.0; 12]);
        let raw: Vec<f32> = (0..12).map(|i| 0.1 * i as f32).collect();
        crate::stateful::record_action(&adapter, &tensor(&raw), &mut buffers, "ep", 0)
            .expect("record");
        let obs = assembled(&adapter, &mut buffers, 1);
        assert_eq!(obs.len(), 45);
        assert_eq!(&obs[33..], raw.as_slice());
    }

    #[test]
    fn a_previous_action_part_permutes_the_actuator_axes_by_label() {
        // The model reads back its own output in Isaac order, and a stand-in
        // of 0.5 before the first action.
        let isaac = labels(&ISAAC);
        let previous = format!(
            r#"{{"role":"action/joint_pos","source":"action","labels":{isaac},"fill":0.5}}"#
        );
        let adapter = go2_full_resolve(&go2_body_env(GO2_BODY_QUAT), &previous).expect("resolves");
        assert!(
            adapter.describe().contains(
                "previous action/joint_pos perm[3,4,5,0,1,2,9,10,11,6,7,8] (fill 0.5)) clip"
            ),
            "got:\n{}",
            adapter.describe()
        );
        let mut buffers = crate::stateful::FrameBuffers::new();
        assert_eq!(&assembled(&adapter, &mut buffers, 0)[33..], &[0.5; 12]);
        let raw: Vec<f32> = (0..12).map(|i| i as f32).collect();
        crate::stateful::record_action(&adapter, &tensor(&raw), &mut buffers, "ep", 0)
            .expect("record");
        let expected: Vec<f32> = ISAAC.iter().map(|&i| i as f32).collect();
        assert_eq!(
            &assembled(&adapter, &mut buffers, 1)[33..],
            expected.as_slice()
        );
        // A label set that differs from the actuator's is a mismatch naming it.
        let err = go2_full_resolve(
            &go2_body_env(GO2_BODY_QUAT),
            r#"{"role":"action/joint_pos","source":"action","labels":["FR_hip_joint","FR_shin"]}"#,
        )
        .expect_err("missing label");
        assert_eq!(err.code, ErrorCode::LabelMismatch);
        assert!(
            err.message.contains(r#"the actuator lacks ["FR_shin"]"#),
            "{}",
            err.message
        );
    }

    #[test]
    fn a_previous_action_part_binds_only_the_models_own_actuator() {
        let env = go2_body_env(GO2_BODY_QUAT);
        let err = go2_full_resolve(&env, r#"{"role":"action/gripper","source":"action"}"#)
            .expect_err("no actuator");
        assert_eq!(err.code, ErrorCode::MissingRole);
        assert!(
            err.message.contains(
                r#"reads the previous action for role "action/gripper" but no actuator emits it"#
            ),
            "{}",
            err.message
        );
        // A declared dim must be the actuator's.
        let err = go2_full_resolve(
            &env,
            r#"{"role":"action/joint_pos","source":"action","dim":6}"#,
        )
        .expect_err("dim");
        assert_eq!(err.code, ErrorCode::DimMismatch);
        assert!(
            err.message
                .contains("declares dim 6 but the actuator emits 12"),
            "{}",
            err.message
        );
        // Never an env role: an observation tag under `action/` is refused at
        // join, before any model part is planned.
        let env = env.replacen(
            r#""role":"command/base_vel""#,
            r#""role":"action/base_vel""#,
            1,
        );
        let err = go2_full_resolve(&env, PREVIOUS).expect_err("action role on an observation");
        assert_eq!(err.code, ErrorCode::ActionRoleOnObservation);
        assert!(
            err.message.contains(
                r#"observation "command" declares role "action/base_vel", an action kind"#
            ),
            "{}",
            err.message
        );
    }

    #[test]
    fn a_sim_publishing_base_rot_twice_needs_the_model_to_pin_a_provenance() {
        let two = format!(
            r#"{GO2_BODY_QUAT}, "base_quat_est":{{"type":"state","role":"proprio/base_rot","encoding":"quat_wxyz","frame":"world","provenance":"estimated"}}"#
        );
        let truth = GO2_BODY_QUAT.replace("sensed", "privileged");
        let two = two.replacen(GO2_BODY_QUAT, &truth, 1);
        let tags: EnvTags = serde_json::from_str(&go2_body_env(&two)).unwrap();
        let obs = go2_body_obs(&[
            "ang_vel",
            "base_quat",
            "base_quat_est",
            "command",
            "joint_pos",
            "joint_vel",
        ]);
        let spec: ModelSpec = serde_json::from_str(&go2_body_model("")).unwrap();
        let err =
            resolve(&tags, &space(&obs), &space(GO2_ACT), &spec, false).expect_err("ambiguous");
        assert_eq!(err.code, ErrorCode::Ambiguous);
        assert!(
            err.message
                .contains(r#"under provenances ["estimated", "privileged"]"#),
            "{}",
            err.message
        );
        let spec: ModelSpec =
            serde_json::from_str(&go2_body_model(r#","provenance":"estimated""#)).unwrap();
        let adapter = resolve(&tags, &space(&obs), &space(GO2_ACT), &spec, false).expect("pinned");
        assert!(
            adapter
                .describe()
                .contains("base_quat_est (quat_wxyz->gravity_xyz)@world#estimated"),
            "{}",
            adapter.describe()
        );
        assert!(
            adapter.advisories().is_empty(),
            "{:?}",
            adapter.advisories()
        );
        let spec: ModelSpec =
            serde_json::from_str(&go2_body_model(r#","provenance":"sensed""#)).unwrap();
        let err = resolve(&tags, &space(&obs), &space(GO2_ACT), &spec, false).expect_err("neither");
        assert_eq!(err.code, ErrorCode::ProvenanceMismatch);
        assert!(
            err.message.contains(r#"accepts provenance ["sensed"] but the env declares it under ["estimated", "privileged"]"#),
            "{}",
            err.message
        );
    }
}

mod provenance_tests {
    use crate::error::ErrorCode;
    use crate::lower::resolve;
    use crate::space_view::SpaceView;
    use crate::spec::{EnvTags, ModelSpec};
    use rlmesh_session_core::spec::role::Severity as AdvisorySeverity;

    const OBS: &str = r#"{"kind":"dict","dtype":"unspecified","keys":["rot"],"children":[
        {"kind":"box","shape":[4],"dtype":"float32"}]}"#;
    const ACT: &str = r#"{"kind":"box","shape":[1],"dtype":"float32"}"#;

    fn env(provenance: &str) -> String {
        format!(
            r#"{{"observation":{{"rot":{{"type":"state","role":"proprio/base_rot","encoding":"quat_wxyz"{provenance}}}}},
                "action":{{"components":[{{"role":"action/gripper","dim":1}}]}}}}"#
        )
    }

    fn model(provenance: &str) -> String {
        format!(
            r#"{{"input":{{"s":{{"type":"state","components":[{{"role":"proprio/base_rot","encoding":"quat_wxyz"{provenance}}}]}}}},
                "output":{{"components":[{{"role":"action/gripper","dim":1}}]}}}}"#
        )
    }

    fn do_resolve(
        env: &str,
        model: &str,
    ) -> Result<crate::plans::Binding, crate::error::AdapterResolutionError> {
        let tags: EnvTags = serde_json::from_str(env).expect("parse env tags");
        let spec: ModelSpec = serde_json::from_str(model).expect("parse model spec");
        let obs: SpaceView = serde_json::from_str(OBS).unwrap();
        let act: SpaceView = serde_json::from_str(ACT).unwrap();
        resolve(&tags, &obs, &act, &spec, false)
    }

    #[test]
    fn agreement_and_env_only_are_silent_and_print_the_value() {
        for model_side in [
            "",
            r#","provenance":"sensed""#,
            r#","provenance":["estimated","sensed"]"#,
        ] {
            let adapter = do_resolve(&env(r#","provenance":"sensed""#), &model(model_side))
                .expect("resolves");
            assert!(
                adapter.describe().contains("concat(rot#sensed)"),
                "{model_side}: {}",
                adapter.describe()
            );
            assert!(
                adapter.advisories().is_empty(),
                "{:?}",
                adapter.advisories()
            );
        }
        let adapter = do_resolve(&env(""), &model("")).expect("resolves");
        assert!(
            adapter.describe().contains("concat(rot)"),
            "{}",
            adapter.describe()
        );
    }

    #[test]
    fn a_model_only_declaration_is_a_caution() {
        let adapter = do_resolve(&env(""), &model(r#","provenance":["sensed","estimated"]"#))
            .expect("resolves");
        let notes = adapter.advisories();
        assert!(
            notes
                .iter()
                .any(|note| note.severity == AdvisorySeverity::Caution
                    && note.message.contains(
                        r#"declares provenance "sensed|estimated" but the env declares none"#
                    )),
            "{notes:?}"
        );
        assert!(
            adapter.describe().contains("rot#sensed|estimated"),
            "{}",
            adapter.describe()
        );
        // Quiet channel: never under `dropped:`.
        assert!(!adapter.describe().contains("dropped:"));
    }

    #[test]
    fn a_disagreement_or_an_unknown_value_is_a_provenance_mismatch() {
        let err = do_resolve(
            &env(r#","provenance":"privileged""#),
            &model(r#","provenance":"sensed""#),
        )
        .expect_err("mismatch");
        assert_eq!(err.code, ErrorCode::ProvenanceMismatch);
        assert!(
            err.message
                .contains(r#"expects provenance ["sensed"] but the env declares "privileged""#),
            "{}",
            err.message
        );
        let err = do_resolve(&env(r#","provenance":"guessed""#), &model("")).expect_err("unknown");
        assert_eq!(err.code, ErrorCode::ProvenanceMismatch);
        assert!(
            err.message.contains("unrecognized provenance \"guessed\""),
            "{}",
            err.message
        );
        let err = do_resolve(
            &env(r#","provenance":"sensed""#),
            &model(r#","provenance":"guessed""#),
        )
        .expect_err("unknown");
        assert_eq!(err.code, ErrorCode::ProvenanceMismatch);
        assert!(
            err.message
                .contains(r#"unrecognized provenance ["guessed"]"#),
            "{}",
            err.message
        );
    }
}

mod projection_tests {
    use crate::error::ErrorCode;
    use crate::lower::resolve;
    use crate::space_view::SpaceView;
    use crate::spec::{EnvTags, ModelSpec};

    /// A LIBERO-style 7-D model on a MetaWorld-style 4-D env: the rotation
    /// delta has no actuator.
    fn resolve_with(
        allow: &str,
    ) -> Result<crate::plans::Binding, crate::error::AdapterResolutionError> {
        let tags: EnvTags = serde_json::from_str(
            r#"{"observation": {"t": {"type": "text", "role": "text/instruction"}},
                "action": {"components": [
                    {"role": "action/delta_eef_pos", "dim": 3},
                    {"role": "action/gripper", "dim": 1}]}}"#,
        )
        .expect("tags");
        let spec: ModelSpec = serde_json::from_str(&format!(
            r#"{{"input": {{"type": "text", "role": "text/instruction"}},
                "output": {{"components": [
                    {{"role": "action/delta_eef_pos", "dim": 3}},
                    {{"role": "action/delta_eef_rot", "dim": 3, "encoding": "axis_angle"}},
                    {{"role": "action/gripper", "dim": 1}}]}}{allow}}}"#
        ))
        .expect("spec");
        let obs: SpaceView = serde_json::from_str(
            r#"{"kind": "dict", "dtype": "unspecified", "keys": ["t"],
                "children": [{"kind": "text", "dtype": "unspecified"}]}"#,
        )
        .expect("obs");
        let action: SpaceView =
            serde_json::from_str(r#"{"kind": "box", "shape": [4], "dtype": "float32"}"#)
                .expect("action");
        resolve(&tags, &obs, &action, &spec, false)
    }

    #[test]
    fn an_unconsumed_model_output_is_refused_unless_drop_action_is_allowed() {
        let refused = resolve_with("").expect_err("refused");
        assert_eq!(refused.code, ErrorCode::MissingRole);
        assert!(
            refused
                .message
                .contains("\"action/delta_eef_rot\" (3 dims) drives no env actuator")
                && refused.message.contains("\"drop_action\""),
            "{}",
            refused.message
        );
        let adapter = resolve_with(r#", "allow": ["drop_action"]"#).expect("allowed");
        assert_eq!(adapter.projections().len(), 1);
        assert!(
            adapter
                .describe()
                .contains("projections:\n  drop_action: model output \"action/delta_eef_rot\" [3, 6) is discarded"),
            "{}",
            adapter.describe()
        );
        let unknown = resolve_with(r#", "allow": ["drop_rot"]"#).expect_err("unknown");
        assert_eq!(unknown.code, ErrorCode::Unsupported);
    }
}

/// The value conventions a gripper and a delta carry: a different `polarity`
/// or `unit` on the two sides is refused unless the model allows the named
/// projection that converts it, and converted exactly when it does.
mod convention_tests {
    use crate::apply::value::{Value, tensor_from_f32, to_f32_vec};
    use crate::error::ErrorCode;
    use crate::lower::resolve;
    use crate::space_view::SpaceView;
    use crate::spec::{EnvTags, ModelSpec};

    /// A one-gripper, one-position-delta env with a one-finger-pair reading,
    /// and a model reading the reading and emitting both commands; `env` and
    /// `model` splice extra attributes into the env's and the model's
    /// actuators (`gripper` / `pos`) and the model's state part (`state`).
    fn resolve_pair(
        env_gripper: &str,
        env_pos: &str,
        env_state: &str,
        model_gripper: &str,
        model_pos: &str,
        model_state: &str,
        allow: &str,
    ) -> Result<crate::plans::Binding, crate::error::AdapterResolutionError> {
        let tags: EnvTags = serde_json::from_str(&format!(
            r#"{{"observation": {{"g": {{"type": "state", "role": "proprio/gripper_pos"{env_state}}}}},
                "action": {{"components": [
                    {{"role": "action/delta_eef_pos", "dim": 3{env_pos}}},
                    {{"role": "action/gripper", "dim": 1, "range": [-1.0, 1.0]{env_gripper}}}]}}}}"#
        ))
        .expect("tags");
        let spec: ModelSpec = serde_json::from_str(&format!(
            r#"{{"input": {{"type": "state", "components": [{{"role": "proprio/gripper_pos"{model_state}}}]}},
                "output": {{"components": [
                    {{"role": "action/delta_eef_pos", "dim": 3{model_pos}}},
                    {{"role": "action/gripper", "dim": 1, "range": [-1.0, 1.0]{model_gripper}}}]}}{allow}}}"#
        ))
        .expect("spec");
        let obs: SpaceView = serde_json::from_str(
            r#"{"kind": "dict", "dtype": "unspecified", "keys": ["g"],
                "children": [{"kind": "box", "shape": [2], "dtype": "float32",
                              "low": [0.0, 0.0], "high": [1.0, 1.0]}]}"#,
        )
        .expect("obs");
        let action: SpaceView =
            serde_json::from_str(r#"{"kind": "box", "shape": [4], "dtype": "float32"}"#)
                .expect("action");
        resolve(&tags, &obs, &action, &spec, false)
    }

    fn act(adapter: &crate::plans::Binding, values: &[f32]) -> Vec<f32> {
        let raw = Value::Tensor(tensor_from_f32(vec![values.len() as i64], values));
        to_f32_vec(&crate::apply::transform_action(adapter, &raw).expect("apply"))
    }

    #[test]
    fn a_reversed_gripper_command_is_refused_unless_flip_gripper_reflects_it() {
        let close = r#", "polarity": "close""#;
        let open = r#", "polarity": "open""#;
        let refused = resolve_pair(close, "", "", open, "", "", "").expect_err("refused");
        assert_eq!(refused.code, ErrorCode::ConventionMismatch);
        assert!(
            refused.message.contains("\"flip_gripper\""),
            "{}",
            refused.message
        );
        // The model's own invert already reverses it: no conversion is needed.
        let inverted = format!(r#"{open}, "invert": true"#);
        let adapter = resolve_pair(close, "", "", &inverted, "", "", "").expect("agrees");
        assert!(adapter.projections().is_empty());
        let adapter = resolve_pair(
            close,
            "",
            "",
            open,
            "",
            "",
            r#", "allow": ["flip_gripper"]"#,
        )
        .expect("allowed");
        assert_eq!(
            act(&adapter, &[0.0, 0.0, 0.0, 0.75]),
            vec![0.0, 0.0, 0.0, -0.75]
        );
        // Declared by one side only, it binds unchecked.
        assert!(resolve_pair(close, "", "", "", "", "", "").is_ok());
    }

    #[test]
    fn a_delta_in_another_unit_is_refused_unless_convert_units_rescales_it() {
        let refused = resolve_pair("", r#", "unit": 0.05"#, "", "", r#", "unit": 1.0"#, "", "")
            .expect_err("refused");
        assert_eq!(refused.code, ErrorCode::ConventionMismatch);
        assert!(
            refused.message.contains("\"convert_units\""),
            "{}",
            refused.message
        );
        // The model's scale bridges it exactly.
        assert!(
            resolve_pair(
                "",
                r#", "unit": 0.05"#,
                "",
                "",
                r#", "unit": 1.0, "scale": 20.0"#,
                "",
                ""
            )
            .expect("agrees")
            .projections()
            .is_empty()
        );
        let adapter = resolve_pair(
            "",
            r#", "unit": 0.05"#,
            "",
            "",
            r#", "unit": 1.0"#,
            "",
            r#", "allow": ["convert_units"]"#,
        )
        .expect("allowed");
        let moved = act(&adapter, &[0.01, -0.02, 0.0, 0.0]);
        assert!(
            (moved[0] - 0.2).abs() < 1e-6 && (moved[1] + 0.4).abs() < 1e-6,
            "{moved:?}"
        );
    }

    #[test]
    fn a_truncated_or_reversed_gripper_reading_is_refused_unless_allowed() {
        let one = r#", "dim": 1"#;
        let refused = resolve_pair("", "", "", "", "", one, "").expect_err("refused");
        assert_eq!(refused.code, ErrorCode::DimMismatch);
        assert!(
            refused.message.contains("\"select_state\""),
            "{}",
            refused.message
        );
        assert!(resolve_pair("", "", "", "", "", one, r#", "allow": ["select_state"]"#).is_ok());

        let refused = resolve_pair(
            "",
            "",
            r#", "polarity": "close""#,
            "",
            "",
            r#", "polarity": "open""#,
            "",
        )
        .expect_err("refused");
        assert_eq!(refused.code, ErrorCode::ConventionMismatch);
        let adapter = resolve_pair(
            "",
            "",
            r#", "polarity": "close""#,
            "",
            "",
            r#", "polarity": "open""#,
            r#", "allow": ["flip_gripper"]"#,
        )
        .expect("allowed");
        let obs = std::collections::BTreeMap::from([(
            "g".to_owned(),
            Value::Tensor(tensor_from_f32(vec![2], &[0.25, 1.0])),
        )]);
        let Value::Tensor(state) =
            crate::apply::transform_obs(&adapter, &obs, &crate::apply::NoCustoms).expect("apply")
        else {
            panic!("a lone state input is its tensor");
        };
        // Closedness 0.25 / 1.0 read as openness 0.75 / 0.0.
        assert_eq!(
            to_f32_vec(&state),
            vec![0.75, 0.0],
            "{}",
            adapter.describe()
        );
    }
}

/// What a declaration alone gets wrong is refused while lowering it.
mod lowering_tests {
    use crate::error::ErrorCode;
    use crate::lower::resolve;
    use crate::space_view::SpaceView;
    use crate::spec::{EnvTags, ModelSpec};

    fn resolve_with(image: &str, action: &str) -> crate::error::AdapterResolutionError {
        let tags: EnvTags = serde_json::from_str(
            r#"{"observation": {"cam": {"type": "image", "role": "image/primary"}},
                "action": {"components": [{"dim": 1}]}}"#,
        )
        .expect("tags");
        let spec: ModelSpec = serde_json::from_str(&format!(
            r#"{{"input": {{"type": "image", "role": "image/primary"{image}}},
                "output": {{"components": [{{"dim": 1}}]{action}}}}}"#
        ))
        .expect("spec");
        let obs: SpaceView = serde_json::from_str(
            r#"{"kind": "dict", "dtype": "unspecified", "keys": ["cam"],
                "children": [{"kind": "box", "shape": [8, 8, 3], "dtype": "uint8"}]}"#,
        )
        .expect("obs");
        let act: SpaceView =
            serde_json::from_str(r#"{"kind": "box", "shape": [1], "dtype": "float32"}"#)
                .expect("action");
        resolve(&tags, &obs, &act, &spec, false).expect_err("refused")
    }

    #[test]
    fn a_contradictory_or_unknown_declaration_is_refused() {
        for (image, action, expect) in [
            (r#", "crop": 0.5, "crop_area": 0.25"#, "", "not both"),
            (r#", "crop_mode": "zooom""#, "", "crop_mode"),
            (r#", "channel_order": "rbg""#, "", "channel_order"),
            (r#", "dtype": "flat32""#, "", "unknown dtype"),
            ("", r#", "clip": [-1.0, 1.0]"#, "env-side actuator clamp"),
        ] {
            let error = resolve_with(image, action);
            assert_eq!(error.code, ErrorCode::Unsupported, "{}", error.message);
            assert!(error.message.contains(expect), "{}", error.message);
        }
    }
}
