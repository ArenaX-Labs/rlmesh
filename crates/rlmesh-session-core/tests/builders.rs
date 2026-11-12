//! The policy-map builders against the shared cases in
//! `fixtures/builders.json`, which the Python builders are checked against
//! too: each case's builder, called with its arguments through the typed
//! setters, gives the case's entry, and a policy built from it is the
//! policy its entry gives as JSON. Every setter is set by some case.

use std::collections::BTreeSet;
use std::time::Duration;

use rlmesh_session_core::spec::actions::{
    self, ChunkStart, GripperCommand, JointCommand, PoseCommand, Reference, SetCommand,
    VelocityCommand, WrenchCommand,
};
use rlmesh_session_core::spec::inputs::{
    self, Allow, Array, DType, ImageFormat, ImageLayout, ImageSize, ImuField, JointField,
    PerElement, PoseField, Resample, Rotation, Source, VelocityField, WrenchField,
};
use rlmesh_session_core::spec::policy::{ActionEntry, InputEntry};
use rlmesh_session_core::spec::{Policy, PolicyBuilder};
use serde::Serialize;
use serde_json::{Map, Value, json};

const CASES: &str = include_str!("fixtures/builders.json");

/// A closed set, picked by its name in the map.
trait Named: Copy + 'static {
    fn all() -> &'static [Self];
    fn name(self) -> &'static str;
}

macro_rules! named {
    ($($ty:ty => $all:expr),* $(,)?) => {
        $(impl Named for $ty {
            fn all() -> &'static [Self] {
                &$all
            }
            fn name(self) -> &'static str {
                <$ty>::name(self)
            }
        })*
    };
}

named! {
    Allow => Allow::ALL,
    DType => DType::ALL,
    Rotation => Rotation::ALL,
    JointField => *JointField::ALL,
    ImuField => *ImuField::ALL,
    WrenchField => *WrenchField::ALL,
    PoseField => *PoseField::ALL,
    VelocityField => *VelocityField::ALL,
    JointCommand => *JointCommand::ALL,
    PoseCommand => *PoseCommand::ALL,
    Reference => *Reference::ALL,
    WrenchCommand => *WrenchCommand::ALL,
    GripperCommand => *GripperCommand::ALL,
    SetCommand => *SetCommand::ALL,
    VelocityCommand => *VelocityCommand::ALL,
    ImageLayout => [ImageLayout::Hwc, ImageLayout::Chw],
    ImageFormat => [ImageFormat::Rgb8, ImageFormat::Bgr8, ImageFormat::Mono8],
    Resample => [
        Resample::Bilinear,
        Resample::BilinearAa,
        Resample::BicubicAa,
        Resample::Lanczos3Aa,
        Resample::Area,
    ],
    Source => [Source::Published, Source::Applied],
    ChunkStart => [ChunkStart::Snapshot, ChunkStart::Send],
}

fn one<T: Named>(v: &Value) -> T {
    let name = text(v);
    T::all()
        .iter()
        .copied()
        .find(|t| t.name() == name)
        .unwrap_or_else(|| panic!("{name:?} is not a {}", std::any::type_name::<T>()))
}

fn each<T: Named>(v: &Value) -> Vec<T> {
    list(v).iter().map(one).collect()
}

fn text(v: &Value) -> String {
    v.as_str()
        .unwrap_or_else(|| panic!("{v} is not a string"))
        .into()
}

fn number(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("{v} is not a number"))
}

fn whole(v: &Value) -> u32 {
    let n = v
        .as_u64()
        .unwrap_or_else(|| panic!("{v} is not a whole number"));
    u32::try_from(n).unwrap_or_else(|_| panic!("{v} is too large"))
}

fn list(v: &Value) -> &Vec<Value> {
    v.as_array().unwrap_or_else(|| panic!("{v} is not a list"))
}

fn object(v: &Value) -> &Map<String, Value> {
    v.as_object()
        .unwrap_or_else(|| panic!("{v} is not an object"))
}

fn texts(v: &Value) -> Vec<String> {
    list(v).iter().map(text).collect()
}

fn dims(v: &Value) -> Vec<u32> {
    list(v).iter().map(whole).collect()
}

fn size(v: &Value) -> ImageSize {
    match v {
        Value::Array(sides) => (whole(&sides[0]), whole(&sides[1])).into(),
        side => whole(side).into(),
    }
}

fn per_element(v: &Value) -> PerElement {
    match v {
        Value::Array(each) => each.iter().map(number).collect::<Vec<_>>().into(),
        all => number(all).into(),
    }
}

fn millis(v: &Value) -> Duration {
    Duration::from_millis(u64::from(whole(v)))
}

fn array(v: &Value) -> Array {
    let pair = list(v);
    Array::new(one(&pair[0]), dims(&pair[1]))
}

fn positional(args: &Map<String, Value>) -> bool {
    args.get("names") == Some(&json!("positional"))
}

/// Calls the setter for the map key `key` on `b` with `v`.
macro_rules! set {
    (size, $b:ident, $v:ident, $args:ident) => {
        $b.size(size($v))
    };
    (layout, $b:ident, $v:ident, $args:ident) => {
        $b.layout(one($v))
    };
    (format, $b:ident, $v:ident, $args:ident) => {
        $b.format(one($v))
    };
    (resample, $b:ident, $v:ident, $args:ident) => {
        $b.resample(one($v))
    };
    (crop, $b:ident, $v:ident, $args:ident) => {
        $b.crop(number($v))
    };
    (upside_down, $b:ident, $v:ident, $args:ident) => {
        $b.upside_down($v.as_bool().expect("bool"))
    };
    (optional, $b:ident, $v:ident, $args:ident) => {
        $b.optional($v.as_bool().expect("bool"))
    };
    (dtype, $b:ident, $v:ident, $args:ident) => {
        $b.dtype(one($v))
    };
    (range, $b:ident, $v:ident, $args:ident) => {
        $b.range(number(&list($v)[0]), number(&list($v)[1]))
    };
    (frame, $b:ident, $v:ident, $args:ident) => {
        $b.frame(text($v))
    };
    (child, $b:ident, $v:ident, $args:ident) => {
        $b.child(text($v))
    };
    (point, $b:ident, $v:ident, $args:ident) => {
        $b.point(text($v))
    };
    (history, $b:ident, $v:ident, $args:ident) => {
        $b.history(whole($v))
    };
    (history_hz, $b:ident, $v:ident, $args:ident) => {
        $b.history_hz(number($v))
    };
    (max_age_ms, $b:ident, $v:ident, $args:ident) => {
        $b.max_age(millis($v))
    };
    (allow, $b:ident, $v:ident, $args:ident) => {
        $b.allow(each::<Allow>($v))
    };
    (source, $b:ident, $v:ident, $args:ident) => {
        $b.source(one($v))
    };
    (names, $b:ident, $v:ident, $args:ident) => {
        if positional($args) {
            $b.positional(whole(&$args["count"]))
        } else {
            $b.names(texts($v))
        }
    };
    (count, $b:ident, $v:ident, $args:ident) => {
        if positional($args) {
            $b
        } else {
            $b.count(whole($v))
        }
    };
    (fields, $b:ident, $v:ident, $args:ident) => {
        $b.fields(each($v))
    };
    (schema, $b:ident, $v:ident, $args:ident) => {
        object($v)
            .iter()
            .fold($b, |b, (name, decl)| b.field(name.as_str(), array(decl)))
    };
    (units, $b:ident, $v:ident, $args:ident) => {
        object($v)
            .iter()
            .fold($b, |b, (field, unit)| b.unit(field.as_str(), text(unit)))
    };
    (scale, $b:ident, $v:ident, $args:ident) => {
        $b.scale(per_element($v))
    };
    (offset, $b:ident, $v:ident, $args:ident) => {
        $b.offset(per_element($v))
    };
    (rotation, $b:ident, $v:ident, $args:ident) => {
        $b.rotation(one($v))
    };
    (shape, $b:ident, $v:ident, $args:ident) => {
        $b.shape(dims($v))
    };
    (command, $b:ident, $v:ident, $args:ident) => {
        $b.command(one($v))
    };
    (custom_command, $b:ident, $v:ident, $args:ident) => {
        $b.command(text($v))
    };
    (reference, $b:ident, $v:ident, $args:ident) => {
        $b.reference(one($v))
    };
    (hz, $b:ident, $v:ident, $args:ident) => {
        $b.hz(number($v))
    };
    (chunk, $b:ident, $v:ident, $args:ident) => {
        $b.chunk(whole($v))
    };
    (t0, $b:ident, $v:ident, $args:ident) => {
        $b.t0(one($v))
    };
    (ttl_ms, $b:ident, $v:ident, $args:ident) => {
        $b.ttl(millis($v))
    };
    (open, $b:ident, $v:ident, $args:ident) => {
        $b.open(number($v))
    };
    (closed, $b:ident, $v:ident, $args:ident) => {
        $b.closed(number($v))
    };
}

/// The map key a `set!` arm sets.
macro_rules! key {
    (schema) => {
        "fields"
    };
    (custom_command) => {
        "command"
    };
    ($key:ident) => {
        stringify!($key)
    };
}

/// `$b` with every argument of the case set, recording which keys the
/// builder has setters for and which the cases set.
macro_rules! apply {
    ($b:expr, $args:expr, $seen:expr; $($key:ident),* $(,)?) => {{
        let mut b = $b;
        let args: &Map<String, Value> = $args;
        let builder = $seen.builder.clone();
        $seen.setters.insert(format!("{builder} at"));
        $($seen.setters.insert(format!("{builder} {}", key!($key)));)*
        for (key, v) in args {
            if key == "kind" {
                continue; // the constructor's argument
            }
            $seen.set.insert(format!("{builder} {key}"));
            b = match key.as_str() {
                "at" => b.at(text(v)),
                $(key!($key) => set!($key, b, v, args),)*
                other => panic!("{builder} has no setter for {other}"),
            };
        }
        b
    }};
}

#[derive(Default)]
struct Seen {
    builder: String,
    setters: BTreeSet<String>,
    set: BTreeSet<String>,
}

/// The entry `b` serialises to, and the policy a one-entry builder gives.
fn input_case(b: impl Serialize + Into<InputEntry>, entry: &Value) -> (Value, Policy) {
    let json = serde_json::to_value(&b).expect("serialise");
    let built = Policy::builder().hz(10).input("x", b).build();
    let parsed = Policy::parse(&json!({"hz": 10, "inputs": {"x": entry}}).to_string());
    assert_eq!(built, parsed, "{entry}");
    (json, built.unwrap_or_else(|err| panic!("{entry}: {err}")))
}

fn action_case(b: impl Serialize + Into<ActionEntry>, entry: &Value) -> (Value, Policy) {
    let json = serde_json::to_value(&b).expect("serialise");
    let built = Policy::builder().hz(10).action("x", b).build();
    let parsed = Policy::parse(&json!({"hz": 10, "actions": {"x": entry}}).to_string());
    assert_eq!(built, parsed, "{entry}");
    (json, built.unwrap_or_else(|err| panic!("{entry}: {err}")))
}

#[test]
fn every_shared_case_gives_its_entry_and_its_policy() {
    let fixture: Value = serde_json::from_str(CASES).expect("fixture");
    let mut seen = Seen::default();
    let cases = list(&fixture["cases"]);
    for case in cases {
        let builder = text(&case["builder"]);
        let args = object(&case["args"]);
        let entry = &case["entry"];
        let kind = || text(&args["kind"]);
        seen.builder = builder.clone();
        let json = match builder.as_str() {
            "inputs.camera" => {
                input_case(
                    apply!(inputs::camera(), args, seen;
                size, layout, format, resample, crop, upside_down, dtype, range, frame,
                optional, max_age_ms, history, history_hz, allow),
                    entry,
                )
                .0
            }
            "inputs.depth" => {
                input_case(
                    apply!(inputs::depth(), args, seen;
                size, crop, upside_down, dtype, units, range, frame,
                optional, max_age_ms, history, history_hz, allow),
                    entry,
                )
                .0
            }
            "inputs.joints" => {
                input_case(
                    apply!(inputs::joints(), args, seen;
                names, count, fields, dtype, units, scale, offset, range, source,
                optional, max_age_ms, history, history_hz, allow),
                    entry,
                )
                .0
            }
            "inputs.hand" => {
                input_case(
                    apply!(inputs::hand(), args, seen;
                names, count, fields, dtype, units, scale, offset, range, source,
                optional, max_age_ms, history, history_hz, allow),
                    entry,
                )
                .0
            }
            "inputs.wrench" => {
                input_case(
                    apply!(inputs::wrench(), args, seen;
                frame, point, fields, dtype, units, scale, offset, range, source,
                optional, max_age_ms, history, history_hz, allow),
                    entry,
                )
                .0
            }
            "inputs.imu" => {
                input_case(
                    apply!(inputs::imu(), args, seen;
                frame, fields, rotation, dtype, units, scale, offset,
                optional, max_age_ms, history, history_hz, allow),
                    entry,
                )
                .0
            }
            "inputs.pose" => {
                input_case(
                    apply!(inputs::pose(), args, seen;
                frame, child, fields, rotation, dtype, units, scale, offset, source,
                optional, max_age_ms, history, history_hz, allow),
                    entry,
                )
                .0
            }
            "inputs.gripper" => {
                input_case(
                    apply!(inputs::gripper(), args, seen;
                dtype, scale, offset, range, source,
                optional, max_age_ms, history, history_hz, allow),
                    entry,
                )
                .0
            }
            "inputs.switch" => {
                input_case(
                    apply!(inputs::switch(), args, seen;
                dtype, source, optional, max_age_ms, history, history_hz, allow),
                    entry,
                )
                .0
            }
            "inputs.value" => {
                input_case(
                    apply!(inputs::value(), args, seen;
                dtype, units, scale, offset, range, source,
                optional, max_age_ms, history, history_hz, allow),
                    entry,
                )
                .0
            }
            "inputs.velocity" => {
                input_case(
                    apply!(inputs::velocity(), args, seen;
                frame, fields, dtype, units, scale, offset, range, source,
                optional, max_age_ms, history, history_hz, allow),
                    entry,
                )
                .0
            }
            "inputs.tactile" => {
                input_case(
                    apply!(inputs::tactile(), args, seen;
                shape, dtype, scale, offset, range,
                optional, max_age_ms, history, history_hz, allow),
                    entry,
                )
                .0
            }
            "inputs.text" => {
                input_case(
                    apply!(inputs::text(), args, seen;
                optional, max_age_ms, allow),
                    entry,
                )
                .0
            }
            "inputs.reward" => {
                input_case(
                    apply!(inputs::reward(), args, seen;
                optional, max_age_ms, allow),
                    entry,
                )
                .0
            }
            "inputs.success" => {
                input_case(
                    apply!(inputs::success(), args, seen;
                optional, max_age_ms, allow),
                    entry,
                )
                .0
            }
            "inputs.custom" => {
                input_case(
                    apply!(inputs::custom(kind()), args, seen;
                schema, frame, units, scale, offset, range, source,
                optional, max_age_ms, history, history_hz, allow),
                    entry,
                )
                .0
            }
            "inputs.of" => {
                input_case(
                    apply!(inputs::of(&case["part"]), args, seen;
                size, layout, format, resample, crop, upside_down, dtype, range, frame, child,
                point, names, count, units, scale, offset, rotation, shape, source,
                optional, max_age_ms, history, history_hz, allow),
                    entry,
                )
                .0
            }
            "actions.joints" => {
                action_case(
                    apply!(actions::joints(), args, seen;
                command, hz, chunk, names, count, units, scale, offset, range,
                t0, ttl_ms, optional, allow),
                    entry,
                )
                .0
            }
            "actions.hand" => {
                action_case(
                    apply!(actions::hand(), args, seen;
                command, hz, chunk, names, count, units, scale, offset, range,
                t0, ttl_ms, optional, allow),
                    entry,
                )
                .0
            }
            "actions.wrench" => {
                action_case(
                    apply!(actions::wrench(), args, seen;
                command, frame, hz, chunk, units, scale, offset, range,
                t0, ttl_ms, optional, allow),
                    entry,
                )
                .0
            }
            "actions.pose" => {
                action_case(
                    apply!(actions::pose(), args, seen;
                command, reference, frame, child, rotation, hz, chunk, units, scale, offset,
                t0, ttl_ms, optional, allow),
                    entry,
                )
                .0
            }
            "actions.gripper" => {
                action_case(
                    apply!(actions::gripper(), args, seen;
                command, hz, chunk, open, closed, scale, offset, range,
                t0, ttl_ms, optional, allow),
                    entry,
                )
                .0
            }
            "actions.switch" => {
                action_case(
                    apply!(actions::switch(), args, seen;
                command, hz, chunk, t0, ttl_ms, optional, allow),
                    entry,
                )
                .0
            }
            "actions.value" => {
                action_case(
                    apply!(actions::value(), args, seen;
                command, hz, chunk, units, scale, offset, range,
                t0, ttl_ms, optional, allow),
                    entry,
                )
                .0
            }
            "actions.velocity" => {
                action_case(
                    apply!(actions::velocity(), args, seen;
                command, frame, hz, chunk, units, scale, offset, range,
                t0, ttl_ms, optional, allow),
                    entry,
                )
                .0
            }
            "actions.custom" => {
                action_case(
                    apply!(actions::custom(kind()), args, seen;
                schema, custom_command, frame, hz, chunk, units, scale, offset, range,
                t0, ttl_ms, optional, allow),
                    entry,
                )
                .0
            }
            "parts.array" => {
                seen.setters
                    .extend(["parts.array dtype".into(), "parts.array shape".into()]);
                seen.set
                    .extend(args.keys().map(|k| format!("parts.array {k}")));
                serde_json::to_value(Array::new(one(&args["dtype"]), dims(&args["shape"])))
                    .expect("serialise")
            }
            other => panic!("no builder {other}"),
        };
        assert_eq!(&json, entry, "{builder}");
    }
    // `of` takes every input setter; the cases need only show it passes them on.
    let unset: Vec<_> = seen
        .setters
        .difference(&seen.set)
        .filter(|s| !s.starts_with("inputs.of ") || s.ends_with(" at"))
        .collect();
    assert!(unset.is_empty(), "no shared case sets {unset:?}");
}

#[test]
fn builders_and_json_mix_in_one_policy() {
    let mixed = Policy::builder()
        .hz(30)
        .input("cam", inputs::camera().at("hallo-cam:cam").size(224))
        .input("task", json!("text"))
        .action(
            "left",
            actions::joints()
                .at("gimbal-left:servos")
                .command(JointCommand::Position)
                .hz(50)
                .chunk(4),
        )
        .action(
            "right",
            json!({"kind": "joints", "at": "gimbal-right:servos", "chunk": 4}),
        )
        .allow([Allow::Resize]);
    let json = r#"{"hz": 30,
        "inputs": {"cam": {"kind": "camera", "at": "hallo-cam:cam", "size": [224, 224]},
                   "task": "text"},
        "actions": {"left": {"kind": "joints", "at": "gimbal-left:servos",
                             "command": "position", "hz": 50, "chunk": 4},
                    "right": {"kind": "joints", "at": "gimbal-right:servos", "chunk": 4}},
        "allow": ["resize"]}"#;
    assert_eq!(mixed.build(), Policy::parse(json));
    let policy = mixed.build().expect("build");
    let names: Vec<_> = policy
        .inputs
        .iter()
        .map(|i| i.entry.name.as_str())
        .collect();
    assert_eq!(names, ["cam", "task"], "entries keep the order added");
    let text = mixed.to_json().expect("json");
    assert_eq!(
        serde_json::from_str::<Value>(&text).expect("json"),
        serde_json::from_str::<Value>(json).expect("json"),
    );
}

/// What the normaliser refuses, it refuses for a built entry in the same
/// words, naming the entry.
#[test]
fn a_built_policy_is_refused_as_its_json_is() {
    let refused = |b: PolicyBuilder, json: &dyn ToString, needle: &str| {
        let err = b.build().expect_err(needle);
        assert_eq!(
            Err(err.clone()),
            Policy::parse(&json.to_string()),
            "{needle}"
        );
        assert!(err.contains(needle), "{err}");
    };
    refused(
        Policy::builder().input("x", inputs::camera().size(0)),
        &json!({"inputs": {"x": {"kind": "camera", "size": [0, 0]}}}),
        "missing field `hz`",
    );
    refused(
        Policy::builder().hz(1).input("x", inputs::camera().size(0)),
        &json!({"hz": 1, "inputs": {"x": {"kind": "camera", "size": [0, 0]}}}),
        "input \"x\": image sides are 1 to 8192",
    );
    refused(
        Policy::builder()
            .hz(1)
            .action("x", actions::joints().chunk(0)),
        &json!({"hz": 1, "actions": {"x": {"kind": "joints", "chunk": 0}}}),
        "action \"x\": chunk is 1 to",
    );
    refused(
        Policy::builder()
            .hz(1)
            .input("x", inputs::joints().names(["a", "a"])),
        &json!({"hz": 1, "inputs": {"x": {"kind": "joints", "names": ["a", "a"]}}}),
        "repeated",
    );
    refused(
        Policy::builder()
            .hz(1)
            .input("x", inputs::custom("rlmesh/x")),
        &json!({"hz": 1, "inputs": {"x": {"kind": "rlmesh/x"}}}),
        "reserved",
    );
    refused(
        Policy::builder().hz(1).input("x", inputs::custom("x/y")),
        &json!({"hz": 1, "inputs": {"x": {"kind": "x/y"}}}),
        "explicit fields schema",
    );
    refused(
        Policy::builder()
            .hz(1)
            .input("x", inputs::text())
            .input("x", inputs::text()),
        &r#"{"hz":1,"inputs":{"x":{"kind":"text"},"x":{"kind":"text"}}}"#,
        "duplicate key",
    );
    refused(
        Policy::builder()
            .hz(1)
            .input("x", inputs::camera().dtype(DType::Utf8)),
        &json!({"hz": 1, "inputs": {"x": {"kind": "camera", "dtype": "utf8"}}}),
        "dtype \"utf8\"",
    );
    refused(
        Policy::builder()
            .hz(1)
            .input("x", inputs::camera().resample(Resample::Nearest)),
        &json!({"hz": 1, "inputs": {"x": {"kind": "camera", "resample": "nearest"}}}),
        "resample \"nearest\"",
    );
    refused(
        Policy::builder().hz(1),
        &json!({"hz": 1}),
        "no inputs and no actions",
    );
}

/// JSON has no NaN or infinity: the builder refuses one before the
/// normaliser, in its words for an out-of-range value.
#[test]
fn a_non_finite_number_is_refused_in_the_normalisers_words() {
    for (b, expected) in [
        (
            Policy::builder().hz(f64::NAN).input("x", inputs::text()),
            "policy: hz NaN is not in (0, 10000]",
        ),
        (
            Policy::builder()
                .hz(1)
                .action("x", actions::joints().hz(f64::INFINITY)),
            "action \"x\": hz inf is not in (0, 10000]",
        ),
        (
            Policy::builder()
                .hz(1)
                .input("x", inputs::camera().history(2).history_hz(f64::NAN)),
            "input \"x\": history_hz NaN is not in (0, 10000]",
        ),
        (
            Policy::builder()
                .hz(1)
                .input("x", inputs::camera().crop(f64::NAN)),
            "input \"x\": crop is a side fraction in (0, 1]",
        ),
        (
            Policy::builder()
                .hz(1)
                .input("x", inputs::value().range(0, f64::NAN)),
            "input \"x\": range is two finite, different numbers",
        ),
        (
            Policy::builder()
                .hz(1)
                .input("x", inputs::joints().scale([1.0, f64::NAN])),
            "input \"x\": scale must be finite numbers",
        ),
        (
            Policy::builder()
                .hz(1)
                .action("x", actions::joints().offset(f64::NEG_INFINITY)),
            "action \"x\": offset must be finite numbers",
        ),
        (
            Policy::builder()
                .hz(1)
                .action("x", actions::gripper().open(f64::NAN)),
            "action \"x\": open must be finite",
        ),
    ] {
        assert_eq!(b.build(), Err(expected.to_string()));
    }
    // Out of range but finite, the normaliser words it the same way.
    assert_eq!(
        Policy::builder().hz(0).input("x", inputs::text()).build(),
        Err("policy: hz 0 is not in (0, 10000]".into())
    );
    let err = serde_json::to_value(inputs::camera().crop(f64::NAN)).expect_err("NaN");
    assert!(err.to_string().contains("crop is a side fraction"), "{err}");
}

#[test]
fn of_refuses_what_the_parts_kind_does_not_take() {
    let eyes = json!({"kind": "joints", "names": ["pan", "tilt"], "hz": 50, "fields": ["pos"]});
    assert_eq!(
        Policy::builder()
            .hz(1)
            .input("x", inputs::of(&eyes).size(4))
            .build(),
        Err("input \"x\": a joints input takes no size".into())
    );
    let err = Policy::builder()
        .hz(1)
        .input("x", inputs::of(&json!("joints")))
        .build()
        .expect_err("not a part");
    assert!(
        err.starts_with("input \"x\": of: \"joints\" is not a part"),
        "{err}"
    );
}

#[test]
fn setting_a_key_again_replaces_it_and_lists_add_up() {
    let built = serde_json::to_value(
        inputs::joints()
            .at("a")
            .at("b")
            .names(["x"])
            .unit("pos", "deg")
            .unit("vel", "deg/s")
            .unit("pos", "rad")
            .allow([Allow::Select])
            .allow([Allow::Units]),
    )
    .expect("serialise");
    assert_eq!(
        built,
        json!({"kind": "joints", "at": "b", "names": ["x"],
               "units": {"pos": "rad", "vel": "deg/s"}, "allow": ["select", "units"]})
    );
    let custom = serde_json::to_value(
        actions::custom("x/y")
            .field("a", Array::new(DType::F32, [2]))
            .field("b", Array::new(DType::Bool, []))
            .field("a", Array::new(DType::F64, [3])),
    )
    .expect("serialise");
    assert_eq!(
        custom,
        json!({"kind": "x/y", "fields": {"a": ["float64", [3]], "b": ["bool", []]}})
    );
}

#[test]
fn whole_numbers_are_written_as_integers() {
    let text = serde_json::to_string(
        &actions::joints()
            .hz(50)
            .scale([0.5, 2.0])
            .ttl(Duration::from_micros(1500)),
    )
    .expect("serialise");
    assert_eq!(
        text,
        r#"{"kind":"joints","hz":50,"scale":[0.5,2],"ttl_ms":1.5}"#
    );
    let wrench = serde_json::to_string(
        &actions::wrench()
            .command(WrenchCommand::Wrench)
            .scale(1e300),
    )
    .expect("serialise");
    assert_eq!(
        wrench,
        r#"{"kind":"wrench","command":"wrench","scale":1e+300}"#
    );
}
