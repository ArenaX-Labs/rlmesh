# Env Tags: `EnvTags` v1 JSON

**Env tags** are the sparse semantics an environment publishes over its observation and action spaces: which observation leaf is the primary camera, which is the joint position vector, what each slice of the action vector commands. A model's adapter spec resolves against them, so a model written for `proprio/eef_pos` binds to whatever key an env publishes that role under.

Tags carry only _meaning_. All _structure_ (shapes, dtypes, bounds) stays in the spaces, and the tags are checked against them.

This page is the grammar for authors who write the JSON by hand: C and C++ envs (`RlmeshEnvConfig.adapter_tags_json`, `rlmesh::EnvConfig::adapter_tags_json`) and anything else that emits it directly. Python authors normally build it with `rlmesh.adapters.tag(...)`, which produces the same document. The source of truth is the Rust codec in `rlmesh-adapters` (`spec/env_tags.rs`, `spec/action.rs`, `spec/strict.rs`, `join.rs`, `roles/registry.rs`); where this page and the code disagree, the code wins.

On the wire the tags travel in the env contract metadata under the key `rlmesh.adapters.v1.env_tags`. The v1 format grows only additively; a breaking change ships under a new key.

## Top Level

```json
{
  "observation": <observation node>,
  "action": <action layout>
}
```

| Key           | Required | Meaning                                                    |
| ------------- | -------- | ---------------------------------------------------------- |
| `observation` | yes      | The observation tree, mirroring the observation space.     |
| `action`      | yes      | The ordered action components, covering the action vector. |

Any other top-level key is a parse error.

## Observation Tree

An observation node is one of three shapes, told apart structurally:

| JSON                             | Node      | Maps to                                                                     |
| -------------------------------- | --------- | --------------------------------------------------------------------------- |
| an array                         | **Tuple** | a `Tuple` space; item `i` tags item `i`, and the arity must match.          |
| an object with a string `"type"` | **Leaf**  | one space leaf (see [Leaf Kinds](#leaf-kinds)).                             |
| any other object                 | **Dict**  | a `Dict` space; each key names a key of the space, and its value is a node. |

- `"type"` is a **reserved key**. An object with a string `"type"` is always a leaf, and an object whose `"type"` is not a string is an error, so a Dict node cannot name a space key called `type`.
- A Dict node need not list every key of its space. Space keys it omits are **untagged**: they carry no semantics and no model can bind them. Every key it does list must exist in the space.
- A single-leaf observation (a bare `Box`) is tagged by a bare leaf object at the top, with no wrapper.

## Leaf Kinds

Every leaf object carries `"type"` plus the fields of its kind.

### `image`

A camera frame. The space leaf must be a 3-D `Box`; height, width, and channels come from its shape.

| Field         | Required | Default | Meaning                                                                                          |
| ------------- | -------- | ------- | ------------------------------------------------------------------------------------------------ |
| `role`        | yes      |         | The role, e.g. `image/primary` (see [Roles](#roles)).                                            |
| `layout`      | no       | `"hwc"` | Axis order of the space's shape: `"hwc"` or `"chw"`. Any other value is a parse error.           |
| `upside_down` | no       | `false` | The camera is mounted rotated 180°.                                                              |
| `part`        | no       |         | The body part the camera sits on, when the role repeats across a body (`left_arm`, `head`, ...). |

A shape whose channel count looks implausible under the declared layout but plausible under the other (for example `[3, 224, 224]` declared `hwc`) produces an advisory, not an error.

### `state`

A numeric vector read whole. The space leaf must be numeric (`Box`, `Discrete`, `MultiBinary`, or `MultiDiscrete`); its width is its element count.

| Field        | Required | Default      | Meaning                                                                                                                                                                  |
| ------------ | -------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `role`       | yes      |              | The role, e.g. `proprio/joint_pos`. An observation may not carry an `action/` role.                                                                                      |
| `encoding`   | no       |              | A rotation encoding (see [Rotation Encodings](#rotation-encodings)): one name, or a non-empty list in preference order. The first recognized entry must match the width. |
| `range`      | no       | space bounds | `[min, max]`, finite, `min <= max`. Use it when the space is unbounded. When the space has finite uniform bounds, a declared range must agree with them.                 |
| `frame`      | no       |              | The coordinate frame of an absolute pose: `world`, `robot_base`, or `tool`.                                                                                              |
| `provenance` | no       |              | Where the numbers come from: `sensed`, `estimated`, or `privileged`.                                                                                                     |
| `part`       | no       |              | The body part this vector belongs to.                                                                                                                                    |
| `labels`     | no       |              | One distinct name per element, in the order the env emits them. A model that names labels is permuted onto these.                                                        |

### `split`

One flat numeric leaf whose fixed index ranges mean different things (for example a 39-element `Box` holding the end-effector position, the gripper, and object poses). The space leaf must be numeric.

| Field    | Required | Meaning                                          |
| -------- | -------- | ------------------------------------------------ |
| `fields` | yes      | A non-empty, ordered array of fields; see below. |

`split` takes no other keys. Fields are laid out in order from offset 0, and their `dim`s must sum to the leaf's width. Each field:

| Field        | Required | Default      | Meaning                                                                                               |
| ------------ | -------- | ------------ | ----------------------------------------------------------------------------------------------------- |
| `dim`        | yes      |              | Elements this field covers, an integer `>= 1`.                                                        |
| `role`       | no       |              | The field's role. A field with no role is a **skip**: it advances the offset but produces no feature. |
| `encoding`   | no       |              | As on `state`; must match `dim`.                                                                      |
| `range`      | no       | slice bounds | As on `state`, checked against this field's slice of the space bounds.                                |
| `frame`      | no       |              | As on `state`.                                                                                        |
| `provenance` | no       |              | As on `state`.                                                                                        |
| `part`       | no       |              | As on `state`.                                                                                        |
| `labels`     | no       |              | As on `state`, exactly `dim` of them.                                                                 |

A skip carries only `dim`. Within one `split`, a `(role, part, provenance)` triple may appear once: one role may repeat across parts or provenances, never within the same one.

### `text`

A string, typically the task instruction. The space leaf must be a `Text` space.

| Field  | Required | Meaning                                |
| ------ | -------- | -------------------------------------- |
| `role` | yes      | The role, normally `text/instruction`. |

## Action Layout

The action space must be a `Box`. The layout is an ordered list of components that tile its flattened vector from offset 0.

| Key          | Required | Default | Meaning                                                                                                    |
| ------------ | -------- | ------- | ---------------------------------------------------------------------------------------------------------- |
| `components` | yes      |         | Ordered array of actuators; their `dim`s must sum to the action width.                                     |
| `clip`       | no       |         | `[min, max]` applied to the whole assembled vector. For a per-component clamp, use `clip` on the actuator. |

Any other key is a parse error. Each actuator:

| Field         | Required | Default      | Meaning                                                                                                                                                                                               |
| ------------- | -------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dim`         | yes      |              | Elements this component covers, a non-negative integer.                                                                                                                                               |
| `role`        | no       |              | What the component commands, e.g. `action/delta_eef_pos`. A component with no role is **opaque** (see below).                                                                                         |
| `encoding`    | no       |              | One rotation encoding name (no list on the action side), or a custom-encoding object `{"base", "name"?, "from_base"?, "to_base"?}` with at least one arm. Must match `dim`; `gravity_xyz` is refused. |
| `range`       | no       | space bounds | `[min, max]`; must agree with the action space's finite bounds over this slice when it has them.                                                                                                      |
| `binary`      | no       | `false`      | An on/off command (a gripper): after the corrections below, the value snaps to `1.0` when `>= 0` and to `-1.0` otherwise.                                                                             |
| `threshold`   | no       |              | Subtracted after `scale`, `offset`, and `invert`, so a `binary` component switches at this value. Requires `binary` (checked at resolve).                                                             |
| `scale`       | no       |              | Multiply the model's value: `value * scale + offset`.                                                                                                                                                 |
| `offset`      | no       |              | Add after `scale`, e.g. the stand pose a joint target is expressed around.                                                                                                                            |
| `axis_scale`  | no       |              | Per-axis `scale`, one finite number per axis. Exclusive with `scale`.                                                                                                                                 |
| `axis_offset` | no       |              | Per-axis `offset`. Exclusive with `offset`.                                                                                                                                                           |
| `invert`      | no       | `false`      | Negate after `scale` and `offset`, e.g. a gripper whose sign is flipped.                                                                                                                              |
| `clip`        | no       | `false`      | Clamp this component to its `range`, declared or taken from the action space's finite bounds; with neither, resolve fails.                                                                            |
| `optional`    | no       | `false`      | If no model output provides the role, fill these dims with `fill` instead of failing resolution.                                                                                                      |
| `fill`        | no       | `0.0`        | Finite constant for an opaque or `optional` component. Must stay `0.0` on a roled, non-optional one.                                                                                                  |
| `frame`       | no       |              | Frame of a Cartesian command's axes: `world`, `robot_base`, or `tool`.                                                                                                                                |
| `reference`   | no       |              | What a delta command is added to: `current` (the measured pose) or `target` (the last commanded target).                                                                                              |
| `part`        | no       |              | The body part this component drives.                                                                                                                                                                  |
| `labels`      | no       |              | One distinct name per axis, exactly `dim` of them.                                                                                                                                                    |

An **opaque** component (no `role`) occupies its dims with the constant `fill` and is matched by no model; it may carry only `dim` and `fill`. A `(role, part)` pair may appear once in a layout.

## Roles

A role is `<kind>/<name>`. The kind is one of `image/`, `proprio/`, `text/`, `action/`, `command/`, or the `x/` escape; any other prefix is refused. A role with no `/` is ad hoc and resolves only on exact-string agreement with a model.

Registered roles, with the width the env must declare where one is fixed:

| Role                 | Width       | Role                   | Width       |
| -------------------- | ----------- | ---------------------- | ----------- |
| `image/primary`      | (image)     | `action/joint_pos`     | any         |
| `image/secondary`    | (image)     | `action/joint_vel`     | any         |
| `image/wrist`        | (image)     | `action/delta_eef_pos` | 3           |
| `text/instruction`   | (text)      | `action/delta_eef_rot` | by encoding |
| `proprio/joint_pos`  | any         | `action/eef_pos`       | 3           |
| `proprio/joint_vel`  | any         | `action/eef_rot`       | by encoding |
| `proprio/eef_pos`    | 3           | `action/gripper`       | any         |
| `proprio/eef_rot`    | by encoding | `proprio/base_ang_vel` | 3           |
| `proprio/gripper`    | any         | `proprio/base_rot`     | by encoding |
| `proprio/eef_wrench` | 6           | `command/base_vel`     | 3           |

A quantity with no registered role takes an `x/` role (`x/target_pos`): explicitly non-standard, matched only by a model that names the same string. An unregistered role outside `x/` still works but produces an advisory. Observation leaves may not use `action/` roles.

Suggested `part` names are `left_arm`, `right_arm`, `head`, `torso`, `base`, `left_leg`, and `right_leg`; any string both sides agree on works.

## Rotation Encodings

| Name             | Width | Notes                                            |
| ---------------- | ----- | ------------------------------------------------ |
| `quat_xyzw`      | 4     |                                                  |
| `quat_wxyz`      | 4     |                                                  |
| `axis_angle`     | 3     |                                                  |
| `rot6d`          | 6     | First two rotation-matrix columns, column-major. |
| `rot6d_rowmajor` | 6     | The same block flattened row-major.              |
| `euler_xyz`      | 3     | Roll, pitch, yaw in radians, extrinsic XYZ.      |
| `gravity_xyz`    | 3     | Projected gravity. Observation only.             |

## Validation

Tags are checked in three stages. An env built with the C API runs all three in `rlmesh_env_new`, which fails with `RLMESH_ERR_INVALID_ARGUMENT` and a message naming the offending key; Python's `adapters.tag()` runs the same checks.

1. **Parse.** The JSON must match the grammar above: required keys present, values of the right type, counts as non-negative integers (`3.0` is refused), ranges as finite `[min, max]` pairs with `min <= max`, a known `layout`, a known action `encoding`, no unknown keys on the top level, the action envelope, or a `split`. Cross-field rules are enforced here too: a skip or opaque component carries nothing else, labels match `dim`, no duplicate roles, `scale` and `axis_scale` are not both set.
2. **Publish gate.** The parser is deliberately tolerant so a newer writer's spec survives an older reader: an unrecognized field on an `image`, `state`, or `text` leaf, a `split` field, or an actuator is captured rather than refused, and a leaf with an unrecognized `"type"` parses as an unknown kind. Publishing then rejects both. An unknown field is refused unless its name starts with `x-` or `ext.`, the namespace a producer uses to mark a field safe to ignore. An unknown leaf kind is refused because this core cannot build it. A role with an undefined kind prefix is refused here too.
3. **Join against the spaces.** Every tagged key exists in the space; each leaf has the space class its kind needs; Tuple arity matches; `split` fields and action components sum to their leaf's width; an encoding's width matches; a fixed-width role has its width; a declared `range` agrees with finite space bounds; the action space is a `Box`.

When a model resolves against published tags (the read side), unknown bare fields still fail, and an unknown leaf kind is ignored with an advisory unless the model's spec asks for its role, which then fails resolution. A `frame`, `reference`, or `provenance` value outside its vocabulary, and a state `encoding` naming no recognized encoding, parse and round-trip but fail at resolve; unrecognized entries in an encoding list are skipped when another entry is recognized.

## Example

The Project Chrono reach env (`examples/chrono/src/reach_env.cpp`) serves this observation `Dict` and a 3-D action `Box(-1, 1)`:

| Key            | Space                   |
| -------------- | ----------------------- |
| `image`        | `Box(uint8, [H, W, 3])` |
| `joint_pos`    | `Box(float32, [6])`     |
| `joint_vel`    | `Box(float32, [6])`     |
| `joint_torque` | `Box(float32, [6])`     |
| `eef_pos`      | `Box(float32, [3])`     |
| `target_pos`   | `Box(float32, [3])`     |
| `instruction`  | `Text`                  |

Its tags:

```json
{
  "observation": {
    "image": { "type": "image", "role": "image/primary", "layout": "hwc" },
    "joint_pos": { "type": "state", "role": "proprio/joint_pos" },
    "joint_vel": { "type": "state", "role": "proprio/joint_vel" },
    "eef_pos": { "type": "state", "role": "proprio/eef_pos" },
    "target_pos": { "type": "state", "role": "x/target_pos" },
    "instruction": { "type": "text", "role": "text/instruction" }
  },
  "action": {
    "components": [{ "role": "action/delta_eef_pos", "dim": 3 }],
    "clip": [-1.0, 1.0]
  }
}
```

`joint_torque` is left untagged. `target_pos` has no registered role, so it takes an `x/` one. `eef_pos` and `action/delta_eef_pos` are fixed at 3 elements, which the spaces satisfy.

A flat-vector env with a quaternion, a gripper, and a 7-D command uses `split` and more actuator fields:

```json
{
  "observation": {
    "state": {
      "type": "split",
      "fields": [
        { "role": "proprio/eef_pos", "dim": 3, "frame": "robot_base" },
        { "role": "proprio/eef_rot", "dim": 4, "encoding": "quat_xyzw", "frame": "robot_base" },
        { "dim": 2 },
        { "role": "proprio/gripper", "dim": 1, "range": [0.0, 0.08] }
      ]
    }
  },
  "action": {
    "components": [
      { "role": "action/delta_eef_pos", "dim": 3, "reference": "current" },
      {
        "role": "action/delta_eef_rot",
        "dim": 3,
        "encoding": "axis_angle",
        "reference": "current"
      },
      { "role": "action/gripper", "dim": 1, "binary": true, "threshold": 0.0, "invert": true }
    ]
  }
}
```

Here `state` is a 10-element `Box` (two elements skipped) and the action space a 7-element `Box`.
