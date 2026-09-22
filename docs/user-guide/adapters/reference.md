# Adapter Reference

The complete field-by-field reference for the declarative adapter specs. Use it to match your environment or model shape to a feature, then look up the exact behavior of every field.

For the concepts (how the two sides connect and why), start with {doc}`/user-guide/adapters`. For logic a spec cannot express, see {doc}`/user-guide/adapters/escape-hatches`. For exact signatures and the autodoc, see {doc}`/api/adapters`. Examples live at {source}`examples/python/adapters`.

Every snippet uses `import rlmesh.adapters as adapt`.

## Role registry

A role is the string that matches an environment feature to a model input. Roles are an **open vocabulary**: any string works as long as the env tag and the model spec agree on it verbatim. The constants below are the well-known conventions RLMesh ships; reach for them so independently authored envs and models line up, and invent your own string for anything they do not cover.

Role strings carry a feature-kind prefix, not a domain prefix (two domains sharing `proprio/joint_pos` is intentional). The kinds are a **closed set of five**: `image/`, `proprio/`, `text/`, `action/`, and `command/` (a numeric setpoint the runner hands the policy, the numeric sibling of `text/instruction`; `command/base_vel` is its first registered role). The set is enforced in code: a role under any other prefix is refused when you author it (`adapt.tag`, `to_dict()`, the publish gate) and when a pair resolves, so a spec written by a newer core still parses and relays but fails loudly here rather than binding to nothing. `x/` remains the escape for a whole role; a role with no `/` names no kind and is simply ad-hoc.

| Constant           | Wire string            | Domain       | Kind       | Typical width / encoding               |
| ------------------ | ---------------------- | ------------ | ---------- | -------------------------------------- |
| `IMAGE_PRIMARY`    | `image/primary`        | core         | `image/`   | H×W×C frame (main camera)              |
| `IMAGE_SECONDARY`  | `image/secondary`      | core         | `image/`   | H×W×C frame (second fixed camera)      |
| `IMAGE_WRIST`      | `image/wrist`          | core         | `image/`   | H×W×C frame (wrist/hand camera)        |
| `INSTRUCTION`      | `text/instruction`     | core         | `text/`    | string (task instruction)              |
| `JOINT_POS`        | `proprio/joint_pos`    | core         | `proprio/` | N joints (embodiment-dependent)        |
| `JOINT_VEL`        | `proprio/joint_vel`    | core         | `proprio/` | N joints                               |
| `ACTION_JOINT_POS` | `action/joint_pos`     | core         | `action/`  | N joints (embodiment-dependent)        |
| `ACTION_JOINT_VEL` | `action/joint_vel`     | core         | `action/`  | N joints                               |
| `EEF_POS`          | `proprio/eef_pos`      | manipulation | `proprio/` | 3 (Cartesian xyz)                      |
| `EEF_ROT`          | `proprio/eef_rot`      | manipulation | `proprio/` | width follows the rotation encoding    |
| `GRIPPER_POS`      | `proprio/gripper`      | manipulation | `proprio/` | 1+ (embodiment-dependent)              |
| `EEF_WRENCH`       | `proprio/eef_wrench`   | manipulation | `proprio/` | 6 (`[fx, fy, fz, tx, ty, tz]`, framed) |
| `ACTION_DELTA_POS` | `action/delta_eef_pos` | manipulation | `action/`  | 3 (Cartesian delta)                    |
| `ACTION_DELTA_ROT` | `action/delta_eef_rot` | manipulation | `action/`  | width follows the rotation encoding    |
| `ACTION_GRIPPER`   | `action/gripper`       | manipulation | `action/`  | 1                                      |
| `ACTION_EEF_POS`   | `action/eef_pos`       | manipulation | `action/`  | 3 (absolute Cartesian target)          |
| `ACTION_EEF_ROT`   | `action/eef_rot`       | manipulation | `action/`  | width follows the rotation encoding    |
| `BASE_ANG_VEL`     | `proprio/base_ang_vel` | body         | `proprio/` | 3 (the IMU gyro, framed)               |
| `BASE_ROT`         | `proprio/base_rot`     | body         | `proprio/` | width follows the rotation encoding    |
| `COMMAND_BASE_VEL` | `command/base_vel`     | body         | `command/` | 3 (`[vx, vy, wz]`, framed)             |

You always pin widths explicitly (`dim`/`index` on a part, `dim` on an actuator); a _registered_ role with a fixed canonical width then **validates** that declared `dim` (e.g. `eef_pos` must be 3-D, a mismatch is a resolve error); it never supplies it. Rotation widths follow the declared encoding (see [Vocabularies](#vocabularies)).

**Registered vs. ad-hoc roles.** A registered role (the table above) is a shared contract: independently authored envs and models line up on it without prior agreement, and the fixed-width ones validate their `dim`. An _ad-hoc_ role (any other `<kind>/<name>` string) still resolves on verbatim agreement, but it draws a non-fatal authoring nudge, and at the managed-service publish boundary a curated tier may reject it (`role_policy="strict"`). When a role is _intentionally_ non-standard (a self-contained env/model pair you own, or a not-yet-blessed domain), mark it with the reserved **`x/` prefix**: an `x/...` role is never nudged and always passes the publish gate, declaring "I know this isn't standard." For an action dim no model reads, prefer a role-less (opaque) actuator over an ad-hoc role.

A role is **registered when both sides of it exist**: an environment that produces the data and a model that reads it. One side alone does not earn a slot — a camera nothing looks at, or a command no policy emits, stays ad-hoc (or `x/`) until its counterpart ships, because until then nothing has pinned what the numbers mean.

A role also names a **raw sensed or commanded quantity**, never a function of one. The three body roles are what a legged robot's base reports and takes: the gyro (`BASE_ANG_VEL`), the IMU orientation (`BASE_ROT`) and the velocity setpoint the runner hands the policy (`COMMAND_BASE_VEL`), all three expressed in a `frame`. The projected gravity every locomotion checkpoint reads is not a fourth role: it is `BASE_ROT` read with `encoding="gravity_xyz"` (see [Vocabularies](#vocabularies)). The base linear velocity is deliberately unregistered: a real robot has only an estimate of it, so a role would silently mean two different things in sim and on hardware; a gait-phase clock is neither sensed nor commanded and stays a `Constant` or an `x/` role. The same law gives manipulation one force role: `EEF_WRENCH` is the six-axis force/torque a wrist sensor reads (force first, `[fx, fy, fz, tx, ty, tz]`), framed like a pose. A wrist-mounted sensor reads it in `tool`; a robot controller's estimate is usually in `robot_base`; an env that has both publishes both under `provenance="sensed"` and `"estimated"`. Its `labels` (`fx`..`tz`) name its own axes and are never held to an embodiment profile. A commanded force is not a role: the controller that tracks it is the env's, and its setpoint is an `x/` role until a second consumer earns it.

### Parts

A role names a quantity; a **part** names where on the body it is measured or commanded when the same role repeats. A bimanual robot has one `proprio/eef_pos` per arm, a humanoid has a camera on its head and a joint vector per leg: each is the same role under a different part. `part` is a keyword-only attribute on `ImageTag`, `StateTag`, `Field`, `Actuator`, `Image`, and a `State`/`Concat` part. It is an **identity key**, not a value the resolver checks: two leaves agree on a part or they do not, and nothing is converted.

```python
tags = adapt.EnvTags(
    observation={
        "left_pos": adapt.StateTag(adapt.EEF_POS, part=adapt.LEFT_ARM),
        "right_pos": adapt.StateTag(adapt.EEF_POS, part=adapt.RIGHT_ARM),
        "head_cam": adapt.ImageTag(adapt.IMAGE_PRIMARY, part=adapt.HEAD),
    },
    action=adapt.Action(
        adapt.Actuator(adapt.ACTION_JOINT_POS, dim=7, part=adapt.LEFT_ARM),
        adapt.Actuator(adapt.ACTION_JOINT_POS, dim=7, part=adapt.RIGHT_ARM),
    ),
)
```

Parts are a registered vocabulary governed exactly like roles. The registered parts are `LEFT_ARM`, `RIGHT_ARM`, `ARM_2`, `HEAD`, `TORSO`, `BASE`, `LEFT_LEG`, and `RIGHT_LEG` (the list is `adapt.PARTS`); a part is a physical place a role can repeat at, never a per-robot name, and the list grows when a real env and model pair needs a slot. Any other part is ad-hoc: it still binds on verbatim agreement, it draws the same non-fatal authoring nudge an ad-hoc role does, and the managed `role_policy="strict"` tier rejects it. Write an intentionally non-standard part under the `x/` prefix (`part="x/tail"`) to declare that and pass the gate. A role-less skip, opaque actuator, or `Constant` carries no part.

A role may repeat across parts on one side but never within one: a `Split`, an `Action`, or an `EnvTags` that declares the same `(role, part)` twice is refused at construction and at the codec. The resolve rules, per leaf:

| Model says | Env has, for that role                   | Outcome                                                                           |
| ---------- | ---------------------------------------- | --------------------------------------------------------------------------------- |
| no part    | a leaf with no part                      | match                                                                             |
| no part    | exactly one leaf, under some part        | binds to it, with an `info` naming the part (declare `part=` to pin it)           |
| no part    | several leaves, under parts              | **resolve error** naming the parts (`declare part=`), even for an `optional` part |
| `part=P`   | a leaf under `P`                         | match                                                                             |
| `part=P`   | no leaf under `P` (whatever else it has) | **resolve error**, or the fill when `optional` -- a named part never rebinds      |

The action side follows the same table with the env actuator as the one looking: an env actuator without a part binds the model's only output of that role under any part, and a named part binds only its own. `explain()` shows the part a leaf was bound under as a `#part` suffix -- `left_pos[:3]#left_arm`, `"action/joint_pos" <- model[0:7]#right_arm` -- and only when a side declared one, so a pre-`part` summary is unchanged.

**The `_2` roles.** The ten second-arm roles that predate parts (`EEF_POS_2`, `EEF_ROT_2`, `GRIPPER_POS_2`, `ACTION_DELTA_POS_2`, `ACTION_DELTA_ROT_2`, `ACTION_GRIPPER_2`, `ACTION_EEF_POS_2`, `ACTION_EEF_ROT_2`, `ACTION_JOINT_POS_2`, `IMAGE_WRIST_2`) are the legacy spelling of the base role under `part=ARM_2`, and the resolver folds them before it matches: a model written against `EEF_POS_2` binds an environment that declares `StateTag(EEF_POS, part=ARM_2)`, and a model that names `part=ARM_2` binds a `_2` environment. `arm_2` means "the second arm", not a side, so a fresh bimanual environment that wants to serve shipped `_2` models names its second arm `ARM_2`; one that names `LEFT_ARM`/`RIGHT_ARM` cannot, and the error says so. A `_2` role may not also carry a part, and no further `_N` role will ever be added. Prefer `part=` for new specs; the `_2` constants remain for the environments and models that already use them.

### Labels

A joint vector is only a list of numbers until something says which joint each one is. A Unitree Go2 reports its twelve joints in SDK motor order (`FR, FL, RR, RL` x `hip, thigh, calf`); Isaac Lab and legged_gym read the same robot in `FL, FR, RL, RR` order, so a checkpoint trained there emits and expects a different permutation, and every deploy script in the wild carries `[3,4,5,0,1,2,9,10,11,6,7,8]` by hand. `labels` puts that fact on the leaf instead. It is a keyword-only tuple of axis names on `StateTag`, `Field`, `Actuator`, and a `State`/`Concat` part, one per element in that side's own order, and the resolver aligns the two sides **by name**: a differing order is a permutation, a subset is a selection, and the permutation is derived, never typed.

```python
import rlmesh.adapters as adapt
from rlmesh.adapters import embodiments

SDK = embodiments.GO2.joints                      # FR_hip, FR_thigh, FR_calf, FL_hip, ...
ISAAC = tuple(SDK[i] for i in (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8))   # FL, FR, RL, RR

tags = adapt.EnvTags(                              # the robot, SDK order on every backend
    observation={
        "joint_pos": adapt.StateTag(adapt.JOINT_POS, labels=SDK),
        "joint_vel": adapt.StateTag(adapt.JOINT_VEL, labels=SDK),
    },
    action=adapt.Action(adapt.Actuator(adapt.ACTION_JOINT_POS, dim=12, labels=SDK)),
)
spec = adapt.ModelSpec(                            # a checkpoint trained in Isaac order
    input={"obs": adapt.Concat(
        adapt.State(adapt.JOINT_POS, labels=ISAAC, offset=tuple(-q for q in (0.0, 0.8, -1.5) * 4)),
        adapt.State(adapt.JOINT_VEL, labels=ISAAC, scale=0.05),
    )},
    output=adapt.Action(adapt.Actuator(
        adapt.ACTION_JOINT_POS, dim=12, labels=ISAAC,
        scale=(0.125, 0.25, 0.25) * 4, offset=(0.0, 0.8, -1.5) * 4,
    )),
)
```

`explain()` on that pairing prints `joint_pos perm[3,4,5,0,1,2,9,10,11,6,7,8]` on both joint parts and the inverse scatter on the actuator; a checkpoint that lists `SDK` instead resolves to the identity and prints `joint_pos[:12]`. Labels fix a part's width (the tuple's length), so a labeled part declares no `dim` and cannot carry `index`. Label spelling is `<part>_<joint>` snake case as the vendor SDK and Isaac Lab name the joints; write the tuple from a shipped profile (below) rather than by hand. A label repeated within one leaf is refused at construction. The resolve rules, per leaf:

| Env says | Model says             | Outcome                                                                                                                                     |
| -------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| nothing  | nothing                | the env's order, as today                                                                                                                   |
| labels   | nothing                | silence: the model reads the env's order, and the env's names annotate any per-axis value it declares                                       |
| nothing  | labels                 | **resolve error** (`LabelMismatch`): a labeled model bound on hope is the bug labels exist to catch; label the env leaf                     |
| labels   | the same set           | gathered by name into the model's order; `explain()` prints `perm[..]` when the order differs, nothing when it is the identity              |
| labels   | a subset (observation) | the named axes, in the model's order; `explain()` prints `select[..]`                                                                       |
| labels   | a subset (action)      | scatters onto the named axes; the rest take the env actuator's `fill` **only if it is `optional`** (with an `info` naming them), else error |
| labels   | a name the env lacks   | **resolve error** (`LabelMismatch`) naming it; an `optional` observation part fills instead                                                 |

The action side is the mirror of the observation side: the model's output is scattered onto the env actuator's axes by name, after the model's own `scale`/`offset` (in the model's order) and before the env's. A width that disagrees with a label count is refused where the width is known: at construction for `Field` and `Actuator` (`dim`), at `join` for a `StateTag` (the space width), and at resolve for a per-axis vector. Labels are verbatim strings with no alias table, so a Go2 model against a Spot environment is a `LabelMismatch`, which is the truthful outcome.

A humanoid with more joints than a checkpoint drives is the subset case end to end. A 29-DoF G1 environment declares `StateTag(JOINT_POS, labels=G1)` and `Actuator(ACTION_JOINT_POS, dim=29, labels=G1, optional=True, fill=STAND_POSE)`; a legs-and-waist checkpoint names its fifteen labels, reads fifteen of twenty-nine on the observation side, and on the action side drives those fifteen while the fourteen arm joints hold the stand pose, with an `info` listing them.

### Per-axis scale, offset and fill

`scale` and `offset` on a `State` part and an `Actuator` take either one float for every axis or a sequence with one value per axis, in the declaring side's own `labels` order (its resolved width when it carries none). A sequence serializes under its own wire key (`axis_scale`, `axis_offset`), so the scalar key never changes type and a spec that uses only scalars is byte-identical to before; declaring both forms for one quantity is a construction error. The Go2 example above is the common case: a per-joint action scale (hips halved) and a stand pose added to every target, and a stand pose subtracted from every observed angle. A per-axis vector whose length differs from the resolved width is a resolve error (`DimMismatch`). Setting `range` alongside an affine keeps the same `info` it always did.

`Actuator` gains a scalar `offset` alongside `scale` (`value * scale + offset`, added before `invert` and `threshold`); like the others it can be declared on either side. `fill` on an env actuator that is `optional` and carries `labels` may be a per-axis sequence too (`axis_fill` on the wire, one value per label): it is what each axis a model's label subset leaves undriven is held at, and what the whole actuator emits when no model output drives it. `explain()` prints every per-axis value as `label:value` when the axes have names, `*[FR_hip:0.125,FR_thigh:0.25,...]`, so the pairing shows which joint gets which number.

### Embodiment profiles

The label tuples ship as data so nobody retypes twelve joint names: `rlmesh.adapters.embodiments` carries one `EmbodimentProfile` (`.name`, `.parts`, `.joints`) per body, defined once in the native crate.

| Profile                    | `.name`            | `.joints`                                                                                                                                                                                   |
| -------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `embodiments.GO2`          | `unitree_go2`      | 12, SDK motor order: `FR_hip, FR_thigh, FR_calf, FL_hip, ..., RR_..., RL_...`                                                                                                               |
| `embodiments.G1_29DOF`     | `unitree_g1_29dof` | 29: `left_hip_pitch` .. `left_ankle_roll`, `right_*` x6, `waist_yaw`, `waist_roll`, `waist_pitch`, `left_shoulder_pitch` .. `left_wrist_yaw`, `right_*` x7                                  |
| `embodiments.FRANKA_PANDA` | `franka_panda`     | 7: `panda_joint1` .. `panda_joint7`                                                                                                                                                         |
| `embodiments.UR5E`         | `ur5e`             | 6, URDF order: `shoulder_pan_joint`, `shoulder_lift_joint`, `elbow_joint`, `wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint` (a Robotiq or other gripper is its own `proprio/gripper` leaf) |

An environment writes `labels=embodiments.GO2.joints`; a model writes the same tuple, a subset, or a reordering. Profiles are data, not a resolver input: the resolver never consults them, two sides agree on label strings or they do not. What consults them is the **label lint**: on a joint role (`proprio/joint_pos`, `proprio/joint_vel`, `action/joint_pos`, `action/joint_vel`), a tuple whose set matches no shipped profile (subsets allowed, order ignored) draws the same non-fatal `info` an ad-hoc role does, `unknown_labels`, naming the closest profile by overlap, and the managed `--require-labels` tier refuses it. A labeled leaf of any other role (the six axes of a wrench, say) names its own axes and is never checked against a profile. A profile grows the way a role does, when a real environment and model pair needs it.

## Vocabularies

Rotation encodings are a closed set (a remote client must resolve a spec with no code). Each has a fixed native width:

| Encoding         | Width | Notes                             |
| ---------------- | ----- | --------------------------------- |
| `quat_xyzw`      | 4     | quaternion, scalar-last           |
| `quat_wxyz`      | 4     | quaternion, scalar-first          |
| `axis_angle`     | 3     | rotation vector                   |
| `rot6d`          | 6     | 6-D continuous (Zhou et al.)      |
| `rot6d_rowmajor` | 6     | 6-D continuous, row-major packing |
| `euler_xyz`      | 3     | Euler angles, XYZ                 |
| `gravity_xyz`    | 3     | projected gravity, a sink (below) |

`gravity_xyz` is the gravity direction expressed in the body frame, `R_world_from_base^T · (0, 0, -1)`: an upright base reads `(0, 0, -1)`, one on its back `(0, 0, 1)`. It is the 3-vector locomotion checkpoints consume instead of an orientation, and it follows a **direction law**: any rotation encoding converts _into_ it, but it carries only a direction (the yaw is gone), so nothing converts _out_ of it. An env that publishes `gravity_xyz` natively binds only a model part that reads `gravity_xyz` (anything else is an `EncodingMismatch`), a `post_rotate` cannot name it (it composes _before_ the projection when the model reads a rotation as gravity), and an actuator cannot be expressed in it (a codec error). It is observation-side only, and because the observation side reads encodings through an accept set, an older core parses it as an unknown entry and fails at resolve naming it; no edition is minted.

Other vocabularies:

| Vocabulary    | Values                              | Default                             | Notes                                                                  |
| ------------- | ----------------------------------- | ----------------------------------- | ---------------------------------------------------------------------- |
| Provenance    | `sensed`, `estimated`, `privileged` | (none)                              | where a state leaf's numbers come from (see [Provenance](#provenance)) |
| Image layout  | `hwc`, `chw`                        | `hwc`                               | axis order of the stored image                                         |
| Fit mode      | `stretch`, `crop`, `pad`            | (none)                              | how to reconcile an aspect mismatch                                    |
| Crop mode     | `zoom`, `slice`                     | `zoom`                              | how a `crop` box is taken                                              |
| Channel order | `rgb`, `bgr`                        | `rgb`                               | channel order the model expects                                        |
| dtype         | any NumPy dtype name                | `uint8` (image) / `float32` (state) | string, e.g. `"float32"`                                               |

### Frames and references

Cartesian numbers mean nothing on their own: `[0.31, -0.02, 0.18]` is a point in _some_ frame, and `[0.01, 0.0, -0.005]` is a step away from _some_ pose. Two keyword-only attributes say which, and they are how the contract catches a pairing whose numbers type-check and whose geometry does not.

| Attribute   | Values                        | Applies to                                                                | Declared on                                            |
| ----------- | ----------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------ |
| `frame`     | `world`, `robot_base`, `tool` | absolute poses (`proprio/eef_*`, `action/eef_*`) and the wrench (9 roles) | `StateTag`, `Field`, `State`/`Concat` part, `Actuator` |
| `reference` | `current`, `target`           | deltas: `action/delta_eef_*` (4 roles)                                    | `Actuator`                                             |

**A delta never carries a `frame`** -- it is expressed in the controller's own frame by definition, and there is nothing to agree about. What a delta _does_ need is the pose it is added to: an env declares what its Cartesian controller integrates against -- the measured pose (`current`) or the last commanded target (`target`) -- and a model declares what it was trained against. That is `reference`, and it is the attribute delta roles get instead of `frame`.

Both attributes follow the same rules, and both are **optional**: a spec that declares neither is valid v1 and serializes exactly as it did before they existed.

| Env says | Model says      | Outcome                                                               |
| -------- | --------------- | --------------------------------------------------------------------- |
| nothing  | nothing         | silence -- there is nothing to check                                  |
| a value  | nothing         | silence -- the env stated a fact the model has no requirement against |
| nothing  | a value         | `caution` -- the model states a requirement nothing can confirm       |
| a value  | the same        | silence -- agreement                                                  |
| a value  | a different one | **resolve error** -- and so is any value outside the vocabulary above |

An unrecognized value still parses and round-trips (a newer peer's vocabulary survives relay), but it fails at resolve: a geometry this core cannot name is a geometry it cannot verify.

`explain()` shows the agreed value as a suffix -- `@robot_base` on a pose, `~target` on a delta -- and only when a side declared one, so a pre-geometry summary is unchanged:

```text
observation:
  "state" <- concat(eef_pos[:3]@robot_base)
action:
  "action/delta_eef_pos" <- model[0:3]~target
```

At the managed publish boundary the `--require-frames` tier turns the optionality off: every role the registry says owes an attribute must declare one. Its twin, `--require-labels`, requires `labels` on every numeric leaf that carries a joint role (`proprio/joint_pos`, `proprio/joint_vel`, `action/joint_pos`, `action/joint_vel`) and requires every label tuple to match a shipped [embodiment profile](#embodiment-profiles). Locally, and by default, declaring nothing stays legal.

Neither attribute is part of a leaf's identity: an env leaf is found by `(role, part, provenance)` (see [Provenance](#provenance)), and its frame or reference is then checked against the model's.

### Provenance

Two envs can publish `proprio/base_rot` as the same four numbers and mean different things: a simulator hands out the true orientation, a robot hands out what its state estimator believes. A checkpoint trained on the first that silently binds the second is the sim-to-real failure nobody sees until the robot moves. `provenance` names where a state leaf's numbers come from, so that pairing is a resolve error instead of a limp.

| Value        | Meaning                                        |
| ------------ | ---------------------------------------------- |
| `sensed`     | a physical sensor, or its simulated equivalent |
| `estimated`  | a state estimator's output                     |
| `privileged` | simulator truth with no hardware counterpart   |

It is keyword-only on `StateTag` and `Field` (one value) and on a `State`/`Concat` part (one value, or a sequence of the ones the checkpoint accepts, serialized as a bare string when one). The rules are the frame table above, verbatim, with the model side a set: both silent, silent; env only, silent; model only, `caution`; the env's value in the model's set, silent; anything else, including a value outside the vocabulary on either side, a `ProvenanceMismatch`.

Unlike `frame`, provenance is part of the leaf's **identity**: an env leaf is keyed `(role, part, provenance)`, so a simulator may publish one role twice, as its truth and as the estimate a robot would have, and a model picks by declaring what it wants. A model that declares none against two such leaves is `Ambiguous` (the error names the provenances; declare `provenance=`); one that declares `"estimated"` binds the estimate; one that accepts several binds the first of them, in its own order, that the env declares. A gyro, an IMU orientation and joint encoders are `sensed` on both a simulated and a real robot by the vocabulary's definition, so a sim/real env pair needs no branch for them; a sim-only privileged leaf is a different observation space and belongs on a `tag_params` branch.

`explain()` shows the agreed value as a `#sensed` suffix after any `#part`, and a model-only set as `#sensed|estimated`:

```text
observation:
  "obs" <- concat(ang_vel (*0.25)@robot_base#sensed, base_quat (quat_wxyz->gravity_xyz)@world#sensed, ...)
```

Normalization is one overloaded field, `normalize`: `False` (off, the default), `True` (the conventional `[0, 1]`), or a `(low, high)` pair (e.g. `(-1.0, 1.0)`) to map into a specific range. One field, so an on/off flag can never disagree with a range, and `False` is an authoritative off-switch.

## The environment side

An environment **tags** its observation and action spaces. Tags are sparse: they carry each entry's role plus the few facts the gymnasium spaces cannot express (image layout, rotation encoding, an explicit range). Keys, widths, dtypes, and bounds are read from the spaces by the native `join` step at resolve time.

```python
import rlmesh.adapters as adapt

tags = adapt.EnvTags(
    observation={
        "pixels": adapt.ImageTag(adapt.IMAGE_PRIMARY),
        "eef_pos": adapt.StateTag(adapt.EEF_POS),
        "eef_quat": adapt.StateTag(adapt.EEF_ROT, encoding="quat_xyzw"),
        "gripper": adapt.StateTag(adapt.GRIPPER_POS),
        "task": adapt.TextTag(adapt.INSTRUCTION),
    },
    action=adapt.Action(
        adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=3, encoding="axis_angle"),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1),
    ),
)
```

{class}`~rlmesh.adapters.EnvTags` takes `observation` and `action`. The observation is a recursive tree whose container type **is** the runtime container type:

| Authored container | Maps a space        | Example                                           |
| ------------------ | ------------------- | ------------------------------------------------- |
| Python `dict`      | `Dict`              | `{"pixels": ImageTag(...), "eef": StateTag(...)}` |
| Python `tuple`     | `Tuple`             | `(ImageTag(...), StateTag(...))`                  |
| bare leaf          | a single space leaf | `Split(...)` or one `StateTag(...)`               |

Nesting is real `dict` nesting that mirrors a nested `Dict` space (`{"agent": {"eef_pos": StateTag(...)}}`); there are no dotted keys. A single-leaf observation is the bare leaf with no dict wrapper.

### ImageTag

{class}`~rlmesh.adapters.ImageTag`: one camera image leaf.

| Field                   | Default | What it declares                             | When to use           |
| ----------------------- | ------- | -------------------------------------------- | --------------------- |
| `role` (1st positional) | --      | the image role to match                      | always                |
| `layout`                | `"hwc"` | axis order of the stored frame               | the env stores `chw`  |
| `upside_down`           | `False` | the camera renders 180° rotated from upright | a known flipped mount |
| `part` (keyword-only)   | `None`  | the body part the camera sits on             | the role repeats      |

### StateTag

{class}`~rlmesh.adapters.StateTag`: one numeric proprioception leaf.

| Field                       | Default | What it declares                                                           | When to use                                                     |
| --------------------------- | ------- | -------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `role` (1st positional)     | --      | the state role to match                                                    | always                                                          |
| `encoding`                  | `None`  | rotation encoding (single, or a native-first preference sequence)          | the role is a rotation                                          |
| `range`                     | `None`  | `(low, high)` bounds where the space is unbounded                          | the space leaves this leaf unbounded                            |
| `frame` (keyword-only)      | `None`  | the coordinate frame these values are expressed in                         | the role is an absolute pose                                    |
| `provenance` (keyword-only) | `None`  | where these numbers come from (see [Provenance](#provenance))              | a sim publishes truth beside an estimate, or a checkpoint cares |
| `part` (keyword-only)       | `None`  | the body part this entry belongs to (see [Parts](#parts))                  | the role repeats across the body                                |
| `labels` (keyword-only)     | `None`  | the axis names in the env's order, one per element (see [Labels](#labels)) | a joint vector                                                  |

`range` only supplies bounds the space lacks. If the space declares finite bounds that disagree with it, resolution errors rather than silently overriding them.

### TextTag

{class}`~rlmesh.adapters.TextTag`: a text leaf (typically the instruction). Single field: `role` (1st positional). Use it when the observation carries a string the model conditions on.

### Split + Field

Some environments expose one flat numeric `Box` with fixed index ranges instead of a key per quantity (Metaworld is the common case). {class}`~rlmesh.adapters.Split` tags that single vector: it is a **leaf**, not a container, and the observation-side mirror of {class}`~rlmesh.adapters.Action`.

```python
adapt.EnvTags(
    observation=adapt.Split(
        adapt.Field(adapt.EEF_POS, dim=3),
        adapt.Field(adapt.EEF_ROT, dim=4, encoding="quat_xyzw"),
        adapt.Field(adapt.GRIPPER_POS, dim=1),
        adapt.Field(dim=10),  # skip the object/goal indices the policy reads from pixels
    ),
    action=adapt.Action(...),
)
```

`Split(*Field)` takes its fields positionally and needs at least one. Field widths must sum to the leaf width (checked at join). A {class}`~rlmesh.adapters.Field`:

| Field                       | Default            | What it declares                                  | When to use                         |
| --------------------------- | ------------------ | ------------------------------------------------- | ----------------------------------- |
| `role` (1st positional)     | `None`             | the role for this slice; `None` is a **skip**     | name it, or skip with `None`        |
| `dim`                       | -- (required, ≥ 1) | element count of the slice                        | always                              |
| `encoding`                  | `None`             | rotation encoding (single or preference sequence) | the slice is a rotation             |
| `range`                     | `None`             | `(low, high)` where the space is unbounded        | the slice is unbounded in the space |
| `frame` (keyword-only)      | `None`             | the coordinate frame this slice is expressed in   | the role is an absolute pose        |
| `provenance` (keyword-only) | `None`             | where this slice's numbers come from              | see [Provenance](#provenance)       |
| `part` (keyword-only)       | `None`             | the body part this slice belongs to               | the role repeats across the body    |
| `labels` (keyword-only)     | `None`             | the axis names of this slice, `dim` of them       | a joint vector                      |

A `role=None` field advances the offset without producing a feature; use it to step over indices the model never reads. A skip carries no encoding, range, frame, provenance, part or labels. Fields are keyed by `(role, part, provenance)`, so one role may appear twice under two provenances.

## The model side

A model **fully specifies** the payload it ingests and the action it emits, in its own conventions. {class}`~rlmesh.adapters.ModelSpec` takes `input` and `output`. The `input` tree's container type **is** the payload container `predict` receives (a `dict`, a `tuple`, or a bare single leaf). A leaf carries no key (its position in the tree is the payload position), and a role may be reused across leaves.

```python
spec = adapt.ModelSpec(
    input={
        "image": adapt.Image(adapt.IMAGE_PRIMARY, size=256, normalize=True),
        "state": adapt.Concat(
            adapt.EEF_POS,
            adapt.State(adapt.EEF_ROT, encoding="rot6d"),
            adapt.GRIPPER_POS,
        ),
        "instruction": adapt.Text(adapt.INSTRUCTION),
    },
    output=adapt.Action(
        adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3),
        adapt.Actuator(adapt.ACTION_DELTA_ROT, dim=6, encoding="rot6d"),
        adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, binary=True),
    ),
)
```

### Image

{class}`~rlmesh.adapters.Image`: a camera input. Every field:

| Field                   | Default      | What it does                                                                  | When to use                                                         |
| ----------------------- | ------------ | ----------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `role` (1st positional) | --           | match an env image                                                            | always                                                              |
| `size`                  | `None`       | sugar that sets `height` **and** `width`                                      | square targets (pass `size` _or_ `height`/`width`, not both)        |
| `height`                | `None`       | target height (keep env height if `None`)                                     | non-square target                                                   |
| `width`                 | `None`       | target width                                                                  | non-square target                                                   |
| `layout`                | `"hwc"`      | axis order the model wants                                                    | the model wants `chw`                                               |
| `channels`              | `None`       | channel count the model wants (3 RGB, 1 gray)                                 | make a channel mismatch an error instead of silent                  |
| `dtype`                 | `"uint8"`    | NumPy dtype of the result                                                     | the model wants floats                                              |
| `normalize`             | `False`      | map 8-bit pixels: `True` → `[0,1]`, or a `(low, high)` pair → that range      | scale `[0,255]`; a pair for signed inputs, e.g. `(-1.0, 1.0)`       |
| `lead_dims`             | `0`          | leading singleton axes to add                                                 | the model wants a batch/time axis                                   |
| `upside_down`           | `False`      | the model was trained on 180°-rotated frames                                  | training-time flip                                                  |
| `resample`              | `"bilinear"` | resize filter: `bilinear`, `bilinear_aa`, `bicubic_aa`, `lanczos3_aa`, `area` | match the training pipeline                                         |
| `allow_upscale`         | `False`      | permit a target larger than the env resolution                                | the model needs more pixels than the camera has                     |
| `fit`                   | `None`       | reconcile an aspect mismatch: `stretch`/`crop`/`pad` or a preference sequence | target aspect differs from the env                                  |
| `optional`              | `False`      | zero-fill a black frame when the env lacks this camera                        | the camera may be absent (needs `height`, `width`, `channels`)      |
| `fill`                  | `None`       | fill value for the blank frame (requires `optional=True`)                     | non-black fill                                                      |
| `stack`                 | `1`          | buffer N frames on a new leading axis                                         | frame history (see [Frame history](#frame-history-stack))           |
| `stride`                | `None`       | sugar: an evenly spaced window (`stack=4, stride=2` -> `(-6,-4,-2,0)`)        | every Nth frame (pass `stride` _or_ `offsets`, not both)            |
| `offsets`               | `None`       | which frames `stack` gathers, as non-positive deltas ending at `0`            | an uneven window; `None` is the contiguous one                      |
| `stack_pad`             | `"first"`    | fills the window at the start of an episode: `first` or `black`               | the model was trained with zeroed frames before step 0              |
| `crop`                  | `None`       | side fraction of the frame a center crop keeps, in `(0, 1]`                   | the training pipeline center-cropped                                |
| `crop_area`             | `None`       | the same crop as an **area** fraction (side = its square root)                | "a 90% center crop" (`crop_area=0.9` → side `0.949`)                |
| `crop_mode`             | `"zoom"`     | how the box is taken: `zoom` (resample the box) or `slice` (integer cut)      | match how the training pipeline cropped                             |
| `jpeg_quality`          | `None`       | round-trip the frame through JPEG at this quality (1-100) before the crop     | the training pipeline stored its frames as JPEG                     |
| `channel_order`         | `"rgb"`      | channel order the model wants; `bgr` swaps red and blue                       | a model trained on OpenCV-ordered frames                            |
| `render`                | `None`       | assert the camera renders at this size (square `int` or `(h, w)`)             | the model needs the env's camera dial moved (see [Render](#render)) |
| `part`                  | `None`       | the body part the wanted camera sits on (see [Parts](#parts))                 | the env has the role on several parts                               |

`size` is the idiomatic square form. `fit` accepts a preference sequence (`("crop", "pad")`); the resolver picks, per env, the first that does not need a disallowed upscale, so one spec can crop a large camera and letterbox a small one.

`resample` names are read by one rule: **un-suffixed is OpenCV/torch semantics, `_aa` is PIL's** (an antialiased filter whose support widens with the downscale factor). So `bilinear` is `cv2.INTER_LINEAR`, `bilinear_aa`/`bicubic_aa`/`lanczos3_aa` are PIL's `BILINEAR`/`BICUBIC`/`LANCZOS`, and `area` is `cv2.INTER_AREA`. Bare `bicubic` and `lanczos3` are not accepted: the two libraries' cubic kernels genuinely differ, so a spec has to say which one it trained against. Pick the one your preprocessing used — a PIL-trained policy fed OpenCV-resized frames is a real, silent accuracy loss.

#### Pixel pipeline

The image steps run in one fixed order, whatever order you write the fields in: **upright → jpeg → crop → resize → channel swap → dtype** (then layout transpose and lead dims). Concretely: the frame is rotated 180° if the env and model disagree on `upside_down`, `jpeg_quality` re-encodes it, the `crop`/`crop_area` box is taken, the result is resized to `height`/`width` under `fit`/`resample`, `channel_order="bgr"` swaps red and blue, and `normalize`/`dtype` map the 8-bit pixels into the model's range.

The two crop modes differ in where the box meets the resize. `crop_mode="zoom"` (the default) hands the _fractional_ box straight to the resampler, which samples it directly onto the target — one pass, no intermediate rounding, and the filter still reaches past the box edge into the neighbouring pixels. That is exactly Pillow's `Image.resize(size, box=...)`, which is what the conformance vectors pin it against. `crop_mode="slice"` instead cuts an _integer_ center box out first (`round(side × fraction)` pixels, the NumPy slice a training pipeline would write) and resizes that. Pick the one your preprocessing did:

```python
# "a 90% center crop, then resize to 224" -- one resample, PIL-style
adapt.Image(adapt.IMAGE_PRIMARY, size=224, crop_area=0.9, resample="lanczos3_aa")
# img[80:400, 80:400] then cv2.resize(..., (448, 448))
adapt.Image(adapt.IMAGE_PRIMARY, size=448, crop=2 / 3, crop_mode="slice", allow_upscale=True)
```

`crop` and `crop_area` are the same box said two ways, so setting both is an error rather than a silent precedence rule. Because the crop is what the resize actually reads, `allow_upscale` measures the target against the _box_, not the camera: cropping a 480×480 camera to 320×320 and asking for 448×448 is an upscale and needs the opt-in.

#### JPEG round-trip

`jpeg_quality` declares that the model was trained on frames that had been through JPEG, and asks the adapter to put the same artifacts back. Several pipelines encode at quality 95 and decode again before cropping, and a model trained that way is measurably worse on the cleaner frame a simulator hands it. Declare the quality and the adapter reproduces the codec instead of the pipeline carrying a TensorFlow or Pillow dependency to do it:

```python
adapt.Image(adapt.IMAGE_PRIMARY, size=224, jpeg_quality=95, crop_area=0.9, resample="lanczos3_aa")
```

It runs on the **upright frame, before the crop and resize** — the order the pipelines encode in, and the only order where the artifacts land at the scale the model saw them. The profile is part of the contract, because a different one is a different picture: baseline sequential, 4:2:0 chroma subsampling with box-averaged chroma, and the standard IJG quantization and Huffman tables scaled by the quality — what `tf.io.encode_jpeg` and Pillow's `save(..., "JPEG", quality=q)` write by default. It needs a 3-channel camera (JPEG subsampling is defined on YCbCr; a grayscale or RGBA frame has none) and is reported by `explain()` as `jpeg q95`.

#### Render

`render` is the one field that is **not** about the frame the model receives — it is about the frame the environment produces. It is an _assertion_: the adapter never resizes a camera to reach it. Declare the resolution your model was trained against (`render=448`, or `render=(480, 640)`), and two things follow.

Before the run, the platform reads every `render` on the spec and binds the environment's camera dial from it — the `renderParams` label an environment package publishes, which names the `make` kwargs that set camera height and width (`{height: cam_height, width: cam_width}`) plus the size the env renders at by default. An environment that publishes no `renderParams` has no dial to bind; a model that must run there drops `render` and declares `size` instead, paying for a resize.

At resolve, the adapter checks the camera it actually bound really does render at that size, and fails with `RenderMismatch` if it does not. So a dial that silently did not move is loud and pre-step, not a quiet accuracy loss. The check follows the camera, not the role: if the lone-camera fallback bound your input to an env camera under another role, the assertion is made against _that_ camera. It is skipped only when there is nothing to compare against — an observation space that does not pin the camera's resolution, or an `optional` input the env did not provide at all (a zero-filled frame is synthesized by the adapter, not rendered by the env). `explain()` prints a checked assertion as the first image step, `render 448x448`.

One camera size, one source. When a model asserts `render`, that assertion is the only thing allowed to set the camera dial for that pairing: pinning `cam_height`/`cam_width` by hand as environment params alongside it is refused rather than silently fighting the binding. When no model asserts anything, a pairing may still pin the dial as an ordinary protocol param.

Perturbations that speak in pixels (a shift in `dx`/`dy`, say) are scaled by the ratio of the bound render size to the environment's default, so a perturbation preset keeps meaning the same physical displacement when a model moves the dial.

### State

{class}`~rlmesh.adapters.State`: the single-part numeric input. Every field:

| Field                       | Default         | What it does                                                                                                                      | When to use                                             |
| --------------------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `role` (1st positional)     | --              | match an env state feature                                                                                                        | always                                                  |
| `encoding`                  | `None`          | rotation encoding: single, preference sequence, or `CustomEncoding`                                                               | the part is a rotation                                  |
| `dim`                       | `None`          | keep the leading N elements                                                                                                       | truncate the source                                     |
| `index`                     | `None`          | select one element after conversion                                                                                               | pick a single scalar                                    |
| `optional`                  | `False`         | zero-fill when the env lacks the role                                                                                             | the role may be absent                                  |
| `range`                     | `None`          | `(low, high)` the model wants; affinely maps from the env range                                                                   | model and env disagree on scale                         |
| `fill`                      | `0.0`           | value contributed when `optional` and the env lacks the role                                                                      | a non-zero stand-in (needs `optional`)                  |
| `post_rotate`               | `None`          | a fixed `Rotation` right-multiplied onto the env's rotation                                                                       | the checkpoint was trained in an offset frame           |
| `scale`                     | `None`          | multiply by this after the range map; a float, or one value per axis                                                              | the model's own units                                   |
| `offset`                    | `None`          | add this after `scale` (`value * scale + offset`); float or per-axis                                                              | a `1 - 2g` gripper (`scale=-2, offset=1`), a stand pose |
| `frame` (keyword-only)      | `None`          | the coordinate frame the checkpoint was trained to read                                                                           | the part is an absolute pose                            |
| `provenance` (keyword-only) | `None`          | where the checkpoint expects the numbers to come from; one value or the accepted set (see [Provenance](#provenance))              | a sim/real pairing, or an env publishing a role twice   |
| `part` (keyword-only)       | `None`          | the body part this part reads (see [Parts](#parts))                                                                               | the env has the role on several parts                   |
| `labels` (keyword-only)     | `None`          | the axis names this part reads, in training order (see [Labels](#labels))                                                         | a joint vector; fixes the width                         |
| `source` (keyword-only)     | `"observation"` | `"action"` reads the model's own previous action for the role instead of an env feature (see [Previous action](#previous-action)) | a policy conditioned on its last command                |
| `pad_to`                    | `None`          | zero-pad the result to this length                                                                                                | fixed-width input                                       |
| `dtype`                     | `"float32"`     | NumPy dtype of the result                                                                                                         | non-default dtype                                       |
| `reshape`                   | `None`          | target shape for the result                                                                                                       | the model wants a specific shape                        |
| `container`                 | `"array"`       | emit a NumPy array or a plain `list`                                                                                              | the model wants a list                                  |
| `clip` (keyword-only)       | `None`          | clamp the assembled vector to `(low, high)`, after every part and before `pad_to`                                                 | legged_gym's `clip_obs`                                 |

`dim` and `index` are mutually exclusive (`dim` keeps the leading N, `index` selects one), and `labels` fixes the width itself (no `dim`, no `index`). When `optional` is set the fill width must be known without an env feature, so set one of `index`, `dim`, `labels`, or `encoding`. `range` is a no-op when the env has no source range to map from; it does not clamp on its own.

The steps run in a fixed order: slice the env feature, gather by `labels`, convert the rotation (with `post_rotate` right-multiplied onto it, then the `gravity_xyz` projection when that is the target), apply `index`/`dim`, map `range`, then apply `scale`/`offset` (scalar or per-axis). The container steps come after the parts are concatenated in order: `clip` clamps the assembled vector, and only then is the result zero-padded to `pad_to`, so neither interacts with a part's own transforms and a pad slot is never clamped into a non-zero value. Setting both `range` and `scale`/`offset` on one part is legal (the affine applies to the range map's result) and raises an `info` advisory, since two rescalings on one value is usually a mistake.

`post_rotate` takes a {class}`~rlmesh.adapters.Rotation`, built from a 3x3 matrix with `Rotation.from_matrix(rows)` (stored as `rot6d`, so the round-trip is exact). It needs a rotation `encoding` and cannot combine with a `CustomEncoding`; the matrix must already be a rotation (orthonormal, `|det - 1| <= 1e-4`).

A `State` is also a valid `Concat` part: its part fields (`role`, `encoding`, `dim`, `index`, `optional`, `range`, `fill`, `post_rotate`, `scale`, `offset`, `frame`, `provenance`, `part`, `labels`, `source`) are taken, and its container fields (`pad_to`, `dtype`, `reshape`, `container`, `clip`) must stay default when used as a part.

### Concat

{class}`~rlmesh.adapters.Concat`: the **multi-part** state leaf, several roles packed into one tensor. `Concat(*parts, pad_to=None, dtype="float32", reshape=None, container="array", clip=None)` needs at least one part. A part is a bare role string (sugar for a role-only `State`), a `State` carrying part fields, or a `Constant` block:

```python
adapt.Concat(
    adapt.EEF_POS,                              # bare role: no options needed
    adapt.State(adapt.EEF_ROT, encoding="rot6d"),  # State part: needs an encoding
    adapt.Constant(dim=1),                      # a slot the env does not produce
    adapt.GRIPPER_POS,
)
```

{class}`~rlmesh.adapters.Constant`: `dim` copies of `fill` (`0.0` by default), read from nothing. Use it for a slot the checkpoint was trained to see but the env has no feature for -- a pad channel, or a proprio entry the training pipeline held fixed. Unlike an absent `optional` part it is authored data, so it never reports as a zero-filled role. At least one part must carry a role: a state of only constants reads nothing from the env and is refused.

| Field  | Default | What it does                    |
| ------ | ------- | ------------------------------- |
| `dim`  | `1`     | width of the constant block     |
| `fill` | `0.0`   | the value every element carries |

Parts are concatenated in order. The container-level fields (`pad_to`, `dtype`, `reshape`, `container`, `clip`) apply to the concatenated result and behave as in `State` -- `clip` clamps the assembled vector once, and `pad_to` is the last step. `explain()` prints a set clip as `clip[-100.0,100.0]` after the parts. A single-role state is `State` directly; `Concat` is the >1-part case (both serialize to the same wire form).

### Previous action

{func}`~rlmesh.adapters.Previous`: the model's own last command as a state part. `Previous(role, part=None, fill=0.0)` is sugar for `State(role, part=part, fill=fill, source="action")`, the declarative form of the `last_action` slot a locomotion checkpoint reads:

```python
adapt.Concat(
    adapt.State(adapt.JOINT_POS, labels=SDK, offset=tuple(-q for q in DEFAULT_POSE)),
    adapt.State(adapt.JOINT_VEL, labels=SDK, scale=0.05),
    adapt.Previous(adapt.ACTION_JOINT_POS),   # the raw output at t-1, 0 at t=0
)
```

**What it binds.** The spec's **own output actuator** with the same `role` and `part`. No env leaf is consulted, and a spec with no such actuator fails resolution (`MissingRole`, "no actuator emits"). The width is the actuator's `dim` (a declared `dim` must agree); `labels` on the part select from the actuator's labels, a subset in the model's own order, the way an observation part gathers from an env leaf, and a name the actuator lacks is a `LabelMismatch`. `explain()` prints the part as `previous action/joint_pos (fill 0.0)`, with `select[..]` when it names a subset.

**What it reads.** The **raw model output** executed at the previous step, in model order, before the actuator's `scale`/`offset`/`range`/scatter: the number the policy emitted, not the command the env received. At the first step of an episode there is no previous action, so the part reads `fill` (`0.0` by default). The window clears on `reset` with the frame windows.

**The replay tick.** Every executed action is recorded under the step it executes at, including a chunk frame the runtime replayed without predicting, so at execution_horizon `H` the re-plan sees the frame executed at the step before it (frame `H - 1` of the last chunk), never the first frame of that chunk. The served engine records each frame of a chunk under its step as it applies the chunk per lane, the Python session records the frame `transform_action` converts on every step, and `observe` counts a replayed step so the frame lands under it. An observation whose previous step recorded no action is an error naming the episode and the step (a `transform_action` or `apply_actions` was skipped), so a missed tick is loud rather than a stale command fed back silently. A model with a `Previous` part therefore holds per-episode history the way a stacked image does: `history_keys()` lists its input, the runtime delivers replayed steps to its route (see [Frame history](#frame-history-stack)), and a prefetched predict carries the rows buffered so far.

**Not an env role.** An environment never publishes `action/` roles on its observation side: a tag under `action/` is refused at `tag` and at resolve with `ActionRoleOnObservation`. An action-source part carries only `part`, `labels`, `dim` and `fill`; `encoding`, `index`, `range`, `optional`, `post_rotate`, `scale`, `offset`, `frame` and `provenance` are refused at construction.

### Text

{class}`~rlmesh.adapters.Text`: a text input.

| Field                   | Default | What it does                                                 | When to use                   |
| ----------------------- | ------- | ------------------------------------------------------------ | ----------------------------- |
| `role` (1st positional) | --      | match an env text feature                                    | always                        |
| `container`             | `"str"` | emit a plain string or a single-element list                 | the model wants a list        |
| `fill`                  | `None`  | value when the obs omits the feature; `None` omits the input | supply a fallback instruction |

Tokenization stays in the model; `Text` delivers the raw string.

### Custom

{class}`~rlmesh.adapters.Custom`: a payload slot computed by host-language code. Set **exactly one** of `transform=` (an in-process callable, local only) or `entrypoint=` (a `"module:callable"` string, imported only under `resolve(..., trust_entrypoints=True)`). The rest of the spec stays declarative. See {doc}`/user-guide/adapters/escape-hatches` for the full pattern and the trust model.

## The action side

{class}`~rlmesh.adapters.Action` is shared by env tags and model specs. `Action(*Actuator, clip=None)` takes its actuators positionally and exposes a `.dim` property (the sum of component dims). `clip` is an optional `(low, high)` applied to the final vector.

{class}`~rlmesh.adapters.Actuator`: one contiguous slice of the action vector:

| Field                   | Default       | What it does                                                                                        | When to use                                      |
| ----------------------- | ------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `role` (1st positional) | `None`        | match the actuator across sides; `None` = opaque (below)                                            | usually                                          |
| `dim`                   | -- (required) | dimensions this component occupies                                                                  | always                                           |
| `encoding`              | `None`        | rotation encoding (or a `CustomEncoding`)                                                           | the component is a rotation                      |
| `range`                 | `None`        | `(low, high)` of the component values                                                               | declare/convert the value range                  |
| `binary`                | `False`       | the component is a binary decision (snap after range map)                                           | a gripper open/close                             |
| `scale`                 | `None`        | multiply the model value; a float, or one value per axis                                            | env actuator is scaled, a per-joint action scale |
| `offset` (kw-only)      | `None`        | add after `scale`; a float, or one value per axis                                                   | a stand pose a joint target sits around          |
| `invert`                | `False`       | negate the model value (explicit `scale=-1`)                                                        | gripper sign correction                          |
| `threshold`             | `None`        | subtract to recenter the decision boundary                                                          | shift a `binary` split off zero                  |
| `clip`                  | `False`       | clamp the mapped value to `range` (requires `range`)                                                | per-dim safety on a mixed-range action           |
| `fill`                  | `0.0`         | constant per dim of an opaque or `optional` actuator; per-axis on a labeled `optional` env actuator | env-required dims no model reads; a held pose    |
| `frame` (keyword-only)  | `None`        | coordinate frame of an **absolute** pose command                                                    | `action/eef_*`                                   |
| `reference` (kw-only)   | `None`        | pose a **delta** is integrated against                                                              | `action/delta_eef_*`                             |
| `part` (kw-only)        | `None`        | the body part this actuator drives (see [Parts](#parts))                                            | one `action/joint_pos` per arm                   |
| `labels` (kw-only)      | `None`        | the axis names this actuator drives, `dim` of them (see [Labels](#labels))                          | a joint vector                                   |

`scale`, `offset`, `invert`, and `threshold` declare a side's actuator convention. They can be set on **either side** and compose as literal transforms applied **after** the declared formats (rotation, range) are bridged, **model-side first** (the model's own output convention, in the model's own axis order), then the scatter by `labels`, then **env-side** (the env's):

```{mermaid}
flowchart LR
  a["rotation / range bridged"] --> b["model: scale → offset → invert → threshold"]
  b --> s["scatter by labels (axis_fill)"]
  s --> c["env: scale → offset → invert → threshold"]
  c --> d["binary"]
  d --> e["clip"]
```

So an env declares its quirk once and every model inherits it; _and_ a model whose own output differs from a **shared** env it cannot edit declares the bridge on its own actuator, e.g. a sign-flipped gripper as `Actuator(ACTION_GRIPPER, dim=1, invert=True)`, or a sigmoid-probability gripper as `binary=True, threshold=0.5`, instead of hardcoding the env's convention in `predict()`. `binary` snaps to a definite side after range mapping: `>= 0` opens (`+1`), below closes (`-1`); a value exactly on the boundary opens rather than emitting an undefined `0`.

`clip` is the exception; it stays **env-side only**: it clamps to the env actuator's `range` (a final safety bound, not a convention), so declaring it on a model actuator is a resolve error.

A **role-less actuator** (`Actuator(dim=N, fill=...)` with no `role`) is _opaque_: it occupies `N` dims of the env action with the constant `fill`, matched by no model output (the action-side mirror of a role-less `Field`). Use it for dims the env requires but no model produces, such as a control-mode selector or base padding. A registered `role` with a fixed canonical dim (e.g. `eef_pos` is 3-D) also validates the declared `dim`; a mismatch is a resolve error.

## Conversion semantics and policy

Each conversion the resolver can perform falls into one of four policies. **Silent** is always applied when declared; **opt-in** is off until you set the flag; **advisory-warn** succeeds but logs data loss; **resolve-error** fails resolution.

| Conversion                             | Policy        | Trigger                                                                                                                                                                                          |
| -------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Image resize (target ≤ env resolution) | SILENT        | a smaller `size`/`height`/`width`                                                                                                                                                                |
| Layout transpose (`hwc` ↔ `chw`)       | SILENT        | model `layout` differs from the env's                                                                                                                                                            |
| Normalize                              | SILENT        | `normalize` set (`True` or a `(low, high)` range)                                                                                                                                                |
| dtype cast                             | SILENT        | model `dtype` differs from the env's                                                                                                                                                             |
| Rotation encoding conversion           | SILENT        | model encoding differs (both known)                                                                                                                                                              |
| Projection to `gravity_xyz`            | SILENT        | model reads a rotation as `gravity_xyz`; **an env `gravity_xyz` read as anything else → resolve error**                                                                                          |
| Container `clip`                       | SILENT        | declared on a `State`/`Concat`; clamps the assembled vector before `pad_to`                                                                                                                      |
| Previous action (`source="action"`)    | SILENT        | `Previous(role)` reads the spec's own actuator's raw output at the previous step, `fill` at the first; **no actuator with that role and part → resolve error**                                   |
| Provenance agreement                   | RESOLVE-ERROR | the env's value outside the model's set, or a value outside the vocabulary; model-only is a `caution`; an env publishing a role under two provenances against a model naming none is `Ambiguous` |
| Range map (affine)                     | SILENT        | model `range` set and env range known                                                                                                                                                            |
| Gather / scatter by `labels`           | SILENT        | both sides name their axes; a differing order is a permutation, a subset a selection                                                                                                             |
| Per-axis `scale` / `offset` / `fill`   | SILENT        | declared as a sequence; **a length other than the resolved width → resolve error**                                                                                                               |
| `binary` + `threshold` snap            | SILENT        | declared on the actuator                                                                                                                                                                         |
| `fit` (aspect-changing resize)         | OPT-IN        | aspect mismatch; **absent `fit` → resolve error**                                                                                                                                                |
| `allow_upscale`                        | OPT-IN        | target > env resolution; **absent → resolve error**                                                                                                                                              |
| `channels` declared                    | OPT-IN        | declaring it turns a channel-count mismatch into a resolve error (silent otherwise)                                                                                                              |
| `optional` / `fill`                    | OPT-IN        | env lacks the camera/role; **absent → resolve error**                                                                                                                                            |
| `crop` / `crop_area`                   | SILENT        | declared; the box is a stated part of the model's preprocessing                                                                                                                                  |
| `channel_order="bgr"`                  | SILENT        | declared; **a non-3-channel camera → resolve error**                                                                                                                                             |
| `jpeg_quality`                         | SILENT        | declared; **a non-3-channel camera → resolve error**                                                                                                                                             |
| Crop                                   | ADVISORY-WARN | `fit="crop"` chosen (pixels discarded)                                                                                                                                                           |
| Pad                                    | ADVISORY-WARN | `fit="pad"` chosen (border added)                                                                                                                                                                |
| Zero-filled camera / state             | ADVISORY-WARN | an `optional` part filled because the env lacks the role                                                                                                                                         |

### Two axes: parsing and resolve

Spec handling has two independent stages. **Parsing** is split: publishing a spec is strict and rejects unknown fields, while reading one back is tolerant and round-trips it, so a newer peer's spec does not break an older reader. **Resolve** is where a spec meets a concrete pair of spaces and the policy table above applies.

The **bare-field taint rule**: an unknown field on a known kind is a resolve error unless its name is prefixed `x-` or `ext-`. Prefixed extension fields are carried through untouched; an un-prefixed unknown field is treated as a typo and rejected.

**Join-time validation** is the final gate: when a tag and its gymnasium space disagree on class, width, encoding, or range, resolution errors rather than guessing.

## Frame history (stack)

A model that conditions on a short history sets `stack=N` on an `Image`. The adapter keeps an **episode-keyed rolling window** of processed frames and emits them on a new leading axis, padding the start of an episode and clearing on `reset`.

```python
adapt.Image(adapt.IMAGE_PRIMARY, size=256, stack=4)
```

`stack` is **image-only**: it exists on `Image` and nowhere else. A model that conditions on its own last command declares a [`Previous`](#previous-action) part, which holds a one-row action window per episode and makes the route a history route the same way; a model that conditions on past proprio has no declarative form yet -- keep that in the model.

### Strided windows

Plenty of policies do not want the last N frames; they want every Nth frame of a longer reach. `stride=` says so:

```python
adapt.Image(adapt.IMAGE_PRIMARY, size=224, stack=4, stride=2)   # offsets (-6, -4, -2, 0)
```

`stride` is construction sugar for `offsets`, the general form: non-positive deltas from the current step, oldest first, always ending at `0` (the current frame is always in the stack). Write `offsets` directly for an uneven window.

`stack` and `offsets` are both declared and neither is inferred from the other -- `len(offsets)` must equal `stack`, or resolution fails. The window's **span** (`1 - offsets[0]`) is how many consecutive frames the adapter holds; the stack is what it gathers out of them. A span above 128 is refused, and a local session refuses a projected history larger than `RLMESH_FRAME_HISTORY_LIMIT_BYTES` (2 GiB by default) before the first step rather than at the allocation that would fail.

### The `stack_pad` law

At the start of an episode the window is not full yet. `stack_pad` says what fills it:

- `"first"` (the default) replicates the first observed frame, so the stack is full from step zero.
- `"black"` pushes a raw 8-bit `0` frame through _this input's own pipeline_. It is black **pixels**, not a zeroed tensor: under `normalize=(-1.0, 1.0)` a black pad frame is `-1.0`. That is the same rule `fill` follows for an absent `optional` camera, so the name never lies about what the model receives.

Use `"black"` when the training pipeline zeroed the pre-episode frames; leave it at `"first"` otherwise.

```python
adapt.Image(adapt.IMAGE_PRIMARY, size=224, stack=6, stride=5, stack_pad="black")
```

```{caution}
Frame stacking is episode state held outside the model, in the adapter core (an episode-keyed window
per vector lane). The spec's `stack`/`offsets`/`stack_pad` round-trip through `to_json`, the window
clears on `reset`, and the env still sends one frame per step -- so no frames leak across episodes or
lanes. The window advances on **every** env step, including one whose action came from a replayed
chunk, so a stacked model sees the same frames at any `execution_horizon`. A Python session ticks the
window itself; the native runtime and a served model negotiate it at resolve, and the runtime then
carries each replayed step's observation to the model as history rows on the next predict (so above
`execution_horizon=1`, stacking does cost wire bytes: see {doc}`../performance`). A `Previous` part
puts a route in the same class: its window is one raw action row per lane, recorded on every executed
step, and the same rows number those steps.
```

## Match your shape

Find the row that matches your environment, then tag it:

| My environment looks like...              | Tag it...                                            |
| ----------------------------------------- | ---------------------------------------------------- |
| `Dict` of cameras + proprio + instruction | a `dict` of `ImageTag` / `StateTag` / `TextTag`      |
| one flat `Box` with fixed index ranges    | a bare `Split(Field(...), ...)`                      |
| a `Tuple` of sub-spaces                   | a Python `tuple` of leaves                           |
| an upside-down camera                     | `ImageTag(role, upside_down=True)`                   |
| quaternion proprioception                 | `StateTag(EEF_ROT, encoding="quat_xyzw")`            |
| two arms                                  | the role with `part=LEFT_ARM` / `RIGHT_ARM`          |
| a camera on the head                      | `ImageTag(IMAGE_PRIMARY, part=HEAD)`                 |
| a joint vector in the robot's own order   | `StateTag(JOINT_POS, labels=embodiments.GO2.joints)` |

Find the row that matches your model, then spec it:

| My model wants...                         | Spec it...                                                                  |
| ----------------------------------------- | --------------------------------------------------------------------------- |
| a resized, normalized image               | `Image(IMAGE_PRIMARY, size=256, normalize=True)`                            |
| channels-first                            | `Image(IMAGE_PRIMARY, size=256, layout="chw")`                              |
| stacked frames                            | `Image(IMAGE_PRIMARY, size=256, stack=4)`                                   |
| every second frame of the last seven      | `Image(IMAGE_PRIMARY, size=256, stack=4, stride=2)`                         |
| a 90% center crop before the resize       | `Image(IMAGE_PRIMARY, size=224, crop_area=0.9)`                             |
| a BGR-trained model                       | `Image(IMAGE_PRIMARY, size=224, channel_order="bgr")`                       |
| a model trained on stored JPEGs           | `Image(IMAGE_PRIMARY, size=224, jpeg_quality=95)`                           |
| concatenated proprio with a rotation      | `Concat(EEF_POS, State(EEF_ROT, encoding="rot6d"), GRIPPER_POS)`            |
| joints in the checkpoint's training order | `State(JOINT_POS, labels=ISAAC, offset=NEG_STAND_POSE)`                     |
| a per-joint action scale and stand pose   | `Actuator(ACTION_JOINT_POS, dim=12, labels=..., scale=(...), offset=(...))` |
| a binary gripper command                  | `Actuator(ACTION_GRIPPER, dim=1, binary=True)`                              |
| an optional second camera                 | `Image(IMAGE_WRIST, size=256, channels=3, optional=True)`                   |
| an instruction string                     | `Text(INSTRUCTION)`                                                         |

### Common pitfalls

| Symptom                           | Cause                                | Fix                                                |
| --------------------------------- | ------------------------------------ | -------------------------------------------------- |
| Wrong channel count slips through | RGB vs grayscale not declared        | set `channels` to make a mismatch an error         |
| Image axes scrambled              | HWC vs CHW mismatch                  | set `layout` to what the model wants               |
| Rotation looks rotated wrong      | `quat_xyzw` vs `quat_wxyz` confusion | match the env's exact encoding                     |
| Values out of range               | scale mismatch                       | set `range` on the model side to map it            |
| Resolve fails on a missing camera | env lacks the role                   | `optional=True` (with `height`/`width`/`channels`) |
| Resolve fails on upscale          | target larger than the camera        | `allow_upscale=True`, or lower the target          |

## Errors and `explain()`

Resolution raises {exc}`~rlmesh.adapters.AdapterResolutionError` when a spec cannot be bridged to the spaces: a required role with no `optional`/zero-fill, a declared channel mismatch, an upscale without `allow_upscale`, an aspect mismatch without `fit`, an unsupported `resample`/`dtype`, an impossible encoding conversion, a bare unknown field on a known kind, a label the other side lacks or a labeled model against an unlabeled env (`LabelMismatch`), a provenance the env contradicts or a value outside its vocabulary (`ProvenanceMismatch`), an env publishing a role under several provenances against a model that pins none (`Ambiguous`), a `Previous` part whose role no actuator of the spec emits (`MissingRole`), an env observation tagged under `action/` (`ActionRoleOnObservation`), or a join-time class/width/encoding/range disagreement between a tag and its space. The message names the offending leaf and what it expected.

Once resolution succeeds, call `adapter.explain()` to print the exact transforms the resolver chose (each resize, layout transpose, encoding conversion, range map, key remap, slice, label permutation, per-axis value, provenance, previous-action part, and clip) before you run a single step. It is the fastest way to confirm the bridge is what you intended.

```python
adapter = adapt.resolve(tags, env.observation_space, env.action_space, spec)
print(adapter.explain())
```
