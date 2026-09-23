import os
import sys
import traceback

import gymnasium as gym
import numpy as np
import rlmesh
import rlmesh.adapters as adapt
from gymnasium.vector.utils import batch_space

FRANKA_USD = "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda"
HOME = np.array([0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741, 0.04, 0.04], np.float32)
ARM = list(range(7))
MAX_DELTA = 0.05  # metres of hand motion per action
SUBSTEPS = 4
SPACING = 2.0  # metres between lanes along y
CAMERA_OFFSET = np.array([1.6, 0.0, 0.45])


class FrankaReachEnv(gym.vector.VectorEnv):
    """`lanes` Frankas in one stage, each reaching a red sphere, stepped in lockstep.

    Lanes that finish reset on their next step (gymnasium NEXT_STEP autoreset),
    and the env truncates at `max_steps` itself.
    """

    metadata = {"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP}

    def __init__(self, app, lanes, image_size, max_steps):
        self.app, self.num_envs, self.max_steps = app, lanes, max_steps
        self.single_observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, (image_size, image_size, 3), np.uint8),
                "joint_pos": gym.spaces.Box(-np.inf, np.inf, (HOME.size,), np.float32),
                "eef_pos": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
                "target_pos": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
            }
        )
        self.single_action_space = gym.spaces.Box(
            -MAX_DELTA, MAX_DELTA, (3,), np.float32
        )
        self.observation_space = batch_space(self.single_observation_space, lanes)
        self.action_space = batch_space(self.single_action_space, lanes)
        self.origins = np.array([[0.0, SPACING * i, 0.0] for i in range(lanes)])
        self.targets = np.zeros((lanes, 3))
        self.steps = np.zeros(lanes, int)
        self.ended = np.zeros(lanes, bool)
        self._build_stage(image_size)

    def _build_stage(self, image_size):
        import isaacsim.core.experimental.utils.stage as stage_utils
        import omni.timeline
        from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
        from isaacsim.core.experimental.objects import Sphere
        from isaacsim.core.experimental.prims import Articulation, RigidPrim
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
        from isaacsim.storage.native import get_assets_root_path

        stage_utils.create_new_stage(template="sunlight")
        lanes = [f"/World/env_{i}" for i in range(self.num_envs)]
        for lane in lanes:
            stage_utils.add_reference_to_stage(
                get_assets_root_path() + FRANKA_USD, path=f"{lane}/robot"
            )
        self.robot = Articulation(
            [f"{lane}/robot" for lane in lanes], positions=self.origins
        )
        self.robot.set_default_state(
            positions=self.origins, dof_positions=np.tile(HOME, (self.num_envs, 1))
        )
        self.hand = RigidPrim([f"{lane}/robot/panda_hand" for lane in lanes])
        self.hand_row = self.robot.get_link_indices("panda_hand").list()[0] - 1
        red = PreviewSurfaceMaterial("/Looks/red")
        red.set_input_values("diffuseColor", [1.0, 0.0, 0.0])
        self.target = Sphere(
            [f"{lane}/target" for lane in lanes],
            radii=[0.03] * self.num_envs,
            reset_xform_op_properties=True,
        )
        self.target.apply_visual_materials(red)
        # Level with each workspace, looking down -X at its robot.
        self.cameras = [
            CameraSensor(
                RtxCamera(
                    f"{lane}/camera",
                    translations=origin + CAMERA_OFFSET,
                    orientations=[0.5, 0.5, 0.5, 0.5],
                ),
                resolution=(image_size, image_size),
                annotators=["rgb"],
            )
            for lane, origin in zip(lanes, self.origins, strict=True)
        ]
        omni.timeline.get_timeline_interface().play()
        self.app.update()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps[:], self.ended[:] = 0, False
        self._reset_lanes(np.arange(self.num_envs))
        return self._observe(), {}

    def step(self, actions):
        resetting = np.flatnonzero(self.ended)
        self._reset_lanes(resetting)
        delta = np.clip(actions, -MAX_DELTA, MAX_DELTA) / SUBSTEPS
        delta[resetting] = 0.0  # NEXT_STEP discards a resetting lane's action
        for _ in range(SUBSTEPS):
            jacobian = self.robot.get_jacobian_matrices().numpy()[
                :, self.hand_row, :3, :7
            ]
            joints = self.robot.get_dof_positions().numpy()[:, ARM]
            # Damped least-squares IK for the hand delta, batched over lanes.
            dq = np.linalg.solve(
                jacobian @ jacobian.transpose(0, 2, 1) + 0.05**2 * np.eye(3),
                delta[..., None],
            )
            self.robot.set_dof_position_targets(
                joints + (jacobian.transpose(0, 2, 1) @ dq)[..., 0], dof_indices=ARM
            )
            self.app.update()
        obs = self._observe()
        distance = np.linalg.norm(obs["eef_pos"] - obs["target_pos"], axis=1)
        self.steps += 1
        self.steps[resetting] = 0
        terminated = distance < 0.03
        truncated = ~terminated & (self.steps >= self.max_steps)
        # A lane reset this step reports a fresh start, not a transition.
        reward = -distance
        reward[resetting] = 0.0
        terminated[resetting] = truncated[resetting] = False
        self.ended = terminated | truncated
        return obs, reward, terminated, truncated, {"is_success": terminated}

    def _reset_lanes(self, lanes):
        if not len(lanes):
            return
        self.targets[lanes] = [0.5, 0.0, 0.4] + self.np_random.uniform(
            -1, 1, (len(lanes), 3)
        ) * [0.15, 0.2, 0.15]
        self.target.set_world_poses(
            positions=self.origins[lanes] + self.targets[lanes], indices=lanes
        )
        self.robot.set_dof_positions(HOME, indices=lanes)
        self.robot.set_dof_velocities(0.0, indices=lanes)
        self.robot.set_dof_position_targets(HOME, indices=lanes)
        for _ in range(SUBSTEPS):
            self.app.update()

    def _observe(self):
        return {
            "image": np.stack([self._frame(camera) for camera in self.cameras]),
            "joint_pos": self.robot.get_dof_positions().numpy().astype(np.float32),
            "eef_pos": (self.hand.get_world_poses()[0].numpy() - self.origins).astype(
                np.float32
            ),
            "target_pos": self.targets.astype(np.float32),
        }

    def _frame(self, camera):
        for _ in range(10):  # the RGB annotator lags a few frames behind a reset
            rgb, _ = camera.get_data("rgb")
            if rgb is not None:
                return rgb.numpy()[..., :3]
            self.app.update()
        return np.zeros(self.single_observation_space["image"].shape, np.uint8)


class FrankaReach(rlmesh.EnvFactory):
    tags = adapt.EnvTags(
        observation={
            "image": adapt.ImageTag(adapt.IMAGE_PRIMARY),
            "joint_pos": adapt.StateTag(adapt.JOINT_POS),
            "eef_pos": adapt.StateTag(adapt.EEF_POS),
        },
        action=adapt.Action(
            adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3, range=(-MAX_DELTA, MAX_DELTA))
        ),
    )

    def prepare(self):
        # Kit starts once per process, on the main thread that serve() keeps
        # the env on. It rejects the `-m rlmesh.serve` argv, so hide it.
        from isaacsim import SimulationApp

        argv, sys.argv = sys.argv, sys.argv[:1]
        try:
            self.app = SimulationApp({"headless": True})
        finally:
            sys.argv = argv
        # Kit's shutdown rewrites a crashing exit code to 0; exit before it can.
        sys.excepthook = lambda *exc: (traceback.print_exception(*exc), os._exit(1))

    def make(self, *, lanes=1, image_size=224, max_steps=100):
        return FrankaReachEnv(self.app, lanes, image_size, max_steps)

    def close(self):
        self.app.close()
