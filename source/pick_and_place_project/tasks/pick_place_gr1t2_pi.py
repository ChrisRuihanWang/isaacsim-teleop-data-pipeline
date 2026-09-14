from __future__ import annotations

import math
import torch

from isaaclab.utils.configclass import configclass
from isaaclab.envs import ManagerBasedRLEnv
from pick_and_place_project.tasks.pick_place_cfg import PickPlaceEnvCfg_PLAY, FRUIT_GEOMETRY


@configclass
class PickPlaceGR1T2PiEnvCfg(PickPlaceEnvCfg_PLAY):
    # Coordinates are relative to each environment origin; angles are degrees.
    fruit_names = ("banana", "apple")
    fruit_xy_range = ((0.40, 0.56), (-0.24, 0.24))
    fruit_yaw_deg: float = 60.0
    basket_xy_range = ((0.72, 0.78), (-0.08, 0.08))
    basket_yaw_deg: float = 20.0
    placement_clearance: float = 0.015
    object_sample_max_tries: int = 300
    # Conservative cavity bounds derived from the KLT's wall/floor colliders.
    basket_inner_half_xy = (0.0875, 0.126)
    basket_floor_z: float = -0.0708
    basket_rim_z: float = 0.071

    robot_joint_noise_std: float = 0.03
    success_hold_steps: int = 15

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 120.0
        self.num_rerenders_on_reset = 2
        for camera in (self.scene.camera, self.scene.wrist_camera, self.scene.oblique_camera):
            camera.height = camera.width = 256


class PickPlaceGR1T2PiEnv(ManagerBasedRLEnv):
    """Franka placing one banana and one apple inside a fixed open KLT bin."""

    def _rand_yaw_quat(self, n, device, yaw_deg):
        yaw = (torch.rand(n, device=device) * 2 - 1) * math.radians(yaw_deg)
        quat = torch.zeros((n, 4), device=device)
        # Isaac Lab uses (w, x, y, z).
        quat[:, 0] = torch.cos(yaw / 2)
        quat[:, 3] = torch.sin(yaw / 2)
        return quat

    def _sample_xy(self, ranges, n):
        bounds = torch.tensor(ranges, device=self.device)
        if torch.any(bounds[:, 1] < bounds[:, 0]):
            raise ValueError("Invalid XY sampling range")
        return bounds[:, 0] + torch.rand((n, 2), device=self.device) * (bounds[:, 1] - bounds[:, 0])

    def _randomize_objects(self, env_ids):
        n = len(env_ids)
        origins = self.scene.env_origins[env_ids]
        basket = self.scene["basket"]
        basket_state = basket.data.default_root_state[env_ids].clone()
        basket_xy = self._sample_xy(self.cfg.basket_xy_range, n)
        basket_state[:, :2] = basket_xy
        basket_state[:, 3:7] = self._rand_yaw_quat(n, self.device, self.cfg.basket_yaw_deg)
        states = [(basket, basket_state)]
        # Circumscribed disks guarantee separation regardless of sampled yaw.
        occupied = [(basket_xy, math.hypot(0.099, 0.149))]
        for name in self.cfg.fruit_names:
            obj = self.scene[name]
            state = obj.data.default_root_state[env_ids].clone()
            radius = math.hypot(*FRUIT_GEOMETRY[name]["half_extents_m"][:2])
            xy = self._sample_xy(self.cfg.fruit_xy_range, n)
            for _ in range(self.cfg.object_sample_max_tries):
                invalid = torch.zeros(n, dtype=torch.bool, device=self.device)
                for other_xy, other_radius in occupied:
                    invalid |= torch.linalg.vector_norm(xy - other_xy, dim=-1) < (
                        radius + other_radius + self.cfg.placement_clearance
                    )
                if not invalid.any():
                    break
                xy[invalid] = self._sample_xy(self.cfg.fruit_xy_range, int(invalid.sum()))
            else:
                raise ValueError("Cannot sample separated fruits and bin; adjust sampling ranges")
            state[:, :2] = xy
            state[:, 3:7] = self._rand_yaw_quat(n, self.device, self.cfg.fruit_yaw_deg)
            states.append((obj, state))
            occupied.append((xy, radius))
        for obj, state in states:
            state[:, :3] += origins
            state[:, 7:] = 0
            obj.write_root_state_to_sim(state, env_ids=env_ids)

    def _randomize_robot(self, env_ids):
        robot = self.scene["robot"]
        q = robot.data.default_joint_pos[env_ids].clone()
        arm_ids, _ = robot.find_joints("panda_joint[1-7]")
        q[:, arm_ids] += torch.randn_like(q[:, arm_ids]) * self.cfg.robot_joint_noise_std
        limits = robot.data.soft_joint_pos_limits[env_ids]
        q = torch.clamp(q, limits[..., 0], limits[..., 1])
        robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        robot.set_joint_position_target(q, env_ids=env_ids)
        robot.set_joint_velocity_target(torch.zeros_like(q), env_ids=env_ids)

    def _reset_idx(self, env_ids):
        # Both explicit reset and automatic episode resets pass through this hook,
        # before rendering and observation computation.
        super()._reset_idx(env_ids)
        self._randomize_objects(env_ids)
        self._randomize_robot(env_ids)
        if not hasattr(self, "success_steps"):
            self.success_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_steps[env_ids] = 0
        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # Termination manager retains pre-reset results for this step.
        info["success"] = self.termination_manager.get_term("success").clone()
        return obs, reward, terminated, truncated, info


def make_env() -> ManagerBasedRLEnv:
    return PickPlaceGR1T2PiEnv(PickPlaceGR1T2PiEnvCfg())
