"""Run with the Isaac Sim Python environment; no keyboard/display required."""
from __future__ import annotations
import argparse
import json
import time
import math
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--resets", type=int, default=8)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, default=Path("data/local_validation"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.resets < 2 or args.steps < 1:
    parser.error("--resets must be >= 2 and --steps >= 1")
args.enable_cameras = True
launcher = AppLauncher(args)
app = launcher.app
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

import torch
from PIL import Image
from pick_and_place_project.tasks.pick_place_gr1t2_pi import PickPlaceGR1T2PiEnv, PickPlaceGR1T2PiEnvCfg
from pick_and_place_project.tasks.pick_place_cfg import FRUIT_GEOMETRY
from pick_and_place_project.tasks.mdp.success import fruits_in_basket, fruit_bounds_in_basket
from isaaclab.utils.math import quat_apply, quat_mul


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"seed": args.seed, "resets": args.resets, "steps_per_reset": args.steps, "checks": [], "poses": []}
    started = time.monotonic()
    cfg = PickPlaceGR1T2PiEnvCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    # Low-cost local profile keeps all three cameras, at 256 square resolution.
    for camera in (cfg.scene.camera, cfg.scene.wrist_camera, cfg.scene.oblique_camera):
        camera.height = camera.width = 256
    env = PickPlaceGR1T2PiEnv(cfg)
    try:
        def poses():
            return torch.cat([env.scene[k].data.root_state_w[:, :7] for k in ("banana", "apple", "basket")], dim=-1).clone()

        def check_obs(obs):
            assert obs["policy"].shape == (1, 23)
            assert torch.isfinite(obs["policy"]).all()
            robot = env.scene["robot"]
            torch.testing.assert_close(obs["policy"][:, :9], robot.data.joint_pos - robot.data.default_joint_pos)
            for name, image in obs["images"].items():
                assert image.shape == (1, 256, 256, 3), (name, image.shape)
                assert image.dtype == torch.uint8, (name, image.dtype)
                assert image.max() > image.min(), f"Blank camera: {name}"

        obs, _ = env.reset(seed=args.seed)
        first = poses()
        joints = env.scene["robot"].data.joint_pos.clone()
        obs, _ = env.reset(seed=args.seed)
        torch.testing.assert_close(first, poses())
        torch.testing.assert_close(joints, env.scene["robot"].data.joint_pos)
        check_obs(obs)
        report["checks"].append("seed_reproducibility_and_fresh_reset_observation")
        action = torch.zeros((1, 7), device=env.device)
        action[:, -1] = -1
        for episode in range(args.resets):
            obs, _ = env.reset()
            check_obs(obs)
            pose = poses()
            report["poses"].append(pose[0].cpu().tolist())
            banana, apple, target = pose[:, :7], pose[:, 7:14], pose[:, 14:]
            radii = [math.hypot(*FRUIT_GEOMETRY[name]["half_extents_m"][:2]) for name in cfg.fruit_names]
            states = [banana, apple, target]
            radii.append(math.hypot(0.099, 0.149))
            for i in range(3):
                for j in range(i):
                    assert torch.linalg.vector_norm(states[i][:, :2] - states[j][:, :2], dim=-1).min() >= radii[i] + radii[j] + cfg.placement_clearance
            for state, ranges, yaw_limit in ((banana, cfg.fruit_xy_range, cfg.fruit_yaw_deg), (apple, cfg.fruit_xy_range, cfg.fruit_yaw_deg), (target, cfg.basket_xy_range, cfg.basket_yaw_deg)):
                local = state[:, :2] - env.scene.env_origins[:, :2]
                bounds = torch.tensor(ranges, device=env.device)
                assert ((local >= bounds[:, 0]) & (local <= bounds[:, 1])).all()
                torch.testing.assert_close(torch.linalg.vector_norm(state[:, 3:7], dim=-1), torch.ones(1, device=env.device))
                assert state[:, 4:6].abs().max() < 1e-6
                assert (2 * torch.atan2(state[:, 6], state[:, 3])).abs().max() <= yaw_limit * torch.pi / 180 + 1e-6
            for _ in range(args.steps):
                obs, _, done, timeout, info = env.step(action)
                assert not done.any() and not timeout.any()
                assert not info["success"].any()
                assert torch.isfinite(obs["policy"]).all()
            for name in cfg.fruit_names:
                state = env.scene[name].data.root_state_w
                assert torch.isfinite(state).all()
                assert state[:, 2].min() > 0.005, f"{name} fell through ground"
                assert torch.linalg.vector_norm(state[:, 7:10], dim=-1).max() < 0.1, f"{name} did not settle"
            torch.testing.assert_close(env.scene["basket"].data.root_state_w[:, :7], target)
            if episode == 0:
                for name, value in obs["images"].items():
                    Image.fromarray(value[0].cpu().numpy()).save(args.output / f"{name}.png")
        samples = torch.tensor(report["poses"])
        assert (samples[:, [0, 1, 6, 7, 8, 13, 14, 15, 20]].std(dim=0) > 1e-5).all()
        report["checks"].append("randomized_xy_yaw_valid_quaternions_separation_static_target_and_rgb")
        # Exercise automatic timeout reset, not only public reset().
        before = poses()
        env.episode_length_buf[:] = env.max_episode_length - 1
        obs, _, done, timeout, info = env.step(action)
        assert timeout.all() and not done.any() and not info["success"].any()
        assert not torch.allclose(before, poses())
        check_obs(obs)
        report["checks"].append("automatic_timeout_randomization")
        # Arrange known placements to verify containment and the two-object condition.
        # These are physics tests, not demonstrations or learned-policy rollouts.
        def put_in_bin(name):
            fruit, basket = env.scene[name], env.scene["basket"]
            state = fruit.data.root_state_w.clone()
            x, y, yaw = (-0.037, 0.0, math.pi / 2) if name == "banana" else (0.037, 0.06, 0.0)
            height = cfg.basket_floor_z + FRUIT_GEOMETRY[name]["half_extents_m"][2] + 0.01
            local = torch.tensor([[x, y, height]], device=env.device)
            state[:, :3] = basket.data.root_pos_w + quat_apply(basket.data.root_quat_w, local)
            q = torch.tensor([[math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]], device=env.device)
            state[:, 3:7] = quat_mul(basket.data.root_quat_w, q)
            state[:, 7:] = 0
            fruit.write_root_state_to_sim(state)

        put_in_bin("banana")
        for _ in range(90):
            obs, _, done, timeout, info = env.step(action)
            assert not done.any() and not info["success"].any()
        report["checks"].append("one_fruit_does_not_trigger_success")
        put_in_bin("apple")
        success = False
        for _ in range(180):
            obs, _, done, timeout, info = env.step(action)
            if done.any():
                assert info["success"].all() and not timeout.any()
                success = True
                break
            if fruits_in_basket(env).all():
                for name, value in obs["images"].items():
                    Image.fromarray(value[0].cpu().numpy()).save(args.output / f"placed_{name}.png")
        if not success:
            report["placement_debug"] = {name: {
                "state": env.scene[name].data.root_state_w.cpu().tolist(),
                "bounds": [x.cpu().tolist() for x in fruit_bounds_in_basket(env, name)]
            } for name in cfg.fruit_names}
            for name, value in obs["images"].items():
                Image.fromarray(value[0].cpu().numpy()).save(args.output / f"failure_{name}.png")
        assert success, f"Both fruits did not settle inside bin: {fruits_in_basket(env)}"
        report["checks"].append("two_fruit_placement_success_and_automatic_reset")
        report["elapsed_seconds"] = time.monotonic() - started
        report["passed"] = True
    except Exception as exc:
        report["passed"] = False
        report["error"] = repr(exc)
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        env.close()
    print(json.dumps(report, indent=2), flush=True)


try:
    main()
finally:
    app.close()
