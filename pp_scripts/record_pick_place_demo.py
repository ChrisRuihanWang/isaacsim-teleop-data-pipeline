"""Headless, physics-driven apple pick-and-place example with MP4 and NPZ recording.

Uses fixed starting poses and scripted TCP waypoints through the same 7D IK/gripper
interface as teleoperation. Only initialization writes object poses. Success is
checked from the settled apple's hull inside the bin, independently of the default
two-fruit task termination. This is a scripted demonstration, not a learned policy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/pick_place_demo"))
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    # Refuse an existing run so stale videos/reports cannot look like new results.
    args.output.mkdir(parents=True, exist_ok=False)
    app = AppLauncher(args).app
    env = None
    writer = None
    report = {"passed": False, "controller": "scripted_tcp_waypoints", "seed": args.seed,
              "task": "Place the apple into the open bin.", "phases": []}
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
    from pick_and_place_project.tasks.pick_place_gr1t2_pi import PickPlaceGR1T2PiEnv, PickPlaceGR1T2PiEnvCfg
    from pick_and_place_project.tasks.pick_place_cfg import FRUIT_GEOMETRY
    from pick_and_place_project.tasks.mdp.success import fruit_bounds_in_basket, fruits_in_basket
    from isaaclab.utils.math import combine_frame_transforms, compute_pose_error
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from dataset_schema import ACTION_NAMES, STATE_NAMES, validate_episode

    try:
        cfg = PickPlaceGR1T2PiEnvCfg()
        cfg.seed = args.seed
        cfg.sim.device = args.device
        cfg.scene.oblique_camera.width = 640
        cfg.scene.oblique_camera.height = 480
        cfg.scene.oblique_camera.spawn.focal_length = 18.0
        env = PickPlaceGR1T2PiEnv(cfg)
        env.reset(seed=args.seed)
        robot, apple, basket = (env.scene[name] for name in ("robot", "apple", "basket"))
        # Fixed scene fixture. Subsequent movement is entirely through physics.
        for name, xyz in (("apple", (0.50, 0.0, FRUIT_GEOMETRY["apple"]["half_extents_m"][2] + 0.005)),
                          ("banana", (0.35, -0.30, FRUIT_GEOMETRY["banana"]["half_extents_m"][2] + 0.005)),
                          ("basket", (0.70, 0.19, 0.074))):
            obj = env.scene[name]
            state = obj.data.default_root_state.clone()
            state[:, :3] = torch.tensor([xyz], device=env.device)
            state[:, 3:7] = torch.tensor([[1., 0., 0., 0.]], device=env.device)
            state[:, 7:] = 0
            obj.write_root_state_to_sim(state)
        env.scene.write_data_to_sim()
        env.sim.forward()
        camera = env.scene.sensors["oblique_camera"]
        camera.set_world_poses_from_view(
            torch.tensor([[1.15, -1.05, 0.95]], device=env.device),
            torch.tensor([[0.36, 0.03, 0.25]], device=env.device),
        )
        hand_id = robot.find_bodies("panda_hand")[0][0]
        fingers = robot.find_joints("panda_finger_joint.*")[0]
        offset = torch.tensor([cfg.actions.arm_action.body_offset.pos], device=env.device)
        target_quat = torch.tensor([[0., 1., 0., 0.]], device=env.device)
        action = torch.zeros((1, 7), device=env.device)
        action[:, -1] = -1
        obs, *_ = env.step(action)
        records = {key: [] for key in ("state", "action", "rgb_raw", "wrist_rgb_raw", "oblique_rgb_raw")}
        phases, trace = [], []
        frame_index = 0
        dt = float(env.step_dt)
        writer = imageio.get_writer(str(args.output / "demo.mp4"), fps=15, codec="libx264", quality=8)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except OSError:
            font = ImageFont.load_default()

        def tcp():
            return combine_frame_transforms(robot.data.body_pos_w[:, hand_id],
                                            robot.data.body_quat_w[:, hand_id], offset)

        def step(target, grip, phase):
            nonlocal obs, frame_index
            pos, quat = tcp()
            dp, dr = compute_pose_error(pos, quat, target, target_quat, rot_error_type="axis_angle")
            dp *= (0.008 / torch.linalg.vector_norm(dp, dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.)
            dr *= (0.06 / torch.linalg.vector_norm(dr, dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.)
            action[:, :3], action[:, 3:6], action[:, 6] = dp, dr, grip
            records["state"].append(obs["policy"][0].cpu().numpy().copy())
            records["action"].append(action[0].cpu().numpy().copy())
            for source, destination in (("rgb", "rgb_raw"), ("wrist_rgb", "wrist_rgb_raw"),
                                        ("oblique_rgb", "oblique_rgb_raw")):
                records[destination].append(obs["images"][source][0].cpu().numpy().copy())
            phases.append(phase)
            # Encode actual sensor pixels, with phase/time labels outside the image.
            if frame_index % 2 == 0:
                canvas = Image.new("RGB", (640, 544), (17, 24, 39))
                canvas.paste(Image.fromarray(records["oblique_rgb_raw"][-1]), (0, 32))
                draw = ImageDraw.Draw(canvas)
                draw.text((14, 5), "FRANKA  /  APPLE PICK & PLACE", font=font, fill="white")
                draw.text((14, 518), f"{phase.upper()}  |  sim {frame_index * dt:04.1f}s  |  scripted IK", font=font, fill="white")
                writer.append_data(np.asarray(canvas))
            obs, _, done, timeout, _ = env.step(action)
            if done.any() or timeout.any():
                raise RuntimeError("Unexpected default task reset during single-object demo")
            trace.append({"phase": phase, "apple": apple.data.root_pos_w[0].cpu().tolist(),
                          "tcp": tcp()[0][0].cpu().tolist(),
                          "gripper_width": float(robot.data.joint_pos[:, fingers].sum())})
            frame_index += 1

        def move(target, grip, phase, steps):
            start = tcp()[0].clone()
            for i in range(steps):
                alpha = min(1.0, (i + 1) / max(1, steps - 60))
                blend = alpha * alpha * (3 - 2 * alpha)
                step(start + (target - start) * blend, grip, phase)
            # Allow the PD/IK cascade to settle before moving to the next phase.
            for _ in range(180):
                if float(torch.linalg.vector_norm(tcp()[0] - target)) < 0.003:
                    break
                step(target, grip, phase)
            error = float(torch.linalg.vector_norm(tcp()[0] - target))
            report["phases"].append({"phase": phase, "end_time": frame_index * dt, "tcp_error_m": error})
            print(f"{phase}: time={frame_index * dt:.1f}s, TCP error={error:.4f}m, target={target.tolist()}, actual={tcp()[0].tolist()}", flush=True)
            if error > 0.015:
                raise AssertionError(f"{phase}: TCP missed target by {error:.4f}m")

        move(tcp()[0].clone(), -1., "settle", 45)
        initial_height = float(apple.data.root_pos_w[0, 2])
        target = apple.data.root_pos_w.clone()
        target[:, 2] = 0.25
        move(target, -1., "approach", 300)
        target[:, 2] = apple.data.root_pos_w[:, 2]
        move(target, -1., "descend", 180)
        for i in range(60):
            step(target, -1 + 2 * (i + 1) / 60, "close gripper")
        move(target, 1., "grasp", 30)
        target[:, 2] = 0.28
        move(target, 1., "lift", 100)
        report["lift_m"] = float(apple.data.root_pos_w[0, 2]) - initial_height
        if report["lift_m"] < 0.12:
            raise AssertionError("Apple was not lifted by physical finger contact")
        held_offset = apple.data.root_pos_w - tcp()[0]
        target[:, :2] = basket.data.root_pos_w[:, :2] - held_offset[:, :2]
        move(target, 1., "transfer to bin", 120)
        target[:, 2] = 0.115 - held_offset[:, 2]
        move(target, 1., "lower into bin", 80)
        for i in range(60):
            step(target, 1 - 2 * (i + 1) / 60, "release")
        target[:, 2] = 0.30
        move(target, -1., "retract", 90)
        stable_steps = 0
        for _ in range(90):
            step(target, -1., "verify placement")
            inside = bool(fruits_in_basket(env)[0, cfg.fruit_names.index("apple")])
            still = float(torch.linalg.vector_norm(apple.data.root_lin_vel_w)) < 0.03
            still &= float(torch.linalg.vector_norm(apple.data.root_ang_vel_w)) < 0.25
            released = float(robot.data.joint_pos[:, fingers].sum()) > 0.06
            stable_steps = stable_steps + 1 if inside and still and released else 0
        report["apple_bounds_in_bin"] = [value[0].cpu().tolist() for value in fruit_bounds_in_basket(env, "apple")]
        report["stable_placement_steps"] = stable_steps
        report["passed"] = stable_steps >= 15
        if not report["passed"]:
            raise AssertionError("Apple did not settle fully inside bin with gripper open")
        report["frames"] = frame_index
        report["simulation_seconds"] = frame_index * dt
        report["control_fps"] = 1 / dt
        report["trace"] = trace
        arrays = {key: np.stack(value) for key, value in records.items()}
        arrays["timestamp"] = np.arange(frame_index, dtype=np.float64) * dt
        arrays["phase"] = np.asarray(phases)
        arrays["metadata_json"] = np.array(json.dumps({
            "schema_version": 1, "T": frame_index, "dt": dt, "episode_id": 0,
            "saved_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
            "task": report["task"], "controller": report["controller"],
            "success": True, "success_scope": "single_apple_example", "end_reason": "script_complete",
            "state_names": STATE_NAMES, "action_names": ACTION_NAMES,
            "action_frame": "robot_base", "alignment": "observation_before_action",
            "timestamp_clock": "simulation_seconds_from_recording_start",
        }))
        validate_episode(arrays)
        np.savez_compressed(args.output / "ep_00000.npz", **arrays)
        Image.fromarray(obs["images"]["oblique_rgb"][0].cpu().numpy()).save(args.output / "final.png")
        print(f"PASS: {args.output} ({frame_index} frames)", flush=True)
    except Exception as exc:
        report["passed"] = False
        report["error"] = repr(exc)
        raise
    finally:
        if writer is not None:
            writer.close()
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
