"""Measure materials and physically approach, close, lift, hold, and release each fruit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('data/grasp_validation'))
parser.add_argument('--fruit', choices=['banana', 'apple', 'both'], default='both')
parser.add_argument('--grasp-z-offset', type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))

import torch
from PIL import Image
from pxr import Usd, UsdPhysics, UsdShade
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_apply
from pick_and_place_project.tasks.pick_place_gr1t2_pi import PickPlaceGR1T2PiEnv, PickPlaceGR1T2PiEnvCfg
from pick_and_place_project.tasks.pick_place_cfg import FRUIT_GEOMETRY


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'trials': []}
    cfg = PickPlaceGR1T2PiEnvCfg()
    cfg.sim.device = args.device
    cfg.seed = 42
    env = PickPlaceGR1T2PiEnv(cfg)
    try:
        env.reset(seed=42)
        robot = env.scene['robot']
        hand_id = robot.find_bodies('panda_hand')[0][0]
        fingers = robot.find_joints('panda_finger_joint.*')[0]
        report['physics_dt'] = env.physics_dt
        report['control_dt'] = env.step_dt
        report['joint_effort_limits'] = robot.data.joint_effort_limits[:, fingers].cpu().tolist()
        report['joint_stiffness'] = robot.data.joint_stiffness[:, fingers].cpu().tolist()
        report['joint_damping'] = robot.data.joint_damping[:, fingers].cpu().tolist()
        report['materials'] = {}
        for name in cfg.fruit_names:
            obj = env.scene[name]
            report['materials'][name] = obj.root_physx_view.get_material_properties().tolist()
            report.setdefault('mass', {})[name] = obj.root_physx_view.get_masses().tolist()
        report['materials']['robot_shapes'] = robot.root_physx_view.get_material_properties().tolist()
        stage = env.sim.stage
        bindings = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0'), Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI) and any(t in str(prim.GetPath()) for t in ('finger', 'Banana', 'Apple')):
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
                bindings.append({'collider': str(prim.GetPath()), 'material': str(material.GetPath()) if material else None})
        report['bindings'] = bindings
        print('AUDIT', json.dumps(report), flush=True)
        offset = torch.tensor([cfg.actions.arm_action.body_offset.pos], device=env.device)
        target_q = torch.tensor([[0., 1., 0., 0.]], device=env.device)
        action = torch.zeros((1, 7), device=env.device)
        trace = []
        active = None

        def tcp():
            return combine_frame_transforms(robot.data.body_pos_w[:, hand_id], robot.data.body_quat_w[:, hand_id], offset)

        def step(target, grip, phase):
            pos, quat = tcp()
            dp, dr = compute_pose_error(pos, quat, target, target_q, rot_error_type='axis_angle')
            # Bound IK increments; do not teleport the arm or attach the fruit.
            dp *= torch.clamp(0.01 / torch.linalg.vector_norm(dp, dim=-1, keepdim=True).clamp_min(1e-8), max=1.)
            dr *= torch.clamp(0.08 / torch.linalg.vector_norm(dr, dim=-1, keepdim=True).clamp_min(1e-8), max=1.)
            action[:, :3], action[:, 3:6], action[:, 6] = dp, dr, grip
            obs, _, done, timeout, _ = env.step(action)
            assert not done.any() and not timeout.any()
            trace.append({'phase': phase, 'tcp': tcp()[0][0].cpu().tolist(),
                          'fruit': env.scene[active].data.root_pos_w[0].cpu().tolist(),
                          'width': float(robot.data.joint_pos[:, fingers].sum()),
                          'finger_effort': robot.data.applied_torque[:, fingers][0].cpu().tolist()})
            return obs

        def move(target, grip, phase, steps):
            obs = None
            for _ in range(steps):
                obs = step(target, grip, phase)
            error = torch.linalg.vector_norm(tcp()[0] - target).item()
            if phase in ('approach', 'descend') and error > 0.012:
                raise AssertionError(f'{active} {phase}: TCP position error {error:.4f} m')
            return obs

        names = cfg.fruit_names if args.fruit == 'both' else [args.fruit]
        for name in names:
            active = name
            trace = []
            env.reset(seed=42)
            obj = env.scene[name]
            state = obj.data.default_root_state.clone()
            state[:, :3] = torch.tensor([[0.50, 0., FRUIT_GEOMETRY[name]['half_extents_m'][2] + 0.005]], device=env.device)
            state[:, 3:7] = torch.tensor([[1., 0., 0., 0.]], device=env.device)
            state[:, 7:] = 0
            obj.write_root_state_to_sim(state)
            other = env.scene['apple' if name == 'banana' else 'banana']
            other_state = other.data.default_root_state.clone()
            other_state[:, 0:2] = torch.tensor([[0.3, -0.4]], device=env.device)
            other_state[:, 7:] = 0
            other.write_root_state_to_sim(other_state)
            hold = tcp()[0].clone()
            move(hold, -1., 'settle', 90)
            start = obj.data.root_pos_w.clone()
            anchor = torch.tensor([FRUIT_GEOMETRY[name].get('grasp_anchor_local_m', [0., 0., 0.])], device=env.device)
            grasp_center = start + quat_apply(obj.data.root_quat_w, anchor)
            target = grasp_center.clone()
            target[:, 2] = 0.25
            move(target, -1., 'approach', 300)
            target[:, 2] = grasp_center[:, 2] + args.grasp_z_offset
            obs = move(target, -1., 'descend', 300)
            print('GRASP_POSE', name, 'tcp', tcp()[0].cpu().tolist(), 'fruit', obj.data.root_pos_w.cpu().tolist(),
                  'fingers', [(b, robot.data.body_pos_w[:, robot.find_bodies(b)[0][0]].cpu().tolist(),
                               robot.data.body_quat_w[:, robot.find_bodies(b)[0][0]].cpu().tolist())
                              for b in ('panda_leftfinger', 'panda_rightfinger')], flush=True)
            for camera, value in obs['images'].items():
                Image.fromarray(value[0].cpu().numpy()).save(args.output / f'{name}_before_{camera}.png')
            for i in range(60):
                step(target, -1. + 2. * (i + 1) / 60, 'close')
            move(target, 1., 'squeeze', 30)
            target[:, 2] += 0.15
            move(target, 1., 'lift', 150)
            obs = move(target, 1., 'hold', 90)
            height = float(obj.data.root_pos_w[0, 2] - start[0, 2])
            held_z = float(obj.data.root_pos_w[0, 2])
            for camera, value in obs['images'].items():
                Image.fromarray(value[0].cpu().numpy()).save(args.output / f'{name}_held_{camera}.png')
            move(target, -1., 'release', 90)
            drop = held_z - float(obj.data.root_pos_w[0, 2])
            trial = {'fruit': name, 'lift_m': height, 'release_drop_m': drop,
                     'passed': height > 0.10 and drop > 0.08, 'trace': trace}
            report['trials'].append(trial)
            print('TRIAL', {k: v for k, v in trial.items() if k != 'trace'}, flush=True)
        report['passed'] = all(t['passed'] for t in report['trials'])
    except Exception as exc:
        report['error'] = repr(exc)
        report['passed'] = False
        raise
    finally:
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        env.close()

try:
    main()
finally:
    app.close()
