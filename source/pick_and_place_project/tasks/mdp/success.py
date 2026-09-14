"""Both fruits must be contained in the KLT cavity, still, and released."""
import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse


def fruit_bounds_in_basket(env, name):
    # The fruit USD roots are centered on their unrotated visual bounding boxes.
    from pick_and_place_project.tasks.pick_place_cfg import FRUIT_GEOMETRY
    fruit, basket = env.scene[name], env.scene["basket"]
    if not hasattr(env, "_fruit_hulls"):
        env._fruit_hulls = {}
    if name not in env._fruit_hulls:
        env._fruit_hulls[name] = torch.tensor(
            FRUIT_GEOMETRY[name]["hull_vertices_m"], device=env.device, dtype=torch.float32
        )
    points = env._fruit_hulls[name].expand(env.num_envs, -1, -1)
    count = points.shape[1]
    world = quat_apply(fruit.data.root_quat_w[:, None, :].expand(-1, count, -1), points)
    world += fruit.data.root_pos_w[:, None, :] - basket.data.root_pos_w[:, None, :]
    local = quat_apply_inverse(basket.data.root_quat_w[:, None, :].expand(-1, count, -1), world)
    return local.amin(dim=1), local.amax(dim=1)


def fruits_in_basket(env):
    contained = []
    for name in env.cfg.fruit_names:
        lower, upper = fruit_bounds_in_basket(env, name)
        half = torch.tensor(env.cfg.basket_inner_half_xy, device=env.device)
        inside = ((lower[:, :2] >= -half) & (upper[:, :2] <= half)).all(dim=-1)
        inside &= (lower[:, 2] >= env.cfg.basket_floor_z - 0.005)
        inside &= (upper[:, 2] <= env.cfg.basket_rim_z - 0.003)
        contained.append(inside)
    return torch.stack(contained, dim=-1)


def placed_on_target(env):
    inside = fruits_in_basket(env).all(dim=-1)
    still = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for name in env.cfg.fruit_names:
        fruit = env.scene[name]
        still &= torch.linalg.vector_norm(fruit.data.root_lin_vel_w, dim=-1) < 0.03
        still &= torch.linalg.vector_norm(fruit.data.root_ang_vel_w, dim=-1) < 0.25
    robot = env.scene["robot"]
    fingers, _ = robot.find_joints("panda_finger_joint.*")
    released = robot.data.joint_pos[:, fingers].sum(dim=-1) > 0.06
    if not hasattr(env, "success_steps"):
        env.success_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env.success_steps = torch.where(inside & still & released, env.success_steps + 1, 0)
    return env.success_steps >= env.cfg.success_hold_steps
