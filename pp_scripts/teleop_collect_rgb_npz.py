import numpy as np
from pathlib import Path
from queue import SimpleQueue
import argparse
import time
import json

try:
    from .dataset_schema import STATE_NAMES, ACTION_NAMES, validate_episode
except ImportError:
    from dataset_schema import STATE_NAMES, ACTION_NAMES, validate_episode


def main():
    parser = argparse.ArgumentParser(description="Collect fruit-v1 RGB demonstrations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/teleop_fruit_v1")
    parser.add_argument("--translation-step", type=float, default=0.005, help="TCP translation command per control step, metres")
    parser.add_argument("--rotation-step", type=float, default=0.03, help="TCP rotation command per control step, radians")
    parser.add_argument("--gripper-rate", type=float, default=1.5, help="Normalized gripper command change per simulation second")
    args = parser.parse_args()
    if min(args.translation_step, args.rotation_step, args.gripper_rate) <= 0:
        parser.error("Control step sizes and gripper rate must be positive")
    from isaaclab.app import AppLauncher
    import torch
    from pynput import keyboard

    # ---- App ----
    app = AppLauncher(headless=False, enable_cameras=True).app

    # ---- Env ----
    from pick_and_place_project.tasks.pick_place_gr1t2_pi import make_env
    env = make_env()

    # ---- Reset + warm-up ----
    obs, info = env.reset(seed=args.seed)
    for _ in range(10):
        app.update()
    a0 = torch.zeros((env.num_envs, 7), device=env.device, dtype=torch.float32)
    a0[:, -1] = -1.0
    obs, *_ = env.step(a0)

    device = env.device

    # ---- Print shapes ----
    if "wrist_rgb" in obs["images"]:
        print("Wrist RGB shape:", obs["images"]["wrist_rgb"].shape, obs["images"]["wrist_rgb"].dtype)
    else:
        print("[WARN] no wrist_rgb in obs['images']!")

    # ---- Save dirs ----
    import re
    from datetime import datetime

    save_dir = args.save_dir.expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = (save_dir / "debug_images").resolve()
    debug_dir.mkdir(parents=True, exist_ok=True)

    def next_episode_id(folder: Path, prefix: str = "ep_", suffix: str = ".npz") -> int:
        pat = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
        max_id = -1
        for p in folder.iterdir():
            if not p.is_file():
                continue
            m = pat.match(p.name)
            if m:
                max_id = max(max_id, int(m.group(1)))
        return max_id + 1

    def safe_episode_path(folder: Path, ep_id: int) -> Path:
        while True:
            p = folder / f"ep_{ep_id:05d}.npz"
            if not p.exists():
                return p
            ep_id += 1

    ep_id = next_episode_id(save_dir)
    print(f"[SAVE] directory: {save_dir}")
    print(f"[SAVE] starting ep_id: {ep_id:05d}")
    print(f"[DEBUG] image debug dir: {debug_dir}")

    # ---- Teleop state ----
    pressed = set()
    recording = False
    pending_reset = False
    commands = SimpleQueue()
    quit_requested = False

    episode_state = []
    episode_rgb_raw = []
    episode_wrist_raw = []
    episode_oblique_raw = []
    episode_action = []

    # scales
    dpos = args.translation_step
    drot = args.rotation_step

    # continuous gripper command in [-1, +1]
    grip_cmd = -1.0
    grip_rate = args.gripper_rate  # per second

    def key_to_str(k):
        try:
            return k.char.lower()
        except Exception:
            return str(k)

   
    def _raw_to_view_u8(x_raw: np.ndarray) -> np.ndarray:
        """Robustly make raw float image viewable as uint8 for debugging."""
        if x_raw.dtype == np.uint8:
            return x_raw.copy()
        x = x_raw.astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mn, mx = float(x.min()), float(x.max())

        # If it already looks like [0,1], keep it
        if mx <= 1.5 and mn >= -0.1:
            y = np.clip(x, 0.0, 1.0)
            return (y * 255.0).astype(np.uint8)

        # Otherwise, percentile stretch in raw space
        lo = np.percentile(x, 1.0)
        hi = np.percentile(x, 99.0)
        if hi <= lo + 1e-8:
            return np.zeros_like(x, dtype=np.uint8)
        y = (x - lo) / (hi - lo)
        y = np.clip(y, 0.0, 1.0)
        return (y * 255.0).astype(np.uint8)

    def save_debug_images(obs_any):
        """Press N -> save current overhead+wrist PNG for human check."""
        if "images" not in obs_any:
            print("[DEBUG] no images in obs.")
            return
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # overhead
        if "rgb" in obs_any["images"]:
            oh = obs_any["images"]["rgb"][0].detach().cpu().numpy()
            oh_u8 = _raw_to_view_u8(oh)
            oh_path = debug_dir / f"overhead_{ts}.png"
            try:
                import imageio.v2 as imageio
                imageio.imwrite(str(oh_path), oh_u8)
            except Exception:
                from PIL import Image
                Image.fromarray(oh_u8).save(str(oh_path))
            print(f"[DEBUG] saved {oh_path.name}  raw[min,max,std]=({oh.min():.4f},{oh.max():.4f},{oh.std():.4f})")

        # wrist
        if "wrist_rgb" in obs_any["images"]:
            w = obs_any["images"]["wrist_rgb"][0].detach().cpu().numpy()
            w_u8 = _raw_to_view_u8(w)
            w_path = debug_dir / f"wrist_{ts}.png"
            try:
                import imageio.v2 as imageio
                imageio.imwrite(str(w_path), w_u8)
            except Exception:
                from PIL import Image
                Image.fromarray(w_u8).save(str(w_path))
            print(f"[DEBUG] saved {w_path.name}  raw[min,max,std]=({w.min():.4f},{w.max():.4f},{w.std():.4f})")
        else:
            print("[DEBUG] no wrist_rgb in obs['images'].")
            
        # oblique
        if "oblique_rgb" in obs_any["images"]:
            ob = obs_any["images"]["oblique_rgb"][0].detach().cpu().numpy()
            ob_u8 = _raw_to_view_u8(ob)
            ob_path = debug_dir / f"oblique_{ts}.png"
            try:
                import imageio.v2 as imageio
                imageio.imwrite(str(ob_path), ob_u8)
            except Exception:
                from PIL import Image
                Image.fromarray(ob_u8).save(str(ob_path))
            print(f"[DEBUG] saved {ob_path.name}  raw[min,max,std]=({ob.min():.4f},{ob.max():.4f},{ob.std():.4f})")

    # ---------- episode save ----------
    def save_episode(current_ep_id: int, reason="manual", success=False) -> int:
        nonlocal episode_state, episode_rgb_raw, episode_wrist_raw, episode_action

        lens = {
            "state": len(episode_state),
            "rgb_raw": len(episode_rgb_raw),
            "wrist_raw": len(episode_wrist_raw),
            "oblique_rgb_raw": len(episode_oblique_raw),
            "action": len(episode_action),
        }
        if len(set(lens.values())) != 1:
            raise ValueError(f"Refusing to save misaligned episode: {lens}")
        T = lens["state"]
        if T <= 5:
            print("[SAVE] skipped (too short / not aligned)", lens)
            episode_state.clear()
            episode_rgb_raw.clear()
            episode_wrist_raw.clear()
            episode_oblique_raw.clear()
            episode_action.clear()
            return current_ep_id

        path = safe_episode_path(save_dir, current_ep_id)

        out = {
            "state": np.stack(episode_state[:T], axis=0).astype(np.float32),
            "rgb_raw": np.stack(episode_rgb_raw[:T], axis=0).astype(np.uint8),
            "wrist_rgb_raw": np.stack(episode_wrist_raw[:T], axis=0).astype(np.uint8),
            "oblique_rgb_raw": np.stack(episode_oblique_raw[:T], axis=0).astype(np.uint8),
            "action": np.stack(episode_action[:T], axis=0).astype(np.float32),
            "timestamp": np.arange(T, dtype=np.float64) * float(env.step_dt),
            "metadata_json": np.array(json.dumps({
                "schema_version": 1,
                "episode_id": int(path.stem.split("_")[1]),
                "state_names": STATE_NAMES,
                "action_names": ACTION_NAMES,
                "action_frame": "robot_base",
                "alignment": "observation_before_action",
                "timestamp_clock": "simulation_seconds_from_recording_start",
                "translation_step_m": args.translation_step,
                "rotation_step_rad": args.rotation_step,
                "gripper_rate_per_second": args.gripper_rate,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "T": int(T),
                "dt": float(env.step_dt),
                "cfg_class": type(env.cfg).__name__,
                "cfg_module": type(env.cfg).__module__,
                "note": "fruit_v1_rgb_uint8_range",
                "task": "Place the banana and the apple into the open bin.",
                "image_range": [0, 255],
                "end_reason": reason,
                "success": bool(success),
                "seed": args.seed,
            })),
        }

        validate_episode(out)
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **out)
        temporary.replace(path)
        print(f"[SAVE] {path.name}  T={T}  keys={list(out.keys())}")

        episode_state.clear()
        episode_rgb_raw.clear()
        episode_wrist_raw.clear()
        episode_oblique_raw.clear()
        episode_action.clear()

        try:
            saved_id = int(path.stem.split("_")[1])
            return saved_id + 1
        except Exception:
            return current_ep_id + 1

    # ---------- keyboard callbacks ----------
    def on_press(k):
        s = key_to_str(k)
        if s in pressed:
            return
        pressed.add(s)
        if s in ("r", "m", "n", "Key.esc"):
            commands.put(s)
        if s == "Key.esc":
            return False

    def on_release(k):
        s = key_to_str(k)
        pressed.discard(s)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print("listener started")
    print("Motion: W/S=X | A/D=Y | Q/E=Z | I/K=Rx | J/L=Ry | U/O=Rz")
    print("Controls: R=record toggle | N=save debug PNGs | M=reset | V/B=gripper close/open | ESC=quit")

    # ---------- main loop ----------
    try:
        while app.is_running() and not quit_requested:
            loop_started = time.perf_counter()
            while not commands.empty():
                command = commands.get()
                if command == "r":
                    recording = not recording
                    if not recording:
                        ep_id = save_episode(ep_id)
                    print(f"[REC] {'ON' if recording else 'OFF'}")
                elif command == "m":
                    pending_reset = True
                elif command == "n":
                    save_debug_images(obs)
                elif command == "Key.esc":
                    quit_requested = True
            if quit_requested:
                break
            if pending_reset:
                pending_reset = False
                ep_id = save_episode(ep_id, reason="manual_reset")
                recording = False
                grip_cmd = -1.0
                print("[RESET] main thread")
                obs, info = env.reset()
                pressed.clear()
                for _ in range(2):
                    app.update()
                continue

            # continuous gripper update
            dt = float(env.step_dt)
            if ("v" in pressed) and ("b" not in pressed):
                grip_cmd = min(1.0, grip_cmd + grip_rate * dt)
            elif ("b" in pressed) and ("v" not in pressed):
                grip_cmd = max(-1.0, grip_cmd - grip_rate * dt)

            # action (7D: dx dy dz dRx dRy dRz grip)
            a = torch.zeros((1, 7), device=device, dtype=torch.float32)

            # translation
            if "w" in pressed: a[0, 0] += dpos
            if "s" in pressed: a[0, 0] -= dpos
            if "a" in pressed: a[0, 1] += dpos
            if "d" in pressed: a[0, 1] -= dpos
            if "q" in pressed: a[0, 2] += dpos
            if "e" in pressed: a[0, 2] -= dpos

            # rotation
            if "i" in pressed: a[0, 3] += drot
            if "k" in pressed: a[0, 3] -= drot
            if "j" in pressed: a[0, 4] += drot
            if "l" in pressed: a[0, 4] -= drot
            if "u" in pressed: a[0, 5] += drot
            if "o" in pressed: a[0, 5] -= drot

            a[0, 6] = float(grip_cmd)

            # record (store obs BEFORE step)
            if recording:
                # state
                episode_state.append(obs["policy"].detach().cpu().numpy()[0].astype(np.float32, copy=True))

                # Preserve sensor RGB bytes without normalization or color stretching.
                oh = obs["images"]["rgb"].detach().cpu().numpy()[0].copy()
                episode_rgb_raw.append(oh)

                episode_wrist_raw.append(obs["images"]["wrist_rgb"][0].detach().cpu().numpy().copy())
                episode_oblique_raw.append(obs["images"]["oblique_rgb"][0].detach().cpu().numpy().copy())

                # action
                episode_action.append(a.detach().cpu().numpy()[0].astype(np.float32, copy=True))

            obs2, rew, done, trunc, info = env.step(a)
            if bool(done.any() or trunc.any()):
                success = bool(info["success"].any())
                ep_id = save_episode(ep_id, reason="success" if success else "timeout", success=success)
                recording = False
                grip_cmd = -1.0
                pressed.clear()
            obs = obs2
            # Keep keyboard input at at most the simulation control frequency.
            time.sleep(max(0.0, env.step_dt - (time.perf_counter() - loop_started)))

    finally:
        listener.stop()
        save_episode(ep_id, reason="exit")
        env.close()
        app.close()


if __name__ == "__main__":
    main()
