"""NPZ contract: each row pairs pre-action observation o_t with applied command a_t.

State preserves the existing 23D observation order (including repeated fingers).
Actions are robot-base-frame TCP translation (m), axis-angle rotation (rad),
and a normalized gripper position command (-1 open, +1 closed).
Timestamps are simulation seconds from the start of each recorded segment.
An episode is one uninterrupted recording segment, closed on stop/reset/done/exit.
RGB arrays are uint8, THWC. Metadata is JSON text, readable without pickle.
"""
import json

import numpy as np

JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]
STATE_NAMES = (
    [f"qrel_{name}" for name in JOINT_NAMES]
    + [f"qdrel_{name}" for name in JOINT_NAMES]
    + ["grip_qrel_finger1", "grip_qrel_finger2", "grip_qdrel_finger1", "grip_qdrel_finger2", "grip_width"]
)
ACTION_NAMES = ["tcp_dx", "tcp_dy", "tcp_dz", "tcp_drx", "tcp_dry", "tcp_drz", "gripper"]
CAMERA_KEYS = ("rgb_raw", "wrist_rgb_raw", "oblique_rgb_raw")


def validate_episode(data, *, legacy_fps=None):
    """Validate alignment and return a summary. Legacy metadata is never unpickled.

Old recordings without timestamps need an explicit legacy_fps when converting.
Inspection may omit it and report unknown FPS rather than guessing.
"""
    required = ("state", "action", *CAMERA_KEYS)
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing arrays: {missing}")
    state, action = data["state"], data["action"]
    if state.ndim != 2 or state.shape[1] != len(STATE_NAMES):
        raise ValueError(f"Expected state (T, 23), got {state.shape}")
    count = len(state)
    if count == 0 or action.shape != (count, len(ACTION_NAMES)):
        raise ValueError(f"Empty or misaligned episode: state={state.shape}, action={action.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Non-finite state/action values")
    if np.any(np.abs(action[:, -1]) > 1.00001):
        raise ValueError("Gripper command outside [-1, 1]")
    cameras = {}
    legacy_images = []
    for key in CAMERA_KEYS:
        frames = data[key]
        if frames.ndim != 4 or frames.shape[0] != count or frames.shape[-1] != 3 or min(frames.shape[1:3]) < 1:
            raise ValueError(f"Invalid or misaligned {key}: {frames.shape}")
        if frames.dtype != np.uint8:
            if not np.issubdtype(frames.dtype, np.floating) or not np.isfinite(frames).all():
                raise ValueError(f"Invalid RGB dtype/values: {key}")
            legacy_images.append(key)
        cameras[key] = list(frames.shape[1:])
    fps = legacy_fps
    metadata = {}
    if "metadata_json" in data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("schema_version") != 1 or metadata.get("T") != count:
            raise ValueError("Invalid schema version or frame count in metadata")
        dt = float(metadata["dt"])
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid control dt")
        fps = 1.0 / dt
        if legacy_images:
            raise ValueError("Schema v1 requires uint8 RGB")
        if "timestamp" not in data:
            raise ValueError("Schema v1 requires timestamps")
    if "timestamp" in data:
        ts = data["timestamp"]
        if ts.shape != (count,) or not np.isfinite(ts).all() or abs(ts[0]) > 1e-8:
            raise ValueError("Invalid timestamps; expected (T,) starting at zero")
        if count > 1:
            delta = np.diff(ts)
            if np.any(delta <= 0) or not np.allclose(delta, delta[0], rtol=1e-5, atol=1e-8):
                raise ValueError("Timestamps must be increasing at a fixed control rate")
            inferred = 1.0 / float(delta[0])
            if fps is not None and not np.isclose(fps, inferred, rtol=1e-5):
                raise ValueError("Timestamp FPS does not match declared FPS")
            fps = inferred
    if fps is not None and (not np.isfinite(fps) or fps <= 0):
        raise ValueError("FPS must be positive and finite")
    return {"frames": count, "fps": fps, "state_dimension": state.shape[1],
            "action_dimension": action.shape[1], "cameras": cameras,
            "legacy_float_images": legacy_images, "metadata": metadata}
