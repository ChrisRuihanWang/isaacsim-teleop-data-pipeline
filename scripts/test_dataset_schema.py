"""Regression checks for demonstration alignment; no simulator required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pp_scripts.dataset_schema import CAMERA_KEYS, validate_episode


def episode():
    return {"state": np.zeros((6, 23), dtype=np.float32),
            "action": np.zeros((6, 7), dtype=np.float32),
            "timestamp": np.arange(6) / 30,
            "metadata_json": np.array(json.dumps({"schema_version": 1, "T": 6, "dt": 1 / 30})),
            **{key: np.zeros((6, 16, 16, 3), dtype=np.uint8) for key in CAMERA_KEYS}}


class DatasetContractTests(unittest.TestCase):
    def test_roundtrip_without_pickle(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ep_00000.npz"
            np.savez_compressed(path, **episode())
            with np.load(path, allow_pickle=False) as data:
                result = validate_episode(data)
            self.assertEqual(result["frames"], 6)
            self.assertAlmostEqual(result["fps"], 30)

    def test_rejects_misaligned_camera(self):
        data = episode()
        data[CAMERA_KEYS[-1]] = data[CAMERA_KEYS[-1]][:-1]
        with self.assertRaisesRegex(ValueError, "misaligned"):
            validate_episode(data)

    def test_rejects_bad_clock(self):
        for clock in (np.arange(6) / 20, np.zeros(6), np.array([0, 1, 2, 4, 5, 6]) / 30):
            with self.subTest(clock=clock):
                data = episode()
                data["timestamp"] = clock
                with self.assertRaises(ValueError):
                    validate_episode(data)

    def test_rejects_nonfinite_state_and_invalid_gripper(self):
        for key, column, value in (("state", 0, np.nan), ("action", -1, 2.0)):
            data = episode()
            data[key][0, column] = value
            with self.assertRaises(ValueError):
                validate_episode(data)

    def test_legacy_metadata_is_not_unpickled(self):
        data = episode()
        del data["metadata_json"], data["timestamp"]
        data["meta"] = np.array([{"dt": 1 / 30}], dtype=object)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.npz"
            np.savez_compressed(path, **data)
            with np.load(path, allow_pickle=False) as loaded:
                self.assertIsNone(validate_episode(loaded)["fps"])
                self.assertEqual(validate_episode(loaded, legacy_fps=30)["fps"], 30)

    def test_v1_rejects_float_images_and_missing_timestamp(self):
        data = episode()
        data[CAMERA_KEYS[0]] = data[CAMERA_KEYS[0]].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "uint8"):
            validate_episode(data)
        data = episode()
        del data["timestamp"]
        with self.assertRaisesRegex(ValueError, "timestamps"):
            validate_episode(data)


if __name__ == "__main__":
    unittest.main()
