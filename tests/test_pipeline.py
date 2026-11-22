import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from turntable_mesh.pipeline import prepare_object_masks


class BackgroundMaskTests(unittest.TestCase):
    def test_empty_background_reference_removes_static_background(self):
        with tempfile.TemporaryDirectory(prefix="test_masks_", dir=Path.cwd()) as folder:
            root = Path(folder)
            background = np.full((120, 160, 3), 30, dtype=np.uint8)
            frame = background.copy()
            cv2.rectangle(frame, (45, 25), (115, 95), (210, 210, 210), -1)
            background_path = root / "empty.png"
            frame_path = root / "frame_000001.png"
            cv2.imwrite(str(background_path), background)
            cv2.imwrite(str(frame_path), frame)
            workspace = root / "workspace"
            workspace.mkdir()
            logs: list[str] = []

            masks = prepare_object_masks(
                [frame_path], workspace, logs.append, background_path
            )

            mask = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(mask)
            assert mask is not None
            self.assertGreater(np.mean(mask[35:85, 55:105]), 200)
            self.assertLess(np.mean(mask[0:20, 0:20]), 25)
            self.assertTrue(any("empty-background reference" in line for line in logs))

    def test_background_reference_rejects_different_aspect_ratio(self):
        with tempfile.TemporaryDirectory(prefix="test_masks_", dir=Path.cwd()) as folder:
            root = Path(folder)
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            background = np.zeros((100, 100, 3), dtype=np.uint8)
            frame_path = root / "frame_000001.png"
            background_path = root / "empty.png"
            cv2.imwrite(str(frame_path), frame)
            cv2.imwrite(str(background_path), background)
            workspace = root / "workspace"
            workspace.mkdir()

            with self.assertRaisesRegex(ValueError, "same aspect ratio"):
                prepare_object_masks(
                    [frame_path], workspace, lambda _message: None, background_path
                )


if __name__ == "__main__":
    unittest.main()
