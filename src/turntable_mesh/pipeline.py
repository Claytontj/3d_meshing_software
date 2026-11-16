"""Frame, mask, and COLMAP processing for one selected video."""

import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import Config

Log = Callable[[str], None]


def new_workspace(config: Config) -> Path:
    config.output_folder.mkdir(parents=True, exist_ok=True)
    workspace = config.output_folder / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    workspace.mkdir()
    return workspace


def extract_frames(config: Config, workspace: Path, log: Log) -> list[Path]:
    """Sample at the configured FPS and apply a fixed crop and size limit."""
    config.validate()
    if config.video_path is None or not config.video_path.is_file():
        raise FileNotFoundError("Choose an existing video file first.")
    capture = cv2.VideoCapture(str(config.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {config.video_path}")
    frame_dir = workspace / "images"
    frame_dir.mkdir()
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise RuntimeError("The video does not report a usable frame rate.")
    selected: list[Path] = []
    next_sample = 0.0
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            seconds = index / source_fps
            index += 1
            if seconds + 1e-8 < next_sample:
                continue
            next_sample += 1.0 / config.frame_sampling_rate
            source_height, source_width = frame.shape[:2]
            x, y, width, height = config.crop
            if width == height == 0:
                x, y, width, height = 0, 0, source_width, source_height
            if x + width > source_width or y + height > source_height:
                raise ValueError("Crop extends outside the video frame.")
            frame = frame[y:y + height, x:x + width]
            scale = min(1.0, config.max_image_dimension / max(frame.shape[:2]))
            if scale < 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            destination = frame_dir / f"frame_{len(selected) + 1:06d}.png"
            if not cv2.imwrite(str(destination), frame):
                raise OSError(f"Could not write {destination}")
            selected.append(destination)
    finally:
        capture.release()
    if len(selected) < 2:
        raise RuntimeError("Fewer than two frames were extracted. Increase FPS or use a longer video.")
    log(f"Extracted {len(selected)} frames to {frame_dir}")
    return selected


def prepare_object_masks(frames: list[Path], workspace: Path, log: Log) -> list[Path]:
    """Create editable initial foreground masks with OpenCV GrabCut."""
    mask_dir = workspace / "masks"
    mask_dir.mkdir()
    masks: list[Path] = []
    for number, frame_path in enumerate(frames, 1):
        image = cv2.imread(str(frame_path))
        if image is None:
            raise RuntimeError(f"Cannot read extracted frame: {frame_path}")
        height, width = image.shape[:2]
        margin_x, margin_y = max(1, width // 20), max(1, height // 20)
        if width <= 2 * margin_x or height <= 2 * margin_y:
            raise ValueError("Frame is too small for mask generation.")
        grabcut = np.zeros((height, width), dtype=np.uint8)
        rectangle = (margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y)
        try:
            cv2.grabCut(image, grabcut, rectangle, None, None, 5, cv2.GC_INIT_WITH_RECT)
            mask = np.where((grabcut == cv2.GC_FGD) | (grabcut == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except cv2.error:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(mask, (margin_x, margin_y), (width - margin_x, height - margin_y), 255, -1)
            log(f"GrabCut could not segment frame {number}; edit its rectangular starter mask.")
        if not np.any(mask):
            cv2.rectangle(mask, (margin_x, margin_y), (width - margin_x, height - margin_y), 255, -1)
            log(f"GrabCut returned an empty mask for frame {number}; edit its rectangular starter mask.")
        destination = mask_dir / f"{frame_path.name}.png"
        if not cv2.imwrite(str(destination), mask):
            raise OSError(f"Could not write {destination}")
        masks.append(destination)
        if number % 10 == 0 or number == len(frames):
            log(f"Generated {number}/{len(frames)} initial masks")
    return masks


def _colmap(config: Config, workspace: Path, log: Log, *args: str) -> None:
    executable = Path(config.colmap_executable)
    if not executable.is_file():
        raise FileNotFoundError(f"COLMAP launcher not found: {executable}")
    command = [str(executable), *args]
    line = subprocess.list2cmdline(command)
    with (workspace / "colmap_commands.log").open("a", encoding="utf-8") as command_log:
        command_log.write(line + "\n")
    log(f"COLMAP command: {line}")
    process = subprocess.Popen(
        command, cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.stdout is not None
    with (workspace / "colmap_output.log").open("a", encoding="utf-8") as output_log:
        for output_line in process.stdout:
            output_log.write(output_line)
            log(output_line.rstrip())
    if process.wait() != 0:
        raise RuntimeError(f"COLMAP command failed: {line}. See colmap_output.log in {workspace}.")


def _prepare_dense_masks(frames: list[Path], masks: list[Path], dense: Path) -> Path:
    """Match mask filenames and sizes to the undistorted dense images."""
    mask_dir = dense / "object_masks"
    mask_dir.mkdir()
    source_masks = {frame.name: mask for frame, mask in zip(frames, masks, strict=True)}
    dense_images = list((dense / "images").rglob("*.png"))
    if not dense_images:
        raise RuntimeError("COLMAP did not produce undistorted images.")
    for image_path in dense_images:
        source = source_masks.get(image_path.name)
        if source is None:
            raise RuntimeError(f"No object mask for dense image {image_path.name}")
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Cannot read dense image or mask: {image_path.name}")
        height, width = image.shape[:2]
        mask_height, mask_width = mask.shape[:2]
        if abs(width / height - mask_width / mask_height) > 0.01:
            raise RuntimeError("Undistorted image shape changed; mask alignment needs review.")
        if (height, width) != (mask_height, mask_width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        destination = mask_dir / image_path.relative_to(dense / "images")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination) + ".png", mask):
            raise OSError(f"Could not write dense mask for {image_path.name}")
    return mask_dir


def run_colmap(frames: list[Path], masks: list[Path], config: Config, workspace: Path, log: Log) -> Path:
    """Run sparse mapping, dense stereo, masked fusion, and Poisson meshing."""
    if len(frames) != len(masks) or not frames:
        raise ValueError("Every frame needs an object mask.")
    for mask in masks:
        if not mask.is_file():
            raise FileNotFoundError(f"Missing object mask: {mask}")
    database = workspace / "database.db"
    image_dir = workspace / "images"
    sparse = workspace / "sparse"
    dense = workspace / "dense"
    sparse.mkdir()
    dense.mkdir()
    _colmap(config, workspace, log, "feature_extractor", "--database_path", str(database),
            "--image_path", str(image_dir), "--ImageReader.mask_path", str(workspace / "masks"),
            "--ImageReader.camera_model", "SIMPLE_PINHOLE", "--ImageReader.single_camera", "1")
    _colmap(config, workspace, log, "exhaustive_matcher", "--database_path", str(database))
    _colmap(config, workspace, log, "mapper", "--database_path", str(database),
            "--image_path", str(image_dir), "--output_path", str(sparse))
    models = sorted((path for path in sparse.iterdir() if path.is_dir()), key=lambda item: int(item.name) if item.name.isdigit() else -1)
    if not models:
        raise RuntimeError("COLMAP did not create a sparse model; check the masks and video overlap.")
    model = models[0]
    _colmap(config, workspace, log, "image_undistorter", "--image_path", str(image_dir),
            "--input_path", str(model), "--output_path", str(dense), "--output_type", "COLMAP")
    dense_masks = _prepare_dense_masks(frames, masks, dense)
    _colmap(config, workspace, log, "patch_match_stereo", "--workspace_path", str(dense),
            "--workspace_format", "COLMAP")
    fused = workspace / "object_fused.ply"
    _colmap(config, workspace, log, "stereo_fusion", "--workspace_path", str(dense),
            "--workspace_format", "COLMAP", "--input_type", "geometric",
            "--StereoFusion.mask_path", str(dense_masks), "--output_path", str(fused))
    mesh = workspace / "object_mesh.ply"
    _colmap(config, workspace, log, "poisson_mesher", "--input_path", str(fused),
            "--output_path", str(mesh))
    return mesh


def collect_mesh_output(mesh: Path, log: Log) -> Path:
    """Report the object-masked mesh produced by COLMAP."""
    if not mesh.is_file() or mesh.stat().st_size == 0:
        raise RuntimeError("COLMAP did not create a mesh file.")
    log(f"Object mesh saved: {mesh}")
    return mesh
