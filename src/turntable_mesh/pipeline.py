"""Planned reconstruction stages. No processing is implemented yet."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cli import Config


def extract_frames(config: "Config") -> list[Path]:
    """Sample, crop, and resize frames from the configured video."""
    raise NotImplementedError("extract_frames")


def prepare_object_masks(frames: list[Path], config: "Config") -> list[Path]:
    """Prepare user-supplied masks that retain the object and exclude the base."""
    raise NotImplementedError("prepare_object_masks")


def run_colmap(frames: list[Path], masks: list[Path], config: "Config") -> Path:
    """Run COLMAP with object masks and return its reconstruction directory."""
    raise NotImplementedError("run_colmap")


def collect_mesh_output(reconstruction_dir: Path, config: "Config") -> Path:
    """Collect the object-only mesh into the configured output folder."""
    raise NotImplementedError("collect_mesh_output")


def run_pipeline(config: "Config") -> Path:
    """Run the planned stages in order once their implementations are added."""
    frames = extract_frames(config)
    masks = prepare_object_masks(frames, config)
    reconstruction_dir = run_colmap(frames, masks, config)
    return collect_mesh_output(reconstruction_dir, config)
