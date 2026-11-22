"""Editable project settings."""

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    video_path: Path | None
    frame_sampling_rate: float
    crop: tuple[int, int, int, int]
    max_image_dimension: int
    colmap_executable: str
    output_folder: Path
    config_root: Path | None = None
    background_image: Path | None = None

    def validate(self) -> None:
        if self.frame_sampling_rate <= 0:
            raise ValueError("Frame sampling rate must be greater than zero.")
        x, y, width, height = self.crop
        if min(x, y, width, height) < 0 or (width == 0) != (height == 0):
            raise ValueError("Crop values must be nonnegative; use width=height=0 for the full frame.")
        if self.max_image_dimension <= 0:
            raise ValueError("Maximum image dimension must be greater than zero.")
        if not self.colmap_executable.strip():
            raise ValueError("Set the COLMAP executable path.")


def _resolve(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def load_config(path: Path) -> Config:
    with path.open("rb") as source:
        data = tomllib.load(source)
    root = path.resolve().parent
    input_data = data["input"]
    raw_video = str(input_data.get("video_path", ""))
    raw_background = str(input_data.get("background_image", ""))
    crop = tuple(int(value) for value in input_data["crop"])
    if len(crop) != 4:
        raise ValueError("Crop must contain x, y, width, height.")
    config = Config(
        video_path=_resolve(root, raw_video) if raw_video else None,
        frame_sampling_rate=float(input_data["frame_sampling_rate"]),
        crop=(crop[0], crop[1], crop[2], crop[3]),
        max_image_dimension=int(input_data["max_image_dimension"]),
        colmap_executable=str(data["colmap"]["executable"]),
        output_folder=_resolve(root, str(data["output"]["folder"])),
        config_root=root,
        background_image=_resolve(root, raw_background) if raw_background else None,
    )
    config.validate()
    return config


def save_config(path: Path, config: Config) -> None:
    config.validate()
    crop = ", ".join(str(value) for value in config.crop)
    content = (
        "[input]\n"
        f"video_path = {json.dumps(str(config.video_path) if config.video_path else '')}\n"
        f"background_image = {json.dumps(str(config.background_image) if config.background_image else '')}\n"
        f"frame_sampling_rate = {config.frame_sampling_rate}\n"
        f"crop = [{crop}] # x, y, width, height; zero width and height use full frame\n"
        f"max_image_dimension = {config.max_image_dimension}\n\n"
        "[colmap]\n"
        f"executable = {json.dumps(config.colmap_executable)}\n\n"
        "[output]\n"
        f"folder = {json.dumps(str(config.output_folder))}\n"
    )
    path.write_text(content, encoding="utf-8")
