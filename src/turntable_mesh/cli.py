"""Command-line entry point and configuration loading."""

import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .pipeline import run_pipeline


@dataclass(frozen=True)
class Config:
    video_path: Path
    frame_sampling_rate: float
    crop: tuple[int, int, int, int]
    max_image_dimension: int
    colmap_executable: str
    output_folder: Path


def load_config(path: Path) -> Config:
    """Load paths relative to the config file's directory."""
    with path.open("rb") as config_file:
        data = tomllib.load(config_file)

    root = path.resolve().parent
    input_settings = data["input"]
    output_settings = data["output"]
    return Config(
        video_path=root / input_settings["video_path"],
        frame_sampling_rate=float(input_settings["frame_sampling_rate"]),
        crop=tuple(input_settings["crop"]),
        max_image_dimension=int(input_settings["max_image_dimension"]),
        colmap_executable=str(data["colmap"]["executable"]),
        output_folder=root / output_settings["folder"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an object-only mesh from a turntable video.")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        run_pipeline(config)
    except NotImplementedError as error:
        parser.exit(1, f"Pipeline stage pending: {error}\n")
    return 0
