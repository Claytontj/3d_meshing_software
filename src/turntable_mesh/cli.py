"""Launch the desktop prototype from the installed command or -m."""

import argparse
from pathlib import Path

from .config import Config, load_config  # Preserve the original public imports.


def main() -> int:
    parser = argparse.ArgumentParser(description="Turntable mesh desktop prototype")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    args = parser.parse_args()
    from .app import launch

    launch(args.config)
    return 0
