"""Launch the desktop prototype from the installed command or -m."""

import argparse
import sys
from pathlib import Path

from .config import Config, load_config  # Preserve the original public imports.


def _default_config_path() -> Path:
    """Find config.toml beside a packaged executable or in the current folder."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "config.toml"
    return Path("config.toml")


def main() -> int:
    parser = argparse.ArgumentParser(description="Turntable mesh desktop prototype")
    parser.add_argument("--config", type=Path, default=_default_config_path())
    args = parser.parse_args()
    from .app import launch

    launch(args.config)
    return 0
