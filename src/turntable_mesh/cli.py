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
    # PyInstaller bundles the whole application as one executable. The mesh
    # cleanup worker re-enters that executable with a private cleanup mode
    # instead of trying to pass a Python script path to the executable.
    if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        from .mesh_cleanup import main as cleanup_main

        sys.argv[1:] = sys.argv[2:]
        return cleanup_main()

    parser = argparse.ArgumentParser(description="Turntable mesh desktop prototype")
    parser.add_argument("--config", type=Path, default=_default_config_path())
    args = parser.parse_args()
    from .app import launch

    launch(args.config)
    return 0
