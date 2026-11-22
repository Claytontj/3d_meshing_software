"""Clean a reconstructed mesh and export binary STL in an isolated process."""

import argparse
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path


def clean_mesh(mesh: Path, destination: Path, log: Callable[[str], None]) -> Path:
    """Apply the object-mesh cleanup recipe without changing the source PLY."""
    try:
        import pymeshlab
    except ImportError as error:
        raise RuntimeError(
            "PyMeshLab is required for STL export. Install it in the app's Python "
            "environment with: python -m pip install pymeshlab"
        ) from error

    ms = pymeshlab.MeshSet()
    log(f"1. Loading mesh: {mesh}")
    ms.load_new_mesh(str(mesh))
    current = ms.current_mesh()
    log(f"Loaded! Vertices: {current.vertex_number():,}, Faces: {current.face_number():,}")
    if current.face_number() == 0:
        raise ValueError("The selected mesh has no triangle faces to export.")

    log("2. Cleaning broken geometry before decimation...")
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    if ms.current_mesh().face_number() == 0:
        raise ValueError("No usable faces remain after mesh cleanup.")

    log("3. Decimating to a target of 50,000 faces...")
    if ms.current_mesh().face_number() > 50_000:
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=50_000,
            optimalplacement=False,
            preservetopology=False,
        )
    else:
        log("Mesh already has 50,000 faces or fewer; skipping decimation.")
    log(f"Face count: {ms.current_mesh().face_number():,}")

    log("4. Final manifold repair for SolidWorks...")
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    if ms.current_mesh().face_number() == 0:
        raise ValueError("No usable faces remain after manifold repair.")

    log("5. Exporting binary STL...")
    # Publish only a complete export, preserving any previous STL on failure.
    with tempfile.TemporaryDirectory(prefix="mesh_export_", dir=destination.parent) as staging:
        temporary = Path(staging) / destination.name
        ms.save_current_mesh(str(temporary), binary=True)
        if not temporary.is_file() or temporary.stat().st_size <= 84:
            raise RuntimeError("PyMeshLab did not create a nonempty binary STL.")
        temporary.replace(destination)
    log(f"STL saved: {destination}")
    return destination


def _cleanup_command(mesh: Path, destination: Path) -> list[str]:
    """Build the child-process command for source and packaged execution."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "--cleanup", str(mesh), str(destination)]
    return [sys.executable, "-u", str(Path(__file__).resolve()), str(mesh), str(destination)]


def export_solidworks_stl(mesh: Path, log: Callable[[str], None]) -> Path:
    """Stream cleanup progress while keeping native library failures out of Tk."""
    mesh = mesh.resolve()
    if not mesh.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh}")
    destination = mesh.parent / "solidworks_ready.stl"
    if mesh == destination:
        raise ValueError("Select the source PLY mesh, rather than the exported STL.")
    log_path = mesh.parent / "mesh_cleanup.log"
    command = _cleanup_command(mesh, destination)
    last_line = "No error details were returned."
    with log_path.open("a", encoding="utf-8") as output_log:
        output_log.write(f"\nCleaning mesh: {mesh}\n")
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                output_log.write(line)
                output_log.flush()
                if line.strip():
                    last_line = line.rstrip()
                    log(last_line)
            returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(
            f"Mesh cleanup failed (exit code {returncode}). {last_line} "
            f"See {log_path} for details."
        )
    if not destination.is_file() or destination.stat().st_size <= 84:
        raise RuntimeError("Mesh cleanup did not produce a nonempty STL file.")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean a mesh and export binary STL")
    parser.add_argument("mesh", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    # Use the same encoding as the parent process, including for non-ASCII paths.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        clean_mesh(args.mesh, args.destination, lambda message: print(message, flush=True))
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
