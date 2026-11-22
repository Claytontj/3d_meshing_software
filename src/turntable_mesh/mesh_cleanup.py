"""Clean a reconstructed mesh and export binary STL in an isolated process."""

import argparse
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StlExports:
    """STL files produced from one source PLY."""

    direct: Path
    cleaned: Path | None


def _load_mesh(mesh: Path):
    try:
        import pymeshlab
    except ImportError as error:
        raise RuntimeError(
            "PyMeshLab is required for STL export. Install it in the app's Python "
            "environment with: python -m pip install pymeshlab"
        ) from error

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh))
    return ms


def _save_binary_stl(mesh_set, destination: Path) -> None:
    # Publish only a complete export, preserving any previous STL on failure.
    with tempfile.TemporaryDirectory(prefix="mesh_export_", dir=destination.parent) as staging:
        temporary = Path(staging) / destination.name
        mesh_set.save_current_mesh(str(temporary), binary=True)
        if not temporary.is_file() or temporary.stat().st_size <= 84:
            raise RuntimeError("PyMeshLab did not create a nonempty binary STL.")
        temporary.replace(destination)


def _export_direct_stl(mesh: Path, destination: Path, log: Callable[[str], None]) -> Path:
    log(f"Direct conversion: loading mesh: {mesh}")
    ms = _load_mesh(mesh)
    current = ms.current_mesh()
    log(f"Loaded! Vertices: {current.vertex_number():,}, Faces: {current.face_number():,}")
    if current.face_number() == 0:
        raise ValueError("The selected mesh has no triangle faces to export.")
    log("Direct conversion: exporting binary STL without cleanup...")
    _save_binary_stl(ms, destination)
    log(f"Direct STL saved: {destination}")
    return destination


def clean_mesh(mesh: Path, destination: Path, log: Callable[[str], None]) -> Path:
    """Apply the object-mesh cleanup recipe without changing the source PLY."""
    log(f"1. Loading mesh: {mesh}")
    ms = _load_mesh(mesh)
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
    _save_binary_stl(ms, destination)
    log(f"STL saved: {destination}")
    return destination


def export_stl_versions(mesh: Path, log: Callable[[str], None], run_cleanup: bool = True) -> StlExports:
    """Create a direct STL and optionally a separate cleaned STL."""
    mesh = mesh.resolve()
    if not mesh.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh}")
    direct = mesh.parent / "solidworks_direct.stl"
    cleaned = mesh.parent / "solidworks_ready.stl"
    _export_direct_stl(mesh, direct, log)
    if not run_cleanup:
        log("Optional cleanup skipped; direct STL is ready for SolidWorks.")
        return StlExports(direct=direct, cleaned=None)
    try:
        clean_mesh(mesh, cleaned, log)
    except Exception as error:
        log(f"WARNING: cleaned STL was not created: {error}")
        log(f"Direct STL remains available: {direct}")
        return StlExports(direct=direct, cleaned=None)
    return StlExports(direct=direct, cleaned=cleaned)


def _cleanup_command(mesh: Path, run_cleanup: bool) -> list[str]:
    """Build the child-process command for source and packaged execution."""
    command = [sys.executable]
    if getattr(sys, "frozen", False):
        command.append("--cleanup")
    else:
        command.extend(["-u", str(Path(__file__).resolve())])
    command.append(str(mesh))
    if not run_cleanup:
        command.append("--skip-cleanup")
    return command


def export_solidworks_stl(mesh: Path, log: Callable[[str], None], run_cleanup: bool = True) -> StlExports:
    """Stream cleanup progress while keeping native library failures out of Tk."""
    mesh = mesh.resolve()
    if not mesh.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh}")
    log_path = mesh.parent / "mesh_cleanup.log"
    command = _cleanup_command(mesh, run_cleanup)
    direct = mesh.parent / "solidworks_direct.stl"
    cleaned = mesh.parent / "solidworks_ready.stl"
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
        if direct.is_file() and direct.stat().st_size > 84:
            log(f"WARNING: optional cleanup process failed; direct STL remains available: {direct}")
            return StlExports(direct=direct, cleaned=None)
        raise RuntimeError(
            f"Mesh cleanup failed (exit code {returncode}). {last_line} "
            f"See {log_path} for details."
        )
    if not direct.is_file() or direct.stat().st_size <= 84:
        raise RuntimeError("Direct STL export did not produce a nonempty STL file.")
    return StlExports(
        direct=direct,
        cleaned=cleaned if run_cleanup and cleaned.is_file() and cleaned.stat().st_size > 84 else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean a mesh and export binary STL")
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--skip-cleanup", action="store_true")
    args = parser.parse_args()
    # Use the same encoding as the parent process, including for non-ASCII paths.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        export_stl_versions(args.mesh, lambda message: print(message, flush=True), run_cleanup=not args.skip_cleanup)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
