"""Exercise real mesh cleanup, binary output, and the desktop export state."""

import hashlib
import importlib.util
import queue
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from turntable_mesh.app import MeshApp
from turntable_mesh.mesh_cleanup import _cleanup_command, export_solidworks_stl


@unittest.skipUnless(importlib.util.find_spec("pymeshlab"), "PyMeshLab is not installed")
class MeshCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="test_mesh_", dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def assert_binary_stl(self, path):
        with path.open("rb") as source:
            header = source.read(84)
        faces = struct.unpack("<I", header[80:84])[0]
        self.assertGreater(faces, 0)
        self.assertLessEqual(faces, 50_000)
        self.assertEqual(path.stat().st_size, 84 + 50 * faces)
        return faces

    def test_large_dirty_mesh_is_reduced_and_source_preserved(self):
        import pymeshlab

        ms = pymeshlab.MeshSet()
        ms.create_sphere(subdiv=6)
        mesh = ms.current_mesh()
        vertices = np.vstack((mesh.vertex_matrix(), [[10.0, 10.0, 10.0]]))
        faces = np.vstack((mesh.face_matrix(), mesh.face_matrix()[:1], [[0, 0, 0]]))
        self.assertGreater(len(faces), 50_000)
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))
        source = self.folder / "large dirty mesh.ply"
        ms.save_current_mesh(str(source))
        original_hash = hashlib.sha256(source.read_bytes()).digest()
        progress = []
        destination = export_solidworks_stl(source, progress.append)
        self.assertEqual(destination, self.folder / "solidworks_ready.stl")
        self.assert_binary_stl(destination)
        self.assertEqual(hashlib.sha256(source.read_bytes()).digest(), original_hash)
        ms.load_new_mesh(str(destination))
        self.assertGreater(ms.current_mesh().face_number(), 0)
        self.assertTrue(any("Final manifold repair" in line for line in progress))
        self.assertIn("STL saved:", (self.folder / "mesh_cleanup.log").read_text(encoding="utf-8"))

    def test_small_mesh_skips_decimation(self):
        import pymeshlab

        ms = pymeshlab.MeshSet()
        ms.create_sphere(subdiv=1)
        count = ms.current_mesh().face_number()
        source = self.folder / "small.ply"
        ms.save_current_mesh(str(source))
        progress = []
        destination = export_solidworks_stl(source, progress.append)
        self.assertEqual(self.assert_binary_stl(destination), count)
        self.assertTrue(any("skipping decimation" in line for line in progress))

    def test_invalid_mesh_reports_failure_and_preserves_existing_export(self):
        source = self.folder / "broken.ply"
        source.write_text("not a PLY mesh", encoding="utf-8")
        destination = self.folder / "solidworks_ready.stl"
        destination.write_bytes(b"previous export")
        with self.assertRaisesRegex(RuntimeError, "Mesh cleanup failed"):
            export_solidworks_stl(source, lambda message: None)
        self.assertEqual(destination.read_bytes(), b"previous export")
        self.assertIn("ERROR:", (self.folder / "mesh_cleanup.log").read_text(encoding="utf-8"))


class ExportInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.app = MeshApp(Path("config.toml"))
        self.app.withdraw()
        self.addCleanup(self.app.destroy)

    def test_export_enabled_when_idle_disabled_during_work(self):
        self.app._update_run_button()
        self.assertEqual(str(self.app.export_button["state"]), "normal")
        self.app.busy = True
        self.app._update_run_button()
        self.assertEqual(str(self.app.export_button["state"]), "disabled")

    def test_export_uses_current_mesh_without_a_picker(self):
        self.app.mesh_path = Path("object_mesh.ply")
        result = Path("solidworks_ready.stl")
        with patch("turntable_mesh.app.export_solidworks_stl", return_value=result) as exporter:
            with patch("turntable_mesh.app.filedialog.askopenfilename") as picker:
                with patch.object(self.app, "_start_worker", side_effect=lambda work: work()):
                    self.app._export_stl()
        picker.assert_not_called()
        self.assertEqual(exporter.call_args.args[0], self.app.mesh_path)
        self.assertEqual(self.app.events.get_nowait(), ("exported", result))

    def test_saved_mesh_picker_can_be_cancelled(self):
        with patch("turntable_mesh.app.filedialog.askopenfilename", return_value=""):
            with patch.object(self.app, "_start_worker") as worker:
                self.app._export_stl()
        worker.assert_not_called()

    def test_saved_mesh_selection_is_exported(self):
        selected = str(Path.cwd() / "previous.ply")
        with patch("turntable_mesh.app.filedialog.askopenfilename", return_value=selected):
            with patch("turntable_mesh.app.export_solidworks_stl", return_value=Path("ready.stl")) as exporter:
                with patch.object(self.app, "_start_worker", side_effect=lambda work: work()):
                    self.app._export_stl()
        self.assertEqual(exporter.call_args.args[0], Path(selected))

    def test_export_error_reenables_button_for_retry(self):
        self.app.busy = True
        self.app.events.put(("export_error", RuntimeError("cleanup failed")))
        with patch("turntable_mesh.app.messagebox.showerror"):
            self.app._poll_events()
        self.assertFalse(self.app.busy)
        self.assertEqual(str(self.app.export_button["state"]), "normal")
        self.assertIn("export failed", self.app.stl_var.get())
        with self.assertRaises(queue.Empty):
            self.app.events.get_nowait()


class PackagedCleanupTests(unittest.TestCase):
    def test_packaged_cleanup_reenters_executable_in_cleanup_mode(self):
        mesh = Path("input.ply")
        destination = Path("output.stl")
        with patch.object(sys, "executable", "TurntableMesh.exe"), patch.object(sys, "frozen", True, create=True):
            self.assertEqual(
                _cleanup_command(mesh, destination),
                ["TurntableMesh.exe", "--cleanup", "input.ply", "output.stl"],
            )


if __name__ == "__main__":
    unittest.main()
