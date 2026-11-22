# turntable-mesh

A small Windows desktop prototype that extracts frames from a turntable video, creates editable object masks, and uses COLMAP for sparse and dense reconstruction. The final PLY is built from points fused through the reviewed object masks; the turntable base should be erased from every mask. A separate cleanup step uses PyMeshLab to export a simplified binary STL for viewing in SolidWorks. Base tracking is a possible later feature, not part of the mesh.

## Setup

For development from source, use Python 3.11 or newer. The source install includes the Python dependencies and the local package; COLMAP can be selected in the app's settings or placed at `colmap\\COLMAP.bat`.

```powershell
conda activate 3d_mesh_app
python -m pip install -r requirements.txt
turntable-mesh --config config.toml
```

For an end-user installation, download the latest `TurntableMesh-Windows-x64.zip` from the [GitHub Releases page](https://github.com/Claytontj/3d_meshing_software/releases). Extract it and run `TurntableMesh.exe`. The release package contains the Python application, its Python dependencies, and a pinned official Windows COLMAP binary; Python, pip, and Git are not required.

The Windows package is built by `.github/workflows/build-windows.yml`. To publish one, push a version tag such as `v0.1.0`:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

The packaged COLMAP build uses CUDA for dense reconstruction, so the end user's computer needs a compatible NVIDIA driver and GPU. COLMAP's Windows binaries are not officially code-signed, so Windows Defender may show a warning the first time they are run.

For a direct launch without installing the package, run from this folder:

```powershell
$env:PYTHONPATH = "src"
python -m turntable_mesh --config config.toml
```

`config.toml` holds the video path, frames per second, fixed crop `[x, y, width, height]`, maximum frame dimension, COLMAP launcher, and output folder. A crop width and height of zero use the whole frame. The desktop form lets you edit and save these settings. Relative paths are resolved from the config file's folder. No video is selected by default.

## Workflow

1. Choose a video in the app. No reconstruction begins on launch or video selection.
2. Adjust FPS and crop, then click **Extract frames + generate masks**. OpenCV GrabCut creates an initial mask per frame.
3. Use **Previous** and **Next** to inspect every mask. Left drag paints the selected **Keep object** or **Erase base/background** mode; right drag always erases. Dark areas are excluded. Changes save when you release the brush or change frames.
4. Check **I reviewed the masks and removed the base**, then click **Run COLMAP**. This runs feature extraction with masks, exhaustive matching, sparse mapping, image undistortion, dense PatchMatch stereo, stereo fusion with masks, and Poisson meshing.
5. The app displays the path to `object_mesh.ply`. Each timestamped output folder also holds frames, masks, COLMAP data, `colmap_commands.log`, and `colmap_output.log`. All COLMAP commands and their output appear in the app log.
6. Click **Export STL files**. The app always saves `solidworks_direct.stl`, a binary STL converted directly from the original PLY. The **Also create cleaned STL** option is enabled by default and additionally saves `solidworks_ready.stl` after removing unreferenced vertices, duplicate faces, and zero-area faces, repairing non-manifold geometry, simplifying toward 50,000 faces, and repairing again. Both files are suitable for SolidWorks mesh import; use the direct file if the cleanup version is not usable. The original PLY is preserved. If no current mesh exists, the button opens a file picker so you can export a previously generated PLY without rerunning COLMAP. Progress appears in the app and in `mesh_cleanup.log` beside the source mesh. Cleanup runs in a separate process so a native PyMeshLab failure does not close the app; the direct STL remains available if optional cleanup fails.

Open the STL in SolidWorks using **File > Open** and select the STL mesh file type. This export is a triangle mesh for viewing and mesh workflows; it does not create a native parametric SolidWorks part or guarantee a watertight solid. No scale conversion is applied: reconstruction has no calibrated physical units, so verify the scale and import units in SolidWorks. See the [SolidWorks mesh import documentation](https://help.solidworks.com/2025/english/SolidWorks/sldworks/c_stl_files.htm?id=18.15.29).

Mask accuracy determines whether the mesh excludes the base. GrabCut is only an initial suggestion; review and correct each frame before reconstruction. Dense reconstruction can require substantial GPU memory and time. If COLMAP cannot find enough matching object features, the app reports the failed stage and leaves the working files and logs in the output folder.
