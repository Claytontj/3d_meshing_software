# turntable-mesh

A Python command-line project skeleton for producing an **object-only** mesh from a turntable video. The processing stages are named stubs and do not yet extract frames, create masks, run COLMAP, or export a mesh.

## Setup

1. Install Python 3.11 or newer and [COLMAP](https://colmap.github.io/).
2. From this folder, create and activate a virtual environment, then install the project:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -e .
   ```

3. Edit `config.toml`. Set the video path, frame sampling rate in frames per second, fixed crop `[x, y, width, height]` in source-video pixels, maximum dimension of each cropped frame, COLMAP executable, and output folder. Relative video and output paths are resolved from the config file's folder.

The intended command is `turntable-mesh --config config.toml`. It currently stops at the first unimplemented stage and reports which stage is pending.

## Intended workflow

1. `extract_frames`: sample the video at the configured rate, apply the same fixed crop to every frame, and resize within the maximum dimension.
2. `prepare_object_masks`: use manually prepared object masks for each extracted frame. Masks should keep the target object and exclude the turntable base and background. Automatic segmentation is outside the current scope.
3. `run_colmap`: use the frames and object masks for feature extraction, reconstruction, and meshing through the configured COLMAP executable.
4. `collect_mesh_output`: place the final **object-only** mesh in the output folder.

Tracking the turntable base may be explored later as an aid to camera motion estimation. The delivered mesh should still contain only the object. This project has no GUI.
