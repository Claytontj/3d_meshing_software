"""Small Tk desktop interface for preparing masks and running COLMAP."""

import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .config import Config, load_config, save_config
from .mesh_cleanup import StlExports, export_solidworks_stl
from .pipeline import (
    collect_mesh_output,
    extract_frames,
    new_workspace,
    prepare_object_masks,
    run_colmap,
)


class MeshApp(tk.Tk):
    def __init__(self, config_path: Path) -> None:
        super().__init__()
        self.config_path = config_path.resolve()
        self.config = load_config(self.config_path)
        self.title("Turntable Mesh")
        self.geometry("1120x900")
        self.minsize(850, 720)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.workspace: Path | None = None
        self.prepared_config: Config | None = None
        self.frames: list[Path] = []
        self.masks: list[Path] = []
        self.frame_index = 0
        self.current_mask: np.ndarray | None = None
        self.current_image: np.ndarray | None = None
        self.preview: ImageTk.PhotoImage | None = None
        self.preview_scale = 1.0
        self.preview_origin = (0, 0)
        self.mask_dirty = False
        self.busy = False
        self.selected_video = False
        self.run_started = False
        self.mesh_path: Path | None = None
        self.direct_stl_path: Path | None = None
        self.stl_path: Path | None = None
        self._build()
        self.after(100, self._poll_events)

    def _build(self) -> None:
        settings = ttk.LabelFrame(self, text="Settings")
        settings.pack(fill="x", padx=10, pady=8)
        self.video_var = tk.StringVar(value=str(self.config.video_path or ""))
        self.background_var = tk.StringVar(value=str(self.config.background_image or ""))
        self.fps_var = tk.StringVar(value=str(self.config.frame_sampling_rate))
        self.crop_vars = [tk.StringVar(value=str(value)) for value in self.config.crop]
        self.max_var = tk.StringVar(value=str(self.config.max_image_dimension))
        self.colmap_var = tk.StringVar(value=self.config.colmap_executable)
        self.output_var = tk.StringVar(value=str(self.config.output_folder))

        ttk.Label(settings, text="Video").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.video_var, state="readonly").grid(row=0, column=1, columnspan=7, sticky="ew", padx=4)
        ttk.Button(settings, text="Choose video", command=self._choose_video).grid(row=0, column=8, padx=4)
        ttk.Label(settings, text="Empty background").grid(row=1, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.background_var).grid(row=1, column=1, columnspan=7, sticky="ew", padx=4)
        ttk.Button(settings, text="Browse", command=self._choose_background).grid(row=1, column=8, padx=4)
        ttk.Label(settings, text="FPS").grid(row=2, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.fps_var, width=8).grid(row=2, column=1, sticky="w")
        for column, (label, variable) in enumerate(zip(("Crop X", "Y", "Width", "Height"), self.crop_vars, strict=True), 2):
            ttk.Label(settings, text=label).grid(row=2, column=column, sticky="e")
            ttk.Entry(settings, textvariable=variable, width=7).grid(row=3, column=column, sticky="w")
        ttk.Label(settings, text="Max dimension").grid(row=2, column=6, sticky="e")
        ttk.Entry(settings, textvariable=self.max_var, width=8).grid(row=3, column=6, sticky="w")
        ttk.Label(settings, text="COLMAP").grid(row=4, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.colmap_var).grid(row=4, column=1, columnspan=8, sticky="ew", padx=4)
        ttk.Label(settings, text="Output").grid(row=5, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.output_var).grid(row=5, column=1, columnspan=7, sticky="ew", padx=4)
        ttk.Button(settings, text="Browse", command=self._choose_output).grid(row=5, column=8, padx=4)
        ttk.Button(settings, text="Save settings", command=self._save_settings).grid(row=6, column=8, pady=5)
        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(7, weight=1)

        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=10)
        self.prepare_button = ttk.Button(controls, text="1. Extract frames + generate masks", command=self._prepare)
        self.prepare_button.pack(side="left", padx=3)
        self.reviewed_var = tk.BooleanVar()
        ttk.Checkbutton(controls, text="I reviewed the masks and removed the base", variable=self.reviewed_var,
                        command=self._update_run_button).pack(side="left", padx=12)
        self.run_button = ttk.Button(controls, text="2. Run COLMAP", command=self._run, state="disabled")
        self.run_button.pack(side="right", padx=3)

        editor = ttk.LabelFrame(self, text="Object mask editor — left drag paints selected mode; right drag erases")
        editor.pack(fill="both", expand=True, padx=10, pady=8)
        toolbar = ttk.Frame(editor)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Previous", command=lambda: self._show_frame(self.frame_index - 1)).pack(side="left")
        ttk.Button(toolbar, text="Next", command=lambda: self._show_frame(self.frame_index + 1)).pack(side="left")
        self.frame_label = ttk.Label(toolbar, text="Choose a video, then generate masks")
        self.frame_label.pack(side="left", padx=12)
        self.paint_mode = tk.StringVar(value="keep")
        ttk.Radiobutton(toolbar, text="Keep object", variable=self.paint_mode, value="keep").pack(side="left")
        ttk.Radiobutton(toolbar, text="Erase base/background", variable=self.paint_mode, value="erase").pack(side="left")
        ttk.Label(toolbar, text="Brush").pack(side="left", padx=(12, 2))
        self.brush_size = tk.IntVar(value=20)
        ttk.Scale(toolbar, from_=2, to=100, variable=self.brush_size, orient="horizontal", length=120).pack(side="left")
        self.canvas = tk.Canvas(editor, bg="#252525", width=900, height=500, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.bind("<Button-1>", lambda event: self._paint(event, self.paint_mode.get()))
        self.canvas.bind("<B1-Motion>", lambda event: self._paint(event, self.paint_mode.get()))
        self.canvas.bind("<Button-3>", lambda event: self._paint(event, "erase"))
        self.canvas.bind("<B3-Motion>", lambda event: self._paint(event, "erase"))
        self.canvas.bind("<ButtonRelease-1>", lambda _event: self._save_mask())
        self.canvas.bind("<ButtonRelease-3>", lambda _event: self._save_mask())
        self.canvas.bind("<Configure>", lambda _event: self._render())

        result = ttk.Frame(self)
        result.pack(fill="x", padx=10)
        self.result_var = tk.StringVar(value="Mesh: waiting for a video")
        ttk.Label(result, textvariable=self.result_var).pack(side="left", fill="x", expand=True)
        ttk.Button(result, text="Open output folder", command=self._open_output).pack(side="right")
        export = ttk.Frame(self)
        export.pack(fill="x", padx=10, pady=4)
        self.export_button = ttk.Button(export, text="3. Export STL files", command=self._export_stl)
        self.export_button.pack(side="left")
        self.cleanup_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(export, text="Also create cleaned STL", variable=self.cleanup_var).pack(side="left", padx=8)
        self.stl_var = tk.StringVar(value="STL: use the current mesh, or choose a saved PLY")
        ttk.Label(export, textvariable=self.stl_var, wraplength=700).pack(side="left", padx=10)
        log_frame = ttk.LabelFrame(self, text="Progress and COLMAP commands")
        log_frame.pack(fill="both", padx=10, pady=8)
        self.log_text = tk.Text(log_frame, height=9, wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True)

    def _settings(self) -> Config:
        root = self.config_path.parent
        output = Path(self.output_var.get().strip()).expanduser()
        if not output.is_absolute():
            output = root / output
        config = Config(
            video_path=Path(self.video_var.get()) if self.video_var.get() else None,
            background_image=Path(self.background_var.get()) if self.background_var.get() else None,
            frame_sampling_rate=float(self.fps_var.get()),
            crop=tuple(int(value.get()) for value in self.crop_vars),  # type: ignore[arg-type]
            max_image_dimension=int(self.max_var.get()),
            colmap_executable=self.colmap_var.get().strip(),
            output_folder=output,
            config_root=root.resolve(),
        )
        config.validate()
        return config

    def _choose_video(self) -> None:
        if self.busy:
            return
        path = filedialog.askopenfilename(title="Choose a turntable video", filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All files", "*.*")])
        if path:
            self.video_var.set(path)
            self.selected_video = True
            self.frames = []
            self.masks = []
            self.workspace = None
            self.prepared_config = None
            self.run_started = False
            self.mesh_path = None
            self.direct_stl_path = None
            self.stl_path = None
            self.stl_var.set("STL: use the current mesh, or choose a saved PLY")
            self.current_image = None
            self.current_mask = None
            self.reviewed_var.set(False)
            self.result_var.set("Mesh: waiting for mask preparation")
            self._render()
            self._update_run_button()
            self._log(f"Selected video: {path}")

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.output_var.set(path)

    def _choose_background(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an empty turntable/background image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.background_var.set(path)

    def _save_settings(self) -> None:
        try:
            config = self._settings()
            save_config(self.config_path, config)
            self.config = config
            self._log(f"Saved settings: {self.config_path}")
        except (OSError, ValueError) as error:
            messagebox.showerror("Settings", str(error))

    def _start_worker(self, work: object) -> None:
        self.busy = True
        self.prepare_button.configure(state="disabled")
        self._update_run_button()
        threading.Thread(target=work, daemon=True).start()

    def _prepare(self) -> None:
        if not self.selected_video or self.busy:
            messagebox.showinfo("Choose video", "Choose a video before preparing frames.")
            return
        try:
            config = self._settings()
            if config.video_path is None or not config.video_path.is_file():
                raise FileNotFoundError("Choose an existing video.")
            if config.background_image is not None and not config.background_image.is_file():
                raise FileNotFoundError(f"Empty-background image not found: {config.background_image}")
        except (OSError, ValueError) as error:
            messagebox.showerror("Settings", str(error))
            return
        self.config = config
        self.prepared_config = config
        self.frames = []
        self.masks = []
        self.run_started = False
        self.mesh_path = None
        self.direct_stl_path = None
        self.stl_path = None
        self.stl_var.set("STL: waiting for reconstruction")
        self.reviewed_var.set(False)
        self._log("Preparing frames and initial masks...")

        def work() -> None:
            try:
                workspace = new_workspace(config)
                frames = extract_frames(config, workspace, self._log_from_worker)
                masks = prepare_object_masks(frames, workspace, self._log_from_worker, config.background_image)
                self.events.put(("prepared", (workspace, frames, masks)))
            except Exception as error:
                self.events.put(("error", error))

        self._start_worker(work)

    def _run(self) -> None:
        if self.busy or self.run_started or not self.reviewed_var.get() or not self.frames or self.workspace is None:
            return
        try:
            if self._settings() != self.prepared_config:
                raise ValueError("Settings changed since frame preparation. Generate frames and masks again.")
        except ValueError as error:
            messagebox.showerror("Settings", str(error))
            return
        if not self._save_mask():
            return
        self.run_started = True
        workspace, frames, masks, config = self.workspace, self.frames[:], self.masks[:], self.prepared_config
        assert config is not None
        self._log("Starting COLMAP reconstruction...")

        def work() -> None:
            try:
                mesh = run_colmap(frames, masks, config, workspace, self._log_from_worker)
                self.events.put(("finished", collect_mesh_output(mesh, self._log_from_worker)))
            except Exception as error:
                self.events.put(("error", error))

        self._start_worker(work)

    def _export_stl(self) -> None:
        if self.busy:
            return
        mesh = self.mesh_path
        if mesh is None:
            selected = filedialog.askopenfilename(
                title="Choose a reconstructed PLY mesh to clean",
                filetypes=[("PLY meshes", "*.ply")],
            )
            if not selected:
                return
            mesh = Path(selected)
        if self.cleanup_var.get():
            self._log(f"Exporting direct STL and optional cleaned STL: {mesh}")
        else:
            self._log(f"Exporting direct STL without cleanup: {mesh}")
        self.stl_path = None
        self.direct_stl_path = None
        self.stl_var.set("STL: exporting...")
        run_cleanup = self.cleanup_var.get()

        def work() -> None:
            try:
                exports = export_solidworks_stl(mesh, self._log_from_worker, run_cleanup=run_cleanup)
                self.events.put(("exported", exports))
            except Exception as error:
                self.events.put(("export_error", error))

        self._start_worker(work)

    def _update_run_button(self) -> None:
        ready = not self.busy and not self.run_started and bool(self.frames) and self.reviewed_var.get()
        self.run_button.configure(state="normal" if ready else "disabled")
        self.export_button.configure(state="disabled" if self.busy else "normal")

    def _show_frame(self, index: int) -> None:
        if not 0 <= index < len(self.frames):
            return
        if not self._save_mask():
            return
        self.frame_index = index
        image = cv2.imread(str(self.frames[index]))
        mask = cv2.imread(str(self.masks[index]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            messagebox.showerror("Mask editor", "Could not read a frame or mask.")
            return
        self.current_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.current_mask = mask
        self.frame_label.configure(text=f"Frame {index + 1} of {len(self.frames)}")
        self._render()

    def _render(self) -> None:
        self.canvas.delete("all")
        if self.current_image is None or self.current_mask is None:
            return
        display = self.current_image.copy()
        background = self.current_mask == 0
        display[background] = (display[background] * 0.35).astype(np.uint8)
        height, width = display.shape[:2]
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        self.preview_scale = min(canvas_width / width, canvas_height / height, 1.0)
        shown_width = max(1, round(width * self.preview_scale))
        shown_height = max(1, round(height * self.preview_scale))
        resized = Image.fromarray(display).resize((shown_width, shown_height), Image.Resampling.BILINEAR)
        self.preview = ImageTk.PhotoImage(resized)
        self.preview_origin = ((canvas_width - shown_width) // 2, (canvas_height - shown_height) // 2)
        self.canvas.create_image(*self.preview_origin, image=self.preview, anchor="nw")

    def _paint(self, event: tk.Event, mode: str) -> None:
        if self.current_mask is None or self.busy:
            return
        origin_x, origin_y = self.preview_origin
        x = int((event.x - origin_x) / self.preview_scale)
        y = int((event.y - origin_y) / self.preview_scale)
        height, width = self.current_mask.shape
        if not 0 <= x < width or not 0 <= y < height:
            return
        cv2.circle(self.current_mask, (x, y), max(1, self.brush_size.get()), 255 if mode == "keep" else 0, -1)
        self.mask_dirty = True
        self.reviewed_var.set(False)
        self._update_run_button()
        self._render()

    def _save_mask(self) -> bool:
        if self.mask_dirty and self.current_mask is not None and self.masks:
            if not cv2.imwrite(str(self.masks[self.frame_index]), self.current_mask):
                messagebox.showerror("Mask editor", "Could not save the edited mask.")
                return False
            else:
                self.mask_dirty = False
        return True

    def _log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _log_from_worker(self, message: str) -> None:
        self.events.put(("log", message))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._log(str(payload))
                elif kind == "prepared":
                    self.workspace, self.frames, self.masks = payload  # type: ignore[misc]
                    self.busy = False
                    self.prepare_button.configure(state="normal")
                    self._show_frame(0)
                    self._update_run_button()
                    self._log("Review and edit each mask, then check the review box.")
                elif kind == "finished":
                    self.mesh_path = payload  # type: ignore[assignment]
                    self.busy = False
                    self.prepare_button.configure(state="normal")
                    self.result_var.set(f"Mesh: {self.mesh_path}")
                    self._update_run_button()
                elif kind == "exported":
                    exports = payload  # type: ignore[assignment]
                    assert isinstance(exports, StlExports)
                    self.direct_stl_path = exports.direct
                    self.stl_path = exports.cleaned or exports.direct
                    self.busy = False
                    self.prepare_button.configure(state="normal")
                    if exports.cleaned:
                        self.stl_var.set(
                            f"Direct STL: {exports.direct}\nCleaned STL: {exports.cleaned}"
                        )
                    else:
                        self.stl_var.set(f"Direct STL: {exports.direct}\nCleaned STL: not created")
                    self._update_run_button()
                elif kind in ("error", "export_error"):
                    if kind == "export_error":
                        self.stl_var.set("STL: export failed; see the log and retry")
                    self.busy = False
                    self.prepare_button.configure(state="normal")
                    self._update_run_button()
                    self._log(f"ERROR: {payload}")
                    messagebox.showerror("Processing stopped", str(payload))
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _open_output(self) -> None:
        folder = self.stl_path.parent if self.stl_path else (
            self.workspace if self.workspace else self.config.output_folder
        )
        if folder.is_dir():
            os.startfile(folder)
        else:
            messagebox.showinfo("Output", "No output folder exists yet.")


def launch(config_path: Path) -> None:
    MeshApp(config_path).mainloop()
