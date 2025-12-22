# -*- coding: utf-8 -*-
"""
3D Tomography Reconstruction and Post-Processing Tool (Tkinter GUI)
(Fixed Version: GUI Parameter for Cutoff)

================================================================================
修正說明 (Fix Log):
1. [Logic] 修正 _convert_to_cartesian 以輸出正方形切片 (N x N x Z)。
2. [Logic] 當 Tolerance <= 0 時，跳過 Polar/FFT 步驟。
3. [Algorithm] FFT 濾波器採用「直接賦值法」保護中心低頻。
4. [GUI] 新增 "FFT_CUTOFF_DIVISOR" (Ring Removal Cutoff) 參數於介面中。
   - 解決了修改程式碼無效的問題 (防止只改到 CPU 區塊卻跑 GPU 的狀況)。
   - 預設值為 64 (即 1/64 影像寬度)。
================================================================================
"""

# --- Standard Library Imports ---
import os
import sys
import glob
import time
import threading
from typing import Optional, Dict, Any

# --- GUI Library Imports ---
import tkinter as tk
from tkinter import filedialog, messagebox, LabelFrame, Frame, scrolledtext

# --- Third-Party Scientific Library Imports ---
import numpy as np
import tifffile
import astra
import cv2
from skimage.transform import downscale_local_mean

# --- Attempt to import CuPy for GPU acceleration ---
try:
    import cupy as cp
    from cupyx.scipy.ndimage import median_filter as cupy_median_filter
    from cupyx.scipy import fft as cupy_fft

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None  # Set to None to prevent NameError
    cupy_median_filter = None
    cupy_fft = None

# --- Import SciPy for CPU-based filtering and as a fallback ---
from scipy.ndimage import median_filter, gaussian_filter
from scipy import fft as scipy_fft

# --- Default Configuration ---
DEFAULT_CONFIG = {
    "FILE_PATTERN": '*.tif',
    "BINNING_FACTOR": 1,
    "TOLERANCE_VALUE": 40,
    "FFT_CUTOFF_DIVISOR": 64.0,  # 新增：控制水平保護區的寬度 (1/N)
    "MEDIAN_FILTER_RADIUS": 3,
    "GAUSSIAN_SIGMA": 2.5,
    "SAVE_RECON_RAW": False,
    "SAVE_POLAR_RAW": False,
    "SAVE_POLAR_FILTERED": False,
    "SAVE_RECON_FILTERED": False,
    "SAVE_RECON_FILTERED_MEDIAN": False,
    "SAVE_RECON_FILTERED_MEDIAN_GAUSS": True,
}


# ==============================================================================
# Helper Classes
# ==============================================================================

class TextRedirector:
    """Redirects print statements to a Tkinter Text widget."""

    def __init__(self, widget: scrolledtext.ScrolledText):
        self.widget = widget

    def write(self, text: str):
        def _write():
            self.widget.configure(state='normal')
            self.widget.insert(tk.END, text)
            self.widget.see(tk.END)
            self.widget.configure(state='disabled')

        self.widget.after_idle(_write)

    def flush(self):
        pass


# ==============================================================================
# Backend Processing Logic
# ==============================================================================

class TomoPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_volume: Optional[np.ndarray] = None
        self.projections: Optional[np.ndarray] = None
        self.angles_rad: Optional[np.ndarray] = None
        self.input_dir: str = ""
        self.base_name: str = ""
        self.file_params: Dict[str, Any] = {}
        self.step_timings: Dict[str, float] = {}

        self.processing_steps = [
            {"name": "load_projections", "method": self._load_projections},
            {"name": "reconstruct_raw", "method": self._reconstruct_volume, "save_key": "SAVE_RECON_RAW",
             "type": "RECON"},
            {"name": "to_polar_raw", "method": self._convert_to_polar, "save_key": "SAVE_POLAR_RAW", "type": "POLAR"},
            {"name": "fft_filter", "method": self._apply_fft_filter, "save_key": "SAVE_POLAR_FILTERED",
             "type": "POLAR"},
            {"name": "to_cartesian_filtered", "method": self._convert_to_cartesian, "save_key": "SAVE_RECON_FILTERED",
             "type": "RECON"},
            {"name": "median_filter", "method": self._apply_median_filter, "save_key": "SAVE_RECON_FILTERED_MEDIAN",
             "type": "RECON"},
            {"name": "gaussian_filter", "method": self._apply_gaussian_filter,
             "save_key": "SAVE_RECON_FILTERED_MEDIAN_GAUSS", "type": "RECON"},
        ]

    def run(self, input_dir: str) -> bool:
        self.input_dir = input_dir
        print("--- Starting 3D Reconstruction and Post-Processing Task ---\n")
        self.base_name = os.path.basename(self.input_dir)
        self.file_params = {"binning": self.config["BINNING_FACTOR"]}
        self.step_timings.clear()
        total_start_time = time.time()

        # Dynamic Step Logic: Skip Polar/FFT if T <= 0
        tolerance = self.config.get("TOLERANCE_VALUE", 0)
        active_steps = list(self.processing_steps)

        if tolerance <= 0:
            print("Tolerance <= 0: Skipping Ring Artifact Removal (Polar/FFT steps).")
            active_steps = [
                s for s in active_steps
                if s["name"] not in ["to_polar_raw", "fft_filter", "to_cartesian_filtered"]
            ]

        # Find last step
        last_step_to_run = None
        for step in reversed(active_steps):
            if self.config.get(step.get("save_key", "")):
                last_step_to_run = step["name"]
                break

        if not last_step_to_run:
            print("Warning: No output selected or step skipped.")
            return False

        # Execute
        for step in active_steps:
            print(f"\n--- Step: {step['name']} ---")
            step_start_time = time.time()
            success = step["method"]()
            self.step_timings[step['name']] = time.time() - step_start_time

            if not success:
                print(f"Error: Step {step['name']} failed.")
                return False

            save_key = step.get("save_key")
            if save_key and self.config.get(save_key):
                self._save_volume(step["type"])

            if step["name"] == last_step_to_run:
                print("\nAll selected processing steps have been completed.")
                break

        self.step_timings["Total Processing Time"] = time.time() - total_start_time
        self._print_timing_summary()
        return True

    def _print_timing_summary(self):
        print("\n\n--- Processing Time Summary ---")
        for name, duration in self.step_timings.items():
            print(f"{name:<35}: {duration:>7.2f} seconds")
        print("---------------------------------------------")

    def _load_projections(self) -> bool:
        print(f"Searching for '{self.config['FILE_PATTERN']}' files in '{self.input_dir}'...")
        search_path = os.path.join(self.input_dir, self.config['FILE_PATTERN'])
        file_list = sorted(glob.glob(search_path))

        if not file_list:
            print(f"Error: No files found at '{search_path}'.")
            return False

        num_files = len(file_list)
        print(f"Found {num_files} projection files.")
        self.angles_rad = np.deg2rad(np.linspace(0, 180, num_files, endpoint=False, dtype=np.float32))

        print(f"Reading images with {self.config['BINNING_FACTOR']}x{self.config['BINNING_FACTOR']} binning...")
        first_image = tifffile.imread(file_list[0])
        bin_factor = self.config['BINNING_FACTOR']

        binned_shape = (downscale_local_mean(first_image, (bin_factor, bin_factor)).shape
                        if bin_factor > 1 else first_image.shape)

        binned_height, binned_width = binned_shape
        print(f"Binned dimensions: {binned_shape}")

        self.projections = np.zeros((num_files, binned_height, binned_width), dtype=np.float32)

        update_interval = max(1, num_files // 10)
        for i, file_path in enumerate(file_list):
            image = tifffile.imread(file_path).astype(np.float32)
            if bin_factor > 1:
                self.projections[i, :, :] = downscale_local_mean(image, (bin_factor, bin_factor))
            else:
                self.projections[i, :, :] = image

            if (i + 1) % update_interval == 0 or (i + 1) == num_files:
                print(f"  > Loading: {i + 1} / {num_files} ({(i + 1) / num_files:.0%})")
        return True

    def _reconstruct_volume(self) -> bool:
        print("Starting ASTRA GPU reconstruction...")
        _, height, width = self.projections.shape
        reconstructed_slices = []

        # Geometry: Width x Width
        vol_geom = astra.create_vol_geom(width, width)
        proj_geom = astra.create_proj_geom('parallel', 1.0, width, self.angles_rad)

        update_interval = max(1, height // 10)
        for y in range(height):
            sinogram_slice = self.projections[:, y, :]
            sino_id = astra.data2d.create('-sino', proj_geom, sinogram_slice)
            reco_id = astra.data2d.create('-vol', vol_geom)

            cfg = astra.astra_dict('FBP_CUDA')
            cfg['ReconstructionDataId'] = reco_id
            cfg['ProjectionDataId'] = sino_id
            alg_id = astra.algorithm.create(cfg)

            astra.algorithm.run(alg_id)
            reco_slice = astra.data2d.get(reco_id)
            reconstructed_slices.append(reco_slice)

            astra.algorithm.delete(alg_id)
            astra.data2d.delete(reco_id)
            astra.data2d.delete(sino_id)

            if (y + 1) % update_interval == 0 or (y + 1) == height:
                print(f"  > Reconstructing: {y + 1} / {height} ({(y + 1) / height:.0%})")

        self.data_volume = np.stack(reconstructed_slices, axis=0).astype(np.float32)
        print("Reconstruction complete.")
        return True

    def _convert_to_polar(self) -> bool:
        print("Converting volume to Polar coordinates...")
        num_slices, y_dim, x_dim = self.data_volume.shape
        center = (x_dim / 2.0, y_dim / 2.0)
        max_radius = x_dim / 2.0

        oversampled_width = int(max_radius)
        oversampled_height = int(x_dim * np.pi)
        if oversampled_height % 2 != 0: oversampled_height += 1
        dsize = (oversampled_width, oversampled_height)

        polar_slices = []
        update_interval = max(1, num_slices // 10)
        for i, s in enumerate(self.data_volume):
            polar_slices.append(cv2.warpPolar(s, dsize, center, max_radius, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS))
            if (i + 1) % update_interval == 0 or (i + 1) == num_slices:
                print(f"  > Polar Transform: {i + 1} / {num_slices} ({(i + 1) / num_slices:.0%})")

        self.data_volume = np.stack(polar_slices, axis=0).astype(np.float32)
        print("Polar conversion complete.")
        return True

    def _apply_fft_filter(self) -> bool:
        """Applies combined Vertical Notch + Direct Assignment for Center Protection."""
        tolerance = self.config["TOLERANCE_VALUE"]
        cutoff_divisor = self.config["FFT_CUTOFF_DIVISOR"]  # 從 GUI 獲取參數
        self.file_params["tolerance"] = int(tolerance)

        if tolerance <= 0:
            print("Tolerance <= 0, skipping FFT.")
            return True

        processed_tolerance = 0.000025 * tolerance
        num_slices = self.data_volume.shape[0]
        update_interval = max(1, num_slices // 10)

        use_cpu_fallback = False
        if CUPY_AVAILABLE:
            try:
                gpu_volume = cp.asarray(self.data_volume)
                filtered_gpu_volume = cp.empty_like(gpu_volume, dtype=cp.float32)

                rows, cols = gpu_volume[0].shape

                # --- Filter Parameter Calculation ---
                sigma = rows * processed_tolerance
                if cutoff_divisor <= 0: cutoff_divisor = 64.0  # Prevent division by zero
                cutoff_threshold = cols / cutoff_divisor
                if cutoff_threshold < 1.0: cutoff_threshold = 1.0

                print(
                    f"FFT Filter Config -> T: {tolerance}, Sigma (Y-width): {sigma:.2f}, Cutoff (X-Protection, 1/{cutoff_divisor}): {cutoff_threshold:.2f} px")

                # Coordinates
                y_coords = cp.arange(rows) - (rows // 2)
                x_coords = cp.arange(cols) - (cols // 2)

                # 1. Create Base Vertical Notch Filter (1D -> 2D Broadcast)
                notch_y_1d = 1.0 - cp.exp(-(y_coords ** 2) / (2 * sigma ** 2))
                notch_y_1d = notch_y_1d.reshape(-1, 1)

                # Create the full 2D filter by tiling the 1D notch across all columns
                filter_2d = cp.tile(notch_y_1d, (1, cols))

                # 2. Direct Assignment for Protection
                center_indices = cp.abs(x_coords) <= cutoff_threshold
                filter_2d[:, center_indices] = 1.0

                print(f"Debug: Filter Mean Value: {cp.mean(filter_2d):.5f}")

                for i, polar_slice_gpu in enumerate(gpu_volume):
                    fft_slice = cupy_fft.fftshift(cupy_fft.fft2(polar_slice_gpu))
                    fft_slice *= filter_2d
                    slice_ifft = cupy_fft.ifft2(cupy_fft.ifftshift(fft_slice))
                    filtered_gpu_volume[i] = cp.real(slice_ifft)
                    if (i + 1) % update_interval == 0 or (i + 1) == num_slices:
                        print(f"  > GPU FFT Filter: {i + 1} / {num_slices} ({(i + 1) / num_slices:.0%})")

                self.data_volume = cp.asnumpy(filtered_gpu_volume)
                cp.get_default_memory_pool().free_all_blocks()

            except Exception as e:
                print(f"GPU failed: {e}. Fallback to CPU.")
                if cp: cp.get_default_memory_pool().free_all_blocks()
                use_cpu_fallback = True

        if not CUPY_AVAILABLE or use_cpu_fallback:
            filtered_slices = []

            # Pre-calc Filter on CPU
            sample_slice = self.data_volume[0]
            rows, cols = sample_slice.shape

            sigma = rows * processed_tolerance
            if cutoff_divisor <= 0: cutoff_divisor = 64.0
            cutoff_threshold = cols / cutoff_divisor
            if cutoff_threshold < 1.0: cutoff_threshold = 1.0

            print(
                f"FFT Filter Config (CPU) -> T: {tolerance}, Sigma: {sigma:.2f}, Cutoff (1/{cutoff_divisor}): {cutoff_threshold:.2f}")

            y_coords = np.arange(rows) - (rows // 2)
            x_coords = np.arange(cols) - (cols // 2)

            # 1. Base Vertical Notch
            notch_y_1d = 1.0 - np.exp(-(y_coords ** 2) / (2 * sigma ** 2))
            notch_y_1d = notch_y_1d.reshape(-1, 1)

            # Create full 2D filter
            filter_2d = np.tile(notch_y_1d, (1, cols))

            # 2. Direct Assignment Protection
            center_indices = np.abs(x_coords) <= cutoff_threshold
            filter_2d[:, center_indices] = 1.0

            print(f"Debug: Filter Mean Value: {np.mean(filter_2d):.5f}")

            for i, polar_slice in enumerate(self.data_volume):
                fft_slice = scipy_fft.fftshift(scipy_fft.fft2(polar_slice))
                fft_slice *= filter_2d
                slice_ifft = scipy_fft.ifft2(scipy_fft.ifftshift(fft_slice))
                filtered_slices.append(np.real(slice_ifft))
                if (i + 1) % update_interval == 0 or (i + 1) == num_slices:
                    print(f"  > CPU FFT Filter: {i + 1} / {num_slices} ({(i + 1) / num_slices:.0%})")
            self.data_volume = np.stack(filtered_slices, axis=0).astype(np.float32)

        print("FFT filtering complete.")
        return True

    def _convert_to_cartesian(self) -> bool:
        print("Converting back to Cartesian...")
        # FIX: Ensure square slice (width x width)
        _, _, width = self.projections.shape

        num_slices = self.data_volume.shape[0]

        center = (width / 2.0, width / 2.0)
        max_radius = width / 2.0
        dsize = (width, width)  # Square

        cartesian_slices = []
        update_interval = max(1, num_slices // 10)
        for i, s in enumerate(self.data_volume):
            cartesian_slices.append(
                cv2.warpPolar(s, dsize, center, max_radius, cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP))
            if (i + 1) % update_interval == 0 or (i + 1) == num_slices:
                print(f"  > Inverse Polar: {i + 1} / {num_slices} ({(i + 1) / num_slices:.0%})")

        self.data_volume = np.stack(cartesian_slices, axis=0).astype(np.float32)
        print("Inverse conversion complete.")
        return True

    def _apply_median_filter(self) -> bool:
        radius = self.config["MEDIAN_FILTER_RADIUS"]
        self.file_params["median"] = radius
        if radius <= 0:
            print("Median filter radius is 0, skipping.")
            return True

        z_radius = radius // 2
        xy_kernel = radius * 2 + 1
        z_kernel = z_radius * 2 + 1
        filter_shape = (z_kernel, xy_kernel, xy_kernel)

        use_cpu_fallback = False
        if CUPY_AVAILABLE:
            try:
                gpu_volume = cp.asarray(self.data_volume)
                filtered_gpu_volume = cupy_median_filter(gpu_volume, size=filter_shape)
                self.data_volume = cp.asnumpy(filtered_gpu_volume)
                cp.get_default_memory_pool().free_all_blocks()
            except Exception as e:
                print(f"GPU Median failed: {e}. Fallback to CPU.")
                if cp: cp.get_default_memory_pool().free_all_blocks()
                use_cpu_fallback = True

        if not CUPY_AVAILABLE or use_cpu_fallback:
            print(f"Starting CPU 3D Median Filter...")
            self.data_volume = median_filter(self.data_volume, size=filter_shape)

        print("Median Filter complete.")
        return True

    def _apply_gaussian_filter(self) -> bool:
        sigma = self.config["GAUSSIAN_SIGMA"]
        self.file_params["gauss"] = sigma
        if sigma <= 0:
            print("Gaussian sigma is 0, skipping.")
            return True

        print(f"Starting CPU 3D Gaussian Filter (Sigma: {sigma})...")
        self.data_volume = gaussian_filter(self.data_volume, sigma=sigma)
        print("Gaussian Filter complete.")
        return True

    def _save_volume(self, volume_type: str):
        output_dir = os.path.dirname(self.input_dir)
        z, y, x = self.data_volume.shape

        param_parts = []
        param_parts.append(f"B{self.file_params['binning']}")
        # Only add T if > 0 (handled by caller logic essentially, but checked here)
        if "tolerance" in self.file_params and self.file_params['tolerance'] > 0:
            param_parts.append(f"T{self.file_params['tolerance']}")
        if "median" in self.file_params and self.file_params['median'] > 0:
            param_parts.append(f"M{self.file_params['median']}")
        if "gauss" in self.file_params and self.file_params['gauss'] > 0:
            param_parts.append(f"G{self.file_params['gauss']:.1f}".replace('.0', ''))
        param_parts.append(f"{x}x{y}x{z}")

        param_str = "_".join(param_parts)
        output_filename = f"{self.base_name}_{volume_type}_{param_str}.tif"
        output_path = os.path.join(output_dir, output_filename)

        print(f"Saving {volume_type} to '{output_path}'...")
        try:
            tifffile.imwrite(output_path, self.data_volume.astype(np.float32))
            print(f"-> Saved.")
        except Exception as e:
            print(f"Error saving: {e}")


# ==============================================================================
# Graphical User Interface (GUI)
# ==============================================================================

class App(tk.Tk):
    def __init__(self, defaults: Dict[str, Any]):
        super().__init__()
        self.defaults = defaults
        self.title("3D Tomography Reconstruction Tool")
        self.resizable(True, True)

        self.entries: Dict[str, tk.StringVar] = {}
        self.save_vars: Dict[str, tk.BooleanVar] = {}
        self.folder_path = tk.StringVar(value="No folder selected")

        self._create_widgets()
        self._center_window()

        sys.stdout = TextRedirector(self.log_text)
        sys.stderr = TextRedirector(self.log_text)

        print("Tool initialized.")
        if CUPY_AVAILABLE:
            print("CuPy found (GPU Acceleration Enabled).")
        else:
            print("CuPy NOT found (CPU Mode - Slower).")

    def _create_widgets(self):
        main_frame = Frame(self, padx=10, pady=10)
        main_frame.pack(fill="both", expand=True)
        main_frame.rowconfigure(3, weight=1)
        main_frame.columnconfigure(0, weight=1)

        param_frame = LabelFrame(main_frame, text="Filter Parameters", padx=10, pady=10)
        param_frame.grid(row=0, column=0, padx=10, pady=5, sticky='ew')
        param_defs = {
            "BINNING_FACTOR": "Binning Factor:",
            "TOLERANCE_VALUE": "FFT Filter Tolerance:",
            "FFT_CUTOFF_DIVISOR": "Ring Removal Cutoff (1/N):",  # 新增 GUI 欄位
            "MEDIAN_FILTER_RADIUS": "Median Filter Radius:",
            "GAUSSIAN_SIGMA": "Gaussian Sigma:"
        }
        for i, (key, label) in enumerate(param_defs.items()):
            tk.Label(param_frame, text=label).grid(row=i, column=0, padx=5, pady=3, sticky='w')
            var = tk.StringVar(value=str(self.defaults.get(key, "")))
            entry = tk.Entry(param_frame, width=15, textvariable=var)
            entry.grid(row=i, column=1, padx=5, pady=3)
            self.entries[key] = var

        save_frame = LabelFrame(main_frame, text="Select Files to Output", padx=10, pady=10)
        save_frame.grid(row=1, column=0, padx=10, pady=5, sticky='ew')
        save_defs = {
            "SAVE_RECON_RAW": "1. Raw Reconstructed Volume", "SAVE_POLAR_RAW": "2. Raw Polar Volume",
            "SAVE_POLAR_FILTERED": "3. FFT-Filtered Polar Volume",
            "SAVE_RECON_FILTERED": "4. FFT-Filtered Reconstructed Volume",
            "SAVE_RECON_FILTERED_MEDIAN": "5. Median-Filtered Volume",
            "SAVE_RECON_FILTERED_MEDIAN_GAUSS": "6. Gaussian-Filtered Volume"
        }
        for key, label in save_defs.items():
            var = tk.BooleanVar(value=self.defaults.get(key, False))
            chk = tk.Checkbutton(save_frame, text=label, variable=var)
            chk.pack(anchor="w")
            self.save_vars[key] = var

        run_frame = LabelFrame(main_frame, text="Execution Control", padx=10, pady=10)
        run_frame.grid(row=2, column=0, padx=10, pady=5, sticky='ew')
        folder_select_frame = Frame(run_frame)
        folder_select_frame.pack(fill='x', pady=5)
        tk.Button(folder_select_frame, text="Select Folder...", command=self._select_folder).pack(side="left")
        folder_label = tk.Label(folder_select_frame, textvariable=self.folder_path, fg="blue", wraplength=300,
                                justify="left")
        folder_label.pack(side="left", padx=10)
        button_frame = Frame(run_frame)
        button_frame.pack(fill='x', pady=10)
        self.start_button = tk.Button(button_frame, text="Start", width=12, command=self._start_processing_thread)
        self.start_button.pack(side="left", padx=10)
        tk.Button(button_frame, text="Exit", width=12, command=self.destroy).pack(side="right", padx=10)

        log_frame = LabelFrame(main_frame, text="Log", padx=10, pady=10)
        log_frame.grid(row=3, column=0, padx=10, pady=5, sticky='nsew')
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, state='disabled', height=15, wrap=tk.WORD, bg="#f0f0f0")
        self.log_text.grid(row=0, column=0, sticky='nsew')

    def _select_folder(self):
        path = filedialog.askdirectory(parent=self, title="Select folder")
        if path: self.folder_path.set(path)

    def _start_processing_thread(self):
        input_dir = self.folder_path.get()
        if not os.path.isdir(input_dir):
            messagebox.showerror("Error", "Invalid folder.")
            return

        try:
            user_params = {
                "BINNING_FACTOR": int(self.entries["BINNING_FACTOR"].get()),
                "TOLERANCE_VALUE": float(self.entries["TOLERANCE_VALUE"].get()),
                "FFT_CUTOFF_DIVISOR": float(self.entries["FFT_CUTOFF_DIVISOR"].get()),  # 讀取新參數
                "MEDIAN_FILTER_RADIUS": int(self.entries["MEDIAN_FILTER_RADIUS"].get()),
                "GAUSSIAN_SIGMA": float(self.entries["GAUSSIAN_SIGMA"].get()),
            }
            for key, var in self.save_vars.items():
                user_params[key] = var.get()
        except ValueError:
            messagebox.showerror("Input Error", "Check numeric values.")
            return

        self.start_button.config(state="disabled")
        config = {**self.defaults, **user_params}
        thread = threading.Thread(target=self._run_pipeline, args=(config, input_dir))
        thread.daemon = True
        thread.start()

    def _run_pipeline(self, config, input_dir):
        pipeline = TomoPipeline(config)
        success = pipeline.run(input_dir)
        self.after_idle(self._on_processing_complete, success)

    def _on_processing_complete(self, success: bool):
        if success:
            messagebox.showinfo("Success", "Task completed!")
        else:
            messagebox.showerror("Error", "Task failed.")
        self.start_button.config(state="normal")

    def _center_window(self):
        self.update_idletasks()
        min_w, min_h = 550, 750
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x, y = (sw - min_w) // 2, (sh - min_h) // 2
        self.geometry(f'{min_w}x{min_h}+{x}+{y}')


def main():
    try:
        app = App(DEFAULT_CONFIG)
        app.mainloop()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()