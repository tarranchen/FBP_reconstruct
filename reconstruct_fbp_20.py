# -*- coding: utf-8 -*-

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
# These values populate the GUI fields on startup.
DEFAULT_CONFIG = {
    "FILE_PATTERN": '*.tif',
    "BINNING_FACTOR": 1,
    "TOLERANCE_VALUE": 40,
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
        """Writes text to the widget in a thread-safe manner."""

        def _write():
            self.widget.configure(state='normal')
            self.widget.insert(tk.END, text)
            self.widget.see(tk.END)  # Auto-scroll to the bottom
            self.widget.configure(state='disabled')

        self.widget.after_idle(_write)

    def flush(self):
        """Flush method is required but does nothing here."""
        pass


# ==============================================================================
# Backend Processing Logic
# ==============================================================================

class TomoPipeline:
    """
    Encapsulates the entire 3D tomography reconstruction and post-processing pipeline.
    This class handles all computational tasks, independent of the GUI.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the TomoPipeline.
        Args:
            config: A dictionary containing all user-defined parameters.
        """
        self.config = config
        self.data_volume: Optional[np.ndarray] = None
        self.projections: Optional[np.ndarray] = None
        self.angles_rad: Optional[np.ndarray] = None
        self.input_dir: str = ""
        self.base_name: str = ""
        self.file_params: Dict[str, Any] = {}
        self.step_timings: Dict[str, float] = {}

        # Define the processing steps in order
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
        """
        Executes the main processing pipeline.
        Args:
            input_dir: Path to the folder containing projection images.
        Returns:
            True if the pipeline completes successfully, False otherwise.
        """
        self.input_dir = input_dir
        print("--- Starting 3D Reconstruction and Post-Processing Task ---\n")
        self.base_name = os.path.basename(self.input_dir)
        self.file_params = {"binning": self.config["BINNING_FACTOR"]}
        self.step_timings.clear()
        total_start_time = time.time()

        # --- [邏輯變更] 判斷是否完全略過去環狀偽影步驟 ---
        # 如果 Tolerance <= 0，我們從執行清單中完全移除 Polar 相關步驟
        tolerance = self.config.get("TOLERANCE_VALUE", 0)
        active_steps = list(self.processing_steps)  # 複製原始步驟清單

        if tolerance <= 0:
            print("Tolerance <= 0: 跳過去環狀偽影步驟 (Polar <-> Cartesian 轉換與 FFT)。")
            # 過濾掉極座標轉換與 FFT 濾波的步驟
            active_steps = [
                s for s in active_steps
                if s["name"] not in ["to_polar_raw", "fft_filter", "to_cartesian_filtered"]
            ]
            # [修改] 移除手動設定 tolerance 為 0，讓檔名不顯示 T0，保持與 M=0/G=0 一致
            # self.file_params["tolerance"] = 0

        # Smart execution: find the last step the user wants to save.
        last_step_to_run = None
        for step in reversed(active_steps):
            if self.config.get(step.get("save_key", "")):
                last_step_to_run = step["name"]
                break

        if not last_step_to_run:
            print("Warning: No output file was selected (or the selected step was skipped due to settings).")
            return False

        # Execute steps sequentially
        for step in active_steps:
            print(f"\n--- Step: {step['name']} ---")
            step_start_time = time.time()
            success = step["method"]()
            self.step_timings[step['name']] = time.time() - step_start_time

            if not success:
                print(f"Error: Step {step['name']} failed. Aborting pipeline.")
                return False

            # Save the volume if the corresponding checkbox was ticked
            save_key = step.get("save_key")
            if save_key and self.config.get(save_key):
                self._save_volume(step["type"])

            # Stop if the last required step has been completed
            if step["name"] == last_step_to_run:
                print("\nAll selected processing steps have been completed.")
                break

        self.step_timings["Total Processing Time"] = time.time() - total_start_time
        self._print_timing_summary()
        return True

    def _print_timing_summary(self):
        """Prints a summary of the time taken for each step."""
        print("\n\n--- Processing Time Summary ---")
        for name, duration in self.step_timings.items():
            print(f"{name:<35}: {duration:>7.2f} seconds")
        print("---------------------------------------------")

    def _load_projections(self) -> bool:
        """Loads, bins, and pre-processes projection images from the source folder."""
        print(f"Searching for '{self.config['FILE_PATTERN']}' files in '{self.input_dir}'...")
        search_path = os.path.join(self.input_dir, self.config['FILE_PATTERN'])
        file_list = sorted(glob.glob(search_path))

        if not file_list:
            print(f"Error: No files found at '{search_path}'. Check the path and file pattern.")
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
        print(f"Original image dimensions: {first_image.shape} -> Binned dimensions: {binned_shape}")

        self.projections = np.zeros((num_files, binned_height, binned_width), dtype=np.float32)

        update_interval = max(1, num_files // 10)  # Update progress every 10%
        for i, file_path in enumerate(file_list):
            image = tifffile.imread(file_path).astype(np.float32)
            if bin_factor > 1:
                self.projections[i, :, :] = downscale_local_mean(image, (bin_factor, bin_factor))
            else:
                self.projections[i, :, :] = image

            if (i + 1) % update_interval == 0 or (i + 1) == num_files:
                print(f"  > Loading & Binning Progress: {i + 1} / {num_files} ({(i + 1) / num_files:.0%})")
        return True

    def _reconstruct_volume(self) -> bool:
        """Performs slice-by-slice FBP reconstruction using the ASTRA Toolbox."""
        print("Starting slice-by-slice GPU reconstruction with ASTRA...")
        _, height, width = self.projections.shape
        reconstructed_slices = []

        # Reconstruction Geometry: Width x Width (Square Slice)
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

            # Clean up ASTRA objects to free GPU memory
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(reco_id)
            astra.data2d.delete(sino_id)

            if (y + 1) % update_interval == 0 or (y + 1) == height:
                print(f"  > GPU Reconstruction Progress: {y + 1} / {height} ({(y + 1) / height:.0%})")

        self.data_volume = np.stack(reconstructed_slices, axis=0).astype(np.float32)
        print("GPU reconstruction complete.")
        return True

    def _convert_to_polar(self) -> bool:
        """Converts the 3D volume from Cartesian to Polar coordinates slice by slice."""
        print("Converting volume to Polar coordinates for ring artifact removal...")
        num_slices, y_dim, x_dim = self.data_volume.shape
        center = (x_dim / 2.0, y_dim / 2.0)
        max_radius = x_dim / 2.0

        # Calculate appropriate dimensions for the polar image
        oversampled_width = int(max_radius)
        oversampled_height = int(x_dim * np.pi)
        if oversampled_height % 2 != 0: oversampled_height += 1
        dsize = (oversampled_width, oversampled_height)

        polar_slices = []
        update_interval = max(1, num_slices // 10)
        for i, s in enumerate(self.data_volume):
            polar_slices.append(cv2.warpPolar(s, dsize, center, max_radius, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS))
            if (i + 1) % update_interval == 0 or (i + 1) == num_slices:
                print(f"  > Polar Transform Progress: {i + 1} / {num_slices} ({(i + 1) / num_slices:.0%})")

        self.data_volume = np.stack(polar_slices, axis=0).astype(np.float32)
        print("Polar coordinate conversion complete.")
        return True

    def _apply_fft_filter(self) -> bool:
        """Applies an FFT band-stop filter to remove vertical stripes in the polar volume."""
        tolerance = self.config["TOLERANCE_VALUE"]
        self.file_params["tolerance"] = int(tolerance)

        if tolerance <= 0:
            print("FFT filter tolerance is 0, skipping this step.")
            return True

        processed_tolerance = 0.000025 * tolerance
        num_slices = self.data_volume.shape[0]
        update_interval = max(1, num_slices // 10)

        use_cpu_fallback = False
        if CUPY_AVAILABLE:
            print(f"Starting GPU FFT filtering to remove vertical stripes (Tolerance: {int(tolerance)})...")
            try:
                gpu_volume = cp.asarray(self.data_volume)
                filtered_gpu_volume = cp.empty_like(gpu_volume, dtype=cp.float32)

                rows, _ = gpu_volume[0].shape
                sigma = rows * processed_tolerance
                y_coords = cp.arange(rows) - (rows // 2)
                gaussian_1d = 1.0 - cp.exp(-(y_coords ** 2) / (2 * sigma ** 2))
                gaussian_1d = gaussian_1d.reshape(-1, 1)

                for i, polar_slice_gpu in enumerate(gpu_volume):
                    fft_slice = cupy_fft.fftshift(cupy_fft.fft2(polar_slice_gpu))
                    fft_slice *= gaussian_1d
                    slice_ifft = cupy_fft.ifft2(cupy_fft.ifftshift(fft_slice))
                    filtered_gpu_volume[i] = cp.real(slice_ifft)
                    if (i + 1) % update_interval == 0 or (i + 1) == num_slices:
                        print(f"  > GPU FFT Filter Progress: {i + 1} / {num_slices} ({(i + 1) / num_slices:.0%})")

                self.data_volume = cp.asnumpy(filtered_gpu_volume)
                cp.get_default_memory_pool().free_all_blocks()

            except Exception as e:
                print(f"GPU FFT filtering failed: {e}. Falling back to CPU.")
                if cp:
                    cp.get_default_memory_pool().free_all_blocks()
                use_cpu_fallback = True

        if not CUPY_AVAILABLE or use_cpu_fallback:
            print(f"Starting CPU FFT filtering to remove vertical stripes (Tolerance: {int(tolerance)})...")
            filtered_slices = []
            for i, polar_slice in enumerate(self.data_volume):
                fft_slice = scipy_fft.fftshift(scipy_fft.fft2(polar_slice))
                rows, _ = fft_slice.shape
                sigma = rows * processed_tolerance
                y_coords = np.arange(rows) - (rows // 2)
                gaussian_1d = 1.0 - np.exp(-(y_coords ** 2) / (2 * sigma ** 2))
                fft_slice *= gaussian_1d.reshape(-1, 1)
                slice_ifft = scipy_fft.ifft2(scipy_fft.ifftshift(fft_slice))
                filtered_slices.append(np.real(slice_ifft))
                if (i + 1) % update_interval == 0 or (i + 1) == num_slices:
                    print(f"  > CPU FFT Filter Progress: {i + 1} / {num_slices} ({(i + 1) / num_slices:.0%})")
            self.data_volume = np.stack(filtered_slices, axis=0).astype(np.float32)

        print("FFT filtering complete.")
        return True

    def _convert_to_cartesian(self) -> bool:
        """Converts the 3D volume back from Polar to Cartesian coordinates."""
        print("Converting filtered polar volume back to Cartesian coordinates...")
        # FIX: Originally using height here caused rectangular output (1277x1080)
        # We need the slice to be Square (Width x Width), e.g., (1277x1277)
        _, _, width = self.projections.shape

        num_slices = self.data_volume.shape[0]

        # FIX: Set both dimensions to 'width' to ensure a square slice
        center = (width / 2.0, width / 2.0)
        max_radius = width / 2.0
        dsize = (width, width)  # Was (width, height)

        cartesian_slices = []
        update_interval = max(1, num_slices // 10)
        for i, s in enumerate(self.data_volume):
            cartesian_slices.append(
                cv2.warpPolar(s, dsize, center, max_radius, cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP))
            if (i + 1) % update_interval == 0 or (i + 1) == num_slices:
                print(f"  > Inverse Polar Transform Progress: {i + 1} / {num_slices} ({(i + 1) / num_slices:.0%})")

        self.data_volume = np.stack(cartesian_slices, axis=0).astype(np.float32)
        print("Inverse conversion complete.")
        return True

    def _apply_median_filter(self) -> bool:
        """Applies a 3D median filter to the volume (GPU-accelerated if CuPy is available)."""
        radius = self.config["MEDIAN_FILTER_RADIUS"]
        self.file_params["median"] = radius
        if radius <= 0:
            print("Median filter radius is 0, skipping this step.")
            return True

        z_radius = radius // 2
        xy_kernel = radius * 2 + 1
        z_kernel = z_radius * 2 + 1
        filter_shape = (z_kernel, xy_kernel, xy_kernel)

        use_cpu_fallback = False
        if CUPY_AVAILABLE:
            print(f"Starting GPU 3D Median Filter (Kernel Shape Z,Y,X: {z_kernel}, {xy_kernel}, {xy_kernel})...")
            try:
                gpu_volume = cp.asarray(self.data_volume)
                filtered_gpu_volume = cupy_median_filter(gpu_volume, size=filter_shape)
                self.data_volume = cp.asnumpy(filtered_gpu_volume)
                cp.get_default_memory_pool().free_all_blocks()
            except Exception as e:
                print(f"GPU Median Filter failed: {e}. Falling back to CPU.")
                if cp:
                    cp.get_default_memory_pool().free_all_blocks()
                use_cpu_fallback = True

        if not CUPY_AVAILABLE or use_cpu_fallback:
            print(f"Starting CPU 3D Median Filter (Kernel Shape Z,Y,X: {z_kernel}, {xy_kernel}, {xy_kernel})...")
            self.data_volume = median_filter(self.data_volume, size=filter_shape)

        print("3D Median Filter complete.")
        return True

    def _apply_gaussian_filter(self) -> bool:
        """Applies a 3D Gaussian filter to the volume (always on CPU with SciPy)."""
        sigma = self.config["GAUSSIAN_SIGMA"]
        self.file_params["gauss"] = sigma
        if sigma <= 0:
            print("Gaussian sigma is 0, skipping this step.")
            return True

        print(f"Starting CPU 3D Gaussian Filter (Sigma: {sigma})...")
        self.data_volume = gaussian_filter(self.data_volume, sigma=sigma)
        print("3D Gaussian Filter complete.")
        return True

    def _save_volume(self, volume_type: str):
        """
        Saves the 3D NumPy array as a TIF file with a standardized name.
        Format: {basename}_{type}_{params}_{dimensions}.tif
        Example: MyData_RECON_B1_T29_M2_G1.8_512x512x512.tif
        """
        output_dir = os.path.dirname(self.input_dir)
        z, y, x = self.data_volume.shape

        param_parts = []
        param_parts.append(f"B{self.file_params['binning']}")
        if "tolerance" in self.file_params: param_parts.append(f"T{self.file_params['tolerance']}")
        if "median" in self.file_params and self.file_params['median'] > 0: param_parts.append(
            f"M{self.file_params['median']}")
        if "gauss" in self.file_params and self.file_params['gauss'] > 0: param_parts.append(
            f"G{self.file_params['gauss']:.1f}".replace('.0', ''))
        param_parts.append(f"{x}x{y}x{z}")

        param_str = "_".join(param_parts)
        output_filename = f"{self.base_name}_{volume_type}_{param_str}.tif"
        output_path = os.path.join(output_dir, output_filename)

        print(f"Saving {volume_type} volume to '{output_path}'...")
        try:
            tifffile.imwrite(output_path, self.data_volume.astype(np.float32))
            print(f"-> File saved successfully.")
        except Exception as e:
            print(f"Error: Failed to save file '{output_path}'. Details: {e}")


# ==============================================================================
# Graphical User Interface (GUI)
# ==============================================================================

class App(tk.Tk):
    """
    The main application window, integrating parameter settings, folder selection,
    and execution control.
    """

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

        # Redirect stdout and stderr to the GUI log widget
        sys.stdout = TextRedirector(self.log_text)
        sys.stderr = TextRedirector(self.log_text)

        # Initial status messages
        print("3D Tomography Reconstruction Tool initialized.")
        if CUPY_AVAILABLE:
            print("CuPy found. Median filter and FFT will be GPU-accelerated.")
        else:
            print("Warning: CuPy not found. Median filter and FFT will run on the CPU.")
            print("Tip: Install CuPy for significant performance improvements.")

    def _create_widgets(self):
        """Creates and lays out all the GUI widgets."""
        main_frame = Frame(self, padx=10, pady=10)
        main_frame.pack(fill="both", expand=True)
        main_frame.rowconfigure(3, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # --- Filter Parameters ---
        param_frame = LabelFrame(main_frame, text="Filter Parameters", padx=10, pady=10)
        param_frame.grid(row=0, column=0, padx=10, pady=5, sticky='ew')
        param_defs = {
            "BINNING_FACTOR": "Binning Factor:", "TOLERANCE_VALUE": "FFT Filter Tolerance:",
            "MEDIAN_FILTER_RADIUS": "Median Filter Radius:", "GAUSSIAN_SIGMA": "Gaussian Sigma:"
        }
        for i, (key, label) in enumerate(param_defs.items()):
            tk.Label(param_frame, text=label).grid(row=i, column=0, padx=5, pady=3, sticky='w')
            var = tk.StringVar(value=str(self.defaults.get(key, "")))
            entry = tk.Entry(param_frame, width=15, textvariable=var)
            entry.grid(row=i, column=1, padx=5, pady=3)
            self.entries[key] = var

        # --- Output Selection ---
        save_frame = LabelFrame(main_frame, text="Select Files to Output (Pipeline runs to the last checked item)",
                                padx=10, pady=10)
        save_frame.grid(row=1, column=0, padx=10, pady=5, sticky='ew')
        save_defs = {
            "SAVE_RECON_RAW": "1. Raw Reconstructed Volume (RECON)", "SAVE_POLAR_RAW": "2. Raw Polar Volume (POLAR)",
            "SAVE_POLAR_FILTERED": "3. FFT-Filtered Polar Volume (POLAR)",
            "SAVE_RECON_FILTERED": "4. FFT-Filtered Reconstructed Volume (RECON)",
            "SAVE_RECON_FILTERED_MEDIAN": "5. Median-Filtered Volume (RECON)",
            "SAVE_RECON_FILTERED_MEDIAN_GAUSS": "6. Gaussian-Filtered Volume (RECON)"
        }
        for key, label in save_defs.items():
            var = tk.BooleanVar(value=self.defaults.get(key, False))
            chk = tk.Checkbutton(save_frame, text=label, variable=var)
            chk.pack(anchor="w")
            self.save_vars[key] = var

        # --- Execution Control ---
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

        # --- Log Output ---
        log_frame = LabelFrame(main_frame, text="Log", padx=10, pady=10)
        log_frame.grid(row=3, column=0, padx=10, pady=5, sticky='nsew')
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, state='disabled', height=15, wrap=tk.WORD, bg="#f0f0f0")
        self.log_text.grid(row=0, column=0, sticky='nsew')

    def _select_folder(self):
        """Opens a dialog to let the user select the source folder."""
        path = filedialog.askdirectory(parent=self, title="Select folder containing projection images")
        if path:
            self.folder_path.set(path)

    def _start_processing_thread(self):
        """Starts the data processing in a background thread to avoid freezing the GUI."""
        input_dir = self.folder_path.get()
        if not os.path.isdir(input_dir):
            messagebox.showerror("Error", "Please select a valid folder first.", parent=self)
            return

        try:
            user_params = {
                "BINNING_FACTOR": int(self.entries["BINNING_FACTOR"].get()),
                "TOLERANCE_VALUE": float(self.entries["TOLERANCE_VALUE"].get()),
                "MEDIAN_FILTER_RADIUS": int(self.entries["MEDIAN_FILTER_RADIUS"].get()),
                "GAUSSIAN_SIGMA": float(self.entries["GAUSSIAN_SIGMA"].get()),
            }
            for key, var in self.save_vars.items():
                user_params[key] = var.get()
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure all parameters are valid numbers.", parent=self)
            return

        self.start_button.config(state="disabled")
        config = {**self.defaults, **user_params}

        thread = threading.Thread(target=self._run_pipeline, args=(config, input_dir))
        thread.daemon = True  # Allows main window to exit even if thread is running
        thread.start()

    def _run_pipeline(self, config, input_dir):
        """This function runs in the background thread and executes the main pipeline."""
        pipeline = TomoPipeline(config)
        success = pipeline.run(input_dir)
        self.after_idle(self._on_processing_complete, success)

    def _on_processing_complete(self, success: bool):
        """This function runs in the main GUI thread after the pipeline finishes."""
        if success:
            messagebox.showinfo("Success", "Task completed successfully!", parent=self)
        else:
            messagebox.showerror("Error", "Task failed. Check the log for details.", parent=self)
        self.start_button.config(state="normal")

    def _center_window(self):
        """Centers the main window on the screen upon startup."""
        self.update_idletasks()
        min_width = 550
        min_height = 750
        self.minsize(min_width, min_height)

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width // 2) - (min_width // 2)
        y = (screen_height // 2) - (min_height // 2)
        self.geometry(f'{min_width}x{min_height}+{x}+{y}')


# ==============================================================================
# Main Execution Block
# ==============================================================================

def main():
    """Main function to create and run the GUI application."""
    try:
        app = App(DEFAULT_CONFIG)
        app.mainloop()
    except Exception as e:
        print(f"\n--- An unexpected critical error occurred ---")
        import traceback
        traceback.print_exc()
        # Fallback to show an error popup if the GUI fails to initialize
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Critical Error",
                             f"An unexpected error occurred and the application will close.\n\nDetails:\n{e}")


if __name__ == '__main__':
    main()