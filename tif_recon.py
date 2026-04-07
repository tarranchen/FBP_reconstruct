# -*- coding: utf-8 -*-
"""
Integrated 3D Tomography Reconstruction and Post-Processing Tool (Auto Pipeline)
"""

import os
import sys
import time
import threading
import struct
import math
from typing import Optional, Dict, Any

import tkinter as tk
from tkinter import filedialog, messagebox, LabelFrame, Frame, scrolledtext

import numpy as np
import tifffile
import astra
import cv2
from skimage.transform import downscale_local_mean

try:
    import cupy as cp
    from cupyx.scipy.ndimage import median_filter as cupy_median_filter
    from cupyx.scipy import fft as cupy_fft
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    cupy_median_filter = None
    cupy_fft = None

from scipy.ndimage import median_filter, gaussian_filter
from scipy import fft as scipy_fft

DEFAULT_CONFIG = {
    "BINNING_FACTOR": 1,
    "TOLERANCE_VALUE": 25,
    "MEDIAN_FILTER_RADIUS": 1,
    "GAUSSIAN_SIGMA": 1,
    "FFT_CUTOFF_EXPONENT": 6,
    "BUTTERWORTH_ORDER": 2,
    "PADDING_DIVISOR_EXPONENT": 2,
    "SAVE_RECON_RAW": False,
    "SAVE_RECON_FILTERED": False,
    "SAVE_RECON_FILTERED_MEDIAN": False,
    "SAVE_RECON_FILTERED_MEDIAN_GAUSS": True,
}

AUTO_RAW_FOLDER_NAME = "auto_raw"
TIF_FOLDER_NAME = "TIF"

def find_center_from_corrected(img0, img180, sample_dir):
    width = img0.shape[1]
    n = int(np.ceil(width * 0.01))
    center_idx = width // 2
    
    results = []
    total_steps = 2 * n + 1
    
    for i in range(total_steps):
        left_crop = i
        right_crop = 2 * n - i
        suffix = center_idx - n + i
        
        if right_crop == 0:
            crop0 = img0[:, left_crop:]
            crop180 = img180[:, left_crop:]
        else:
            crop0 = img0[:, left_crop:-right_crop]
            crop180 = img180[:, left_crop:-right_crop]
            
        crop180_flipped = np.fliplr(crop180)
        diff = crop0 - crop180_flipped
        abs_diff_sum = np.sum(np.abs(diff), dtype=np.float64)
        results.append((suffix, abs_diff_sum))
        
    best_suffix, min_sum = min(results, key=lambda x: x[1])
    
    output_file = os.path.join(sample_dir, f"{best_suffix}.auto_detected_center")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for suffix, val in results:
                f.write(f"{suffix}, {int(val)}\n")
    except Exception as e:
        print(f"Failed to save center file: {e}")
        
    return best_suffix

def crop_image(img, center, original_width):
    default_center = original_width // 2
    diff = center - default_center
    if diff < 0:
        crop_pixels = 2 * abs(diff)
        return img[:, :-crop_pixels]
    elif diff > 0:
        crop_pixels = 2 * diff
        return img[:, crop_pixels:]
    return img

class TextRedirector:
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

class TomoPipeline:
    def __init__(self, config: Dict[str, Any], sample_dir: str, log_func):
        self.config = config
        self.sample_dir = sample_dir
        self.log = log_func
        
        self.auto_raw_dir = os.path.join(sample_dir, AUTO_RAW_FOLDER_NAME)
        self.tif_dir = os.path.join(sample_dir, TIF_FOLDER_NAME)
        self.base_name = os.path.basename(sample_dir)
        
        self.data_volume = None
        self.projections = None
        self.angles_rad = None
        
        self.file_params = {"binning": self.config["BINNING_FACTOR"]}
        self.step_timings = {}

        self.processing_steps = [
            {"name": "reconstruct_raw", "method": self._reconstruct_volume, "save_key": "SAVE_RECON_RAW", "type": "RECON"},
            {"name": "to_polar_raw", "method": self._convert_to_polar, "type": "POLAR"},
            {"name": "fft_filter", "method": self._apply_fft_filter, "type": "POLAR"},
            {"name": "to_cartesian_filtered", "method": self._convert_to_cartesian, "save_key": "SAVE_RECON_FILTERED", "type": "RECON"},
            {"name": "median_filter", "method": self._apply_median_filter, "save_key": "SAVE_RECON_FILTERED_MEDIAN", "type": "RECON"},
            {"name": "gaussian_filter", "method": self._apply_gaussian_filter, "save_key": "SAVE_RECON_FILTERED_MEDIAN_GAUSS", "type": "RECON"},
        ]

    def run(self) -> bool:
        self.log(f"--- Starting Pipeline for {self.base_name} ---")
        total_start_time = time.time()
        
        os.makedirs(self.auto_raw_dir, exist_ok=True)
        
        step_start = time.time()
        if not self._prepare_projections():
            return False
        self.step_timings["Prepare Projections (Convert, Center, Crop, Bin)"] = time.time() - step_start

        tolerance = self.config.get("TOLERANCE_VALUE", 0)
        active_steps = list(self.processing_steps)

        if tolerance <= 0:
            self.log("Tolerance <= 0: Skipping Ring Artifact Removal (Polar/FFT steps).")
            active_steps = [s for s in active_steps if s["name"] not in ["to_polar_raw", "fft_filter", "to_cartesian_filtered"]]

        last_step_to_run = None
        for step in reversed(active_steps):
            if self.config.get(step.get("save_key", "")):
                last_step_to_run = step["name"]
                break

        if not last_step_to_run:
            self.log("Warning: No output selected or step skipped.")
            return False

        for step in active_steps:
            self.log(f"\n--- Step: {step['name']} ---")
            step_start = time.time()
            success = step["method"]()
            self.step_timings[step['name']] = time.time() - step_start

            if not success:
                self.log(f"Error: Step {step['name']} failed.")
                return False

            save_key = step.get("save_key")
            if save_key and self.config.get(save_key):
                self._save_volume(step["type"])

            if step["name"] == last_step_to_run:
                self.log("\nAll selected processing steps have been completed.")
                break

        self.step_timings["Total Processing Time"] = time.time() - total_start_time
        self._print_timing_summary()
        return True

    def _prepare_projections(self) -> bool:
        self.log("Preparing projections from TIF files...")
        
        if not os.path.isdir(self.tif_dir):
            self.log(f"Error: TIF directory not found at '{self.tif_dir}'. Aborting.")
            return False
            
        tif_files = sorted([f for f in os.listdir(self.tif_dir) if f.lower().endswith(('.tif', '.tiff'))])
        if len(tif_files) != 721:
            self.log(f"Error: Expected 721 TIF files in {self.tif_dir}, but found {len(tif_files)}. Aborting.")
            return False

        center_idx = None
        center_file = next((f for f in os.listdir(self.sample_dir) if f.endswith('.auto_detected_center')), None)

        if center_file:
            try:
                center_idx = int(center_file.split('.')[0])
                self.log(f"Found existing center file. Using center: {center_idx}")
            except (ValueError, IndexError):
                self.log(f"Warning: Could not parse center from filename '{center_file}'. Will recalculate.")
                center_file = None

        if not center_file:
            self.log("No center file found. Calculating center of rotation from TIF files...")
            proj0_path = os.path.join(self.tif_dir, tif_files[0])
            proj180_path = os.path.join(self.tif_dir, tif_files[-1])

            try:
                img0 = tifffile.imread(proj0_path).astype(np.float32)
                img180 = tifffile.imread(proj180_path).astype(np.float32)
            except Exception as e:
                self.log(f"Error reading TIF files for center detection: {e}")
                return False

            center_idx = find_center_from_corrected(img0, img180, self.sample_dir)
            self.log(f"Detected center: {center_idx}")

        first_tif_path = os.path.join(self.tif_dir, tif_files[0])
        temp_img = tifffile.imread(first_tif_path)
        original_height, original_width = temp_img.shape

        test_crop = crop_image(temp_img, center_idx, original_width)
        bin_factor = self.config['BINNING_FACTOR']
        if bin_factor > 1:
            test_binned = downscale_local_mean(test_crop, (bin_factor, bin_factor))
        else:
            test_binned = test_crop

        binned_height, binned_width = test_binned.shape
        self.log(f"Binned dimensions: {binned_width}x{binned_height}")

        self.projections = np.zeros((721, binned_height, binned_width), dtype=np.float32)
        self.angles_rad = np.deg2rad(np.linspace(0, 180, 721, endpoint=False, dtype=np.float32))

        self.log(f"Processing 721 projections (Load TIF -> Crop -> Bin -> Log)...")
        update_interval = max(1, 721 // 10)

        for i, tif_name in enumerate(tif_files):
            tif_path = os.path.join(self.tif_dir, tif_name)
            try:
                corrected = tifffile.imread(tif_path).astype(np.float32)
            except Exception as e:
                self.log(f"Error reading {tif_name}: {e}. Filling with zeros.")
                continue

            cropped = crop_image(corrected, center_idx, original_width)

            if bin_factor > 1:
                binned = downscale_local_mean(cropped, (bin_factor, bin_factor))
            else:
                binned = cropped

            absorption_data = -np.log(np.maximum(binned, 1e-9))
            self.projections[i, :, :] = absorption_data

            if (i + 1) % update_interval == 0 or (i + 1) == 721:
                self.log(f"  > Processed: {i + 1} / 721 ({(i + 1) / 721:.0%})")

        return True

    def _reconstruct_volume(self) -> bool:
        self.log("Starting ASTRA GPU reconstruction...")
        _, height, width = self.projections.shape
        reconstructed_slices = []

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
                self.log(f"  > Reconstructing: {y + 1} / {height} ({(y + 1) / height:.0%})")

        self.data_volume = np.stack(reconstructed_slices, axis=0).astype(np.float32)
        self.log("Reconstruction complete.")
        return True

    def _convert_to_polar(self) -> bool:
        self.log("Converting volume to Polar coordinates...")
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
                self.log(f"  > Polar Transform: {i + 1} / {num_slices} ({(i + 1) / num_slices:.0%})")

        self.data_volume = np.stack(polar_slices, axis=0).astype(np.float32)
        self.log("Polar conversion complete.")
        return True

    def _apply_fft_filter(self) -> bool:
        tolerance = self.config["TOLERANCE_VALUE"]
        cutoff_exponent = self.config.get("FFT_CUTOFF_EXPONENT", 6.0)
        padding_divisor_exponent = self.config.get("PADDING_DIVISOR_EXPONENT", 2.0)
        butterworth_order = self.config.get("BUTTERWORTH_ORDER", 2)

        power_of_2_cutoff = 2.0 ** cutoff_exponent
        padding_ratio = 1.0 / (2.0 ** padding_divisor_exponent)

        self.file_params["tolerance"] = int(tolerance)

        if tolerance <= 0:
            self.log("Tolerance <= 0, skipping FFT.")
            return True

        processed_tolerance = 0.000025 * tolerance
        num_slices = self.data_volume.shape[0]
        update_interval = max(1, num_slices // 10)

        rows, cols_orig = self.data_volume[0].shape
        pad_width = int(cols_orig * padding_ratio)
        if pad_width < 1: pad_width = 1
        if pad_width > cols_orig: pad_width = cols_orig
        cols_padded = cols_orig + pad_width
        sigma = rows * processed_tolerance
        cutoff_threshold = cols_padded / power_of_2_cutoff
        if cutoff_threshold < 1.0: cutoff_threshold = 1.0

        self.log(f"FFT Config -> T:{tolerance}, Sigma:{sigma:.2f}, n:{cutoff_exponent}, RelOrder:{padding_divisor_exponent}, Order:{butterworth_order}")
        self.log(f"  -> Pad Width: {pad_width}px (Left), Cutoff: {cutoff_threshold:.2f}px")

        use_cpu_fallback = False
        if CUPY_AVAILABLE:
            try:
                gpu_volume = cp.asarray(self.data_volume)
                filtered_gpu_volume = cp.empty_like(gpu_volume, dtype=cp.float32)

                y_coords = cp.arange(rows) - (rows // 2)
                x_coords = cp.arange(cols_padded) - (cols_padded // 2)
                notch_y = 1.0 - cp.exp(-(y_coords ** 2) / (2 * sigma ** 2))
                notch_y = notch_y.reshape(-1, 1)
                epsilon = 1e-6
                order = int(butterworth_order)
                protect_x = 1.0 / (1.0 + (cp.abs(x_coords) / (cutoff_threshold + epsilon)) ** (2 * order))
                protect_x = protect_x.reshape(1, -1)
                filter_2d = cp.maximum(notch_y, protect_x)

                for i, polar_slice_gpu in enumerate(gpu_volume):
                    left_strip = polar_slice_gpu[:, :pad_width]
                    mirrored_left = cp.flip(left_strip, axis=1)
                    combined_slice = cp.concatenate((mirrored_left, polar_slice_gpu), axis=1)

                    fft_slice = cupy_fft.fftshift(cupy_fft.fft2(combined_slice))
                    fft_slice *= filter_2d
                    slice_ifft = cupy_fft.ifft2(cupy_fft.ifftshift(fft_slice))
                    result_real = cp.real(slice_ifft)

                    filtered_gpu_volume[i] = result_real[:, pad_width:].astype(cp.float32)

                    if (i + 1) % update_interval == 0 or (i + 1) == num_slices:
                        self.log(f"  > GPU FFT Filter: {i + 1} / {num_slices} ({(i + 1) / num_slices:.0%})")

                self.data_volume = cp.asnumpy(filtered_gpu_volume)

            except Exception as e:
                self.log(f"GPU failed: {e}. Fallback to CPU.")
                use_cpu_fallback = True
            finally:
                if cp: cp.get_default_memory_pool().free_all_blocks()

        if not CUPY_AVAILABLE or use_cpu_fallback:
            filtered_slices = []
            y_coords = np.arange(rows) - (rows // 2)
            x_coords = np.arange(cols_padded) - (cols_padded // 2)

            notch_y = 1.0 - np.exp(-(y_coords ** 2) / (2 * sigma ** 2))
            notch_y = notch_y.reshape(-1, 1)

            epsilon = 1e-6
            order = int(butterworth_order)
            protect_x = 1.0 / (1.0 + (np.abs(x_coords) / (cutoff_threshold + epsilon)) ** (2 * order))
            protect_x = protect_x.reshape(1, -1)

            filter_2d = np.maximum(notch_y, protect_x)

            for i, polar_slice in enumerate(self.data_volume):
                left_strip = polar_slice[:, :pad_width]
                mirrored_left = np.fliplr(left_strip)
                combined_slice = np.concatenate((mirrored_left, polar_slice), axis=1)

                fft_slice = scipy_fft.fftshift(scipy_fft.fft2(combined_slice))
                fft_slice *= filter_2d
                slice_ifft = scipy_fft.ifft2(scipy_fft.ifftshift(fft_slice))
                result_real = np.real(slice_ifft)

                filtered_slices.append(result_real[:, pad_width:])

                if (i + 1) % update_interval == 0 or (i + 1) == num_slices:
                    self.log(f"  > CPU FFT Filter: {i + 1} / {num_slices} ({(i + 1) / num_slices:.0%})")
            self.data_volume = np.stack(filtered_slices, axis=0).astype(np.float32)

        self.log("FFT filtering complete.")
        return True

    def _convert_to_cartesian(self) -> bool:
        self.log("Converting back to Cartesian...")
        _, _, width = self.projections.shape
        num_slices = self.data_volume.shape[0]

        center = (width / 2.0, width / 2.0)
        max_radius = width / 2.0
        dsize = (width, width)

        cartesian_slices = []
        update_interval = max(1, num_slices // 10)
        for i, s in enumerate(self.data_volume):
            cartesian_slices.append(cv2.warpPolar(s, dsize, center, max_radius, cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP))
            if (i + 1) % update_interval == 0 or (i + 1) == num_slices:
                self.log(f"  > Inverse Polar: {i + 1} / {num_slices} ({(i + 1) / num_slices:.0%})")

        self.data_volume = np.stack(cartesian_slices, axis=0).astype(np.float32)
        self.log("Inverse conversion complete.")
        return True

    def _apply_median_filter(self) -> bool:
        radius = self.config["MEDIAN_FILTER_RADIUS"]
        self.file_params["median"] = radius
        if radius <= 0:
            self.log("Median filter radius is 0, skipping.")
            return True

        z_radius = radius // 2
        xy_kernel = radius * 2 + 1
        z_kernel = z_radius * 2 + 1
        filter_shape = (z_kernel, xy_kernel, xy_kernel)

        filter_backend = None
        device = "CPU"
        if CUPY_AVAILABLE:
            try:
                cp.cuda.runtime.getDeviceCount()
                filter_backend = cupy_median_filter
                device = "GPU"
            except Exception as e:
                self.log(f"GPU Median failed: {e}. Fallback to CPU.")
        
        if filter_backend is None:
            filter_backend = median_filter

        self.log(f"Starting 3D Median Filter on {device}...")
        if device == "GPU":
            gpu_volume = cp.asarray(self.data_volume)
            filtered_gpu_volume = filter_backend(gpu_volume, size=filter_shape)
            self.data_volume = cp.asnumpy(filtered_gpu_volume)
            cp.get_default_memory_pool().free_all_blocks()
        else: # CPU
            self.data_volume = median_filter(self.data_volume, size=filter_shape)

        self.log("Median Filter complete.")
        return True

    def _apply_gaussian_filter(self) -> bool:
        sigma = self.config["GAUSSIAN_SIGMA"]
        self.file_params["gauss"] = sigma
        if sigma <= 0:
            self.log("Gaussian sigma is 0, skipping.")
            return True

        self.log(f"Starting CPU 3D Gaussian Filter (Sigma: {sigma})...")
        self.data_volume = gaussian_filter(self.data_volume, sigma=sigma)
        self.log("Gaussian Filter complete.")
        return True

    def _save_volume(self, volume_type: str):
        z, y, x = self.data_volume.shape

        param_parts = []
        param_parts.append(f"B{self.file_params['binning']}")
        if "tolerance" in self.file_params and self.file_params['tolerance'] > 0:
            param_parts.append(f"T{self.file_params['tolerance']}")
        if "median" in self.file_params and self.file_params['median'] > 0:
            param_parts.append(f"M{self.file_params['median']}")
        if "gauss" in self.file_params and self.file_params['gauss'] > 0:
            param_parts.append(f"G{self.file_params['gauss']:.1f}".replace('.0', ''))
        param_parts.append(f"{x}x{y}x{z}")

        param_str = "_".join(param_parts)

        if volume_type == "RECON":
            output_filename = f"{self.base_name}_{param_str}.tif"
        else:
            output_filename = f"{self.base_name}_{volume_type}_{param_str}.tif"

        output_path = os.path.join(self.auto_raw_dir, output_filename)

        self.log(f"Saving {volume_type} to '{output_path}'...")
        try:
            tifffile.imwrite(output_path, self.data_volume.astype(np.float32))
            self.log("-> Saved.")
        except Exception as e:
            self.log(f"Error saving: {e}")

    def _print_timing_summary(self):
        self.log("\n--- Processing Time Summary ---")
        for name, duration in self.step_timings.items():
            self.log(f"{name:<45}: {duration:>7.2f} seconds")
        self.log("---------------------------------------------")

class App(tk.Tk):
    def __init__(self, defaults: Dict[str, Any]):
        super().__init__()
        self.defaults = defaults
        self.title("Auto Tomography Pipeline Tool")
        self.resizable(True, True)

        self.entries: Dict[str, tk.StringVar] = {}
        self.save_vars: Dict[str, tk.BooleanVar] = {}
        self.folder_path = tk.StringVar(value="No folder selected")
        self.is_monitoring = threading.Event()

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

        top_container = Frame(main_frame)
        top_container.grid(row=0, column=0, sticky='ew')
        top_container.columnconfigure(0, weight=1)
        top_container.columnconfigure(1, weight=1)

        param_frame = LabelFrame(top_container, text="Filter Parameters", padx=10, pady=10)
        param_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')

        param_defs = {
            "BINNING_FACTOR": "Binning Factor:",
            "TOLERANCE_VALUE": "FFT Filter Tolerance:",
            "MEDIAN_FILTER_RADIUS": "Median Filter Radius:",
            "GAUSSIAN_SIGMA": "Gaussian Sigma:"
        }
        for i, (key, label) in enumerate(param_defs.items()):
            tk.Label(param_frame, text=label).grid(row=i, column=0, padx=5, pady=3, sticky='w')
            val = self.defaults.get(key, "")
            val_str = str(int(val)) if isinstance(val, float) and val.is_integer() else str(val)
            var = tk.StringVar(value=val_str)
            entry = tk.Entry(param_frame, width=15, textvariable=var)
            entry.grid(row=i, column=1, padx=5, pady=3)
            self.entries[key] = var

        adv_frame = LabelFrame(top_container, text="Advanced Parameters", padx=10, pady=10)
        adv_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')

        adv_defs = {
            "FFT_CUTOFF_EXPONENT": "Ring Cutoff Order:",
            "BUTTERWORTH_ORDER": "Butterworth Order:",
            "PADDING_DIVISOR_EXPONENT": "Padding Relative Order:"
        }
        for i, (key, label) in enumerate(adv_defs.items()):
            tk.Label(adv_frame, text=label).grid(row=i, column=0, padx=5, pady=3, sticky='w')
            val = self.defaults.get(key, "")
            val_str = str(int(val)) if isinstance(val, float) and val.is_integer() else str(val)
            var = tk.StringVar(value=val_str)
            entry = tk.Entry(adv_frame, width=15, textvariable=var)
            entry.grid(row=i, column=1, padx=5, pady=3)
            self.entries[key] = var

        save_frame = LabelFrame(main_frame, text="Select Files to Output", padx=10, pady=10)
        save_frame.grid(row=1, column=0, padx=10, pady=5, sticky='ew')

        save_defs = {
            "SAVE_RECON_RAW": "1. Raw Reconstructed Volume",
            "SAVE_RECON_FILTERED": "2. FFT Filtered Volume",
            "SAVE_RECON_FILTERED_MEDIAN": "3. Median-Filtered Volume",
            "SAVE_RECON_FILTERED_MEDIAN_GAUSS": "4. Gaussian-Filtered Volume"
        }
        for key, label in save_defs.items():
            var = tk.BooleanVar(value=self.defaults.get(key, False))
            chk = tk.Checkbutton(save_frame, text=label, variable=var)
            chk.pack(anchor="w")
            self.save_vars[key] = var

        run_frame = LabelFrame(main_frame, text="Execution Control", padx=10, pady=10)
        run_frame.grid(row=2, column=0, padx=10, pady=5, sticky='ew')
        
        folder_select_frame = Frame(run_frame)
        folder_select_frame.pack(fill='x', pady=(5, 2))

        folder_label = tk.Label(folder_select_frame, textvariable=self.folder_path, fg="blue", wraplength=450, justify="left")
        folder_label.pack(side="left", padx=10)

        monitor_frame = Frame(run_frame, pady=5)
        monitor_frame.pack(fill='x')

        self.monitor_button = tk.Button(monitor_frame, text="Select Folder & Monitor", command=self._select_folder_and_monitor)
        self.monitor_button.pack(side="left", padx=(10, 5))

        self.stop_button = tk.Button(monitor_frame, text="Stop Monitoring", command=self._stop_monitoring, state="disabled")
        self.stop_button.pack(side="left")

        log_frame = LabelFrame(main_frame, text="Log", padx=10, pady=10)
        log_frame.grid(row=3, column=0, padx=10, pady=5, sticky='nsew')
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, state='disabled', height=15, wrap=tk.WORD, bg="#f0f0f0")
        self.log_text.grid(row=0, column=0, sticky='nsew')

    def log(self, message: str):
        print(message)

    def _get_user_params(self):
        try:
            user_params = {
                "BINNING_FACTOR": int(self.entries["BINNING_FACTOR"].get()),
                "TOLERANCE_VALUE": float(self.entries["TOLERANCE_VALUE"].get()),
                "MEDIAN_FILTER_RADIUS": int(self.entries["MEDIAN_FILTER_RADIUS"].get()),
                "GAUSSIAN_SIGMA": float(self.entries["GAUSSIAN_SIGMA"].get()),
                "FFT_CUTOFF_EXPONENT": float(self.entries["FFT_CUTOFF_EXPONENT"].get()),
                "BUTTERWORTH_ORDER": int(self.entries["BUTTERWORTH_ORDER"].get()),
                "PADDING_DIVISOR_EXPONENT": float(self.entries["PADDING_DIVISOR_EXPONENT"].get()),
            }
            for key, var in self.save_vars.items():
                user_params[key] = var.get()
            return user_params
        except ValueError:
            return None

    def _select_folder_and_monitor(self):
        path = filedialog.askdirectory(parent=self, title="Select Monitoring Folder")
        if path:
            self.folder_path.set(path)
            self.is_monitoring.clear()

            self.monitor_button.config(state="disabled")
            self.stop_button.config(state="normal")

            thread = threading.Thread(target=self._monitor_folder, args=(path,))
            thread.daemon = True
            thread.start()

    def _monitor_folder(self, folder_path: str, check_interval: int = 5):
        self.after_idle(self.log, f"Monitoring folder '{folder_path}' for new samples...")
        try:
            while not self.is_monitoring.is_set():
                sample_to_process = None
                for item in os.listdir(folder_path):
                    sample_dir = os.path.join(folder_path, item)
                    if not os.path.isdir(sample_dir):
                        continue
                    
                    tif_dir = os.path.join(sample_dir, TIF_FOLDER_NAME)
                    if os.path.isdir(tif_dir):
                        tif_files = [f for f in os.listdir(tif_dir) if f.lower().endswith(('.tif', '.tiff'))]
                        if len(tif_files) == 721:
                            auto_raw_dir = os.path.join(sample_dir, AUTO_RAW_FOLDER_NAME)
                            has_output = False
                            if os.path.isdir(auto_raw_dir):
                                output_files = [f for f in os.listdir(auto_raw_dir) if f.lower().endswith(('.tif', '.tiff'))]
                                if len(output_files) > 0:
                                    has_output = True
                            
                            if not has_output:
                                sample_to_process = sample_dir
                                break

                if sample_to_process:
                    self.after_idle(self.log, f"Trigger conditions met for sample: {os.path.basename(sample_to_process)}.")
                    self._run_pipeline_sync(sample_to_process)
                else:
                    self.is_monitoring.wait(timeout=check_interval)
        finally:
            self.after_idle(self._on_monitoring_stopped)

    def _run_pipeline_sync(self, sample_dir):
        config = self._get_user_params()
        if not config:
            self.after_idle(self.log, "Error parsing parameters. Stopping monitor.")
            self.is_monitoring.set()
            return
            
        def log_safe(msg):
            self.after_idle(self.log, msg)

        pipeline = TomoPipeline(config, sample_dir, log_safe)
        success = pipeline.run()
        if success:
            self.after_idle(self._on_processing_complete, True)
        else:
            self.after_idle(self.log, f"Pipeline failed for {os.path.basename(sample_dir)}.")

    def _stop_monitoring(self):
        self.is_monitoring.set()

    def _on_monitoring_stopped(self):
        self.log("Monitoring stopped.")
        self.monitor_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def _on_processing_complete(self, success: bool):
        if success:
            self.bell()
            top = tk.Toplevel(self)
            top.title("Success")
            tk.Label(top, text="Task completed!", padx=20, pady=20).pack()
            x = self.winfo_x() + (self.winfo_width() // 2) - 75
            y = self.winfo_y() + (self.winfo_height() // 2) - 40
            top.geometry(f"+{x}+{y}")
            self.after(1000, top.destroy)
        else:
            messagebox.showerror("Error", "Task failed.")

    def _center_window(self):
        self.update_idletasks()
        min_w, min_h = 600, 800
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