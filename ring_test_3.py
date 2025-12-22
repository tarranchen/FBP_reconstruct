# -*- coding: utf-8 -*-
"""
FFT Ring Artifact Filter Test Tool V3
--------------------------------------------------
A GUI-based tool to evaluate the effect of FFT filtering on ring artifact removal
with varying intensities.

Core Algorithm Update (Mirror Padding + Cropping):
1. Mirror Padding: Flips and concatenates the image [Flip | Original] to eliminate FFT boundary effects.
2. Filter: Combines "Vertical Notch (Gaussian)" and "Horizontal Butterworth Protection (Low-pass)".
   - Uses Maximum overlay logic.
   - Calculates filter mask based on the padded width.
3. Cropping: Crops back to the original region after filtering.

Features:
1. Reads 3D TIF files.
2. Specifies Z-slice, cutoff frequency parameter, and Butterworth order for testing.
3. Automatically executes filtering loop for intensities 1-100.
4. Supports GPU (CuPy) acceleration; automatically falls back to CPU (NumPy) if unavailable.
5. Outputs a stacked 3D TIF file for comparison.
"""

import os
import threading
import numpy as np
import tifffile
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# --- Try importing CuPy for GPU acceleration ---
try:
    import cupy as cp
    from cupyx.scipy import fft as cupy_fft

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    cupy_fft = None


class FFTFilterProcessor:
    """
    Encapsulates the complete backend logic from reading, filtering to saving.
    """

    def __init__(self, input_path: str, output_path: str, z_index: int, cutoff_exponent: float,
                 butterworth_order: float, progress_callback, status_callback):
        self.input_path = input_path
        self.output_path = output_path
        self.z_index = z_index
        self.cutoff_exponent = cutoff_exponent  # Cutoff frequency exponent n
        self.butterworth_order = butterworth_order  # Butterworth order
        self.progress_callback = progress_callback
        self.status_callback = status_callback

    def _convert_to_polar(self, image: np.ndarray) -> np.ndarray:
        """Converts a single Cartesian image to a polar image."""
        y_dim, x_dim = image.shape
        center = (x_dim / 2.0, y_dim / 2.0)
        max_radius = x_dim / 2.0

        oversampled_width = int(max_radius)
        oversampled_height = int(x_dim * np.pi)
        # Ensure height is even for FFT processing
        if oversampled_height % 2 != 0:
            oversampled_height += 1
        dsize = (oversampled_width, oversampled_height)

        # Conversion result: X-axis is Radius, Y-axis is Angle
        # Thus, ring artifacts appear as vertical stripes in the polar image
        polar_image = cv2.warpPolar(
            image.astype(np.float32), dsize, center, max_radius,
            cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
        )
        return polar_image

    def _convert_to_cartesian(self, polar_image: np.ndarray, original_shape: tuple) -> np.ndarray:
        """Converts a single polar image back to Cartesian coordinates."""
        height, width = original_shape
        center = (width / 2.0, height / 2.0)
        max_radius = width / 2.0
        dsize = (width, height)

        cartesian_image = cv2.warpPolar(
            polar_image.astype(np.float32), dsize, center, max_radius,
            cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
        return cartesian_image

    def _compute_filter_mask(self, xp, shape, tolerance):
        """
        [Core Algorithm]
        Calculates the filter mask combining Vertical Notch (removes vertical stripes)
        and Horizontal Butterworth Protection (protects wide objects).

        Args:
            xp: numpy or cupy module
            shape: Shape of the padded image (rows, cols_padded)
            tolerance: Current filter intensity (1-100)
        """
        rows, cols_padded = shape
        processed_tolerance = 0.000025 * tolerance

        # --- Parameter Calculation ---
        # Calculate actual divisor 2^n
        cutoff_divisor = 2.0 ** self.cutoff_exponent
        if cutoff_divisor <= 0: cutoff_divisor = 64.0

        # Cutoff calculated based on Padded Width
        cutoff_threshold = cols_padded / cutoff_divisor
        if cutoff_threshold < 1.0: cutoff_threshold = 1.0

        # 1. Vertical Notch (Gaussian) - Removes vertical lines
        sigma = rows * processed_tolerance
        if sigma == 0: sigma = 1e-6

        y_coords = xp.arange(rows) - (rows // 2)
        notch_y = 1.0 - xp.exp(-(y_coords ** 2) / (2 * sigma ** 2))
        notch_y = notch_y.reshape(-1, 1)  # Reshape to column vector

        # 2. Horizontal Butterworth Protection (Low-pass)
        # Uses coordinate system of Padded Width
        x_coords = xp.arange(cols_padded) - (cols_padded // 2)
        epsilon = 1e-6
        order = self.butterworth_order  # Uses user-defined order

        # Low-pass Butterworth Formula: 1 / (1 + (w / wc)^(2n))
        protect_x = 1.0 / (1.0 + (xp.abs(x_coords) / (cutoff_threshold + epsilon)) ** (2 * order))
        protect_x = protect_x.reshape(1, -1)  # Reshape to row vector

        # 3. Overlay Logic (Maximum)
        filter_2d = xp.maximum(notch_y, protect_x)

        return filter_2d

    def run(self):
        """Executes the full processing pipeline (Mirror Padding -> FFT -> Crop)."""
        try:
            self.status_callback("Reading input 3D TIF file...")
            volume = tifffile.imread(self.input_path)

            num_slices = volume.shape[0]
            if not (0 <= self.z_index < num_slices):
                raise ValueError(f"Specified Z-slice {self.z_index} is out of range (0 to {num_slices - 1}).")

            target_slice = volume[self.z_index, :, :]
            original_shape = target_slice.shape

            self.status_callback(f"Selected slice {self.z_index}, starting polar conversion...")
            polar_slice = self._convert_to_polar(target_slice)

            filtered_results = []
            tolerance_range = np.arange(1, 101, 1)

            use_cpu_fallback = False

            # --- GPU Processing Block ---
            if CUPY_AVAILABLE:
                self.status_callback(
                    f"CuPy found, using GPU acceleration (n={self.cutoff_exponent}, order={self.butterworth_order})...")
                try:
                    polar_slice_gpu = cp.asarray(polar_slice)
                    rows, cols_orig = polar_slice_gpu.shape

                    # Mirror Padding: Width becomes 2x
                    cols_padded = cols_orig * 2

                    for i, tolerance in enumerate(tolerance_range):
                        self.status_callback(f"Processing on GPU... Intensity: {tolerance} ({i + 1}/100)")

                        if tolerance <= 0:
                            filtered_polar_gpu = polar_slice_gpu
                        else:
                            # 1. Prepare Mirror Padding Image
                            # [Flip | Original]
                            mirrored_part = cp.flip(polar_slice_gpu, axis=1)
                            combined_slice = cp.concatenate((mirrored_part, polar_slice_gpu), axis=1)

                            # 2. Compute Filter Mask (Based on Padded Shape)
                            filter_2d = self._compute_filter_mask(cp, (rows, cols_padded), tolerance)

                            # 3. Apply FFT
                            fft_slice = cupy_fft.fftshift(cupy_fft.fft2(combined_slice))
                            fft_slice *= filter_2d

                            # 4. Apply IFFT
                            slice_ifft = cupy_fft.ifft2(cupy_fft.ifftshift(fft_slice))
                            result_real = cp.real(slice_ifft)

                            # 5. Cropping: Keep original region (right half)
                            filtered_polar_gpu = result_real[:, cols_orig:]

                        filtered_polar_cpu = cp.asnumpy(filtered_polar_gpu)
                        final_slice = self._convert_to_cartesian(filtered_polar_cpu, original_shape)
                        filtered_results.append(final_slice)
                        self.progress_callback((i + 1) * 100 / len(tolerance_range))

                    cp.get_default_memory_pool().free_all_blocks()

                except Exception as e:
                    self.status_callback(f"GPU FFT failed: {e}. Switching to CPU.")
                    if cp: cp.get_default_memory_pool().free_all_blocks()
                    use_cpu_fallback = True
                    filtered_results = []

            # --- CPU Processing Block (Fallback) ---
            if not CUPY_AVAILABLE or use_cpu_fallback:
                if not use_cpu_fallback:
                    self.status_callback(
                        f"CuPy not found. Using CPU processing (n={self.cutoff_exponent}, order={self.butterworth_order})...")

                rows, cols_orig = polar_slice.shape
                cols_padded = cols_orig * 2

                for i, tolerance in enumerate(tolerance_range):
                    self.status_callback(f"Processing on CPU... Intensity: {tolerance} ({i + 1}/100)")

                    if tolerance <= 0:
                        filtered_polar = polar_slice
                    else:
                        # 1. Mirror Padding [Flip | Original]
                        mirrored_part = np.fliplr(polar_slice)
                        combined_slice = np.concatenate((mirrored_part, polar_slice), axis=1)

                        # 2. Compute Filter Mask
                        filter_2d = self._compute_filter_mask(np, (rows, cols_padded), tolerance)

                        # 3. FFT -> Mult -> IFFT
                        fft_slice = np.fft.fftshift(np.fft.fft2(combined_slice))
                        fft_slice *= filter_2d
                        slice_ifft = np.fft.ifft2(np.fft.ifftshift(fft_slice))
                        result_real = np.real(slice_ifft)

                        # 4. Cropping
                        filtered_polar = result_real[:, cols_orig:]

                    final_slice = self._convert_to_cartesian(filtered_polar, original_shape)
                    filtered_results.append(final_slice)
                    self.progress_callback((i + 1) * 100 / len(tolerance_range))

            self.status_callback("Stacking 100 images and saving...")
            output_volume = np.stack(filtered_results, axis=0).astype(np.float32)

            tifffile.imwrite(self.output_path, output_volume)
            self.status_callback(f"Processing complete! File saved to:\n{self.output_path}")
            messagebox.showinfo("Success", f"Processing complete! File saved to:\n{self.output_path}")

        except Exception as e:
            error_message = f"Error occurred: {e}"
            self.status_callback(error_message)
            messagebox.showerror("Error", error_message)


class App(tk.Tk):
    """Main Application GUI."""

    def __init__(self):
        super().__init__()
        self.title("FFT Ring Artifact Filter Test Tool V3")
        self.geometry("650x450")
        self.resizable(False, False)

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()

        # [UPDATED Defaults] Z=300, Exponent=6, Order=2
        self.z_index_var = tk.StringVar(value="300")
        self.cutoff_exponent_var = tk.StringVar(value="6")
        self.butterworth_order_var = tk.StringVar(value="2")

        # Bind variable changes to auto-update filename
        self.z_index_var.trace("w", self._update_output_filename)
        self.cutoff_exponent_var.trace("w", self._update_output_filename)
        self.butterworth_order_var.trace("w", self._update_output_filename)

        initial_status = "CuPy found, using GPU acceleration." if CUPY_AVAILABLE else "CuPy not found, using CPU processing."
        self.status_var = tk.StringVar(value=initial_status)

        self._create_widgets()

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)

        # File Settings Block
        path_frame = ttk.LabelFrame(main_frame, text="File Settings", padding="10")
        path_frame.pack(fill="x", pady=5)

        ttk.Button(path_frame, text="Select Input 3D TIF...", command=self._select_input_file).grid(row=0, column=0,
                                                                                                    padx=5, pady=5,
                                                                                                    sticky="w")
        ttk.Label(path_frame, textvariable=self.input_path, wraplength=450).grid(row=0, column=1, padx=5, pady=5,
                                                                                 sticky="w")

        ttk.Button(path_frame, text="Set Output 3D TIF...", command=self._select_output_file).grid(row=1, column=0,
                                                                                                   padx=5, pady=5,
                                                                                                   sticky="w")
        ttk.Label(path_frame, textvariable=self.output_path, wraplength=450).grid(row=1, column=1, padx=5, pady=5,
                                                                                  sticky="w")

        # Parameter Settings Block
        param_frame = ttk.LabelFrame(main_frame, text="Parameter Settings", padding="10")
        param_frame.pack(fill="x", pady=5)

        # Row 0: Z-Slice
        ttk.Label(param_frame, text="Target Z Slice (0-based):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        z_entry = ttk.Entry(param_frame, textvariable=self.z_index_var, width=10)
        z_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(param_frame, text="(Auto-runs intensity threshold)").grid(row=0, column=2, padx=10, pady=5, sticky="w")

        # Row 1: Cutoff Exponent
        ttk.Label(param_frame, text="Cutoff Exponent n (2^n):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        cutoff_entry = ttk.Entry(param_frame, textvariable=self.cutoff_exponent_var, width=10)
        cutoff_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(param_frame, text="(Protection width: smaller value = more protection)").grid(row=1, column=2,
                                                                                                padx=10, pady=5,
                                                                                                sticky="w")

        # Row 2: Butterworth Order
        ttk.Label(param_frame, text="Butterworth Order:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        order_entry = ttk.Entry(param_frame, textvariable=self.butterworth_order_var, width=10)
        order_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(param_frame, text="(Transition steepness: larger value = steeper)").grid(row=2, column=2, padx=10,
                                                                                           pady=5, sticky="w")

        # Execution Control Block
        run_frame = ttk.LabelFrame(main_frame, text="Execution Control", padding="10")
        run_frame.pack(fill="both", expand=True, pady=5)

        self.start_button = ttk.Button(run_frame, text="Start Processing", command=self._start_processing)
        self.start_button.pack(pady=5)

        self.progress_bar = ttk.Progressbar(run_frame, orient="horizontal", length=500, mode="determinate")
        self.progress_bar.pack(pady=5)

        status_label = ttk.Label(run_frame, textvariable=self.status_var, wraplength=600, justify="center")
        status_label.pack(pady=10)

    def _update_output_filename(self, *args):
        """Automatically updates output filename based on input and parameters."""
        if not self.input_path.get():
            return

        try:
            input_p = self.input_path.get()
            base, ext = os.path.splitext(input_p)
            z_val = self.z_index_var.get().strip()
            c_val = self.cutoff_exponent_var.get().strip()
            o_val = self.butterworth_order_var.get().strip()

            # Filename rule: Base_ringtest_Z{Z}_C{n}_O{Order}.tif
            new_name = f"{base}_ringtest_Z{z_val}_C{c_val}_O{o_val}{ext}"
            self.output_path.set(new_name)
        except Exception:
            pass  # Ignore temporary errors during typing

    def _select_input_file(self):
        path = filedialog.askopenfilename(title="Select Input 3D TIF File", filetypes=[("TIF Files", "*.tif *.tiff")])
        if path:
            self.input_path.set(path)
            self._update_output_filename()

    def _select_output_file(self):
        path = filedialog.asksaveasfilename(title="Set Output 3D TIF File", defaultextension=".tif",
                                            filetypes=[("TIF Files", "*.tif *.tiff")])
        if path:
            self.output_path.set(path)

    def _update_progress(self, value):
        self.progress_bar['value'] = value
        self.update_idletasks()

    def _update_status(self, text):
        self.status_var.set(text)
        self.update_idletasks()

    def _start_processing(self):
        input_p = self.input_path.get()
        output_p = self.output_path.get()

        if not input_p or not output_p:
            messagebox.showwarning("Warning", "Please set both input and output file paths.")
            return

        try:
            z_idx = int(self.z_index_var.get())
            cutoff_exp = float(self.cutoff_exponent_var.get())
            order_val = float(self.butterworth_order_var.get())
        except ValueError:
            messagebox.showerror("Error", "Z-slice must be an integer. Exponent and Order must be numbers.")
            return

        self.start_button.config(state="disabled")
        self.progress_bar['value'] = 0

        processor = FFTFilterProcessor(
            input_p, output_p, z_idx, cutoff_exp, order_val,
            self._update_progress, self._update_status
        )
        thread = threading.Thread(
            target=lambda: (processor.run(), self.after_idle(self.start_button.config, {'state': 'normal'})))
        thread.daemon = True
        thread.start()


if __name__ == '__main__':
    app = App()
    app.mainloop()