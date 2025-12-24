# -*- coding: utf-8 -*-
import os
import glob
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import numpy as np
import tifffile
import astra

# --- Check for GPU availability (CUDA is recommended for ASTRA) ---
try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not detected. ASTRA might run in CPU mode.")


class SingleLayerCTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Single Layer CT COR Tool")
        self.root.geometry("500x550")

        self.setup_ui()

    def setup_ui(self):
        # 1. Parameters Area
        frame_params = tk.LabelFrame(self.root, text="Parameters", padx=10, pady=10)
        frame_params.pack(fill="x", padx=10, pady=10)

        # Target Y Layer
        tk.Label(frame_params, text="Target Y Layer:").grid(row=0, column=0, sticky="e")
        self.entry_y_target = tk.Entry(frame_params, width=15)
        self.entry_y_target.insert(0, "500")  # Default 500
        self.entry_y_target.grid(row=0, column=1, sticky="w", padx=5)
        tk.Label(frame_params, text="(0 to ImageHeight-1)").grid(row=0, column=2, sticky="w")

        # Center Search Range
        tk.Label(frame_params, text="Center Search Range (N):").grid(row=1, column=0, sticky="e")
        self.entry_range = tk.Entry(frame_params, width=15)
        self.entry_range.insert(0, "30")  # Default 30
        self.entry_range.grid(row=1, column=1, sticky="w", padx=5)
        tk.Label(frame_params, text="(Range: -(N-1) to +N)").grid(row=1, column=2, sticky="w")

        # 2. Action Area
        frame_actions = tk.Frame(self.root, pady=10)
        frame_actions.pack(fill="x", padx=10)

        # Combined button: Browse -> Auto Start
        # Removed custom colors (bg/fg)
        self.btn_run = tk.Button(frame_actions, text="Select Folder & Start Reconstruction",
                                 font=("Arial", 12, "bold"),
                                 command=self.on_browse_and_run)
        self.btn_run.pack(fill="x", ipady=5)

        # 3. Log Area
        self.log_area = scrolledtext.ScrolledText(self.root, height=20)
        self.log_area.pack(fill="both", expand=True, padx=10, pady=10)

    def log(self, message):
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)

    def show_success_auto_close(self, title, message, timeout=3000):
        """Displays a popup window that closes automatically after 'timeout' ms."""
        top = tk.Toplevel(self.root)
        top.title(title)

        # Try to center relative to root
        try:
            x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 175
            y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 75
            top.geometry(f"350x150+{x}+{y}")
        except:
            top.geometry("350x150")

        lbl = tk.Label(top, text=message, wraplength=330, justify="center", padx=10, pady=10)
        lbl.pack(expand=True, fill="both")

        # Auto close after timeout (ms)
        top.after(timeout, top.destroy)

    def on_browse_and_run(self):
        # 1. Validate Parameters first
        try:
            target_y = int(self.entry_y_target.get())
            range_val = int(self.entry_range.get())

            if target_y < 0:
                raise ValueError("Y Layer must be a positive integer.")
            if range_val <= 0:
                raise ValueError("Range N must be greater than 0.")

        except ValueError as e:
            messagebox.showerror("Parameter Error", str(e))
            return

        # 2. Select Directory
        input_dir = filedialog.askdirectory()
        if not input_dir:
            return  # User cancelled

        # 3. Determine Output Path (Parent Directory)
        # Handle potential trailing slashes
        norm_path = os.path.normpath(input_dir)
        folder_name = os.path.basename(norm_path)
        parent_dir = os.path.dirname(norm_path)

        # Filename: {FolderName}_centertest_Y{Y_Layer}.tif
        output_filename = f"{folder_name}_centertest_Y{target_y}.tif"
        output_file = os.path.join(parent_dir, output_filename)

        self.log(f"Selected Folder: {input_dir}")
        self.log(f"Output File: {output_file}")

        # 4. Start Processing in Thread
        self.btn_run.config(state=tk.DISABLED)
        thread = threading.Thread(target=self.process_reconstruction,
                                  args=(input_dir, output_file, target_y, range_val))
        thread.start()

    def process_reconstruction(self, input_dir, output_file, target_y, range_val):
        try:
            self.log("--- Starting COR Scan Reconstruction ---")

            # Define search range
            # Input N -> Range -(N-1) to N
            shift_min = -(range_val - 1)
            shift_max = range_val

            self.log(f"Range Parameter N={range_val}")
            self.log(f"Center Offset Range: {shift_min} to +{shift_max}")

            # 1. Search Files
            file_pattern = "*.tif*"
            files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
            if not files:
                self.log("Error: No TIFF files found in directory.")
                return

            num_angles = len(files)
            self.log(f"Projection Count: {num_angles}")

            # 2. Read Dimensions
            with tifffile.TiffFile(files[0]) as tif:
                shape = tif.pages[0].shape
                orig_h, orig_w = shape

            self.log(f"Original Dimensions: {orig_w} x {orig_h}")
            orig_center = orig_w // 2
            self.log(f"Geometric Center (Width/2): {orig_center}")

            if target_y >= orig_h:
                self.log(f"Error: Target Y ({target_y}) exceeds image height ({orig_h}).")
                return

            # 3. Read Sinogram
            sinogram_full = np.zeros((num_angles, orig_w), dtype=np.float32)
            self.log(f"Reading Data for Y-Layer {target_y}...")

            start_time = time.time()
            for i, fname in enumerate(files):
                try:
                    img = tifffile.imread(fname)
                    sinogram_full[i, :] = img[target_y, :]
                except Exception as e:
                    self.log(f"Read Failed ({i}): {e}")
                    return
                if (i + 1) % 200 == 0:
                    self.log(f"Read {i + 1} / {num_angles}")

            self.log(f"Read Complete ({time.time() - start_time:.1f}s)")

            # 4. Prepare Output Stack
            shifts = range(shift_min, shift_max + 1)
            num_slices = len(shifts)

            # Use original width for canvas, crop later
            output_stack = np.zeros((num_slices, orig_w, orig_w), dtype=np.float32)

            self.log(f"Reconstructing {num_slices} different center points...")

            # 5. Loop Reconstruction
            for idx, shift_val in enumerate(shifts):
                # Calculate new target center
                target_center = orig_center + shift_val

                # Calculate symmetric crop radius
                radius = min(target_center, orig_w - 1 - target_center)

                crop_start = target_center - radius
                crop_end = target_center + radius + 1
                current_width = crop_end - crop_start

                if idx == 0 or idx == num_slices - 1 or idx == num_slices // 2:
                    self.log(f"Slice {idx} (Shift {shift_val}): Center={target_center}, Width={current_width}")

                # Crop Sinogram
                sino_crop = sinogram_full[:, crop_start:crop_end]

                # ASTRA Configuration
                vol_geom = astra.create_vol_geom(current_width, current_width)
                proj_geom = astra.create_proj_geom('parallel', 1.0, current_width,
                                                   np.linspace(0, np.pi, num_angles, endpoint=False))

                sino_id = astra.data2d.create('-sino', proj_geom, sino_crop)
                reco_id = astra.data2d.create('-vol', vol_geom)

                cfg = astra.astra_dict('FBP_CUDA')
                cfg['ReconstructionDataId'] = reco_id
                cfg['ProjectionDataId'] = sino_id
                alg_id = astra.algorithm.create(cfg)

                astra.algorithm.run(alg_id)
                rec_result = astra.data2d.get(reco_id)

                # Cleanup ASTRA
                astra.algorithm.delete(alg_id)
                astra.data2d.delete(reco_id)
                astra.data2d.delete(sino_id)

                # Center the result in the output stack
                pad_y = (orig_w - current_width) // 2
                pad_x = (orig_w - current_width) // 2

                output_stack[idx, pad_y: pad_y + current_width, pad_x: pad_x + current_width] = rec_result

                if (idx + 1) % 20 == 0:
                    self.log(f"Progress: {idx + 1} / {num_slices} done")

            # 6. Crop Outer Border & Save
            self.log(f"Cropping outer {range_val} pixels...")

            if range_val > 0 and (orig_w - 2 * range_val) > 0:
                final_stack = output_stack[:, range_val:-range_val, range_val:-range_val]
            else:
                self.log("Warning: Crop range too large or invalid, keeping original size.")
                final_stack = output_stack

            self.log(f"Final Stack Size: {final_stack.shape}")
            self.log(f"Saving 3D TIFF ({final_stack.nbytes / 1e9:.2f} GB)...")

            tifffile.imwrite(output_file, final_stack, photometric='minisblack')

            self.log("All Process Completed!")
            # Replaced blocking messagebox with auto-closing popup
            success_msg = f"File saved to:\n{output_file}\n(Total {num_slices} slices)"
            self.root.after(0, lambda: self.show_success_auto_close("Success", success_msg))

        except Exception as e:
            self.log(f"Critical Error: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

        finally:
            self.btn_run.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = SingleLayerCTApp(root)
    root.mainloop()