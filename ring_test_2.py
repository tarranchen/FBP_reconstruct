# -*- coding: utf-8 -*-
"""
此腳本為一個專門用於測試FFT環狀偽影(Ring Artifacts)濾波強度的工具。

功能流程如下：
1.  **讀取3D TIF檔案**：使用者透過GUI選擇一個已經過斷層掃描重建的3D TIF檔案。
2.  **指定Z層**：使用者輸入欲進行濾波測試的特定Z軸層數。
3.  **執行濾波迴圈 (GPU加速)**：
    -   對指定的Z層影像進行極座標轉換。
    -   若偵測到 CuPy 函式庫，將使用 GPU 加速執行一個從 1 到 100 (間隔1) 的迴圈，
        對極座標影像進行不同強度的FFT濾波。若無 CuPy 或 GPU 執行失敗，則自動切換回 CPU 處理。
    -   將每一張濾波後的極座標影像反向轉換回笛卡爾座標。
4.  **堆疊並儲存**：將迴圈中產生的100張2D影像堆疊成一個新的3D TIF檔案，並儲存至使用者指定的位置。

所需函式庫：
- tkinter: Python內建的GUI函式庫。
- os: 檔案與路徑處理。
- numpy: 高效的數值與陣列運算。
- tifffile: 讀取與寫入 TIF 格式影像。
- opencv-python (cv2): 用於影像的極座標與笛卡爾座標轉換。
- threading: 避免GUI在處理期間無回應。
- cupy (cupyx): (選用) 推薦安裝以啟用GPU加速FFT濾波。
"""

import os
import threading
import numpy as np
import tifffile
import cv2

# --- 嘗試導入 CuPy 以啟用 GPU 加速 ---
try:
    import cupy as cp
    from cupyx.scipy import fft as cupy_fft
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None  # 為避免未安裝 CuPy 時產生 NameError
    cupy_fft = None

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

class FFTFilterProcessor:
    """
    封裝了從讀取、濾波到儲存的完整後端處理邏輯。
    """
    def __init__(self, input_path: str, output_path: str, z_index: int, progress_callback, status_callback):
        self.input_path = input_path
        self.output_path = output_path
        self.z_index = z_index
        self.progress_callback = progress_callback
        self.status_callback = status_callback

    def _convert_to_polar(self, image: np.ndarray) -> np.ndarray:
        """將單張笛卡爾座標影像轉換為極座標影像。"""
        y_dim, x_dim = image.shape
        center = (x_dim / 2.0, y_dim / 2.0)
        max_radius = x_dim / 2.0

        oversampled_width = int(max_radius)
        oversampled_height = int(x_dim * np.pi)
        if oversampled_height % 2 != 0:
            oversampled_height += 1
        dsize = (oversampled_width, oversampled_height)

        polar_image = cv2.warpPolar(
            image.astype(np.float32), dsize, center, max_radius,
            cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
        )
        return polar_image

    def _apply_fft_filter(self, polar_image: np.ndarray, tolerance: float) -> np.ndarray:
        """(CPU版本) 對單張極座標影像應用FFT帶阻濾波器。"""
        processed_tolerance = 0.000025 * tolerance
        if processed_tolerance <= 0:
            return polar_image

        fft_slice = np.fft.fftshift(np.fft.fft2(polar_image))
        rows, _ = fft_slice.shape
        
        sigma = (rows * processed_tolerance)
        y_coords = np.arange(rows) - (rows // 2)
        gaussian_1d = 1.0 - np.exp(-(y_coords ** 2) / (2 * sigma ** 2))
        
        fft_slice *= gaussian_1d.reshape(-1, 1)
        
        slice_ifft = np.fft.ifft2(np.fft.ifftshift(fft_slice))
        
        return np.real(slice_ifft)

    def _convert_to_cartesian(self, polar_image: np.ndarray, original_shape: tuple) -> np.ndarray:
        """將單張極座標影像反向轉換回笛卡爾座標。"""
        height, width = original_shape
        center = (width / 2.0, height / 2.0)
        max_radius = width / 2.0
        dsize = (width, height)

        cartesian_image = cv2.warpPolar(
            polar_image.astype(np.float32), dsize, center, max_radius,
            cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
        return cartesian_image

    def run(self):
        """執行完整處理流程，若CuPy可用則優先使用GPU。"""
        try:
            self.status_callback("正在讀取輸入的3D TIF檔案...")
            volume = tifffile.imread(self.input_path)
            
            num_slices = volume.shape[0]
            if not (0 <= self.z_index < num_slices):
                raise ValueError(f"指定的Z層數 {self.z_index} 超出範圍 (0 到 {num_slices - 1})。")

            target_slice = volume[self.z_index, :, :]
            original_shape = target_slice.shape
            
            self.status_callback(f"已成功選取第 {self.z_index} 層，開始極座標轉換...")
            polar_slice = self._convert_to_polar(target_slice)
            
            filtered_results = []
            tolerance_range = np.arange(1, 101, 1)

            use_cpu_fallback = False
            if CUPY_AVAILABLE:
                self.status_callback("CuPy 已找到，嘗試使用 GPU 進行 FFT 濾波...")
                try:
                    polar_slice_gpu = cp.asarray(polar_slice)

                    for i, tolerance in enumerate(tolerance_range):
                        self.status_callback(f"GPU 處理中... 濾波強度: {tolerance} ({i+1}/100)")
                        
                        processed_tolerance = 0.000025 * tolerance
                        if processed_tolerance <= 0:
                            filtered_polar_gpu = polar_slice_gpu
                        else:
                            fft_slice_gpu = cupy_fft.fftshift(cupy_fft.fft2(polar_slice_gpu))
                            rows, _ = fft_slice_gpu.shape
                            sigma = rows * processed_tolerance
                            y_coords = cp.arange(rows, dtype=cp.float32) - (rows // 2)
                            gaussian_1d = 1.0 - cp.exp(-(y_coords ** 2) / (2 * (sigma ** 2)))
                            
                            fft_slice_gpu *= gaussian_1d.reshape(-1, 1)
                            
                            slice_ifft_gpu = cupy_fft.ifft2(cupy_fft.ifftshift(fft_slice_gpu))
                            filtered_polar_gpu = cp.real(slice_ifft_gpu)

                        filtered_polar_cpu = cp.asnumpy(filtered_polar_gpu)
                        final_slice = self._convert_to_cartesian(filtered_polar_cpu, original_shape)
                        filtered_results.append(final_slice)
                        self.progress_callback((i + 1) * 100 / len(tolerance_range))
                    
                    cp.get_default_memory_pool().free_all_blocks()
                
                except Exception as e:
                    self.status_callback(f"GPU FFT 濾波失敗: {e}。將切換至 CPU 處理。")
                    if cp: cp.get_default_memory_pool().free_all_blocks()
                    use_cpu_fallback = True
                    filtered_results = []

            if not CUPY_AVAILABLE or use_cpu_fallback:
                if not use_cpu_fallback:
                     self.status_callback("CuPy 未找到。開始使用 CPU 進行 FFT 濾波...")
                for i, tolerance in enumerate(tolerance_range):
                    self.status_callback(f"CPU 處理中... 濾波強度: {tolerance} ({i+1}/100)")
                    
                    filtered_polar = self._apply_fft_filter(polar_slice, tolerance)
                    
                    final_slice = self._convert_to_cartesian(filtered_polar, original_shape)
                    filtered_results.append(final_slice)
                    self.progress_callback((i + 1) * 100 / len(tolerance_range))

            self.status_callback("正在堆疊100張影像並儲存...")
            output_volume = np.stack(filtered_results, axis=0).astype(np.float32)
            
            tifffile.imwrite(self.output_path, output_volume)
            self.status_callback(f"處理完成！檔案已成功儲存至:\n{self.output_path}")
            messagebox.showinfo("成功", f"處理完成！檔案已成功儲存至:\n{self.output_path}")

        except Exception as e:
            error_message = f"發生錯誤: {e}"
            self.status_callback(error_message)
            messagebox.showerror("錯誤", error_message)


class App(tk.Tk):
    """主應用程式GUI視窗。"""
    def __init__(self):
        super().__init__()
        self.title("FFT環狀偽影濾波測試工具 (GPU加速版)")
        self.geometry("550x350")
        self.resizable(False, False)

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.z_index_var = tk.StringVar(value="0")

        initial_status = "CuPy 已找到，將使用 GPU 加速。" if CUPY_AVAILABLE else "CuPy 未找到，將使用 CPU 處理。"
        self.status_var = tk.StringVar(value=initial_status)

        self._create_widgets()

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)

        path_frame = ttk.LabelFrame(main_frame, text="檔案設定", padding="10")
        path_frame.pack(fill="x", pady=5)

        ttk.Button(path_frame, text="選擇輸入3D TIF檔...", command=self._select_input_file).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Label(path_frame, textvariable=self.input_path, wraplength=350).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Button(path_frame, text="設定輸出3D TIF檔...", command=self._select_output_file).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Label(path_frame, textvariable=self.output_path, wraplength=350).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        param_frame = ttk.LabelFrame(main_frame, text="參數設定", padding="10")
        param_frame.pack(fill="x", pady=5)

        ttk.Label(param_frame, text="指定Z層數 (從0開始):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        z_entry = ttk.Entry(param_frame, textvariable=self.z_index_var, width=10)
        z_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        run_frame = ttk.LabelFrame(main_frame, text="執行控制", padding="10")
        run_frame.pack(fill="both", expand=True, pady=5)

        self.start_button = ttk.Button(run_frame, text="開始處理", command=self._start_processing)
        self.start_button.pack(pady=10)
        
        self.progress_bar = ttk.Progressbar(run_frame, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(pady=5)
        
        status_label = ttk.Label(run_frame, textvariable=self.status_var, wraplength=500, justify="center")
        status_label.pack(pady=10)

    def _select_input_file(self):
        path = filedialog.askopenfilename(title="選擇輸入的3D TIF檔案", filetypes=[("TIF Files", "*.tif *.tiff")])
        if path:
            self.input_path.set(path)
            if not self.output_path.get():
                base, ext = os.path.splitext(path)
                self.output_path.set(f"{base}_FFT_TestStack{ext}")


    def _select_output_file(self):
        path = filedialog.asksaveasfilename(title="設定輸出的3D TIF檔案", defaultextension=".tif", filetypes=[("TIF Files", "*.tif *.tiff")])
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
            messagebox.showwarning("警告", "請先設定輸入與輸出檔案路徑。")
            return
            
        try:
            z_idx = int(self.z_index_var.get())
        except ValueError:
            messagebox.showerror("錯誤", "Z層數必須是一個有效的整數。")
            return

        self.start_button.config(state="disabled")
        self.progress_bar['value'] = 0
        
        processor = FFTFilterProcessor(input_p, output_p, z_idx, self._update_progress, self._update_status)
        thread = threading.Thread(target=lambda: (processor.run(), self.after_idle(self.start_button.config, {'state': 'normal'})))
        thread.daemon = True
        thread.start()


if __name__ == '__main__':
    app = App()
    app.mainloop()

