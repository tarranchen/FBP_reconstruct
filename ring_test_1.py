# -*- coding: utf-8 -*-
"""
此腳本為一個專門用於測試FFT環狀偽影(Ring Artifacts)濾波強度的工具。

功能流程如下：
1.  **讀取3D TIF檔案**：使用者透過GUI選擇一個已經過斷層掃描重建的3D TIF檔案。
2.  **指定Z層**：使用者輸入欲進行濾波測試的特定Z軸層數。
3.  **執行濾波迴圈**：
    -   對指定的Z層影像進行極座標轉換。
    -   使用一個從 0.1 到 10.0 (間隔0.1) 的迴圈，對極座標影像進行不同強度的FFT濾波。
    -   將每一張濾波後的極座標影像反向轉換回笛卡爾座標。
4.  **堆疊並儲存**：將迴圈中產生的100張2D影像堆疊成一個新的3D TIF檔案，並儲存至使用者指定的位置。

所需函式庫：
- tkinter: Python內建的GUI函式庫。
- os: 檔案與路徑處理。
- numpy: 高效的數值與陣列運算。
- tifffile: 讀取與寫入 TIF 格式影像。
- opencv-python (cv2): 用於影像的極座標與笛卡爾座標轉換。
- threading: 避免GUI在處理期間無回應。
"""

import os
import threading
import numpy as np
import tifffile
import cv2
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

        # 為了避免失真，設定合理的輸出尺寸
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
        """對單張極座標影像應用FFT帶阻濾波器。"""
        processed_tolerance = 0.00025 * tolerance
        if processed_tolerance <= 0:
            return polar_image

        # 進行傅立葉變換
        fft_slice = np.fft.fftshift(np.fft.fft2(polar_image))
        rows, _ = fft_slice.shape

        # 建立一維高斯濾波器 (帶阻)
        sigma = (rows * processed_tolerance)
        y_coords = np.arange(rows) - (rows // 2)
        gaussian_1d = 1.0 - np.exp(-(y_coords ** 2) / (2 * sigma ** 2))

        # 將一維濾波器應用於二維頻譜
        fft_slice *= gaussian_1d.reshape(-1, 1)

        # 進行反傅立葉變換
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
        """執行完整處理流程。"""
        try:
            self.status_callback("正在讀取輸入的3D TIF檔案...")
            volume = tifffile.imread(self.input_path)

            num_slices = volume.shape[0]
            if not (0 <= self.z_index < num_slices):
                raise ValueError(f"指定的Z層數 {self.z_index} 超出範圍 (0 到 {num_slices - 1})。")

            # 提取指定的單一Z層
            target_slice = volume[self.z_index, :, :]
            original_shape = target_slice.shape

            self.status_callback(f"已成功選取第 {self.z_index} 層，開始極座標轉換...")
            polar_slice = self._convert_to_polar(target_slice)

            filtered_results = []
            tolerance_range = np.arange(0.1, 10.01, 0.1)

            for i, tolerance in enumerate(tolerance_range):
                self.status_callback(f"正在處理... 濾波強度: {tolerance:.1f} ({i + 1}/100)")

                # 步驟 5: FFT濾波
                filtered_polar = self._apply_fft_filter(polar_slice, tolerance)

                # 步驟 6: 反向轉換
                final_slice = self._convert_to_cartesian(filtered_polar, original_shape)
                filtered_results.append(final_slice)

                # 更新進度條
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
        self.title("FFT環狀偽影濾波測試工具")
        self.geometry("550x350")
        self.resizable(False, False)

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.z_index_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar(value="請設定參數並開始處理")

        self._create_widgets()

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)

        # --- 檔案路徑設定 ---
        path_frame = ttk.LabelFrame(main_frame, text="檔案設定", padding="10")
        path_frame.pack(fill="x", pady=5)

        ttk.Button(path_frame, text="選擇輸入3D TIF檔...", command=self._select_input_file).grid(row=0, column=0,
                                                                                                 padx=5, pady=5,
                                                                                                 sticky="w")
        ttk.Label(path_frame, textvariable=self.input_path, wraplength=350).grid(row=0, column=1, padx=5, pady=5,
                                                                                 sticky="w")

        ttk.Button(path_frame, text="設定輸出3D TIF檔...", command=self._select_output_file).grid(row=1, column=0,
                                                                                                  padx=5, pady=5,
                                                                                                  sticky="w")
        ttk.Label(path_frame, textvariable=self.output_path, wraplength=350).grid(row=1, column=1, padx=5, pady=5,
                                                                                  sticky="w")

        # --- 參數設定 ---
        param_frame = ttk.LabelFrame(main_frame, text="參數設定", padding="10")
        param_frame.pack(fill="x", pady=5)

        ttk.Label(param_frame, text="指定Z層數 (從0開始):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        z_entry = ttk.Entry(param_frame, textvariable=self.z_index_var, width=10)
        z_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # --- 執行與進度 ---
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
            # 自動建議輸出檔名
            if not self.output_path.get():
                base, ext = os.path.splitext(path)
                self.output_path.set(f"{base}_FFT_TestStack{ext}")

    def _select_output_file(self):
        path = filedialog.asksaveasfilename(title="設定輸出的3D TIF檔案", defaultextension=".tif",
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
            messagebox.showwarning("警告", "請先設定輸入與輸出檔案路徑。")
            return

        try:
            z_idx = int(self.z_index_var.get())
        except ValueError:
            messagebox.showerror("錯誤", "Z層數必須是一個有效的整數。")
            return

        self.start_button.config(state="disabled")
        self.progress_bar['value'] = 0

        # 使用執行緒來處理，避免GUI凍結
        processor = FFTFilterProcessor(input_p, output_p, z_idx, self._update_progress, self._update_status)
        thread = threading.Thread(target=lambda: (processor.run(), self.start_button.config(state="normal")))
        thread.daemon = True
        thread.start()


if __name__ == '__main__':
    app = App()
    app.mainloop()
