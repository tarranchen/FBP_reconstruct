# -*- coding: utf-8 -*-
"""
此腳本為一個完整的3D斷層掃描重建與後處理工具 (使用 Tkinter GUI)。

主要功能流程如下：
1.  **單一主視窗**：程式啟動時，會顯示一個整合的GUI視窗，讓使用者設定所有參數、勾選輸出並選擇資料夾。
2.  **讀取與預處理**：點擊開始後，讀取所有投影影像，並根據設定進行像素合併(Binning)。
3.  **GPU重建**：使用 ASTRA Toolbox 的 FBP_CUDA 演算法，在GPU上高效地逐一切片重建3D斷層影像。
4.  **極座標轉換**：將重建後的笛卡爾座標(Cartesian)3D體積轉換為極座標(Polar)。
    -   **目的**：此轉換是為了將影像中的環狀偽影(Ring Artifacts)變為垂直於半徑軸的直線條紋。
5.  **FFT濾波**：對極座標影像進行FFT(快速傅立葉變換)濾波，以高效去除上一步中產生的垂直條紋。
6.  **反向轉換**：將濾波後的極座標影像反向轉換回笛卡爾座標。
7.  **3D濾波 (GPU加速)**：
    -   **中值濾波**：對體積進行3D中值濾波，以去除椒鹽雜訊(Salt-and-pepper noise)。
    -   **高斯模糊**：對體積進行3D高斯模糊，以平滑影像並抑制高頻雜訊。
    -   **若偵測到 CuPy 函式庫，此步驟將自動在 GPU 上執行以大幅提升速度。**
8.  **智慧儲存**：根據使用者的勾選，僅儲存必要的結果，並自動跳過後續不需要的運算，節省處理時間。

所需函式庫：
- tkinter: Python內建的GUI函式庫。
- os, glob: 檔案與路徑處理。
- numpy: 高效的數值與陣列運算。
- tifffile: 讀取與寫入 TIF 格式影像。
- tqdm: 提供美觀的進度條，方便監控處理進度。
- astra-toolbox: 高效能的斷層掃描重建工具箱 (GPU加速)。
- opencv-python (cv2): 用於影像的極座標與笛卡爾座標轉換。
- scikit-image (skimage): 用於影像降採樣 (Binning)。
- cupy (cupyx): (可選，推薦) 用於GPU加速的3D濾波。
- scipy: 若無CuPy，則用於CPU上的3D濾波。
"""

import os
import sys
import glob
import numpy as np
import tifffile
from tqdm import tqdm
import astra  # ASTRA Toolbox
import cv2  # OpenCV 函式庫
from skimage.transform import downscale_local_mean
from typing import Optional, Dict, Any

import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, LabelFrame, Frame

# --- 嘗試導入 CuPy 以啟用 GPU 濾波 ---
try:
    import cupy as cp
    from cupyx.scipy.ndimage import median_filter as cupy_median_filter
    from cupyx.scipy.ndimage import gaussian_filter as cupy_gaussian_filter

    CUPY_AVAILABLE = True
    print("成功載入 CuPy，將使用 GPU 進行濾波加速。")
except ImportError:
    from scipy.ndimage import median_filter, gaussian_filter

    CUPY_AVAILABLE = False
    print("警告：未找到 CuPy。濾波將在 CPU 上執行，速度可能較慢。")

# --- 預設組態設定 ---
# 這些值將作為GUI視窗中顯示的預設值。
DEFAULT_CONFIG = {
    "FILE_PATTERN": '*.tif',
    "BINNING_FACTOR": 1,
    "TOLERANCE_VALUE": 28,
    "MEDIAN_FILTER_SIZE": 7,
    "GAUSSIAN_SIGMA": 2.5,
    "SAVE_RECON_RAW": False,
    "SAVE_POLAR_RAW": False,
    "SAVE_POLAR_FILTERED": False,
    "SAVE_RECON_FILTERED": False,
    "SAVE_RECON_FILTERED_MEDIAN": False,
    "SAVE_RECON_FILTERED_MEDIAN_GAUSS": True,
}


# -----------------------------------------


class TomoPipeline:
    """
    封裝了整個3D斷層掃描重建與後處理的流程。(後端處理邏輯，與GUI無關)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_volume: Optional[np.ndarray] = None
        self.projections: Optional[np.ndarray] = None
        self.angles_rad: Optional[np.ndarray] = None
        self.input_dir: str = ""
        self.base_name: str = ""
        self.file_params: Dict[str, Any] = {}

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
        print("--- 開始 3D 重建與後處理任務 ---\n")
        self.base_name = os.path.basename(self.input_dir)
        self.file_params = {"binning": self.config["BINNING_FACTOR"]}

        last_step_to_run = None
        for step in reversed(self.processing_steps):
            if self.config.get(step.get("save_key", "")):
                last_step_to_run = step["name"]
                break

        if not last_step_to_run:
            print("警告：未勾選任何存檔選項，程式將不會產生任何輸出。")
            return False

        for step in self.processing_steps:
            print(f"\n--- 步驟：{step['name']} ---")
            success = step["method"]()
            if not success:
                print(f"錯誤：步驟 {step['name']} 執行失敗，流程中止。")
                return False

            save_key = step.get("save_key")
            if save_key and self.config.get(save_key):
                self._save_volume(step["type"])

            if step["name"] == last_step_to_run:
                print("\n已完成所有指定步驟，處理流程結束。")
                break

        return True

    def _load_projections(self) -> bool:
        """從指定資料夾讀取、合併並預處理投影影像。"""
        print(f"在 '{self.input_dir}' 資料夾中搜尋 '{self.config['FILE_PATTERN']}' 檔案...")
        search_path = os.path.join(self.input_dir, self.config['FILE_PATTERN'])
        file_list = sorted(glob.glob(search_path))

        if not file_list:
            print(f"錯誤：在 '{search_path}' 中找不到任何檔案。請檢查路徑和檔案模式。")
            return False

        num_files = len(file_list)
        print(f"找到 {num_files} 個投影檔案。")
        self.angles_rad = np.deg2rad(np.linspace(0, 180, num_files, endpoint=False, dtype=np.float32))

        print(f"正在讀取影像並進行 {self.config['BINNING_FACTOR']}x{self.config['BINNING_FACTOR']} 像素合併...")
        first_image = tifffile.imread(file_list[0])
        bin_factor = self.config['BINNING_FACTOR']

        if bin_factor > 1:
            binned_shape = downscale_local_mean(first_image, (bin_factor, bin_factor)).shape
        else:
            binned_shape = first_image.shape

        binned_height, binned_width = binned_shape
        print(f"原始影像維度: {first_image.shape} -> 合併後維度: {binned_shape}")

        self.projections = np.zeros((num_files, binned_height, binned_width), dtype=np.float32)
        for i, file_path in enumerate(tqdm(file_list, desc="讀取與合併進度")):
            image = tifffile.imread(file_path).astype(np.float32)
            if bin_factor > 1:
                self.projections[i, :, :] = downscale_local_mean(image, (bin_factor, bin_factor))
            else:
                self.projections[i, :, :] = image

        return True

    def _reconstruct_volume(self) -> bool:
        """使用 ASTRA Toolbox 逐切片進行濾波反投影(FBP)重建。"""
        print("開始使用 ASTRA 進行逐切片 GPU 重建...")
        _, height, width = self.projections.shape
        reconstructed_slices = []

        vol_geom = astra.create_vol_geom(width, width)
        proj_geom = astra.create_proj_geom('parallel', 1.0, width, self.angles_rad)

        for y in tqdm(range(height), desc="GPU 重建進度"):
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

        self.data_volume = np.stack(reconstructed_slices, axis=0).astype(np.float32)
        print("GPU 重建完成。")
        return True

    def _convert_to_polar(self) -> bool:
        """將笛卡爾座標系的3D體積逐切片轉換為極座標系。"""
        print("將體積轉換為極座標以處理環狀偽影...")
        _, y_dim, x_dim = self.data_volume.shape
        center = (x_dim / 2.0, y_dim / 2.0)
        max_radius = x_dim / 2.0

        oversampled_width = int(max_radius)
        oversampled_height = int(x_dim * np.pi)
        if oversampled_height % 2 != 0: oversampled_height += 1
        dsize = (oversampled_width, oversampled_height)

        polar_slices = [
            cv2.warpPolar(s, dsize, center, max_radius, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
            for s in tqdm(self.data_volume, desc="極座標轉換進度")
        ]
        self.data_volume = np.stack(polar_slices, axis=0).astype(np.float32)
        print("極座標轉換完成。")
        return True

    def _apply_fft_filter(self) -> bool:
        """使用FFT帶阻濾波器去除極座標體積中的垂直條紋。"""
        tolerance = self.config["TOLERANCE_VALUE"]
        processed_tolerance = 0.000025 * tolerance
        self.file_params["tolerance"] = tolerance

        if processed_tolerance <= 0: return True

        print(f"開始FFT濾波以去除垂直條紋 (強度: {tolerance})...")
        filtered_slices = []
        for polar_slice in tqdm(self.data_volume, desc="FFT濾波進度"):
            fft_slice = np.fft.fftshift(np.fft.fft2(polar_slice))
            rows, _ = fft_slice.shape
            sigma = (rows * processed_tolerance)
            y_coords = np.arange(rows) - (rows // 2)
            gaussian_1d = 1.0 - np.exp(-(y_coords ** 2) / (2 * sigma ** 2))
            fft_slice *= gaussian_1d.reshape(-1, 1)
            slice_ifft = np.fft.ifft2(np.fft.ifftshift(fft_slice))
            filtered_slices.append(np.real(slice_ifft))

        self.data_volume = np.stack(filtered_slices, axis=0).astype(np.float32)
        print("FFT濾波完成。")
        return True

    def _convert_to_cartesian(self) -> bool:
        """將極座標系的3D體積反向轉換回笛卡爾座標系。"""
        print("將濾波後的極座標體積轉回笛卡爾座標...")
        _, height, width = self.projections.shape
        center = (width / 2.0, height / 2.0)
        max_radius = width / 2.0
        dsize = (width, height)

        cartesian_slices = [
            cv2.warpPolar(s, dsize, center, max_radius, cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            for s in tqdm(self.data_volume, desc="反向極座標轉換進度")
        ]
        self.data_volume = np.stack(cartesian_slices, axis=0).astype(np.float32)
        print("反向轉換完成。")
        return True

    def _apply_median_filter(self) -> bool:
        """對3D體積應用中值濾波器 (若CuPy可用則使用GPU)。"""
        size = self.config["MEDIAN_FILTER_SIZE"]
        self.file_params["median"] = size
        if size <= 1: return True

        # 計算Z軸的核心大小：取XY大小的一半，並確保為最接近的奇數
        z_size = size // 2
        if z_size % 2 == 0:  # 如果是偶數
            z_size += 1  # 加1使其變為奇數
        z_size = max(1, z_size)  # 確保最小為1
        filter_shape = (z_size, size, size)

        if CUPY_AVAILABLE:
            print(f"開始 GPU 3D中值濾波 (核心大小: {filter_shape})...")
            try:
                gpu_volume = cp.asarray(self.data_volume)
                filtered_gpu_volume = cupy_median_filter(gpu_volume, size=filter_shape)
                self.data_volume = cp.asnumpy(filtered_gpu_volume)
                cp.get_default_memory_pool().free_all_blocks()  # 釋放GPU記憶體
            except Exception as e:
                print(f"GPU中值濾波失敗: {e}。將嘗試使用CPU。")
                self.data_volume = median_filter(self.data_volume, size=filter_shape)  # Fallback to CPU
        else:
            print(f"開始 CPU 3D中值濾波 (核心大小: {filter_shape})...")
            self.data_volume = median_filter(self.data_volume, size=filter_shape)

        print("3D中值濾波完成。")
        return True

    def _apply_gaussian_filter(self) -> bool:
        """對3D體積應用高斯模糊 (若CuPy可用則使用GPU)。"""
        sigma = self.config["GAUSSIAN_SIGMA"]
        self.file_params["gauss"] = sigma
        if sigma <= 0: return True

        if CUPY_AVAILABLE:
            print(f"開始 GPU 3D高斯模糊 (Sigma: {sigma})...")
            try:
                gpu_volume = cp.asarray(self.data_volume)
                filtered_gpu_volume = cupy_gaussian_filter(gpu_volume, sigma=sigma)
                self.data_volume = cp.asnumpy(filtered_gpu_volume)
                cp.get_default_memory_pool().free_all_blocks()  # 釋放GPU記憶體
            except Exception as e:
                print(f"GPU高斯模糊失敗: {e}。將嘗試使用CPU。")
                self.data_volume = gaussian_filter(self.data_volume, sigma=sigma)  # Fallback to CPU
        else:
            print(f"開始 CPU 3D高斯模糊 (Sigma: {sigma})...")
            self.data_volume = gaussian_filter(self.data_volume, sigma=sigma)

        print("3D高斯模糊完成。")
        return True

    def _save_volume(self, volume_type: str):
        """將 3D NumPy 陣列儲存為 TIF 檔案，使用標準化命名格式。"""
        output_dir = os.path.dirname(self.input_dir)
        z, y, x = self.data_volume.shape
        param_parts = []
        if "tolerance" in self.file_params: param_parts.append(f"T{self.file_params['tolerance']}")
        if "median" in self.file_params: param_parts.append(f"M{self.file_params['median']}")
        if "gauss" in self.file_params: param_parts.append(f"G{self.file_params['gauss']}")
        param_parts.append(f"B{self.file_params['binning']}")
        param_parts.append(f"{x}x{y}x{z}")

        param_str = "_".join(param_parts)
        output_filename = f"{self.base_name}_{volume_type}_{param_str}.tif"
        output_path = os.path.join(output_dir, output_filename)

        print(f"正在將 {volume_type} 體積儲存至 '{output_path}'...")
        try:
            tifffile.imwrite(output_path, self.data_volume)
            print(f"-> 檔案已成功儲存。")
        except Exception as e:
            print(f"錯誤：儲存檔案 '{output_path}' 失敗。詳細資訊: {e}")


class App(tk.Tk):
    """
    主應用程式視窗，整合了參數設定、資料夾選擇和執行控制。
    """

    def __init__(self, defaults: Dict[str, Any]):
        super().__init__()
        self.defaults = defaults

        self.title("3D斷層掃描重建工具")
        self.resizable(False, False)

        self.entries: Dict[str, tk.StringVar] = {}
        self.save_vars: Dict[str, tk.BooleanVar] = {}
        self.folder_path = tk.StringVar()

        self._create_widgets()
        self._center_window()

    def _create_widgets(self):
        main_frame = Frame(self, padx=10, pady=10)
        main_frame.pack(fill="both", expand=True)

        # --- 參數輸入區 ---
        param_frame = LabelFrame(main_frame, text="濾波器參數", padx=10, pady=10)
        param_frame.pack(padx=10, pady=5, fill="x")

        param_defs = {
            "BINNING_FACTOR": "像素合併因子 (Binning):", "TOLERANCE_VALUE": "FFT 濾波強度 (Tolerance):",
            "MEDIAN_FILTER_SIZE": "中值濾波核心大小 (Median):", "GAUSSIAN_SIGMA": "高斯模糊標準差 (Gauss):"
        }
        for i, (key, label) in enumerate(param_defs.items()):
            tk.Label(param_frame, text=label).grid(row=i, column=0, padx=5, pady=3, sticky='w')
            var = tk.StringVar(value=str(self.defaults.get(key, "")))
            entry = tk.Entry(param_frame, width=15, textvariable=var)
            entry.grid(row=i, column=1, padx=5, pady=3)
            self.entries[key] = var

        # --- 存檔選項區 ---
        save_frame = LabelFrame(main_frame, text="選擇要輸出的檔案 (流程將執行至最後一個勾選項)", padx=10, pady=10)
        save_frame.pack(padx=10, pady=5, fill="x")

        save_defs = {
            "SAVE_RECON_RAW": "1. 原始重建檔案 (RECON)", "SAVE_POLAR_RAW": "2. 原始極座標檔案 (POLAR)",
            "SAVE_POLAR_FILTERED": "3. FFT濾波後極座標 (POLAR)", "SAVE_RECON_FILTERED": "4. FFT濾波後重建 (RECON)",
            "SAVE_RECON_FILTERED_MEDIAN": "5. 中值濾波後檔案 (RECON)",
            "SAVE_RECON_FILTERED_MEDIAN_GAUSS": "6. 高斯模糊後檔案 (RECON)"
        }
        for key, label in save_defs.items():
            var = tk.BooleanVar(value=self.defaults.get(key, False))
            chk = tk.Checkbutton(save_frame, text=label, variable=var)
            chk.pack(anchor="w")
            self.save_vars[key] = var

        # --- 執行控制區 ---
        run_frame = LabelFrame(main_frame, text="執行控制", padx=10, pady=10)
        run_frame.pack(padx=10, pady=5, fill="x")

        # 資料夾選擇
        folder_select_frame = Frame(run_frame)
        folder_select_frame.pack(fill='x', pady=5)
        tk.Button(folder_select_frame, text="選擇資料夾...", command=self._select_folder).pack(side="left")
        folder_label = tk.Label(folder_select_frame, textvariable=self.folder_path, fg="blue", wraplength=300,
                                justify="left")
        folder_label.pack(side="left", padx=10)
        self.folder_path.set("尚未選擇資料夾")

        # 主按鈕
        button_frame = Frame(run_frame)
        button_frame.pack(fill='x', pady=10)
        self.start_button = tk.Button(button_frame, text="開始重建", width=12, command=self._start_processing)
        self.start_button.pack(side="left", padx=10)
        tk.Button(button_frame, text="離開", width=12, command=self.destroy).pack(side="right", padx=10)

    def _select_folder(self):
        path = filedialog.askdirectory(parent=self, title="請選擇包含投影影像的資料夾")
        if path:
            self.folder_path.set(path)

    def _start_processing(self):
        input_dir = self.folder_path.get()
        if not os.path.isdir(input_dir):
            messagebox.showerror("錯誤", "請先選擇一個有效的資料夾。", parent=self)
            return

        try:
            # 獲取並驗證參數
            median_size = int(self.entries["MEDIAN_FILTER_SIZE"].get())
            if median_size % 2 == 0 and median_size > 1:
                median_size += 1
                print(f"提示：中值濾波核心大小已自動調整為奇數 {median_size}")

            user_params = {
                "BINNING_FACTOR": int(self.entries["BINNING_FACTOR"].get()),
                "TOLERANCE_VALUE": float(self.entries["TOLERANCE_VALUE"].get()),
                "MEDIAN_FILTER_SIZE": median_size,
                "GAUSSIAN_SIGMA": float(self.entries["GAUSSIAN_SIGMA"].get()),
            }
            for key, var in self.save_vars.items():
                user_params[key] = var.get()
        except ValueError:
            messagebox.showerror("輸入錯誤", "請確保所有參數均為有效的數字。", parent=self)
            return

        # 禁用按鈕，防止重複點擊
        self.start_button.config(state="disabled")
        self.update_idletasks()  # 立即更新UI

        # 執行後端處理
        config = self.defaults.copy()
        config.update(user_params)

        pipeline = TomoPipeline(config)
        success = pipeline.run(input_dir)

        # 顯示結果
        if success:
            messagebox.showinfo("成功", "任務成功！\n3D 重建與後處理任務已成功完成。", parent=self)
        else:
            messagebox.showerror("錯誤", "任務失敗！\n處理過程中發生錯誤，請查看終端機輸出以獲取詳細資訊。", parent=self)

        # 重新啟用按鈕
        self.start_button.config(state="normal")

    def _center_window(self):
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')


def main():
    """主執行函式，啟動GUI應用程式。"""
    app = App(DEFAULT_CONFIG)
    app.mainloop()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n--- 程式執行期間發生未預期的嚴重錯誤 ---")
        import traceback

        traceback.print_exc()
