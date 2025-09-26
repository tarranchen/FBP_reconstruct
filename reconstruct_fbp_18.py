# -*- coding: utf-8 -*-
"""
3D斷層掃描重建與後處理工具 (Tkinter GUI)

================================================================================
工作流程總覽：
================================================================================
本腳本提供一個圖形化使用者介面 (GUI)，整合了從投影影像到最終三維體積的完整
斷層掃描重建與影像後處理流程。

主要處理步驟如下：
1.  **GUI 介面**：
    -   啟動時顯示一個主視窗，使用者可在此設定所有參數、勾選要儲存的
        中間/最終結果，並選擇包含投影影像的資料夾。

2.  **資料讀取與預處理 (CPU)**：
    -   點擊「開始重建」後，程式會讀取指定資料夾中的所有投影影像 (TIF格式)。
    -   根據使用者設定的「像素合併因子 (Binning)」，對影像進行降採樣以加速運算。

3.  **GPU 重建 (ASTRA Toolbox)**：
    -   使用 ASTRA Toolbox 的 `FBP_CUDA` 演算法，在 GPU 上逐一切片進行
        濾波反投影 (Filtered Back-Projection)，高效重建出 3D 體積。

4.  **環狀偽影校正 (極座標轉換 + FFT 濾波)**：
    a.  **笛卡爾轉極座標 (CPU - OpenCV)**：
        -   將重建後的 3D 體積逐一切片從笛卡爾座標 (Cartesian) 轉換為
            極座標 (Polar)。此轉換的目的是將影像中的同心圓「環狀偽影」
            (Ring Artifacts) 變換為垂直於半徑軸的直線條紋。
    b.  **FFT 帶阻濾波 (GPU/CPU)**：
        -   在頻域 (Fourier Domain) 中，對極座標影像進行快速傅立葉變換 (FFT)。
        -   應用一個高斯帶阻濾波器，專門抑制上一步中產生的垂直條紋所對應的
            頻率成分。
        -   若偵測到 CuPy 函式庫，此步驟將在 GPU 上執行以大幅提升速度；
            否則將使用 SciPy 在 CPU 上執行。
    c.  **極座標轉笛卡爾 (CPU - OpenCV)**：
        -   將濾波後的極座標影像反向轉換回笛卡爾座標。

5.  **3D 體積濾波 (降噪與平滑)**：
    a.  **3D 中值濾波 (GPU/CPU)**：
        -   對整個 3D 體積應用中值濾波器，有效去除椒鹽雜訊 (Salt-and-pepper noise)。
        -   同樣地，若 CuPy 可用，則在 GPU 上執行，否則使用 SciPy 在 CPU 執行。
    b.  **3D 高斯模糊 (CPU - SciPy)**：
        -   對 3D 體積進行高斯模糊，以平滑影像並抑制高頻雜訊。
        -   此步驟固定使用 Scipy 在 CPU 上執行，以確保結果的穩定性。

6.  **智慧儲存與執行**：
    -   程式會根據使用者在 GUI 中勾選的存檔選項，自動判斷需要執行的最後一個步驟。
    -   流程會執行到該步驟為止，並跳過後續所有不必要的運算，以節省處理時間。
    -   所有輸出檔案將以標準化格式命名，包含所有應用的濾波器參數。

================================================================================
所需函式庫：
================================================================================
- tkinter:          Python 內建的 GUI 函式庫。
- os, glob, sys:    用於檔案與系統路徑處理。
- threading:        用於在背景執行緒中處理數據，避免 GUI 凍結。
- typing:           提供型別提示，增強程式碼可讀性。
- numpy:            核心科學計算函式庫，用於高效的陣列運算。
- tifffile:         讀取與寫入 TIF 格式影像。
- astra-toolbox:    高效能的斷層掃描重建工具箱 (GPU 加速)。
- opencv-python:    用於影像的極座標與笛卡爾座標轉換 (cv2)。
- scikit-image:     用於影像降採樣 (Binning)。
- cupy (cupyx):     (選用，強力推薦) 用於 GPU 加速 3D 中值濾波與 FFT。
- scipy:            用於 CPU 上的 3D 濾波與 FFT 運算。
"""

# --- 標準函式庫導入 ---
import os
import sys
import glob
import time
import threading
from typing import Optional, Dict, Any

# --- GUI 相關函式庫導入 ---
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, LabelFrame, Frame, scrolledtext

# --- 第三方科學計算函式庫導入 ---
import numpy as np
import tifffile
import astra
import cv2
from skimage.transform import downscale_local_mean

# --- 嘗試導入 CuPy 以啟用 GPU 加速 ---
try:
    import cupy as cp
    from cupyx.scipy.ndimage import median_filter as cupy_median_filter
    from cupyx.scipy import fft as cupy_fft

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    # 將 cupy 相關變數設為 None，避免後續 NameError
    cp = None
    cupy_median_filter = None
    cupy_fft = None

# --- 導入 Scipy 作為 CPU 備援方案與高斯濾波器 ---
# 高斯模糊固定使用 Scipy；中值濾波和 FFT 在 CuPy 不可用時也使用 Scipy
from scipy.ndimage import median_filter, gaussian_filter
from scipy import fft as scipy_fft

# --- 預設組態設定 ---
# 這些值將作為 GUI 視窗中顯示的預設值。
DEFAULT_CONFIG = {
    "FILE_PATTERN": '*.tif',
    "BINNING_FACTOR": 1,
    "TOLERANCE_VALUE": 29,
    "MEDIAN_FILTER_RADIUS": 2,
    "GAUSSIAN_SIGMA": 2,
    "SAVE_RECON_RAW": False,
    "SAVE_POLAR_RAW": False,
    "SAVE_POLAR_FILTERED": False,
    "SAVE_RECON_FILTERED": False,
    "SAVE_RECON_FILTERED_MEDIAN": False,
    "SAVE_RECON_FILTERED_MEDIAN_GAUSS": True,
}


# ==============================================================================
# 輔助類別 (Helper Classes)
# ==============================================================================

class TextRedirector:
    """一個將 print 輸出重新導向到 Tkinter Text Widget 的類別。

    這個類別實作了 `write` 和 `flush` 方法，使其可以像檔案物件一樣被
    `sys.stdout` 和 `sys.stderr` 使用。所有寫入的文字都會被安全地插入
    到指定的 GUI 元件中。
    """

    def __init__(self, widget: scrolledtext.ScrolledText):
        self.widget = widget

    def write(self, text: str):
        """將文字寫入 Text Widget，並確保執行緒安全。"""

        # 使用 after_idle 確保 GUI 更新操作總是在主執行緒中執行
        def _write():
            self.widget.configure(state='normal')
            self.widget.insert(tk.END, text)
            self.widget.see(tk.END)  # 自動捲動到最底部
            self.widget.configure(state='disabled')

        self.widget.after_idle(_write)

    def flush(self):
        """flush 方法是必要的，但在這裡不需要做任何事，因為 Text Widget 會自動更新。"""
        pass


# ==============================================================================
# 後端處理邏輯 (Backend Logic)
# ==============================================================================

class TomoPipeline:
    """
    封裝整個 3D 斷層掃描重建與後處理的流程。

    這個類別負責處理所有與計算相關的任務，完全獨立於 GUI。
    它接收一個設定字典，並按照預先定義的順序執行所有處理步驟。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 TomoPipeline。

        Args:
            config (Dict[str, Any]): 包含所有使用者設定參數的字典。
        """
        self.config = config
        self.data_volume: Optional[np.ndarray] = None
        self.projections: Optional[np.ndarray] = None
        self.angles_rad: Optional[np.ndarray] = None
        self.input_dir: str = ""
        self.base_name: str = ""
        self.file_params: Dict[str, Any] = {}
        self.step_timings: Dict[str, float] = {}

        # 定義處理流程中的每一步，包含其名稱、對應的方法和儲存選項的鍵
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
        執行完整處理流程的主函式。

        Args:
            input_dir (str): 包含投影影像的資料夾路徑。

        Returns:
            bool: 如果流程成功完成則返回 True，否則返回 False。
        """
        self.input_dir = input_dir
        print("--- 開始 3D 重建與後處理任務 ---\n")
        self.base_name = os.path.basename(self.input_dir)
        self.file_params = {"binning": self.config["BINNING_FACTOR"]}
        self.step_timings.clear()
        total_start_time = time.time()

        # 智慧執行：從後往前檢查，找到使用者勾選的最後一個儲存步驟
        last_step_to_run = None
        for step in reversed(self.processing_steps):
            if self.config.get(step.get("save_key", "")):
                last_step_to_run = step["name"]
                break

        if not last_step_to_run:
            print("警告：未勾選任何存檔選項，程式將不會產生任何輸出。")
            return False

        # 依序執行處理步驟
        for step in self.processing_steps:
            print(f"\n--- 步驟：{step['name']} ---")
            step_start_time = time.time()
            success = step["method"]()
            self.step_timings[step['name']] = time.time() - step_start_time

            if not success:
                print(f"錯誤：步驟 {step['name']} 執行失敗，流程中止。")
                return False

            # 如果目前步驟對應的儲存選項被勾選，則儲存當前的 data_volume
            save_key = step.get("save_key")
            if save_key and self.config.get(save_key):
                self._save_volume(step["type"])

            # 如果已完成最後一個需要執行的步驟，則提前結束流程
            if step["name"] == last_step_to_run:
                print("\n已完成所有指定步驟，處理流程結束。")
                break

        self.step_timings["Total Processing Time"] = time.time() - total_start_time
        self._print_timing_summary()
        return True

    def _print_timing_summary(self):
        """印出每個步驟的耗時摘要。"""
        print("\n\n--- 處理時間摘要 ---")
        for name, duration in self.step_timings.items():
            print(f"{name:<35}: {duration:>7.2f} 秒")
        print("---------------------------------------------")

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

        binned_shape = (downscale_local_mean(first_image, (bin_factor, bin_factor)).shape
                        if bin_factor > 1 else first_image.shape)

        binned_height, binned_width = binned_shape
        print(f"原始影像維度: {first_image.shape} -> 合併後維度: {binned_shape}")

        self.projections = np.zeros((num_files, binned_height, binned_width), dtype=np.float32)

        update_interval = max(1, num_files // 10)  # 每 10% 更新一次進度
        for i, file_path in enumerate(file_list):
            image = tifffile.imread(file_path).astype(np.float32)
            if bin_factor > 1:
                self.projections[i, :, :] = downscale_local_mean(image, (bin_factor, bin_factor))
            else:
                self.projections[i, :, :] = image

            if (i + 1) % update_interval == 0 or (i + 1) == num_files:
                print(f"  > 讀取與合併進度: {i + 1} / {num_files} ({(i + 1) / num_files:.0%})")
        return True

    def _reconstruct_volume(self) -> bool:
        """使用 ASTRA Toolbox 逐切片進行濾波反投影 (FBP) 重建。"""
        print("開始使用 ASTRA 進行逐切片 GPU 重建...")
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

            # 清理 ASTRA 物件以釋放 GPU 記憶體
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(reco_id)
            astra.data2d.delete(sino_id)

            if (y + 1) % update_interval == 0 or (y + 1) == height:
                print(f"  > GPU 重建進度: {y + 1} / {height} ({(y + 1) / height:.0%})")

        self.data_volume = np.stack(reconstructed_slices, axis=0).astype(np.float32)
        print("GPU 重建完成。")
        return True

    def _convert_to_polar(self) -> bool:
        """將笛卡爾座標系的 3D 體積逐切片轉換為極座標系。"""
        print("將體積轉換為極座標以處理環狀偽影...")
        num_slices, y_dim, x_dim = self.data_volume.shape
        center = (x_dim / 2.0, y_dim / 2.0)
        max_radius = x_dim / 2.0

        # 計算合理的極座標影像尺寸
        oversampled_width = int(max_radius)
        oversampled_height = int(x_dim * np.pi)
        if oversampled_height % 2 != 0: oversampled_height += 1
        dsize = (oversampled_width, oversampled_height)

        polar_slices = []
        update_interval = max(1, num_slices // 10)
        for i, s in enumerate(self.data_volume):
            polar_slices.append(cv2.warpPolar(s, dsize, center, max_radius, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS))
            if (i + 1) % update_interval == 0 or (i + 1) == num_slices:
                print(f"  > 極座標轉換進度: {i + 1} / {num_slices} ({(i + 1) / num_slices:.0%})")

        self.data_volume = np.stack(polar_slices, axis=0).astype(np.float32)
        print("極座標轉換完成。")
        return True

    def _apply_fft_filter(self) -> bool:
        """使用 FFT 帶阻濾波器去除極座標體積中的垂直條紋。"""
        tolerance = self.config["TOLERANCE_VALUE"]
        self.file_params["tolerance"] = int(tolerance)

        if tolerance <= 0:
            print("FFT 濾波強度為 0，跳過此步驟。")
            return True

        processed_tolerance = 0.000025 * tolerance
        num_slices = self.data_volume.shape[0]
        update_interval = max(1, num_slices // 10)

        use_cpu_fallback = False
        if CUPY_AVAILABLE:
            print(f"開始 GPU FFT 濾波以去除垂直條紋 (強度: {int(tolerance)})...")
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
                        print(f"  > GPU FFT 濾波進度: {i + 1} / {num_slices} ({(i + 1) / num_slices:.0%})")

                self.data_volume = cp.asnumpy(filtered_gpu_volume)
                cp.get_default_memory_pool().free_all_blocks()

            except Exception as e:
                print(f"GPU FFT 濾波失敗: {e}。將改用 CPU 執行。")
                if cp:
                    cp.get_default_memory_pool().free_all_blocks()
                use_cpu_fallback = True

        if not CUPY_AVAILABLE or use_cpu_fallback:
            print(f"開始 CPU FFT 濾波以去除垂直條紋 (強度: {int(tolerance)})...")
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
                    print(f"  > CPU FFT 濾波進度: {i + 1} / {num_slices} ({(i + 1) / num_slices:.0%})")
            self.data_volume = np.stack(filtered_slices, axis=0).astype(np.float32)

        print("FFT 濾波完成。")
        return True

    def _convert_to_cartesian(self) -> bool:
        """將極座標系的 3D 體積反向轉換回笛卡爾座標系。"""
        print("將濾波後的極座標體積轉回笛卡爾座標...")
        _, height, width = self.projections.shape
        num_slices = self.data_volume.shape[0]
        center = (width / 2.0, height / 2.0)
        max_radius = width / 2.0
        dsize = (width, height)

        cartesian_slices = []
        update_interval = max(1, num_slices // 10)
        for i, s in enumerate(self.data_volume):
            cartesian_slices.append(
                cv2.warpPolar(s, dsize, center, max_radius, cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP))
            if (i + 1) % update_interval == 0 or (i + 1) == num_slices:
                print(f"  > 反向極座標轉換進度: {i + 1} / {num_slices} ({(i + 1) / num_slices:.0%})")

        self.data_volume = np.stack(cartesian_slices, axis=0).astype(np.float32)
        print("反向轉換完成。")
        return True

    def _apply_median_filter(self) -> bool:
        """對 3D 體積應用中值濾波器 (若 CuPy 可用則使用 GPU)。"""
        radius = self.config["MEDIAN_FILTER_RADIUS"]
        self.file_params["median"] = radius
        if radius <= 0:
            print("中值濾波半徑為 0，跳過此步驟。")
            return True

        z_radius = radius // 2
        xy_kernel = radius * 2 + 1
        z_kernel = z_radius * 2 + 1
        filter_shape = (z_kernel, xy_kernel, xy_kernel)

        use_cpu_fallback = False
        if CUPY_AVAILABLE:
            print(f"開始 GPU 3D 中值濾波 (核心 X, Y, Z: {xy_kernel}, {xy_kernel}, {z_kernel})...")
            try:
                gpu_volume = cp.asarray(self.data_volume)
                filtered_gpu_volume = cupy_median_filter(gpu_volume, size=filter_shape)
                self.data_volume = cp.asnumpy(filtered_gpu_volume)
                cp.get_default_memory_pool().free_all_blocks()
            except Exception as e:
                print(f"GPU 中值濾波失敗: {e}。將嘗試使用 CPU。")
                if cp:
                    cp.get_default_memory_pool().free_all_blocks()
                use_cpu_fallback = True

        if not CUPY_AVAILABLE or use_cpu_fallback:
            print(f"開始 CPU 3D 中值濾波 (核心 X, Y, Z: {xy_kernel}, {xy_kernel}, {z_kernel})...")
            self.data_volume = median_filter(self.data_volume, size=filter_shape)

        print("3D 中值濾波完成。")
        return True

    def _apply_gaussian_filter(self) -> bool:
        """對 3D 體積應用高斯模糊 (固定使用 CPU 上的 Scipy)。"""
        sigma = self.config["GAUSSIAN_SIGMA"]
        self.file_params["gauss"] = sigma
        if sigma <= 0:
            print("高斯模糊標準差為 0，跳過此步驟。")
            return True

        print(f"開始 CPU 3D 高斯模糊 (Sigma: {sigma})...")
        self.data_volume = gaussian_filter(self.data_volume, sigma=sigma)
        print("3D 高斯模糊完成。")
        return True

    def _save_volume(self, volume_type: str):
        """
        將 3D NumPy 陣列儲存為 TIF 檔案，使用包含參數的標準化命名格式。
        命名格式: {basename}_{type}_{params}_{dimensions}.tif
        例如: MyData_RECON_B1_T28_M3_G1.8_512x512x512.tif
        """
        # 輸出檔案將儲存在輸入資料夾的上一層目錄
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

        print(f"正在將 {volume_type} 體積儲存至 '{output_path}'...")
        try:
            tifffile.imwrite(output_path, self.data_volume.astype(np.float32))
            print(f"-> 檔案已成功儲存。")
        except Exception as e:
            print(f"錯誤：儲存檔案 '{output_path}' 失敗。詳細資訊: {e}")


# ==============================================================================
# 圖形化使用者介面 (GUI)
# ==============================================================================

class App(tk.Tk):
    """
    主應用程式視窗，整合了參數設定、資料夾選擇和執行控制。
    """

    def __init__(self, defaults: Dict[str, Any]):
        super().__init__()
        self.defaults = defaults
        self.title("3D 斷層掃描重建工具")
        self.resizable(True, True)

        self.entries: Dict[str, tk.StringVar] = {}
        self.save_vars: Dict[str, tk.BooleanVar] = {}
        self.folder_path = tk.StringVar(value="尚未選擇資料夾")

        self._create_widgets()
        self._center_window()

        # 將 stdout 和 stderr 重新導向到 GUI 中的 log widget
        # 這必須在 widget 建立之後執行
        sys.stdout = TextRedirector(self.log_text)
        sys.stderr = TextRedirector(self.log_text)

        # 初始訊息
        print("3D 斷層掃描重建工具已啟動。")
        if CUPY_AVAILABLE:
            print("成功載入 CuPy，將使用 GPU 進行中值濾波與 FFT 加速。")
        else:
            print("警告：未找到 CuPy。FFT 與中值濾波將在 CPU 上執行。")
            print("提示：若要啟用 GPU 加速，請安裝 CuPy 函式庫。")

    def _create_widgets(self):
        """建立並佈局所有 GUI 元件。"""
        main_frame = Frame(self, padx=10, pady=10)
        main_frame.pack(fill="both", expand=True)
        main_frame.rowconfigure(3, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # --- 參數輸入區 ---
        param_frame = LabelFrame(main_frame, text="濾波器參數", padx=10, pady=10)
        param_frame.grid(row=0, column=0, padx=10, pady=5, sticky='ew')
        param_defs = {
            "BINNING_FACTOR": "像素合併因子 (Binning):", "TOLERANCE_VALUE": "FFT 濾波強度 (Tolerance):",
            "MEDIAN_FILTER_RADIUS": "中值濾波半徑 (Median Radius):", "GAUSSIAN_SIGMA": "高斯模糊標準差 (Gauss):"
        }
        for i, (key, label) in enumerate(param_defs.items()):
            tk.Label(param_frame, text=label).grid(row=i, column=0, padx=5, pady=3, sticky='w')
            var = tk.StringVar(value=str(self.defaults.get(key, "")))
            entry = tk.Entry(param_frame, width=15, textvariable=var)
            entry.grid(row=i, column=1, padx=5, pady=3)
            self.entries[key] = var

        # --- 存檔選項區 ---
        save_frame = LabelFrame(main_frame, text="選擇要輸出的檔案 (流程將執行至最後一個勾選項)", padx=10, pady=10)
        save_frame.grid(row=1, column=0, padx=10, pady=5, sticky='ew')
        save_defs = {
            "SAVE_RECON_RAW": "1. 原始重建檔案 (RECON)", "SAVE_POLAR_RAW": "2. 原始極座標檔案 (POLAR)",
            "SAVE_POLAR_FILTERED": "3. FFT 濾波後極座標 (POLAR)", "SAVE_RECON_FILTERED": "4. FFT 濾波後重建 (RECON)",
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
        run_frame.grid(row=2, column=0, padx=10, pady=5, sticky='ew')
        folder_select_frame = Frame(run_frame)
        folder_select_frame.pack(fill='x', pady=5)
        tk.Button(folder_select_frame, text="選擇資料夾...", command=self._select_folder).pack(side="left")
        folder_label = tk.Label(folder_select_frame, textvariable=self.folder_path, fg="blue", wraplength=300,
                                justify="left")
        folder_label.pack(side="left", padx=10)
        button_frame = Frame(run_frame)
        button_frame.pack(fill='x', pady=10)
        self.start_button = tk.Button(button_frame, text="開始重建", width=12, command=self._start_processing_thread)
        self.start_button.pack(side="left", padx=10)
        tk.Button(button_frame, text="離開", width=12, command=self.destroy).pack(side="right", padx=10)

        # --- Log 輸出區 ---
        log_frame = LabelFrame(main_frame, text="執行日誌", padx=10, pady=10)
        log_frame.grid(row=3, column=0, padx=10, pady=5, sticky='nsew')
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, state='disabled', height=15, wrap=tk.WORD, bg="#f0f0f0")
        self.log_text.grid(row=0, column=0, sticky='nsew')

    def _select_folder(self):
        """開啟對話框讓使用者選擇資料夾。"""
        path = filedialog.askdirectory(parent=self, title="請選擇包含投影影像的資料夾")
        if path:
            self.folder_path.set(path)

    def _start_processing_thread(self):
        """啟動一個背景執行緒來處理數據，避免 GUI 凍結。"""
        input_dir = self.folder_path.get()
        if not os.path.isdir(input_dir):
            messagebox.showerror("錯誤", "請先選擇一個有效的資料夾。", parent=self)
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
            messagebox.showerror("輸入錯誤", "請確保所有參數均為有效的數字。", parent=self)
            return

        self.start_button.config(state="disabled")
        config = self.defaults.copy()
        config.update(user_params)

        thread = threading.Thread(target=self._run_pipeline, args=(config, input_dir))
        thread.daemon = True  # 設置為守護執行緒，主視窗關閉時會自動退出
        thread.start()

    def _run_pipeline(self, config, input_dir):
        """此函式在背景執行緒中運行，執行主要的處理流程。"""
        pipeline = TomoPipeline(config)
        success = pipeline.run(input_dir)
        self.after_idle(self._on_processing_complete, success)

    def _on_processing_complete(self, success: bool):
        """此函式在主執行緒中運行，用於顯示最終訊息並重新啟用按鈕。"""
        if success:
            messagebox.showinfo("成功", "任務成功！\n3D 重建與後處理任務已成功完成。", parent=self)
        else:
            messagebox.showerror("錯誤", "任務失敗！\n處理過程中發生錯誤，請查看日誌視窗以獲取詳細資訊。", parent=self)
        self.start_button.config(state="normal")

    def _center_window(self):
        """計算並設定視窗的啟動位置，使其在螢幕上置中。"""
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
# 主程式進入點 (Main Entry Point)
# ==============================================================================

def main():
    """主執行函式，建立並啟動 GUI 應用程式。"""
    try:
        app = App(DEFAULT_CONFIG)
        app.mainloop()
    except Exception as e:
        print(f"\n--- 程式執行期間發生未預期的嚴重錯誤 ---")
        import traceback
        traceback.print_exc()
        # 即使在 GUI 初始化失敗時，也嘗試彈出錯誤視窗
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("嚴重錯誤", f"發生未預期的錯誤，程式即將關閉。\n\n詳細資訊:\n{e}")


if __name__ == '__main__':
    main()

