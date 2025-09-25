# -*- coding: utf-8 -*-
"""
此腳本為一個完整的3D斷層掃描重建與後處理工具。

主要功能流程如下：
1. 讓使用者透過圖形介面選擇包含連續2D投影影像（例如.tif格式）的資料夾。
2. 讀取所有投影影像，並可選地進行像素合併（Binning）以降低資料量與雜訊。
3. 使用 ASTRA Toolbox 的 FBP_CUDA 演算法，在GPU上高效地逐一切片重建3D斷層影像。
4. 將重建後的笛卡爾座標（Cartesian）3D體積轉換為極座標（Polar coordinates）。
5. 對極座標影像進行FFT濾波，以去除因探測器不一致造成的環狀偽影(Ring Artifacts)。
6. 將濾波後的極座標影像反向轉換回笛卡爾座標。
7. 根據設定，選擇性地儲存每一步驟的結果檔案。

所需函式庫：
- os, glob: 檔案路徑處理。
- numpy: 高效的數值計算。
- tifffile: 讀取與寫入 TIF 格式影像。
- tqdm: 提供進度條，方便監控處理進度。
- astra-toolbox: 高效能的斷層掃描重建工具箱。
- opencv-python (cv2): 用於影像處理，此處主要用於極座標轉換。
- scikit-image (skimage): 用於影像處理，此處用於像素合併。
- tkinter: Python內建的GUI函式庫，用於彈出資料夾選擇視窗。
"""

import os
import glob
import numpy as np
import tifffile
from tqdm import tqdm
import astra  # ASTRA Toolbox
import cv2  # OpenCV 函式庫
from skimage.transform import downscale_local_mean
import tkinter as tk
from tkinter import filedialog
from typing import Tuple, Optional

# --- 組態設定 (使用者可調整此處參數) ---
CONFIG = {
    # 檔案搜尋模式：用來尋找所有投影影像檔案
    "FILE_PATTERN": '*.tif',

    # 像素合併因子 (Binning Factor)
    # 設為 1 表示不進行合併。設為 2 表示將 2x2 的像素合併為 1 個像素。
    "BINNING_FACTOR": 1,

    # FFT 環狀偽影濾波強度 (強度參數)
    # 此為一個基礎強度參數，程式會透過公式 `0.001 * (強度參數 ** 2.5)` 轉換為實際濾波百分比。
    # 數值越高，濾波越強。建議從 5 開始測試。
    "TOLERANCE_VALUE": 5,

    # --- 各步驟存檔選項 ---
    # 設定為 True 表示要儲存該步驟的檔案，False 則不儲存。
    "SAVE_RECON_RAW": False,  # 是否儲存原始重建的檔案 (recon)
    "SAVE_POLAR_RAW": False,  # 是否儲存轉換後的極座標檔案 (polar)
    "SAVE_POLAR_FILTERED": False,  # 是否儲存濾波後的極座標檔案 (polar_filtered)
    "SAVE_RECON_FILTERED": True,  # 是否儲存最終濾波後的重建檔案 (recon_filtered)
}


# -----------------------------------------


def select_folder(title: str = "請選擇資料夾") -> Optional[str]:
    """
    彈出一個圖形化視窗，讓使用者選擇一個資料夾。

    Args:
        title (str): 選擇視窗的標題。

    Returns:
        Optional[str]: 如果使用者選擇了資料夾，則返回該資料夾的路徑字串；
                       如果使用者取消了選擇，則返回 None。
    """
    root = tk.Tk()
    root.withdraw()  # 隱藏不必要的主視窗
    folder_path = filedialog.askdirectory(title=title)
    if not folder_path:
        print("使用者未選擇任何資料夾，程式終止。")
        return None
    return folder_path


def load_projections(
        input_dir: str,
        file_pattern: str,
        binning_factor: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int], Optional[int]]:
    """
    從指定資料夾讀取、合併並預處理投影影像。

    Args:
        input_dir (str): 包含投影影像的資料夾路徑。
        file_pattern (str): 尋找影像檔案的模式 (例如 '*.tif')。
        binning_factor (int): 像素合併的因子。大於1時會縮小影像。

    Returns:
        tuple: 包含以下四個元素的元組：
               - projections_stack (np.ndarray | None): 預處理後的投影堆疊 (sinograms)。
               - angles_rad (np.ndarray | None): 投影角度陣列 (弧度)。
               - binned_height (int | None): 合併後影像的高度。
               - binned_width (int | None): 合併後影像的寬度。
               如果找不到任何檔案，則所有元素皆為 None。
    """
    print(f"在 '{input_dir}' 資料夾中搜尋 '{file_pattern}' 檔案...")
    search_path = os.path.join(input_dir, file_pattern)
    file_list = sorted(glob.glob(search_path))

    if not file_list:
        print(f"錯誤：在 '{search_path}' 中找不到任何檔案。請檢查路徑和檔案模式。")
        return None, None, None, None

    num_files = len(file_list)
    print(f"   找到 {num_files} 個投影檔案。")

    # 假設為 0-180 度線性掃描，自動計算每個投影的角度。
    # endpoint=False 表示不包含 180 度，符合常規掃描設定。
    angles_rad = np.linspace(0, 180, num_files, endpoint=False, dtype=np.float32)
    angles_rad = np.deg2rad(angles_rad)
    print(f"   -> 自動設定 0-180 度掃描，共 {num_files} 個角度。")

    print(f"正在讀取影像並進行 {binning_factor}x{binning_factor} 像素合併...")
    # 先讀取第一張影像，以確定合併後的影像維度
    first_image = tifffile.imread(file_list[0])
    if binning_factor > 1:
        # 使用局部平均值進行降採樣，效果比直接取點更好
        binned_first_image = downscale_local_mean(first_image, (binning_factor, binning_factor))
    else:
        binned_first_image = first_image

    binned_height, binned_width = binned_first_image.shape
    print(f"   原始影像維度: {first_image.shape}")
    print(f"   合併後影像維度: {(binned_height, binned_width)}")

    # 根據合併後維度，建立一個空的 NumPy 陣列來存放所有投影資料
    projections_stack = np.zeros((num_files, binned_height, binned_width), dtype=np.float32)

    # 迭代讀取所有影像，進行合併後存入陣列
    for i, file_path in enumerate(tqdm(file_list, desc="讀取與合併進度")):
        image = tifffile.imread(file_path).astype(np.float32)
        if binning_factor > 1:
            projections_stack[i, :, :] = downscale_local_mean(image, (binning_factor, binning_factor))
        else:
            projections_stack[i, :, :] = image

    print("   影像預處理完成。")
    return projections_stack, angles_rad, binned_height, binned_width


def reconstruct_volume_with_astra(
        projections_stack: np.ndarray,
        angles_rad: np.ndarray,
) -> np.ndarray:
    """
    使用 ASTRA Toolbox 逐一切片進行濾波反投影（FBP）重建。

    Args:
        projections_stack (np.ndarray): 預處理後的投影堆疊 (維度：[角度數量, 高度, 寬度])。
        angles_rad (np.ndarray): 對應的投影角度陣列 (弧度)。

    Returns:
        np.ndarray: 重建後的 3D 體積 (維度：[高度, 寬度, 寬度])。
    """
    print("開始使用 ASTRA 進行逐切片 GPU 重建...")
    num_projections, height, width = projections_stack.shape
    reconstructed_slices = []

    # 定義重建體積的幾何形狀 (一個寬x寬的正方形)
    vol_geom = astra.create_vol_geom(width, width)
    # 定義投影的幾何形狀 (平行光束，偵測器像素大小為1.0)
    proj_geom = astra.create_proj_geom('parallel', 1.0, width, angles_rad)

    # 對於每一層高度，取出一條 sinogram 進行 2D 重建
    for y in tqdm(range(height), desc="GPU 重建進度"):
        sinogram_slice = projections_stack[:, y, :]

        # 將單一 sinogram 數據傳輸到 ASTRA 的記憶體中
        sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram_slice)
        # 在 ASTRA 中建立一個空間來存放重建結果
        reco_id = astra.data2d.create('-vol', vol_geom)

        # 設定重建演算法
        cfg = astra.astra_dict('FBP_CUDA')  # 使用 CUDA 加速的 FBP 演算法
        cfg['ReconstructionDataId'] = reco_id
        cfg['ProjectionDataId'] = sinogram_id

        # 建立並執行演算法
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        # 從 ASTRA 取回重建後的 2D 切片
        reco_slice = astra.data2d.get(reco_id)
        reconstructed_slices.append(reco_slice)

        # 清理 ASTRA 記憶體，為下一個切片做準備，避免記憶體洩漏
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(reco_id)
        astra.data2d.delete(sinogram_id)

    # 將所有重建後的 2D 切片堆疊成一個 3D 體積
    reconstructed_volume = np.stack(reconstructed_slices, axis=0).astype(np.float32)
    print("   GPU 重建完成。")
    return reconstructed_volume


def convert_volume_to_polar(cartesian_volume: np.ndarray) -> np.ndarray:
    """
    將笛卡爾座標系的3D體積逐切片轉換為極座標系。

    Args:
        cartesian_volume (np.ndarray): 重建後的笛卡爾座標3D體積。

    Returns:
        np.ndarray: 轉換後的極座標3D體積。
    """
    print("開始後處理：將體積轉換為極座標...")

    z_dim, y_dim, x_dim = cartesian_volume.shape
    center_x = x_dim / 2.0
    center_y = y_dim / 2.0  # 以切片高度計算中心
    max_radius = x_dim / 2.0

    # 為了得到較好的轉換品質，設定超取樣的極座標影像尺寸
    # 寬度等於半徑，高度約等於圓周長，以確保長寬比失真較小
    oversampled_width = int(max_radius)
    oversampled_height = int(x_dim * np.pi)

    # 確保高度為偶數，某些後續處理可能需要
    if oversampled_height % 2 != 0:
        oversampled_height += 1

    oversampled_dsize = (oversampled_width, oversampled_height)
    print(f"   極座標超取樣尺寸 (寬x高): {oversampled_dsize}")

    polar_slices_list = []
    # 逐一切片進行座標轉換
    for cartesian_slice in tqdm(cartesian_volume, desc="極座標轉換進度"):
        # 使用 OpenCV 的 warpPolar 函式進行轉換
        polar_slice = cv2.warpPolar(
            cartesian_slice,
            oversampled_dsize,
            (center_x, center_y),
            max_radius,
            cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
        )
        polar_slices_list.append(polar_slice)

    # 將所有極座標切片堆疊成一個3D體積
    polar_volume = np.stack(polar_slices_list, axis=0).astype(np.float32)
    print("   極座標轉換完成。")
    return polar_volume


def fft_filter_vertical_stripes(
        polar_volume: np.ndarray,
        tolerance_percent: float = 0.1
) -> np.ndarray:
    """
    使用FFT帶阻濾波器(band-stop filter)去除極座標體積中的垂直條紋。
    環狀偽影(ring artifacts)在極座標下會表現為垂直條紋，此函式可有效抑制它們。

    Args:
        polar_volume (np.ndarray): 輸入的極座標3D體積 (維度: [Z, 角度, 半徑])。
        tolerance_percent (float): 濾波容忍度（百分比）。這個值決定了在頻域中
                                   要抑制的中心水平帶的寬度。5%是一個常用的起始值。

    Returns:
        np.ndarray: 去除條紋後的3D體積。
    """
    print(f"開始FFT濾波以去除垂直條紋 (容忍度: {tolerance_percent:.6f}%) ...")
    filtered_slices_list = []

    for polar_slice in tqdm(polar_volume, desc="FFT濾波進度"):
        # 步驟 1: 2D FFT 並將零頻率移到中心
        fft_slice = np.fft.fftshift(np.fft.fft2(polar_slice))

        # 步驟 2: 建立一個高斯漸弱的遮罩
        rows, cols = fft_slice.shape
        center_row = rows // 2

        # 垂直條紋對應到頻域是在水平軸(u-axis)上產生高能量。
        # 我們建立一個中心衰減的高斯遮罩來平滑地抑制這些頻率，以避免產生振鈴效應(Ringing artifact)。

        # 根據 tolerance_percent 計算高斯函數的標準差 (sigma)。
        # tolerance_percent 越高，高斯曲線越寬，抑制的頻帶也越寬。
        if tolerance_percent > 0:
            sigma = (rows * tolerance_percent / 100.0)

            # 建立一個垂直方向的座標陣列，中心點為 0
            y_coords = np.arange(rows) - center_row

            # 計算一維的高斯函數，這是一個反向高斯，中心為0，兩側為1
            gaussian_1d = 1.0 - np.exp(-(y_coords ** 2) / (2 * sigma ** 2))

            # 將一維遮罩擴展為二維(利用Numpy廣播機制)，使其能應用於整個頻譜影像
            gaussian_mask_2d = gaussian_1d.reshape(-1, 1)

            # 將遮罩應用於頻譜
            fft_slice *= gaussian_mask_2d

        # 步驟 3: 逆FFT
        slice_ifft = np.fft.ifft2(np.fft.ifftshift(fft_slice))

        # 步驟 4: 取實部
        filtered_slice = np.real(slice_ifft)
        filtered_slices_list.append(filtered_slice)

    filtered_volume = np.stack(filtered_slices_list, axis=0).astype(np.float32)
    print("   FFT濾波完成。")
    return filtered_volume


def convert_polar_to_cartesian(
        polar_volume: np.ndarray,
        target_dims: Tuple[int, int]
) -> np.ndarray:
    """
    將極座標系的3D體積逐切片反向轉換回笛卡爾座標系。

    Args:
        polar_volume (np.ndarray): 輸入的極座標3D體積。
        target_dims (Tuple[int, int]): 目標笛卡爾座標影像的尺寸 (高度, 寬度)。

    Returns:
        np.ndarray: 轉換回的笛卡爾座標3D體積。
    """
    print("開始後處理：將濾波後的極座標體積轉回笛卡爾座標...")

    target_height, target_width = target_dims
    center_x = target_width / 2.0
    center_y = target_height / 2.0
    max_radius = target_width / 2.0
    dsize = (target_width, target_height)

    cartesian_slices_list = []
    for polar_slice in tqdm(polar_volume, desc="反向極座標轉換進度"):
        # 使用 WARP_INVERSE_MAP 旗標進行反向轉換
        cartesian_slice = cv2.warpPolar(
            polar_slice,
            dsize,
            (center_x, center_y),
            max_radius,
            cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP + cv2.WARP_FILL_OUTLIERS
        )
        cartesian_slices_list.append(cartesian_slice)

    cartesian_volume = np.stack(cartesian_slices_list, axis=0).astype(np.float32)
    print("   反向轉換完成。")
    return cartesian_volume


def save_volume_as_tif(
        volume_data: np.ndarray,
        base_name: str,
        suffix: str,
        binning_factor: int,
        tolerance: Optional[float] = None
):
    """
    將 3D NumPy 陣列儲存為 TIF 檔案。

    Args:
        volume_data (np.ndarray): 要儲存的3D體積資料。
        base_name (str): 基礎檔名 (通常來自輸入資料夾名稱)。
        suffix (str): 檔名後綴，用以區分不同處理結果 (例如 "recon" 或 "polar")。
        binning_factor (int): 像素合併因子，用於記錄在檔名中。
        tolerance (Optional[float]): (可選) FFT濾波強度，用於記錄在檔名中。
    """
    # 取得體積維度 (Z, Y, X)
    z, y, x = volume_data.shape

    # 如果提供了tolerance值，建立對應的檔名部分
    tolerance_str = f"_T{tolerance}" if tolerance is not None else ""

    # 組合輸出檔名，格式為：基礎檔名_後綴_T(強度)_B(合併因子)_(維度X)x(維度Y)x(維度Z).tif
    output_filename = f"{base_name}_{suffix}{tolerance_str}_B{binning_factor}_{x}x{y}x{z}.tif"

    print(f"正在將 {suffix} 體積儲存至 '{output_filename}'...")
    try:
        tifffile.imwrite(output_filename, volume_data)
        print(f"   -> 檔案已成功儲存。")
    except Exception as e:
        print(f"   錯誤：儲存檔案 '{output_filename}' 失敗。詳細資訊: {e}")


def main():
    """主執行函式，協調整個處理流程。"""
    print("--- 開始 3D 重建與後處理任務 ---\n")

    # 步驟 1: 讓使用者選擇輸入資料夾
    input_dir = select_folder(title="請選擇包含投影影像的資料夾")
    if not input_dir:
        return  # 如果使用者取消選擇，則終止程式

    folder_name = os.path.basename(input_dir)

    # 步驟 2: 讀取與預處理影像
    projections, angles, _, _ = load_projections(
        input_dir,
        CONFIG["FILE_PATTERN"],
        CONFIG["BINNING_FACTOR"]
    )
    if projections is None:
        return  # 如果找不到檔案，則終止程式

    # 步驟 3: 使用 GPU 進行重建
    reconstructed_volume = reconstruct_volume_with_astra(projections, angles)

    # 步驟 4: (可選) 儲存原始的笛卡爾座標重建結果
    if CONFIG["SAVE_RECON_RAW"]:
        save_volume_as_tif(
            reconstructed_volume,
            folder_name,
            suffix="recon",
            binning_factor=CONFIG["BINNING_FACTOR"]
        )

    # 步驟 5: 將重建體積轉換為極座標
    polar_volume = convert_volume_to_polar(reconstructed_volume)

    # 步驟 6: (可選) 儲存極座標體積結果
    if CONFIG["SAVE_POLAR_RAW"]:
        save_volume_as_tif(
            polar_volume,
            folder_name,
            suffix="polar",
            binning_factor=CONFIG["BINNING_FACTOR"]
        )

    # 步驟 7: 對極座標體積進行FFT濾波，去除條紋
    # 讀取使用者輸入的基礎強度參數
    raw_tolerance_value = CONFIG["TOLERANCE_VALUE"]
    # 根據公式轉換為實際用於濾波的百分比
    processed_tolerance_percent = 0.001 * (raw_tolerance_value ** 2.5)

    filtered_polar_volume = fft_filter_vertical_stripes(
        polar_volume,
        tolerance_percent=processed_tolerance_percent
    )

    # 步驟 8: (可選) 儲存濾波後的極座標體積結果
    if CONFIG["SAVE_POLAR_FILTERED"]:
        save_volume_as_tif(
            filtered_polar_volume,
            folder_name,
            suffix="polar_filtered",
            binning_factor=CONFIG["BINNING_FACTOR"],
            tolerance=raw_tolerance_value
        )

    # 步驟 9: 將濾波後的極座標體積轉回笛卡爾座標
    # 取得原始重建的切片尺寸，用於反向轉換
    _, y_dim, x_dim = reconstructed_volume.shape
    filtered_cartesian_volume = convert_polar_to_cartesian(
        filtered_polar_volume, target_dims=(y_dim, x_dim)
    )

    # 步驟 10: (可選) 儲存最終轉回笛卡爾座標的濾波後體積
    if CONFIG["SAVE_RECON_FILTERED"]:
        save_volume_as_tif(
            filtered_cartesian_volume,
            folder_name,
            suffix="recon_filtered",
            binning_factor=CONFIG["BINNING_FACTOR"],
            tolerance=raw_tolerance_value
        )

    print("\n--- 3D 重建與後處理任務成功完成 ---")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n--- 程式執行期間發生未預期的嚴重錯誤 ---")
        import traceback

        traceback.print_exc()

