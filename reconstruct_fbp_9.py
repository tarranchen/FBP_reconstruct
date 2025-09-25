import os
import glob
import numpy as np
import tifffile
from tqdm import tqdm
import astra  # ASTRA 工具箱
import cv2  # OpenCV 函式庫，用於極座標轉換
from skimage.transform import downscale_local_mean
import tkinter as tk
from tkinter import filedialog

# --- 參數設定 (請根據您的需求修改) ---

# 檔案搜尋模式：用來尋找所有投影影像檔案
FILE_PATTERN = '*.tif'

# 像素合併因子 (Binning Factor)
# 設為 1 表示不進行合併
BINNING_FACTOR = 1


def select_folder(title="請選擇資料夾"):
    """
    彈出視窗讓使用者選擇資料夾。

    Args:
        title (str): 視窗標題。

    Returns:
        str: 所選資料夾的路徑，如果取消則返回 None。
    """
    root = tk.Tk()
    root.withdraw()  # 隱藏主視窗
    folder_path = filedialog.askdirectory(title=title)
    if not folder_path:
        print("未選擇任何資料夾，程式終止。")
        return None
    return folder_path


def load_projections(input_dir, file_pattern, binning_factor):
    """
    從指定資料夾讀取、合併並預處理投影影像。

    Args:
        input_dir (str): 包含投影影像的資料夾路徑。
        file_pattern (str): 尋找影像檔案的模式 (例如 '*.tif')。
        binning_factor (int): 像素合併的因子。

    Returns:
        tuple: 包含 (投影堆疊, 角度陣列, 合併後高度, 合併後寬度) 的元組。
               如果找不到檔案則返回 (None, None, None, None)。
    """
    print(f"1. 在 '{input_dir}' 資料夾中搜尋檔案...")
    search_path = os.path.join(input_dir, file_pattern)
    file_list = sorted(glob.glob(search_path))

    if not file_list:
        print(f"錯誤：在 '{search_path}' 中找不到任何檔案。請檢查路徑和檔案模式。")
        return None, None, None, None

    num_files = len(file_list)
    print(f"   找到 {num_files} 個投影檔案。")

    # 根據檔案數量自動判斷角度設定 (假設為 0-180 度線性掃描)
    angles_rad = np.deg2rad(np.linspace(0, 180, num_files, endpoint=False, dtype=np.float32))
    print(f"   -> 自動設定 0-180 度掃描，共 {num_files} 個角度。")

    print(f"2. 正在讀取並進行 {binning_factor}x{binning_factor} 像素合併...")
    # 讀取第一張影像以獲取維度資訊
    first_image = tifffile.imread(file_list[0])
    if binning_factor > 1:
        binned_first_image = downscale_local_mean(first_image, (binning_factor, binning_factor))
    else:
        binned_first_image = first_image

    height, width = binned_first_image.shape
    print(f"   原始影像維度: {first_image.shape}")
    print(f"   合併後影像維度: {(height, width)}")

    # 建立空的 NumPy 陣列以存放所有投影
    projections_stack = np.zeros((num_files, height, width), dtype=np.float32)

    # 讀取所有影像並進行合併
    for i, file_path in enumerate(tqdm(file_list, desc="讀取與合併進度")):
        image = tifffile.imread(file_path).astype(np.float32)
        if binning_factor > 1:
            projections_stack[i, :, :] = downscale_local_mean(image, (binning_factor, binning_factor))
        else:
            projections_stack[i, :, :] = image

    print("   影像預處理完成。")
    return projections_stack, angles_rad, height, width


def reconstruct_with_astra(projections_stack, angles_rad, height, width):
    """
    使用 ASTRA 工具箱逐切片進行 FBP 重建。

    Args:
        projections_stack (np.ndarray): 預處理後的投影堆疊 (sinograms)。
        angles_rad (np.ndarray): 投影角度 (弧度)。
        height (int): 投影影像的高度 (切片數量)。
        width (int): 投影影像的寬度 (偵測器寬度)。

    Returns:
        np.ndarray: 重建後的 3D 體積。
    """
    print("3. 開始使用 ASTRA 進行逐切片 GPU 重建...")
    reconstructed_slices = []

    # 定義重建與投影的幾何形狀
    vol_geom = astra.create_vol_geom(width, width)
    proj_geom = astra.create_proj_geom('parallel', 1.0, width, angles_rad)

    for y in tqdm(range(height), desc="GPU 重建進度"):
        sinogram_slice = projections_stack[:, y, :]

        # 將 Sinogram 數據傳輸到 ASTRA
        sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram_slice)

        # 建立存放重建結果的空間
        reco_id = astra.data2d.create('-vol', vol_geom)

        # 設定 FBP_CUDA 重建演算法
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = reco_id
        cfg['ProjectionDataId'] = sinogram_id

        # 建立並執行演算法
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        # 從 ASTRA 取回重建切片
        reco_slice = astra.data2d.get(reco_id)
        reconstructed_slices.append(reco_slice)

        # 清理 ASTRA 記憶體以供下一個切片使用
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(reco_id)
        astra.data2d.delete(sinogram_id)

    reconstructed_volume = np.stack(reconstructed_slices, axis=0).astype(np.float32)
    print("   GPU 重建完成。")
    return reconstructed_volume


def process_and_save_polar_volume(volume, binning, folder_name):
    """
    將重建後的體積轉換為極座標，進行裁切、翻轉、拼接、旋轉並儲存。

    Args:
        volume (np.ndarray): 重建後的 3D 體積。
        binning (int): 像素合併因子，用於命名。
        folder_name (str): 輸入資料夾名稱，用於命名。
    """
    print("4. 開始進行後處理：轉換為極座標並拼接...")

    center_x = volume.shape[2] / 2.0
    center_y = volume.shape[1] / 2.0
    max_radius = center_x

    # --- 步驟 1: 將所有切片轉換至超取樣極座標 ---
    print("   - 步驟 1/3: 將所有切片轉換至超取樣極座標...")
    # 計算超取樣尺寸
    oversampled_width = int(center_x)
    oversampled_height = int(volume.shape[1] * np.pi)
    if oversampled_height % 2 != 0:
        oversampled_height += 1  # 確保為偶數
    oversampled_dsize = (oversampled_width, oversampled_height)
    print(f"     極座標超取樣尺寸 (寬x高): {oversampled_dsize}")

    polar_images_list = []
    for cartesian_slice in tqdm(volume, desc="極座標轉換進度"):
        polar_image = cv2.warpPolar(cartesian_slice, oversampled_dsize, (center_x, center_y), max_radius,
                                    cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        polar_images_list.append(polar_image)
    polar_volume = np.stack(polar_images_list, axis=0).astype(np.float32)

    # --- 步驟 2: 裁切、翻轉、拼接與旋轉極座標體積 ---
    print("   - 步驟 2/3: 裁切、翻轉、拼接與旋轉極座標體積...")
    polar_height = polar_volume.shape[1]
    mid_point_y = polar_height // 2

    top_half = polar_volume[:, :mid_point_y, :]
    bottom_half = polar_volume[:, mid_point_y:, :]

    # 新增：對 bottom_half 進行垂直翻轉
    bottom_half_vert_flipped = bottom_half[:, ::-1, :]

    # 對垂直翻轉後的下半部進行水平翻轉
    bottom_half_double_flipped = bottom_half_vert_flipped[:, :, ::-1]

    # 拼接
    joined_volume = np.concatenate((bottom_half_double_flipped, top_half), axis=2)
    print(f"     拼接後體積維度: {joined_volume.shape}")

    # 新增：將拼接後的體積順時針旋轉90度
    joined_volume = np.rot90(joined_volume, k=-1, axes=(1, 2))
    print(f"     順時針旋轉90度後體積維度: {joined_volume.shape}")

    # --- 步驟 3: 儲存拼接並旋轉後的極座標體積 ---
    print("   - 步驟 3/3: 儲存處理後的檔案...")
    z, y, x = joined_volume.shape
    # 修改：調整檔名中的維度順序為 X x Y x Z
    output_file = f"{folder_name}_polar_B{binning}_{x}x{y}x{z}.tif"
    print(f"5. 正在將結果儲存至 '{output_file}'...")
    try:
        tifffile.imwrite(output_file, joined_volume)
        print(f"--- 3D 重建與後處理任務成功完成 ---")
    except Exception as e:
        print(f"錯誤：儲存檔案失敗。詳細資訊: {e}")


def main():
    """主執行函式"""
    print("--- 開始 3D 重建與後處理任務 ---")

    # 步驟 1: 讓使用者選擇輸入資料夾
    input_dir = select_folder(title="請選擇包含投影影像的資料夾")
    if not input_dir:
        return  # 如果使用者取消選擇，則終止程式

    folder_name = os.path.basename(input_dir)

    # 步驟 2: 讀取與預處理影像
    projections, angles, height, width = load_projections(input_dir, FILE_PATTERN, BINNING_FACTOR)
    if projections is None:
        return  # 如果找不到檔案，則終止程式

    # 步驟 3: 使用 GPU 進行重建
    reconstructed_volume = reconstruct_with_astra(projections, angles, height, width)

    # 新增：儲存步驟 3 的原始重建結果
    print("   -> 正在儲存原始重建體積...")
    z_raw, y_raw, x_raw = reconstructed_volume.shape
    # 修改：調整檔名中的維度順序為 X x Y x Z
    raw_output_file = f"{folder_name}_recon_B{BINNING_FACTOR}_{x_raw}x{y_raw}x{z_raw}.tif"
    try:
        tifffile.imwrite(raw_output_file, reconstructed_volume)
        print(f"      原始重建體積已儲存至 '{raw_output_file}'")
    except Exception as e:
        print(f"      錯誤：儲存原始重建檔案失敗。詳細資訊: {e}")

    # 步驟 4: 進行極座標轉換、拼接、旋轉並儲存
    process_and_save_polar_volume(reconstructed_volume, BINNING_FACTOR, folder_name)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n--- 程式執行期間發生未預期的錯誤 ---")
        import traceback

        traceback.print_exc()

