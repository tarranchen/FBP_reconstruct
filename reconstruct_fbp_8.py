import os
import glob
import numpy as np
import tifffile
from tqdm import tqdm
import tomopy  # 用於條紋去除
import astra   # ASTRA 工具箱
import cv2     # OpenCV 函式庫，用於極座標轉換
from skimage.transform import downscale_local_mean
import tkinter as tk
from tkinter import filedialog

# --- 參數設定 (請根據您的需求修改) ---

# 檔案搜尋模式：用來尋找所有投影影像檔案
FILE_PATTERN = '*.tif'

# 像素合併因子 (Binning Factor)
BINNING_FACTOR = 1

# --- 環狀偽影校正優化參數 (在極座標空間中作用) ---
# tomopy.remove_stripe_fw 的參數
STRIPE_REMOVAL_SIGMA = 2
STRIPE_REMOVAL_LEVEL = 10
STRIPE_REMOVAL_WNAME = 'db5'


def select_input_folder():
    """彈出視窗讓使用者選擇輸入資料夾。"""
    root = tk.Tk()
    root.withdraw()  # 隱藏主視窗
    folder_path = filedialog.askdirectory(title="請選擇包含投影影像的資料夾")
    if not folder_path:
        print("未選擇資料夾，程式終止。")
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
    print(f"1. 在 '{input_dir}' 資料夾中尋找檔案...")
    search_path = os.path.join(input_dir, file_pattern)
    file_list = sorted(glob.glob(search_path))

    if not file_list:
        print(f"錯誤：在 '{search_path}' 中找不到任何檔案。請檢查路徑和檔案模式。")
        return None, None, None, None

    num_files = len(file_list)
    print(f"   找到 {num_files} 個投影檔案。")

    # 根據檔案數量自動判斷角度設定 (0-180度)
    angles_rad = np.deg2rad(np.linspace(0, 180, num_files, endpoint=False, dtype=np.float32))
    print(f"   -> 自動設定角度為 0-180 度，共 {num_files} 個角度。")

    print(f"2. 正在讀取並進行 {binning_factor}x{binning_factor} 像素合併...")
    first_image = tifffile.imread(file_list[0])
    if binning_factor > 1:
        binned_first_image = downscale_local_mean(first_image, (binning_factor, binning_factor))
    else:
        binned_first_image = first_image

    height, width = binned_first_image.shape
    print(f"   原始影像維度: {first_image.shape}")
    print(f"   合併後影像維度: {(height, width)}")

    projections_stack = np.zeros((num_files, height, width), dtype=np.float32)

    for i, file_path in enumerate(tqdm(file_list, desc="讀取與合併")):
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
        projections_stack (np.ndarray): 預處理後的投影堆疊。
        angles_rad (np.ndarray): 投影角度 (弧度)。
        height (int): 投影影像的高度。
        width (int): 投影影像的寬度。

    Returns:
        np.ndarray: 重建後的 3D 體積，帶有環狀偽影。
    """
    print("3. 開始使用 ASTRA 進行逐切片 GPU 重建...")
    reconstructed_slices = []

    for y in tqdm(range(height), desc="GPU 重建進度"):
        sinogram = projections_stack[:, y, :]

        # 定義重建與投影的幾何形狀
        vol_geom = astra.create_vol_geom(width, width)
        proj_geom = astra.create_proj_geom('parallel', 1.0, width, angles_rad)

        # 將 Sinogram 數據傳輸到 ASTRA
        sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)

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

        # 清理 ASTRA 記憶體
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(reco_id)
        astra.data2d.delete(sinogram_id)

    reconstructed_volume = np.stack(reconstructed_slices, axis=0).astype(np.float32)
    print("   GPU 重建完成，得到帶有環狀偽影的體積。")
    return reconstructed_volume


def correct_rings_post_process(volume, sigma, level, wname, binning, folder_name):
    """
    對重建後的體積進行後處理，以校正環狀偽影。
    此方法將影像轉換到極座標，將其切半、翻轉拼接、旋轉後進行水平條紋去除，然後再還原並轉回直角座標。

    Args:
        volume (np.ndarray): 帶有環狀偽影的 3D 體積。
        sigma (float): tomopy.remove_stripe_fw 的 sigma 參數。
        level (int): tomopy.remove_stripe_fw 的 level 參數。
        wname (str): tomopy.remove_stripe_fw 的小波名稱。
        binning (int): 像素合併因子，用於命名。
        folder_name (str): 輸入資料夾名稱，用於命名。

    Returns:
        np.ndarray: 校正環狀偽影後的 3D 體積。
    """
    print("4. 開始進行後處理環狀偽影校正...")

    center_x = volume.shape[2] / 2.0
    center_y = volume.shape[1] / 2.0
    max_radius = center_x
    original_size = volume[0].shape

    # 步驟 1: 將所有切片轉換至超取樣極座標
    print("   - 步驟 1/8: 將所有切片轉換至超取樣極座標...")

    # 新的超取樣尺寸計算
    oversampled_width = int(center_x)
    oversampled_height = int(volume.shape[1] * np.pi)

    if oversampled_height % 2 != 0:
        oversampled_height += 1  # 確保為偶數

    oversampled_dsize = (oversampled_width, oversampled_height)
    print(f"      極座標超取樣尺寸 (寬x高): {oversampled_dsize}")

    polar_images_list = []
    for original_slice in tqdm(volume, desc="極座標轉換進度"):
        polar_image = cv2.warpPolar(original_slice, oversampled_dsize, (center_x, center_y), max_radius, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        polar_images_list.append(polar_image)

    polar_volume = np.stack(polar_images_list, axis=0).astype(np.float32)

    # 步驟 2: 裁切、翻轉並拼接極座標體積
    print("   - 步驟 2/8: 裁切、翻轉並拼接極座標體積...")
    polar_height = polar_volume.shape[1]
    mid_point_y = polar_height // 2
    top_half = polar_volume[:, :mid_point_y, :]
    bottom_half = polar_volume[:, mid_point_y:, :]
    bottom_half_flipped = bottom_half[:, :, ::-1]
    joined_volume = np.concatenate((bottom_half_flipped, top_half), axis=2)
    print(f"      拼接後體積維度: {joined_volume.shape}")

    # 步驟 2.5: 儲存暫存檔
    print("   - 步驟 2.5/8: 儲存暫存檔...")
    z_j, y_j, x_j = joined_volume.shape
    temp_filename = f"{folder_name}_polar0_B{binning}S{sigma}L{level}_{z_j}_{y_j}_{x_j}.tif"
    print(f"      >> 正在儲存暫存檔: {temp_filename}")
    try:
        tifffile.imwrite(temp_filename, joined_volume)
    except Exception as e:
        print(f"      >> 儲存暫存檔失敗: {e}")

    # 步驟 3: 將整個體積旋轉90度
    print("   - 步驟 3/8: 將整個體積旋轉90度...")
    rotated_volume = np.rot90(joined_volume, k=1, axes=(1, 2))

    # 步驟 4: 對拼接體積進行條紋去除
    print("   - 步驟 4/8: 對拼接體積進行條紋去除...")
    corrected_rotated_volume = tomopy.remove_stripe_fw(
        rotated_volume,
        sigma=sigma,
        level=level,
        wname=wname
    )
    print("      條紋去除完成。")

    # 步驟 5: 將整個體積旋轉90度回來
    print("   - 步驟 5/8: 將整個體積旋轉90度回來...")
    corrected_joined_volume = np.rot90(corrected_rotated_volume, k=-1, axes=(1, 2))

    # 步驟 6: 儲存暫存檔
    print("   - 步驟 6/8: 儲存暫存檔...")
    z_j, y_j, x_j = corrected_joined_volume.shape
    temp_filename = f"{folder_name}_polar1_B{binning}S{sigma}L{level}_{z_j}_{y_j}_{x_j}.tif"
    print(f"      >> 正在儲存暫存檔: {temp_filename}")
    try:
        tifffile.imwrite(temp_filename, corrected_joined_volume)
    except Exception as e:
        print(f"      >> 儲存暫存檔失敗: {e}")

    # 步驟 7: 還原拼接的體積
    print("   - 步驟 7/8: 還原拼接的體積...")
    mid_point_x = corrected_joined_volume.shape[2] // 2
    corrected_bottom_flipped = corrected_joined_volume[:, :, :mid_point_x]
    corrected_top = corrected_joined_volume[:, :, mid_point_x:]
    corrected_bottom = corrected_bottom_flipped[:, :, ::-1]
    corrected_polar_volume = np.concatenate((corrected_top, corrected_bottom), axis=1)

    # 步驟 8: 將所有切片轉回直角座標
    print("   - 步驟 8/8: 將所有切片轉回直角座標...")
    final_corrected_slices = []
    original_dsize = (original_size[1], original_size[0])
    for corrected_polar_slice in tqdm(corrected_polar_volume, desc="直角座標還原進度"):
        corrected_slice = cv2.warpPolar(corrected_polar_slice, original_dsize, (center_x, center_y), max_radius, cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR)
        final_corrected_slices.append(corrected_slice)

    final_volume = np.stack(final_corrected_slices, axis=0).astype(np.float32)
    return final_volume


def main():
    """主執行函式"""
    print("--- 開始 3D 重建與後處理任務 ---")

    # 讓使用者選擇輸入資料夾
    input_dir = select_input_folder()
    if not input_dir:
        return  # 如果使用者取消選擇，則終止程式

    # 從路徑中獲取資料夾名稱
    folder_name = os.path.basename(input_dir)

    # 讀取與預處理
    projections, angles, height, width = load_projections(input_dir, FILE_PATTERN, BINNING_FACTOR)
    if projections is None:
        return  # 如果找不到檔案，則終止程式

    # GPU 重建
    reco_with_rings = reconstruct_with_astra(projections, angles, height, width)

    # 環狀偽影校正
    final_reco = correct_rings_post_process(
        reco_with_rings,
        STRIPE_REMOVAL_SIGMA,
        STRIPE_REMOVAL_LEVEL,
        STRIPE_REMOVAL_WNAME,
        BINNING_FACTOR,
        folder_name
    )

    # 儲存結果
    print("5. 正在將所有校正後的切片組合成最終 3D 體積...")
    z, y, x = final_reco.shape
    print(f"   最終體積維度: {(z, y, x)}")

    # 動態產生輸出檔名
    output_file = f"{folder_name}_recon1_B{BINNING_FACTOR}S{STRIPE_REMOVAL_SIGMA}L{STRIPE_REMOVAL_LEVEL}_{z}_{y}_{x}.tif"

    print(f"6. 正在將結果儲存至 '{output_file}'...")
    try:
        tifffile.imwrite(output_file, final_reco)
        print("--- 3D 重建與後處理任務成功完成 ---")
    except Exception as e:
        print(f"錯誤：儲存檔案失敗。{e}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("\n--- 發生嚴重錯誤 ---")
        print(f"錯誤訊息: {e}")

