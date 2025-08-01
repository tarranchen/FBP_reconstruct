import os
import glob
import numpy as np
import tifffile
from tqdm import tqdm
import tomopy  # 用於條紋去除
import astra  # 用於 GPU 重建
import cv2  # 引入 OpenCV 函式庫，用於極座標轉換

# --- 參數設定 (請根據您的需求修改) ---

# 輸入資料夾：包含 181 張 .tiff 投影影像的資料夾路徑
INPUT_DIR = './projections'

# 檔案搜尋模式：用來尋找所有投影影像檔案
FILE_PATTERN = '*.tif'

# 輸出檔案：重建後的 3D 體積將儲存為一個多頁的 TIFF 檔案
OUTPUT_FILE = 'reconstructed_volume_gpu_postprocessed_B2S005L1.tif'

# 像素合併因子 (Binning Factor)
BINNING_FACTOR = 2

# 角度設定：從 0 到 180 度，每 1 度一張，共 181 個角度
ANGLES_RAD = np.deg2rad(np.arange(0, 180.25, 1))

# --- 條紋去除優化參數 (在極座標空間中作用) ---
STRIPE_REMOVAL_SIGMA = 0.05
STRIPE_REMOVAL_LEVEL = 1
STRIPE_REMOVAL_WNAME = 'db5'


def reconstruct_and_correct(input_dir, file_pattern, output_file, angles_rad, binning_factor):
    """
    使用 GPU 進行 FBP 重建，然後在重建後的影像上進行環狀偽影校正。
    """
    print(f"--- 開始 3D 重建與後處理任務 (GPU FBP + Post-Processing Ring Removal) ---")

    # 1. 尋找並排序投影檔案
    print(f"1. 在 '{input_dir}' 資料夾中尋找檔案...")
    search_path = os.path.join(input_dir, file_pattern)
    file_list = sorted(glob.glob(search_path))

    if not file_list:
        print(f"錯誤：在 '{search_path}' 中找不到任何檔案。請檢查路徑和檔案模式。")
        return

    if len(file_list) != len(angles_rad):
        print(f"警告：找到的檔案數量 ({len(file_list)}) 與設定的角度數量 ({len(angles_rad)}) 不符。")
        return

    print(f"   找到 {len(file_list)} 個投影檔案。")

    # 2. 讀取、合併並堆疊影像 (預處理)
    print(f"2. 正在讀取並進行 {binning_factor}x{binning_factor} 像素合併...")
    from skimage.transform import downscale_local_mean

    first_image = tifffile.imread(file_list[0])
    if binning_factor > 1:
        binned_first_image = downscale_local_mean(first_image, (binning_factor, binning_factor))
    else:
        binned_first_image = first_image

    height, width = binned_first_image.shape

    print(f"   原始影像維度: {first_image.shape}")
    print(f"   合併後影像維度: {(height, width)}")

    projections_stack = np.zeros((len(file_list), height, width), dtype=np.float32)

    for i, file_path in enumerate(tqdm(file_list, desc="讀取與合併")):
        image = tifffile.imread(file_path).astype(np.float32)
        if binning_factor > 1:
            projections_stack[i, :, :] = downscale_local_mean(image, (binning_factor, binning_factor))
        else:
            projections_stack[i, :, :] = image
    print("   影像預處理完成。")

    # 3. 逐切片進行 GPU FBP 重建
    print("3. 開始逐切片進行 FBP_CUDA 重建...")

    reconstructed_slices_with_rings = []
    reco_width = width

    for y in tqdm(range(height), desc="GPU 重建進度"):
        sinogram = projections_stack[:, y, :]
        vol_geom = astra.create_vol_geom(reco_width, reco_width)
        proj_geom = astra.create_proj_geom('parallel', 1.0, width, angles_rad)
        sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
        reco_id = astra.data2d.create('-vol', vol_geom)
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = reco_id
        cfg['ProjectionDataId'] = sinogram_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        reco_slice = astra.data2d.get(reco_id)
        reconstructed_slices_with_rings.append(reco_slice)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(reco_id)
        astra.data2d.delete(sinogram_id)

    reconstructed_volume_with_rings = np.stack(reconstructed_slices_with_rings, axis=0).astype(np.float32)
    print("   GPU 重建完成，得到帶有環狀偽影的體積。")

    # 4. 進行後處理環狀偽影校正 (使用 OpenCV)
    print("4. 開始進行後處理校正 (使用 OpenCV)...")

    center_x = reconstructed_volume_with_rings.shape[2] / 2.0
    center_y = reconstructed_volume_with_rings.shape[1] / 2.0
    max_radius = np.sqrt(center_x ** 2 + center_y ** 2)
    output_size = reconstructed_volume_with_rings[0].shape

    # 步驟 1: 將所有切片批次轉換為極座標影像
    print("   - 步驟 1/3: 將所有切片轉換至極座標...")
    polar_images_list = []
    for original_slice in tqdm(reconstructed_volume_with_rings, desc="極座標轉換進度"):
        polar_image = cv2.warpPolar(original_slice, output_size, (center_x, center_y), max_radius,
                                    cv2.WARP_FILL_OUTLIERS)
        polar_images_list.append(polar_image)

    polar_volume = np.stack(polar_images_list, axis=0).astype(np.float32)

    # 步驟 2: 對整個極座標體積進行條紋去除
    print("   - 步驟 2/3: 對極座標體積進行條紋去除...")
    polar_volume_transposed = np.swapaxes(polar_volume, 1, 2)

    corrected_polar_transposed = tomopy.remove_stripe_fw(
        polar_volume_transposed,
        sigma=STRIPE_REMOVAL_SIGMA,
        level=STRIPE_REMOVAL_LEVEL,
        wname=STRIPE_REMOVAL_WNAME
    )

    corrected_polar_volume = np.swapaxes(corrected_polar_transposed, 1, 2)
    print("   條紋去除完成。")

    # 步驟 3: 將校正後的極座標影像批次轉回直角座標
    print("   - 步驟 3/3: 將所有切片轉回直角座標...")
    final_corrected_slices = []
    for corrected_polar_slice in tqdm(corrected_polar_volume, desc="直角座標還原進度"):
        corrected_slice = cv2.warpPolar(corrected_polar_slice, output_size, (center_x, center_y), max_radius,
                                        cv2.WARP_INVERSE_MAP)
        final_corrected_slices.append(corrected_slice)

    # 5. 組合與儲存
    print("5. 正在將所有校正後的切片組合成最終 3D 體積...")
    final_reconstructed_volume = np.stack(final_corrected_slices, axis=0).astype(np.float32)
    print(f"   最終體積維度: {final_reconstructed_volume.shape}")

    print(f"6. 正在將結果儲存至 '{output_file}'...")
    try:
        tifffile.imwrite(output_file, final_reconstructed_volume)
        print("--- 3D 重建與後處理任務成功完成 ---")
    except Exception as e:
        print(f"錯誤：儲存檔案失敗。{e}")


if __name__ == '__main__':
    try:
        reconstruct_and_correct(
            input_dir=INPUT_DIR,
            file_pattern=FILE_PATTERN,
            output_file=OUTPUT_FILE,
            angles_rad=ANGLES_RAD,
            binning_factor=BINNING_FACTOR
        )
    except Exception as e:
        print("\n--- 發生嚴重錯誤 ---")
        print("錯誤訊息:", e)
