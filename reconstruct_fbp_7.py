import os
import glob
import numpy as np
import tifffile
from tqdm import tqdm
import tomopy  # 用於條紋去除與重建
import astra  # ASTRA 工具箱
import cv2  # 引入 OpenCV 函式庫，用於極座標轉換

# Scipy 已被移除，因為不再需要背景擬合功能

# --- 參數設定 (請根據您的需求修改) ---

# 輸入資料夾：包含 .tiff 投影影像的資料夾路徑
INPUT_DIR = './projections'

# 檔案搜尋模式：用來尋找所有投影影像檔案
FILE_PATTERN = '*.tif'

# 輸出檔案：重建後的 3D 體積將儲存為一個多頁的 TIFF 檔案
OUTPUT_FILE = 'reconstructed_volume_dual_correction_iterative.tif'

# 像素合併因子 (Binning Factor)
BINNING_FACTOR = 2

# --- 條紋去除優化參數 ---
# 由於環狀偽影嚴重，我們在重建前後都進行校正
# level 參數越高，濾波效果越強、越激進。可根據需求調整。
STRIPE_REMOVAL_WNAME = 'db5'

# *** 新增：迭代次數控制 ***
# 1. 重建前校正的迭代次數
PRE_RECON_ITERATIONS = 3
# 2. 重建後校正的迭代次數
POST_RECON_ITERATIONS = 3

# *** 新增：批次大小控制 ***
# 一次處理的切片數量。可根據您的記憶體大小調整 (例如 32, 64, 128)
PRE_RECON_CHUNK_SIZE = 32
POST_RECON_CHUNK_SIZE = 32

# 1. 重建前參數 (在正弦圖上作用)
# 注意：level 設得非常高會大幅增加計算時間與記憶體，建議從 10 開始嘗試
PRE_RECON_STRIPE_LEVEL = 10
PRE_RECON_STRIPE_SIGMA = 1

# 2. 重建後參數 (在極座標空間中作用)
POST_RECON_STRIPE_LEVEL = 10
POST_RECON_STRIPE_SIGMA = 1


def reconstruct_and_correct(input_dir, file_pattern, output_file, binning_factor):
    """
    使用 GPU 進行 FBP 重建，並在重建前後都進行迭代式環狀偽影校正。
    角度會根據檔案數量自動判斷。
    """
    print(f"--- 開始 3D 重建與雙重迭代校正任務 ---")

    # 1. 尋找並排序投影檔案
    print(f"1. 在 '{input_dir}' 資料夾中尋找檔案...")
    search_path = os.path.join(input_dir, file_pattern)
    file_list = sorted(glob.glob(search_path))

    if not file_list:
        print(f"錯誤：在 '{search_path}' 中找不到任何檔案。請檢查路徑和檔案模式。")
        return

    num_files = len(file_list)
    print(f"    找到 {num_files} 個投影檔案。")
    angles_rad = np.deg2rad(np.linspace(0, 180, num_files, dtype=np.float32))
    print(f"    -> 自動設定角度為 0-180 度，共 {num_files} 個角度。")

    # 2. 讀取、合併並堆疊影像 (預處理)
    print(f"2. 正在讀取並進行 {binning_factor}x{binning_factor} 像素合併...")
    from skimage.transform import downscale_local_mean

    first_image = tifffile.imread(file_list[0])
    if binning_factor > 1:
        binned_first_image = downscale_local_mean(first_image, (binning_factor, binning_factor))
    else:
        binned_first_image = first_image
    height, width = binned_first_image.shape
    print(f"    原始影像維度: {first_image.shape}")
    print(f"    合併後影像維度: {(height, width)}")

    projections_stack = np.zeros((len(file_list), height, width), dtype=np.float32)
    for i, file_path in enumerate(tqdm(file_list, desc="讀取與合併")):
        image = tifffile.imread(file_path).astype(np.float32)
        if binning_factor > 1:
            projections_stack[i, :, :] = downscale_local_mean(image, (binning_factor, binning_factor))
        else:
            projections_stack[i, :, :] = image
    print("    影像預處理完成。")

    # 3. 進行重建前校正 (逐批次迭代式)
    print(f"3. 開始進行逐批次重建前校正 (批次大小: {PRE_RECON_CHUNK_SIZE}, 迭代 {PRE_RECON_ITERATIONS} 次)...")
    projections_stack_transposed = np.transpose(projections_stack, (1, 0, 2))

    corrected_sinogram_chunks = []
    num_sinograms = projections_stack_transposed.shape[0]

    for i in tqdm(range(0, num_sinograms, PRE_RECON_CHUNK_SIZE), desc="重建前校正批次進度"):
        chunk_start = i
        chunk_end = min(i + PRE_RECON_CHUNK_SIZE, num_sinograms)
        sino_chunk = projections_stack_transposed[chunk_start:chunk_end]

        # 對當前批次進行迭代校正
        for _ in range(PRE_RECON_ITERATIONS):
            sino_chunk = tomopy.remove_stripe_fw(
                sino_chunk,
                level=PRE_RECON_STRIPE_LEVEL,
                sigma=PRE_RECON_STRIPE_SIGMA,
                wname=STRIPE_REMOVAL_WNAME
            )
        corrected_sinogram_chunks.append(sino_chunk)

    sinograms_corrected_stack = np.concatenate(corrected_sinogram_chunks, axis=0)
    print("    - 重建前校正完成。")

    # 4. 逐切片進行 GPU 重建
    print("4. 開始逐切片進行 GPU 重建...")
    reconstructed_slices_with_rings = []
    reco_width = width

    for y in tqdm(range(height), desc="GPU 重建進度"):
        sinogram_corrected = sinograms_corrected_stack[y, :, :]
        vol_geom = astra.create_vol_geom(reco_width, reco_width)
        proj_geom = astra.create_proj_geom('parallel', 1.0, width, angles_rad)
        sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram_corrected)
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
    print("    GPU 重建完成。")

    # 5. 進行後處理環狀偽影校正 (逐批次迭代式，分區處理)
    print(f"5. 開始進行逐批次後處理校正 (批次大小: {POST_RECON_CHUNK_SIZE}, 迭代 {POST_RECON_ITERATIONS} 次)...")

    center_x = reconstructed_volume_with_rings.shape[2] / 2.0
    center_y = reconstructed_volume_with_rings.shape[1] / 2.0
    max_radius = np.sqrt(center_x ** 2 + center_y ** 2)
    original_size = reconstructed_volume_with_rings[0].shape
    oversampled_width = int(original_size[1] * np.sqrt(2))
    oversampled_height = int(original_size[0] * np.pi)
    oversampled_dsize = (oversampled_width, oversampled_height)

    final_corrected_slices = []
    num_slices = reconstructed_volume_with_rings.shape[0]

    for i in tqdm(range(0, num_slices, POST_RECON_CHUNK_SIZE), desc="後處理批次進度"):
        chunk_start = i
        chunk_end = min(i + POST_RECON_CHUNK_SIZE, num_slices)
        reco_chunk = reconstructed_volume_with_rings[chunk_start:chunk_end]

        # 步驟 1: 將整個批次轉換為極座標
        polar_chunk_list = []
        for slice_in_chunk in reco_chunk:
            polar_image = cv2.warpPolar(slice_in_chunk, oversampled_dsize, (center_x, center_y), max_radius,
                                        cv2.WARP_FILL_OUTLIERS)
            polar_chunk_list.append(polar_image)
        polar_chunk = np.stack(polar_chunk_list, axis=0).astype(np.float32)

        # 步驟 2: 分區處理
        mid_y = polar_chunk.shape[1] // 2
        top_half = polar_chunk[:, :mid_y, :]
        bottom_half = polar_chunk[:, mid_y:, :]

        # 對上下兩部分別進行迭代校正
        for _ in range(POST_RECON_ITERATIONS):
            top_half = tomopy.remove_stripe_fw(
                top_half,
                sigma=POST_RECON_STRIPE_SIGMA,
                level=POST_RECON_STRIPE_LEVEL,
                wname=STRIPE_REMOVAL_WNAME
            )
            bottom_half = tomopy.remove_stripe_fw(
                bottom_half,
                sigma=POST_RECON_STRIPE_SIGMA,
                level=POST_RECON_STRIPE_LEVEL,
                wname=STRIPE_REMOVAL_WNAME
            )

        # 合併校正後的上下兩部分
        corrected_polar_chunk = np.concatenate((top_half, bottom_half), axis=1)

        # 步驟 3: 將校正後的批次轉回直角座標
        corrected_chunk_list = []
        original_dsize = (original_size[1], original_size[0])
        for corrected_polar_slice in corrected_polar_chunk:
            corrected_slice = cv2.warpPolar(corrected_polar_slice, original_dsize, (center_x, center_y), max_radius,
                                            cv2.WARP_INVERSE_MAP)
            corrected_chunk_list.append(corrected_slice)

        final_corrected_slices.extend(corrected_chunk_list)

    print("    後處理校正完成。")

    # 6. 組合與儲存
    print("6. 正在將所有校正後的切片組合成最終 3D 體積...")
    final_reconstructed_volume = np.stack(final_corrected_slices, axis=0).astype(np.float32)
    print(f"    最終體積維度: {final_reconstructed_volume.shape}")

    print(f"7. 正在將結果儲存至 '{output_file}'...")
    try:
        tifffile.imwrite(output_file, final_reconstructed_volume)
        print("--- 3D 重建與雙重迭代校正任務成功完成 ---")
    except Exception as e:
        print(f"錯誤：儲存檔案失敗。{e}")


if __name__ == '__main__':
    try:
        reconstruct_and_correct(
            input_dir=INPUT_DIR,
            file_pattern=FILE_PATTERN,
            output_file=OUTPUT_FILE,
            binning_factor=BINNING_FACTOR
        )
    except Exception as e:
        print("\n--- 發生嚴重錯誤 ---")
        print("錯誤訊息:", e)
