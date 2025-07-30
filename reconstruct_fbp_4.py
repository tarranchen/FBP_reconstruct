import os
import glob
import numpy as np
import tifffile
from tqdm import tqdm
import tomopy  # 用於環狀偽影校正
import astra  # 用於 GPU 重建

# --- 參數設定 (請根據您的需求修改) ---

# 輸入資料夾：包含 181 張 .tiff 投影影像的資料夾路徑
INPUT_DIR = './P'

# 檔案搜尋模式：用來尋找所有投影影像檔案
FILE_PATTERN = '*.tif'

# 輸出檔案：重建後的 3D 體積將儲存為一個多頁的 TIFF 檔案
OUTPUT_FILE = 'reconstructed_volume_gpu_optimized_N13_35B1.tif'

# 像素合併因子 (Binning Factor)
# 2 代表 2x2 的像素合併
BINNING_FACTOR = 1

# 角度設定：從 0 到 180 度，共 181 個角度
# ASTRA 需要的角度單位是弧度 (radians)
ANGLES_RAD = np.deg2rad(np.arange(0, 180.25, 0.25))

# --- 環狀偽影校正優化參數 ---
WAVELET_SIGMA = 3
WAVELET_LEVEL = 5
WAVELET_NAME = 'db5'


def reconstruct_volume_gpu_optimized(input_dir, file_pattern, output_file, angles_rad, binning_factor):
    """
    整合像素合併、環狀偽影校正與 GPU 加速的 FBP 重建。
    """
    print(f"--- 開始 GPU 3D 重建任務 (Binning: {binning_factor}x{binning_factor}, Ring Correction: ON) ---")

    # 1. 尋找並排序投影檔案
    print(f"1. 在 '{input_dir}' 資料夾中尋找檔案...")
    search_path = os.path.join(input_dir, file_pattern)
    file_list = sorted(glob.glob(search_path))

    if not file_list:
        print(f"錯誤：在 '{search_path}' 中找不到任何檔案。請檢查路徑和檔案模式。")
        return

    # 2. 讀取、合併並堆疊影像
    print(f"2. 正在讀取並進行 {binning_factor}x{binning_factor} 像素合併...")
    from skimage.transform import downscale_local_mean  # 僅在此處引入

    first_image = tifffile.imread(file_list[0])
    binned_first_image = downscale_local_mean(first_image, (binning_factor, binning_factor))
    height, width = binned_first_image.shape

    print(f"   原始影像維度: {first_image.shape}")
    print(f"   合併後影像維度: {(height, width)}")

    projections_stack = np.zeros((len(file_list), height, width), dtype=np.float32)

    for i, file_path in enumerate(tqdm(file_list, desc="讀取與合併")):
        image = tifffile.imread(file_path).astype(np.float32)
        projections_stack[i, :, :] = downscale_local_mean(image, (binning_factor, binning_factor))

    print("   所有影像處理完畢。")

    # 3. 環狀偽影校正 (CPU, TomoPy)
    print("3. 正在使用優化演算法 (Wavelet-FFT) 進行環狀偽影校正...")
    print(f"   使用參數: sigma={WAVELET_SIGMA}, level={WAVELET_LEVEL}, wname='{WAVELET_NAME}'")

    projections_stack_swapped = np.swapaxes(projections_stack, 0, 1)
    projections_stack_corrected_swapped = tomopy.remove_stripe_fw(
        projections_stack_swapped,
        sigma=WAVELET_SIGMA,
        level=WAVELET_LEVEL,
        wname=WAVELET_NAME
    )
    projections_stack_corrected = np.swapaxes(projections_stack_corrected_swapped, 0, 1)
    print("   環狀偽影校正完成。")

    # 4. 逐切片進行 GPU 重建 (ASTRA Toolbox)
    print("4. 開始逐切片進行 FBP_CUDA 重建...")

    reconstructed_slices = []
    reco_width = width  # 重建後的寬度與投影寬度相同

    for y in tqdm(range(height), desc="GPU 重建進度"):
        sinogram = projections_stack_corrected[:, y, :]

        # a. 定義重建與投影的幾何形狀
        vol_geom = astra.create_vol_geom(reco_width, reco_width)
        proj_geom = astra.create_proj_geom('parallel', 1.0, width, angles_rad)

        # b. 將 Sinogram 數據傳輸到 ASTRA
        sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)

        # c. 建立一個空間來存放重建結果
        reco_id = astra.data2d.create('-vol', vol_geom)

        # d. 設定重建演算法的參數
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = reco_id
        cfg['ProjectionDataId'] = sinogram_id

        # e. 建立並執行演算法
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        # f. 從 ASTRA 取回重建完成的切片數據
        reco_slice = astra.data2d.get(reco_id)
        reconstructed_slices.append(reco_slice)

        # g. 清理 ASTRA 記憶體 (非常重要!)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(reco_id)
        astra.data2d.delete(sinogram_id)

    # 5. 組合與儲存
    print("5. 正在將所有重建切片組合成 3D 體積...")
    reconstructed_volume = np.stack(reconstructed_slices, axis=0).astype(np.float32)
    print(f"   重建完成的 3D 體積維度: {reconstructed_volume.shape}")

    print(f"6. 正在將結果儲存至 '{output_file}'...")
    try:
        tifffile.imwrite(output_file, reconstructed_volume)
        print("--- GPU 重建任務成功完成 ---")
    except Exception as e:
        print(f"錯誤：儲存檔案失敗。{e}")


if __name__ == '__main__':
    # 直接執行 GPU 重建函式。
    # 如果 CUDA 或 astra-toolbox 設定有問題，程式會在執行時回報更精確的錯誤。
    try:
        reconstruct_volume_gpu_optimized(
            input_dir=INPUT_DIR,
            file_pattern=FILE_PATTERN,
            output_file=OUTPUT_FILE,
            angles_rad=ANGLES_RAD,
            binning_factor=BINNING_FACTOR
        )
    except Exception as e:
        print("\n--- 發生錯誤 ---")
        print("錯誤訊息:", e)
        print("\n可能原因：")
        print("1. NVIDIA 驅動程式未正確安裝。")
        print("2. CUDA Toolkit 版本與 astra-toolbox 不相容或未安裝。")
        print("3. astra-toolbox 套件安裝不完整。")
        print("請檢查您的環境設定。")

