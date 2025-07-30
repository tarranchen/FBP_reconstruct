import os
import glob
import numpy as np
import tifffile
from tqdm import tqdm
from skimage.transform import iradon, downscale_local_mean
import tomopy  # 引入 tomopy 函式庫

# --- 參數設定 (請根據您的需求修改) ---

# 輸入資料夾：包含 181 張 .tiff 投影影像的資料夾路徑
INPUT_DIR = './projections'

# 檔案搜尋模式：用來尋找所有投影影像檔案
FILE_PATTERN = '*.tif'

# 輸出檔案：重建後的 3D 體積將儲存為一個多頁的 TIFF 檔案
OUTPUT_FILE = 'reconstructed_volume_binned_corrected_optimized053.tif'

# 像素合併因子 (Binning Factor)
# 2 代表 2x2 的像素合併
BINNING_FACTOR = 2

# 角度設定：從 0 到 180 度，共 181 個角度
# iradon 需要的角度單位是度 (degrees)
ANGLES_DEG = np.arange(0, 181, 1)

# --- 環狀偽影校正優化參數 ---
# 您可以調整這些參數來優化校正效果
WAVELET_SIGMA = 0.5  # 濾波器的強度。數值越小，濾波越強。建議範圍 1-5。
WAVELET_LEVEL = 3  # 小波分解的級別。可以根據條紋的寬度調整，通常 5-8 之間效果不錯。
WAVELET_NAME = 'db5'  # 小波的類型，'db5' 是一個常用的選項。


def reconstruct_volume_fbp_binned_corrected(input_dir, file_pattern, output_file, angles_deg, binning_factor):
    """
    對投影影像進行像素合併與環狀偽影校正後，使用 CPU 進行濾波反投影重建。

    Args:
        input_dir (str): 包含投影 TIFF 檔案的資料夾路徑。
        file_pattern (str): 用於尋找影像檔案的 glob 模式。
        output_file (str): 儲存重建體積的輸出檔案路徑。
        angles_deg (np.ndarray): 每個投影對應的角度陣列（單位：度）。
        binning_factor (int): 像素合併的因子。
    """
    print(f"--- 開始 CPU 3D 重建任務 (使用 {binning_factor}x{binning_factor} Binning 並校正環狀偽影) ---")

    # 1. 尋找並排序投影檔案
    print(f"1. 在 '{input_dir}' 資料夾中尋找檔案...")
    search_path = os.path.join(input_dir, file_pattern)
    file_list = sorted(glob.glob(search_path))

    if not file_list:
        print(f"錯誤：在 '{search_path}' 中找不到任何檔案。請檢查路徑和檔案模式。")
        return

    num_projections = len(file_list)
    print(f"   找到 {num_projections} 個投影檔案。")

    # 2. 讀取所有投影影像，進行像素合併，然後堆疊
    print(f"2. 正在讀取並進行 {binning_factor}x{binning_factor} 像素合併...")

    first_image = tifffile.imread(file_list[0])
    binned_first_image = downscale_local_mean(first_image, (binning_factor, binning_factor))
    height, width = binned_first_image.shape

    print(f"   原始影像維度: {first_image.shape}")
    print(f"   合併後影像維度: {(height, width)}")

    projections_stack = np.zeros((num_projections, height, width), dtype=np.float32)

    for i, file_path in enumerate(tqdm(file_list, desc="讀取與合併")):
        image = tifffile.imread(file_path).astype(np.float32)
        binned_image = downscale_local_mean(image, (binning_factor, binning_factor))
        projections_stack[i, :, :] = binned_image

    print("   所有影像處理完畢。")
    print(f"   投影堆疊維度: {projections_stack.shape}")

    # --- 優化: 使用小波-傅立葉變換法進行環狀偽影校正 ---
    print("3. 正在使用優化演算法 (Wavelet-FFT) 進行環狀偽影校正...")
    print(f"   使用參數: sigma={WAVELET_SIGMA}, level={WAVELET_LEVEL}, wname='{WAVELET_NAME}'")

    # tomopy.remove_stripe_fw 預期輸入一個 3D 陣列 (切片高度, 角度, 探測器寬度)
    # 所以我們需要先將資料堆疊的軸進行交換 (0,1,2) -> (1,0,2)
    projections_stack_swapped = np.swapaxes(projections_stack, 0, 1)

    # 執行校正 (使用 tomopy.remove_stripe_fw，這是推薦的強效函式)
    projections_stack_corrected_swapped = tomopy.remove_stripe_fw(
        projections_stack_swapped,
        sigma=WAVELET_SIGMA,
        level=WAVELET_LEVEL,
        wname=WAVELET_NAME
    )

    # 將校正後的資料軸交換回來
    projections_stack_corrected = np.swapaxes(projections_stack_corrected_swapped, 0, 1)
    print("   環狀偽影校正完成。")

    # 4. 逐切片進行 CPU 重建
    print("4. 開始逐切片進行 FBP 重建...")

    reconstructed_slices = []

    # 使用校正後的資料堆疊進行迴圈
    for y in tqdm(range(height), desc="CPU 重建進度"):
        # 從 *校正後* 的投影堆疊中提取正弦圖 (Sinogram)
        sinogram = projections_stack_corrected[:, y, :]

        # 將 sinogram 轉置以符合 iradon 的輸入格式 (探測器寬度, 角度)
        sinogram_transposed = sinogram.T

        # 執行濾波反投影 (Inverse Radon Transform)
        reco_slice = iradon(sinogram_transposed, theta=angles_deg, circle=True, filter_name='ramp')

        reconstructed_slices.append(reco_slice)

    # 5. 將所有重建的 2D 切片堆疊成最終的 3D 體積
    print("5. 正在將所有重建切片組合成 3D 體積...")
    reconstructed_volume = np.stack(reconstructed_slices, axis=0).astype(np.float32)
    print(f"   重建完成的 3D 體積維度: {reconstructed_volume.shape}")

    # 6. 儲存 3D 體積檔案
    print(f"6. 正在將結果儲存至 '{output_file}'...")
    try:
        tifffile.imwrite(output_file, reconstructed_volume)
        print("--- CPU 重建任務成功完成 ---")
    except Exception as e:
        print(f"錯誤：儲存檔案失敗。{e}")


if __name__ == '__main__':
    reconstruct_volume_fbp_binned_corrected(
        input_dir=INPUT_DIR,
        file_pattern=FILE_PATTERN,
        output_file=OUTPUT_FILE,
        angles_deg=ANGLES_DEG,
        binning_factor=BINNING_FACTOR
    )
