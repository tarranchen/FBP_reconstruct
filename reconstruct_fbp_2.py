import os
import glob
import numpy as np
import tifffile
from tqdm import tqdm
from skimage.transform import iradon, downscale_local_mean

# --- 參數設定 (請根據您的需求修改) ---

# 輸入資料夾：包含 181 張 .tiff 投影影像的資料夾路徑
INPUT_DIR = './TIF17'

# 檔案搜尋模式：用來尋找所有投影影像檔案
FILE_PATTERN = '*.tif'

# 輸出檔案：重建後的 3D 體積將儲存為一個多頁的 TIFF 檔案
OUTPUT_FILE = 'reconstructed_volume_binned.tif'

# 像素合併因子 (Binning Factor)
# 2 代表 2x2 的像素合併
BINNING_FACTOR = 2

# 角度設定：從 0 到 180 度，共 181 個角度
# iradon 需要的角度單位是度 (degrees)
ANGLES_DEG = np.arange(0, 181, 1)

def reconstruct_volume_fbp_binned(input_dir, file_pattern, output_file, angles_deg, binning_factor):
    """
    對投影影像進行像素合併後，使用 CPU 進行濾波反投影重建。

    Args:
        input_dir (str): 包含投影 TIFF 檔案的資料夾路徑。
        file_pattern (str): 用於尋找影像檔案的 glob 模式。
        output_file (str): 儲存重建體積的輸出檔案路徑。
        angles_deg (np.ndarray): 每個投影對應的角度陣列（單位：度）。
        binning_factor (int): 像素合併的因子。
    """
    print(f"--- 開始 CPU 3D 重建任務 (使用 {binning_factor}x{binning_factor} Binning) ---")

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
    
    # 先讀取第一張影像以取得原始維度
    first_image = tifffile.imread(file_list[0])
    
    # 計算合併後的新維度
    # 使用 downscale_local_mean 進行平均值合併，效果等同於 binning
    binned_first_image = downscale_local_mean(first_image, (binning_factor, binning_factor))
    height, width = binned_first_image.shape
    
    print(f"   原始影像維度: {first_image.shape}")
    print(f"   合併後影像維度: {(height, width)}")

    # 建立一個 NumPy 陣列來存放所有合併後的投影
    projections_stack = np.zeros((num_projections, height, width), dtype=np.float32)

    for i, file_path in enumerate(tqdm(file_list, desc="讀取與合併")):
        image = tifffile.imread(file_path).astype(np.float32)
        # 進行像素合併
        binned_image = downscale_local_mean(image, (binning_factor, binning_factor))
        projections_stack[i, :, :] = binned_image

    print("   所有影像處理完畢。")
    print(f"   投影堆疊維度: {projections_stack.shape}")
    
    # 3. 逐切片進行 CPU 重建
    print("3. 開始逐切片進行 FBP 重建...")
    
    reconstructed_slices = []
    
    # 迴圈遍歷物件的每一個垂直切片 (y-axis)
    for y in tqdm(range(height), desc="CPU 重建進度"):
        # 從合併後的投影堆疊中提取正弦圖 (Sinogram)
        sinogram = projections_stack[:, y, :]
        
        # 將 sinogram 轉置以符合 iradon 的輸入格式
        sinogram_transposed = sinogram.T
        
        # 執行濾波反投影 (Inverse Radon Transform)
        reco_slice = iradon(sinogram_transposed, theta=angles_deg, circle=True, filter_name='ramp')
        
        reconstructed_slices.append(reco_slice)

    # 4. 將所有重建的 2D 切片堆疊成最終的 3D 體積
    print("4. 正在將所有重建切片組合成 3D 體積...")
    reconstructed_volume = np.stack(reconstructed_slices, axis=0).astype(np.float32)
    print(f"   重建完成的 3D 體積維度: {reconstructed_volume.shape}")

    # 5. 儲存 3D 體積檔案
    print(f"5. 正在將結果儲存至 '{output_file}'...")
    try:
        tifffile.imwrite(output_file, reconstructed_volume)
        print("--- CPU 重建任務成功完成 ---")
    except Exception as e:
        print(f"錯誤：儲存檔案失敗。{e}")


if __name__ == '__main__':
    reconstruct_volume_fbp_binned(
        input_dir=INPUT_DIR,
        file_pattern=FILE_PATTERN,
        output_file=OUTPUT_FILE,
        angles_deg=ANGLES_DEG,
        binning_factor=BINNING_FACTOR
    )
