import os
import glob
import numpy as np
from skimage.transform import iradon
from skimage.io import imread
import tifffile
from tqdm import tqdm

# --- 參數設定 (請根據您的需求修改) ---

# 輸入資料夾：包含 181 張 .tiff 投影影像的資料夾路徑
INPUT_DIR = 'TIF17'

# 檔案搜尋模式：用來尋找所有投影影像檔案
# 假設檔案命名為 proj_000.tif, proj_001.tif ...
FILE_PATTERN = '*.tif'

# 輸出檔案：重建後的 3D 體積將儲存為一個多頁的 TIFF 檔案
OUTPUT_FILE = 'reconstructed_volume.tif'

# 角度設定：從 0 到 180 度，共 181 個角度
# np.arange(start, stop, step)
# 這裡 stop 是 181，因為 np.arange 不包含 stop 值，所以會產生 0, 1, ..., 180
ANGLES_DEG = np.arange(0, 181, 1)

# --- 腳本主體 ---

def reconstruct_volume_fbp(input_dir, file_pattern, output_file, angles):
    """
    使用濾波反投影演算法，從一系列 2D 投影重建 3D 體積。

    Args:
        input_dir (str): 包含投影 TIFF 檔案的資料夾路徑。
        file_pattern (str): 用於尋找影像檔案的 glob 模式。
        output_file (str): 儲存重建體積的輸出檔案路徑。
        angles (np.ndarray): 每個投影對應的角度陣列（單位：度）。
    """
    print("--- 開始 3D 重建任務 ---")

    # 1. 尋找並排序投影檔案
    print(f"1. 在 '{input_dir}' 資料夾中尋找檔案...")
    search_path = os.path.join(input_dir, file_pattern)
    file_list = sorted(glob.glob(search_path))

    if not file_list:
        print(f"錯誤：在 '{search_path}' 中找不到任何檔案。請檢查路徑和檔案模式。")
        return

    num_projections = len(file_list)
    print(f"   找到 {num_projections} 個投影檔案。")

    if num_projections != len(angles):
        print(f"警告：找到的檔案數量 ({num_projections}) 與設定的角度數量 ({len(angles)}) 不符。")
        # 仍繼續執行，但使用者應注意此問題

    # 2. 讀取所有投影影像並堆疊成一個 3D NumPy 陣列
    print("2. 正在讀取所有投影影像...")
    
    # 先讀取第一張影像以取得維度資訊
    first_image = imread(file_list[0])
    if first_image.dtype != np.float32:
        print(f"警告：影像的資料型態為 {first_image.dtype}，而非預期的 float32。")

    # 取得影像的高度和寬度 (detector_pixels_y, detector_pixels_x)
    height, width = first_image.shape
    
    # 建立一個 NumPy 陣列來存放所有投影
    # 維度為 (num_projections, height, width)
    projections_stack = np.zeros((num_projections, height, width), dtype=np.float32)

    for i, file_path in enumerate(tqdm(file_list, desc="讀取影像")):
        projections_stack[i, :, :] = imread(file_path).astype(np.float32)

    print("   所有影像讀取完畢。")
    print(f"   投影堆疊維度: {projections_stack.shape}")

    # 3. 逐切片進行濾波反投影重建
    print("3. 開始逐切片進行 FBP 重建...")
    
    reconstructed_slices = []
    
    # 迴圈遍歷物件的每一個垂直切片 (y-axis)
    for y in tqdm(range(height), desc="重建進度"):
        # 從投影堆疊中提取該切片對應的正弦圖 (Sinogram)
        # sinogram 維度為 (num_projections, width)
        sinogram = projections_stack[:, y, :]
        
        # scikit-image 的 iradon 函數需要 sinogram 的維度為 (detector_pixels, num_angles)
        # 所以我們需要將其轉置
        sinogram_transposed = sinogram.T
        
        # 執行濾波反投影 (Inverse Radon Transform)
        # circle=True 表示重建區域會被限制在一個圓形內，這是CT重建的標準做法
        reco_slice = iradon(sinogram_transposed, theta=angles, circle=True, filter_name='ramp')
        
        reconstructed_slices.append(reco_slice)

    # 4. 將所有重建的 2D 切片堆疊成最終的 3D 體積
    print("4. 正在將所有重建切片組合成 3D 體積...")
    
    # iradon 預設輸出 float64，我們將其轉回 float32
    # stack 預設會在 axis=0 堆疊，形成 (height, reco_width, reco_width) 的 3D 體積
    reconstructed_volume = np.stack(reconstructed_slices, axis=0).astype(np.float32)
    
    print(f"   重建完成的 3D 體積維度: {reconstructed_volume.shape}")

    # 5. 儲存 3D 體積檔案
    print(f"5. 正在將結果儲存至 '{output_file}'...")
    try:
        # 使用 tifffile 函式庫，它可以很好地處理大型 3D float tiff 檔案
        tifffile.imwrite(output_file, reconstructed_volume)
        print("--- 重建任務成功完成 ---")
    except Exception as e:
        print(f"錯誤：儲存檔案失敗。{e}")


if __name__ == '__main__':
    # 執行主函式
    reconstruct_volume_fbp(
        input_dir=INPUT_DIR,
        file_pattern=FILE_PATTERN,
        output_file=OUTPUT_FILE,
        angles=ANGLES_DEG
    )