import os
import glob
import numpy as np
import tifffile
from tqdm import tqdm
import tomopy  # 用於條紋去除與重建
import astra  # ASTRA 工具箱
import cv2  # 引入 OpenCV 函式庫，用於極座標轉換
from scipy.optimize import curve_fit
from scipy.special import erf

# --- 參數設定 (請根據您的需求修改) ---

# 輸入資料夾：包含 .tiff 投影影像的資料夾路徑
INPUT_DIR = './P'

# 檔案搜尋模式：用來尋找所有投影影像檔案
FILE_PATTERN = '*.tif'

# 輸出檔案：重建後的 3D 體積將儲存為一個多頁的 TIFF 檔案
OUTPUT_FILE = 'reconstructed_volume_v6_B2S1L1.tif'

# 像素合併因子 (Binning Factor)
BINNING_FACTOR = 2

# --- 條紋去除優化參數 (在極座標空間中作用) ---
STRIPE_REMOVAL_SIGMA = 1
STRIPE_REMOVAL_LEVEL = 1
STRIPE_REMOVAL_WNAME = 'db5'


def erf_model(x, a, b, c, d):
    """高斯誤差函數模型，用於擬合背景。"""
    return a * erf((x - b) / c) + d


def reconstruct_and_correct(input_dir, file_pattern, output_file, binning_factor):
    """
    使用 GPU 進行 FBP 重建，然後在重建後的影像上進行環狀偽影校正。
    角度會根據檔案數量自動判斷。
    """
    print(f"--- 開始 3D 重建與後處理任務 (GPU FBP + Post-Processing Ring Removal) ---")

    # 1. 尋找並排序投影檔案
    print(f"1. 在 '{input_dir}' 資料夾中尋找檔案...")
    search_path = os.path.join(input_dir, file_pattern)
    file_list = sorted(glob.glob(search_path))

    if not file_list:
        print(f"錯誤：在 '{search_path}' 中找不到任何檔案。請檢查路徑和檔案模式。")
        return

    # 根據檔案數量自動判斷角度設定
    num_files = len(file_list)
    print(f"   找到 {num_files} 個投影檔案。")

    # 使用 np.linspace 來確保角度數量與檔案數量完全一致，並指定 dtype
    angles_rad = np.deg2rad(np.linspace(0, 180, num_files, dtype=np.float32))
    print(f"   -> 自動設定角度為 0-180 度，共 {num_files} 個角度。")

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

    # 3. 直接使用 ASTRA 工具箱進行逐切片 GPU FBP 重建
    print("3. 開始直接使用 ASTRA 進行逐切片 GPU 重建...")

    reconstructed_slices_with_rings = []
    reco_width = width

    for y in tqdm(range(height), desc="GPU 重建進度"):
        # 提取該切片對應的 2D 正弦圖 (sinogram)
        sinogram = projections_stack[:, y, :]

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
        reconstructed_slices_with_rings.append(reco_slice)

        # g. 清理 ASTRA 記憶體 (非常重要!)
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
    original_size = reconstructed_volume_with_rings[0].shape

    # 定義超取樣後的大小 (OpenCV 的 dsize 是 (width, height))
    # Theta 方向 (角度, 對應高度) 超取樣 Pi 倍
    # r 方向 (半徑, 對應寬度) 超取樣 Sqrt(2) 倍
    oversampled_width = int(original_size[1] * np.sqrt(2))  # X-axis is r
    oversampled_height = int(original_size[0] * np.pi)  # Y-axis is theta
    oversampled_dsize = (oversampled_width, oversampled_height)

    # 步驟 1: 將所有切片批次轉換為超取樣的極座標影像
    print(f"   - 步驟 1/4: 將所有切片轉換至 {oversampled_dsize} 的超取樣極座標...")
    polar_images_list = []
    for original_slice in tqdm(reconstructed_volume_with_rings, desc="極座標轉換進度"):
        polar_image = cv2.warpPolar(original_slice, oversampled_dsize, (center_x, center_y), max_radius,
                                    cv2.WARP_FILL_OUTLIERS)
        polar_images_list.append(polar_image)

    polar_volume = np.stack(polar_images_list, axis=0).astype(np.float32)

    # 新增：儲存極座標體積的臨時檔
    try:
        print("   >> 正在儲存臨時檔：temp_polar_volume.tif ...")
        #tifffile.imwrite('temp_polar_volume.tif', polar_volume)
    except Exception as e:
        print(f"   >> 儲存臨時檔失敗: {e}")

    # 優化步驟：對超取樣的極座標體積的每一行進行更精確的背景扣除
    print("   - 步驟 2/4: 對極座標體積進行背景扣除 (erf擬合)...")
    polar_volume_bg_subtracted = np.copy(polar_volume)
    num_slices, num_rows, num_cols = polar_volume.shape
    x_data = np.arange(num_rows)

    # 固定參數 b (中心點) 和 c (寬度)
    center_b = num_rows / 2.0
    fixed_c = num_rows * 0.02  # 轉變區寬度固定為總像素數的 2%

    for i in tqdm(range(num_slices), desc="背景扣除進度"):
        for j in range(num_cols):  # 遍歷每一垂直行
            y_data = polar_volume[i, :, j]

            # 固定參數 d (垂直位移) 為該行數據的平均值
            fixed_d = np.mean(y_data)

            # 定義一個只擬合參數 a (振幅) 的 lambda 函式
            model_to_fit = lambda x, a: erf_model(x, a, center_b, fixed_c, fixed_d)

            # 判斷是上半部亮還是下半部亮
            mid_point = num_rows // 2
            top_half_mean = np.mean(y_data[mid_point:])
            bottom_half_mean = np.mean(y_data[:mid_point])

            try:
                if top_half_mean > bottom_half_mean:  # 上半部較亮 -> 遞減函數
                    # 振幅 'a' 必須為負
                    p0 = [-(top_half_mean - bottom_half_mean) / 2]
                    bounds = ([-np.inf], [0])
                else:  # 上半部較暗 -> 遞增函數
                    # 振幅 'a' 必須為正
                    p0 = [(bottom_half_mean - top_half_mean) / 2]
                    bounds = ([0], [np.inf])

                # 擬合時只擬合 a 一個參數
                params, _ = curve_fit(model_to_fit, x_data, y_data, p0=p0, bounds=bounds, maxfev=500)
                # 從原始數據中減去擬合出的背景曲線
                fit_curve = model_to_fit(x_data, *params)
                polar_volume_bg_subtracted[i, :, j] -= fit_curve
            except RuntimeError:
                # 如果擬合失敗，則保持該行數據不變
                pass

    # 步驟 3: 對整個極座標體積進行條紋去除
    print("   - 步驟 3/4: 對極座標體積進行條紋去除...")

    # 根據使用者陳述，極座標影像的條紋已為垂直，因此直接處理
    corrected_polar_volume = tomopy.remove_stripe_fw(
        polar_volume_bg_subtracted,
        sigma=STRIPE_REMOVAL_SIGMA,
        level=STRIPE_REMOVAL_LEVEL,
        wname=STRIPE_REMOVAL_WNAME
    )

    print("   條紋去除完成。")

    # 新增：儲存校正後極座標體積的臨時檔
    try:
        print("   >> 正在儲存臨時檔：temp_corrected_polar_volume.tif ...")
        tifffile.imwrite('temp_corrected_polar_volume.tif', corrected_polar_volume)
    except Exception as e:
        print(f"   >> 儲存臨時檔失敗: {e}")

    # 步驟 4: 將校正後的極座標影像批次轉回原始大小的直角座標
    print("   - 步驟 4/4: 將所有切片轉回直角座標...")
    final_corrected_slices = []
    # 目標大小是原始的 (width, height)
    original_dsize = (original_size[1], original_size[0])
    for corrected_polar_slice in tqdm(corrected_polar_volume, desc="直角座標還原進度"):
        corrected_slice = cv2.warpPolar(corrected_polar_slice, original_dsize, (center_x, center_y), max_radius,
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
            binning_factor=BINNING_FACTOR
        )
    except Exception as e:
        print("\n--- 發生嚴重錯誤 ---")
        print("錯誤訊息:", e)
