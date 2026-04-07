import os
import numpy as np
import tifffile
import struct
import time
import shutil
import tkinter as tk
import tempfile
from tkinter import filedialog

# --- 設定 ---
# PXM 檔案所在的資料夾名稱
PXM_FOLDER_NAME = "PXM"
# PXM 檔案的標頭大小 (bytes)
HEADER_SIZE = 217912
# 觸發處理程序的 PXM 檔案數量
PROCESSING_TRIGGER_COUNT = 723
# 監控資料夾的檢查間隔 (秒)
POLL_INTERVAL_SECONDS = 10

def get_image_dimensions(file_path):
    """
    根據檔案大小判斷影像的寬和高。

    Args:
        file_path (str): 檔案的路徑。

    Returns:
        tuple: (寬, 高) 或 (None, None) 如果檔案大小不符合預期。
    """
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if 1.8 <= file_size_mb <= 2.2:
        return 1280, 1080  # 2MB 左右的檔案
    elif 7.8 <= file_size_mb <= 8.2:
        return 2560, 2160  # 8MB 左右的檔案
    else:
        return None, None

def process_pxm_file(pxm_path, tif_path):
    """
    讀取單一 PXM 檔案，轉換並儲存為 TIF 檔案。

    Args:
        pxm_path (str): 來源 PXM 檔案的路徑。
        tif_path (str): 目標 TIF 檔案的路徑。
    """
    width, height = get_image_dimensions(pxm_path)
    if not width:
        print(f"\n警告：跳過檔案 {os.path.basename(pxm_path)}，因為其大小 ({os.path.getsize(pxm_path) / (1024*1024):.2f} MB) 不符合預期。")
        return

    # 呼叫此函數的地方會顯示進度，這裡不再重複輸出
    try:
        with open(pxm_path, 'rb') as f:
            # 略過檔案開頭的標頭
            f.seek(HEADER_SIZE)
            
            # 讀取剩下的資料內容
            raw_data = f.read()

        # 預期像素總數
        total_pixels = width * height
        
        # 存放解碼後的 16-bit 像素資料
        pixel_data = []

        # 每 3 個 bytes 為一組進行處理
        # 這會產生 2 個 16-bit 的資料點
        for i in range(0, len(raw_data), 3):
            # 確保不會讀取到檔案結尾的不完整資料組
            if i + 2 >= len(raw_data):
                break

            # 使用 struct.unpack 讀取 3 個 unsigned bytes
            byte1, byte2, byte3 = struct.unpack_from('<BBB', raw_data, i)

            # --- 根據規則重組資料點 ---
            # 資料點 1:
            # bits 13-16 來自 byte2 的前 4 bits
            # bits 5-12 來自 byte1
            # bits 1-4 為 0 (透過左移 4 bits 實現)
            datapoint1 = (byte1 << 4) | (byte2 >> 4)

            # 資料點 2:
            # bits 13-16 來自 byte2 的後 4 bits
            # bits 5-12 來自 byte3
            # bits 1-4 為 0 (透過左移 4 bits 實現)
            datapoint2 = (byte3 << 4) | (byte2 & 0x0F)

            pixel_data.append(datapoint1)
            pixel_data.append(datapoint2)

        # 檢查解析出的像素數量是否正確
        if len(pixel_data) != total_pixels:
            print(f"\n警告：檔案 {os.path.basename(pxm_path)} 的資料長度不符。")
            print(f"  預期像素數: {total_pixels}, 實際解析出: {len(pixel_data)}")
            # 如果像素數量不足，用 0 填補剩餘部分
            pixel_data.extend([0] * (total_pixels - len(pixel_data)))


        # 將 list 轉換為 NumPy array，並設定資料型態為 16-bit unsigned integer
        image_array = np.array(pixel_data, dtype=np.uint16)
        
        # 將一維陣列重塑為 (height, width) 的二維影像
        image_array = image_array.reshape((height, width))

        # 使用 tifffile 將 NumPy array 寫入 TIF 檔案
        tifffile.imwrite(tif_path, image_array)

    except FileNotFoundError:
        print(f"\n錯誤：找不到檔案 {pxm_path}")
    except Exception as e:
        print(f"\n處理檔案 {os.path.basename(pxm_path)} 時發生錯誤：{e}")


def calculate_center_of_rotation(temp_tif_dir, sample_dir):
    """
    執行扣背、滑動窗口裁切、左右翻轉、影像相減，最後計算絕對值總和以尋找旋轉中心。
    """
    print("  正在讀取暫存的 TIF 檔案...")
    # 讀取 4 個轉換好的 TIF
    bg0_path = os.path.join(temp_tif_dir, "PXMs_00_0000_0000.tif")
    bg1_path = os.path.join(temp_tif_dir, "PXMs_00_0000_0001.tif")
    proj0_path = os.path.join(temp_tif_dir, "PXMs_04_0000_0000.tif")
    proj180_path = os.path.join(temp_tif_dir, "PXMs_04_0000_0720.tif")

    try:
        bg0 = tifffile.imread(bg0_path).astype(np.float32)
        bg1 = tifffile.imread(bg1_path).astype(np.float32)
        proj0 = tifffile.imread(proj0_path).astype(np.float32)
        proj180 = tifffile.imread(proj180_path).astype(np.float32)
    except Exception as e:
        print(f"  讀取檔案失敗: {e}")
        return

    # 避免除以零
    bg0[bg0 == 0] = 1.0
    bg1[bg1 == 0] = 1.0

    print("  正在執行扣背...")
    img0 = proj0 / bg0
    img180 = proj180 / bg1

    # 存成 000.tif, 180.tif 暫存檔
    tifffile.imwrite(os.path.join(temp_tif_dir, "000.tif"), img0)
    tifffile.imwrite(os.path.join(temp_tif_dir, "180.tif"), img180)

    # 計算 n (寬度的 1% 並無條件進位)
    width = img0.shape[1]
    n = int(np.ceil(width * 0.01))
    center_idx = width // 2

    results = []
    total_steps = 2 * n + 1

    print(f"  影像寬度: {width} pixel, n={n}, 將產生 {total_steps} 組檔案進行比對...")

    for i in range(total_steps):
        left_crop = i
        right_crop = 2 * n - i
        suffix = center_idx - n + i

        # 裁切
        if right_crop == 0:
            crop0 = img0[:, left_crop:]
            crop180 = img180[:, left_crop:]
        else:
            crop0 = img0[:, left_crop:-right_crop]
            crop180 = img180[:, left_crop:-right_crop]

        # 存檔 000_*.tif, 180_*.tif
        tifffile.imwrite(os.path.join(temp_tif_dir, f"000_{suffix}.tif"), crop0)
        tifffile.imwrite(os.path.join(temp_tif_dir, f"180_{suffix}.tif"), crop180)

        # 左右鏡像翻轉 -> 180m_*.tif
        crop180_flipped = np.fliplr(crop180)
        tifffile.imwrite(os.path.join(temp_tif_dir, f"180m_{suffix}.tif"), crop180_flipped)

        # 影像相減 (000 扣掉對應的 180m) -> convolution_*.tif
        diff = crop0 - crop180_flipped
        tifffile.imwrite(os.path.join(temp_tif_dir, f"convolution_{suffix}.tif"), diff)

        # 計算絕對值總和 (使用 float64 避免溢位)
        abs_diff_sum = np.sum(np.abs(diff), dtype=np.float64)
        results.append((suffix, abs_diff_sum))

    # 尋找總和最低值 (最佳中心點)
    best_suffix, min_sum = min(results, key=lambda x: x[1])

    # 輸出表格
    output_file = os.path.join(sample_dir, f"{best_suffix}.auto_detected_center")
    print(f"  正在儲存結果至 {os.path.basename(output_file)} ...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for suffix, val in results:
                f.write(f"{suffix}, {int(val)}\n")
    except Exception as e:
        print(f"  儲存結果失敗: {e}")

    print(f"  最佳旋轉中心計算完成: {best_suffix} (總和誤差: {int(min_sum)})")


def process_sample_folder(sample_dir):
    """
    對單一樣品資料夾執行完整的處理流程：
    1. 讀取指定的 4 個 PXM 檔案並轉換為暫存 TIF。
    2. 執行扣背、滑動窗口裁切、翻轉與相減計算。
    3. 輸出結果表格 (*.auto_detected_center)。
    4. 清理暫存檔案。
    """
    print(f"\n--- 開始處理樣品資料夾: {os.path.basename(sample_dir)} ---")

    pxm_dir = os.path.join(sample_dir, PXM_FOLDER_NAME)
    temp_tif_dir = None  # 確保在 finally 區塊中可用

    try:
        # 在本機的系統暫存區建立一個唯一的資料夾
        temp_tif_dir = tempfile.mkdtemp(prefix="pxm_processing_")

        print(f"暫存 TIF 資料夾 (本機): {temp_tif_dir}")

        # --- 步驟 1 & 2: 讀取指定 PXM 並轉換 ---
        target_files = [
            "PXMs_00_0000_0000.pxm",
            "PXMs_00_0000_0001.pxm",
            "PXMs_04_0000_0000.pxm",
            "PXMs_04_0000_0720.pxm"
        ]
        print(f"\n步驟 1 & 2: 正在轉換指定的 4 個 PXM 檔案...")

        for filename in target_files:
            pxm_file_path = os.path.join(pxm_dir, filename)
            if not os.path.exists(pxm_file_path):
                print(f"  錯誤：找不到必要檔案 {filename}")
                return

            tif_filename = os.path.splitext(filename)[0] + ".tif"
            temp_tif_path = os.path.join(temp_tif_dir, tif_filename)

            print(f"  正在轉換 {filename}...")
            process_pxm_file(pxm_file_path, temp_tif_path)

        # --- 步驟 3 ~ 8: 執行尋找旋轉中心流程 ---
        print("\n步驟 3 ~ 8: 正在執行扣背與尋找旋轉中心...")
        calculate_center_of_rotation(temp_tif_dir, sample_dir)

        print(f"--- 樣品資料夾 '{os.path.basename(sample_dir)}' 處理完成 ---\n")
    finally:
        # --- 清理暫存資料夾 ---
        if temp_tif_dir and os.path.isdir(temp_tif_dir):
            print("\n正在清理暫存檔案...")
            try:
                shutil.rmtree(temp_tif_dir)
                print(f"成功刪除本機暫存資料夾: {temp_tif_dir}")
            except Exception as e:
                print(f"錯誤：刪除本機暫存資料夾 {temp_tif_dir} 時失敗: {e}")


def monitor_folder(monitoring_dir):
    """
    監控指定資料夾，尋找符合條件的樣品資料夾並進行處理。
    """
    print(f"開始監控資料夾: {monitoring_dir}")
    print(f"每隔 {POLL_INTERVAL_SECONDS} 秒檢查一次。按 Ctrl+C 停止程式。")
    
    try:
        while True:
            # 取得監控資料夾中的所有子目錄 (樣品資料夾)
            for item in os.listdir(monitoring_dir):
                sample_dir = os.path.join(monitoring_dir, item)
                
                # 檢查是否為一個尚未處理的資料夾 (檢查是否存在 .auto_detected_center 檔案)
                if os.path.isdir(sample_dir) and not any(f.endswith(".auto_detected_center") for f in os.listdir(sample_dir)):
                    
                    # 檢查 PXM 資料夾是否存在
                    pxm_dir = os.path.join(sample_dir, PXM_FOLDER_NAME)
                    if not os.path.isdir(pxm_dir):
                        continue

                    # 檢查 PXM 資料夾中的檔案數量是否達到觸發條件
                    try:
                        files = [f for f in os.listdir(pxm_dir) if os.path.isfile(os.path.join(pxm_dir, f))]
                        if len(files) == PROCESSING_TRIGGER_COUNT:
                            print(f"\n偵測到 '{os.path.basename(sample_dir)}/{PXM_FOLDER_NAME}' 中有 {len(files)} 個檔案，觸發處理程序。")
                            process_sample_folder(sample_dir)
                    except Exception as e:
                        print(f"處理資料夾 {sample_dir} 時發生錯誤: {e}")

            # 等待一段時間後再次檢查
            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n使用者手動停止監控，程式結束。")

def main():
    """
    主執行函數。
    """
    # 確保在執行前已安裝必要的套件
    try:
        import numpy
        import tifffile
    except ImportError:
        print("錯誤：缺少必要的套件。請先安裝 numpy 和 tifffile。")
        print("您可以使用以下指令安裝：")
        print("pip install numpy tifffile")
        return

    # 使用 Tkinter GUI 讓使用者選擇要監控的資料夾
    root = tk.Tk()
    root.withdraw()  # 隱藏主視窗
    
    print("請在彈出視窗中選擇一個要監控的資料夾...")
    monitoring_dir = filedialog.askdirectory(title="請選擇要監控的資料夾")

    if not monitoring_dir:
        print("未選擇任何資料夾，程式即將結束。")
        return

    # 開始監控指定的資料夾
    monitor_folder(monitoring_dir)

if __name__ == "__main__":
    main()
