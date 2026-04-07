import os
import numpy as np
import tifffile
import shutil
import tkinter as tk
import tempfile
from tkinter import filedialog

def calculate_center_of_rotation(tif_path0, tif_path180):
    """
    執行滑動窗口裁切、左右翻轉、影像相減，最後計算絕對值總和以尋找旋轉中心。
    """
    print(f"\n正在讀取 TIF 檔案:\n1: {tif_path0}\n2: {tif_path180}")

    try:
        img0 = tifffile.imread(tif_path0).astype(np.float32)
        img180 = tifffile.imread(tif_path180).astype(np.float32)
    except Exception as e:
        print(f"  讀取檔案失敗: {e}")
        return

    # 計算 n (寬度的 1% 並無條件進位)
    width = img0.shape[1]
    n = int(np.ceil(width * 0.01))
    center_idx = width // 2

    results = []
    total_steps = 2 * n + 1

    print(f"  影像寬度: {width} pixel, n={n}, 將產生 {total_steps} 組檔案進行比對...")

    temp_tif_dir = tempfile.mkdtemp(prefix="tif_center_processing_")
    print(f"  暫存處理資料夾: {temp_tif_dir}")

    try:
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

        # 將表格輸出到 tif_path0 所在目錄的上一層資料夾
        parent_dir = os.path.dirname(os.path.dirname(tif_path0))
        output_file = os.path.join(parent_dir, f"{best_suffix}.auto_detected_center")
        
        print(f"  正在儲存結果至 {output_file} ...")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for suffix, val in results:
                    f.write(f"{suffix}, {int(val)}\n")
        except Exception as e:
            print(f"  儲存結果失敗: {e}")

        print(f"  最佳旋轉中心計算完成: {best_suffix} (總和誤差: {int(min_sum)})")
        
    finally:
        # --- 清理暫存資料夾 ---
        if os.path.isdir(temp_tif_dir):
            try:
                shutil.rmtree(temp_tif_dir)
                print(f"  暫存資料夾已清理。")
            except Exception as e:
                print(f"  錯誤：刪除暫存資料夾失敗: {e}")

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

    # 使用 Tkinter GUI 讓使用者選擇檔案
    root = tk.Tk()
    root.withdraw()  # 隱藏主視窗
    
    print("請在彈出視窗中選擇兩個已扣好背的 32-bit TIF 檔案 (分別代表0度與180度)...")
    file_paths = filedialog.askopenfilenames(
        title="請選擇兩個 TIF 檔案",
        filetypes=[("TIF Files", "*.tif;*.tiff"), ("All Files", "*.*")]
    )

    if not file_paths or len(file_paths) != 2:
        print("您必須選擇剛好兩個 TIF 檔案，程式即將結束。")
        return

    tif_path0, tif_path180 = file_paths[0], file_paths[1]
    calculate_center_of_rotation(tif_path0, tif_path180)

if __name__ == "__main__":
    main()
