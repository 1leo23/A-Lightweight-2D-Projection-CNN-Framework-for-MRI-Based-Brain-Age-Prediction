import ants
import os

def ants_n4_correction(input_path, output_path, intensity_truncation=(0.025, 0.975, 256), mask=None):
    """
    使用 ANTsPy 進行 N4 偏場校正。

    參數:
    - input_path: str, 輸入 NIfTI 檔案路徑
    - output_path: str, 輸出 NIfTI 檔案路徑
    - intensity_truncation: tuple, (下界百分比, 上界百分比, 數量級)，用於裁剪強度範圍（預設: (0.025, 0.975, 256)）
    - mask: ants.ANTsImage 或 None，自訂腦區範圍 mask，若無會使用整張影像

    儲存:
    - 校正後的影像會存為 NIfTI(.nii.gz)
    """
    print(f":arrows_counterclockwise: 正在處理: {os.path.basename(input_path)}")

    # 讀影像
    image = ants.image_read(input_path)

    # 強度裁剪 (避免極端值影響)
    image_truncated = ants.iMath(
        image,
        "TruncateIntensity",
        intensity_truncation[0],
        intensity_truncation[1],
        intensity_truncation[2]
    )

    # N4 校正
    corrected_image = ants.n4_bias_field_correction(
        image_truncated,
        mask=mask,
        shrink_factor=4,  # 預設 4 倍降解析度加速
        convergence={'iters': [50, 50, 50, 50], 'tol': 1e-7},
        verbose=True
    )

    # 儲存
    ants.image_write(corrected_image, output_path)
    print(f":white_check_mark: 已儲存: {output_path}\n")

# ===================== 批次範例 =====================

if __name__ == '__main__':
    input_folder = r"D:\6_30_hsu\new\MNI\temp\IXI_mni"
    output_folder = r"D:\6_30_hsu\new\MNI\temp\IXI_mni_n4"
    os.makedirs(output_folder, exist_ok=True)

    nii_files = [f for f in os.listdir(input_folder) if f.endswith((".nii", ".nii.gz"))]

    for f in nii_files:
        input_path = os.path.join(input_folder, f)
        output_path = os.path.join(output_folder, f.replace(".nii", "_n4.nii").replace(".nii.gz", "_n4.nii.gz"))
        ants_n4_correction(input_path, output_path)