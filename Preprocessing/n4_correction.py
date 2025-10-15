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
    image = ants.image_read(input_path)

    image_truncated = ants.iMath(
        image,
        "TruncateIntensity",
        intensity_truncation[0],
        intensity_truncation[1],
        intensity_truncation[2]
    )

    corrected_image = ants.n4_bias_field_correction(
        image_truncated,s
        mask=mask,
        shrink_factor=4,
        convergence={'iters': [50, 50, 50, 50], 'tol': 1e-7},
        verbose=False
    )

    ants.image_write(corrected_image, output_path)