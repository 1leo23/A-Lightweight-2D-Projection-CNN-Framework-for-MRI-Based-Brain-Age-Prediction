import os
import SimpleITK as sitk
import numpy as np

def adaptive_normal(image_path, output_path):
    """
    Normalize MRI image to [-1, 1] range, excluding background voxels in statistics.
    """
    min_p = 0.001
    max_p = 0.999

    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    imgArray = np.float32(image_array)

    # 選取強度>=0的像素
    imgPixel = imgArray[imgArray >= 0]
    imgPixel.sort()

    # 找到0.1%最小值位置
    index = int(round((len(imgPixel) - 1) * min_p + 0.5))
    index = np.clip(index, 0, len(imgPixel) - 1)
    value_min = imgPixel[index]

    # 找到99.9%最大值位置
    index = int(round((len(imgPixel) - 1) * max_p + 0.5))
    index = np.clip(index, 0, len(imgPixel) - 1)
    value_max = imgPixel[index]

    # 均值與標準差計算
    mean = (value_max + value_min) / 2.0
    stddev = (value_max - value_min) / 2.0

    # 進行歸一化
    imgArray = (imgArray - mean) / stddev
    imgArray[imgArray > 1] = 1.0
    imgArray[imgArray < -1] = -1.0

    # 儲存成新的 NIfTI
    norm_image = sitk.GetImageFromArray(imgArray)
    norm_image.CopyInformation(image)
    sitk.WriteImage(norm_image, output_path)

def batch_adaptive_normal(input_dir, output_dir):
    """
    Batch process all .nii and .nii.gz files in input_dir, save results to output_dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fname in os.listdir(input_dir):
        if fname.endswith(".nii") or fname.endswith(".nii.gz"):
            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, fname)
            print(f"\u2728 正在處理: {fname}")
            adaptive_normal(input_path, output_path)
    print("\n\u2705 全部處理完成！")

if __name__ == "__main__":
    # 修改這裡來設定你的資料夾路徑
    input_folder = r"D:\data\ADNI\50~59"
    output_folder = r"D:\data\icbm\10~19\minmax2"

    batch_adaptive_normal(input_folder, output_folder)