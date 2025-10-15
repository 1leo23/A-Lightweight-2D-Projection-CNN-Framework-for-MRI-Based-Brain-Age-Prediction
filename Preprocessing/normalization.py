import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib

def zscore_normalization(input_path: str, output_path: str):
    """Z-score 標準化"""
    nb = nib.load(input_path)
    data = nb.get_fdata()
    zdata = (data - np.mean(data)) / np.std(data)
    nib.save(nib.Nifti1Image(zdata, nb.affine, nb.header), output_path)

def adaptive_normal(image_path: str, output_path: str):
    """
    Normalize MRI image to [-1, 1] range, excluding background voxels in statistics.
    """
    min_p = 0.001
    max_p = 0.999

    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    imgArray = np.float32(image_array)

    imgPixel = imgArray[imgArray >= 0]
    imgPixel.sort()

    index = int(round((len(imgPixel) - 1) * min_p + 0.5))
    index = np.clip(index, 0, len(imgPixel) - 1)
    value_min = imgPixel[index]

    index = int(round((len(imgPixel) - 1) * max_p + 0.5))
    index = np.clip(index, 0, len(imgPixel) - 1)
    value_max = imgPixel[index]

    mean = (value_max + value_min) / 2.0
    stddev = (value_max - value_min) / 2.0

    imgArray = (imgArray - mean) / stddev
    imgArray[imgArray > 1] = 1.0
    imgArray[imgArray < -1] = -1.0

    norm_image = sitk.GetImageFromArray(imgArray)
    norm_image.CopyInformation(image)
    sitk.WriteImage(norm_image, output_path)