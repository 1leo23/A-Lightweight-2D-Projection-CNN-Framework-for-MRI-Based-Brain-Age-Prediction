#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from ipywidgets import interact
import numpy as np
import SimpleITK as sitk
import cv2
import os
import ants
from antspynet.utilities import brain_extraction

def explore_3D_array(arr: np.ndarray, cmap: str = 'gray'):
    def fn(SLICE):
        plt.figure(figsize=(7, 7))
        plt.imshow(arr[SLICE, :, :], cmap=cmap)
    interact(fn, SLICE=(0, arr.shape[0] - 1))

def explore_3D_array_comparison(arr_before: np.ndarray, arr_after: np.ndarray, cmap: str = 'gray'):
    assert arr_after.shape == arr_before.shape
    def fn(SLICE):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10, 10))
        ax1.set_title('Before', fontsize=15)
        ax1.imshow(arr_before[SLICE, :, :], cmap=cmap)
        ax2.set_title('After', fontsize=15)
        ax2.imshow(arr_after[SLICE, :, :], cmap=cmap)
        plt.tight_layout()
    interact(fn, SLICE=(0, arr_before.shape[0] - 1))

def explore_3D_array_with_mask_contour(arr: np.ndarray, mask: np.ndarray, thickness: int = 1):
    assert arr.shape == mask.shape
    def rescale_linear(array, new_min, new_max):
        minimum, maximum = np.min(array), np.max(array)
        m = (new_max - new_min) / (maximum - minimum)
        b = new_min - m * minimum
        return m * array + b

    _arr = rescale_linear(arr, 0, 1)
    _mask = rescale_linear(mask, 0, 1).astype(np.uint8)
    def fn(SLICE):
        arr_rgb = cv2.cvtColor(_arr[SLICE, :, :], cv2.COLOR_GRAY2RGB)
        contours, _ = cv2.findContours(_mask[SLICE, :, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        arr_with_contours = cv2.drawContours(arr_rgb, contours, -1, (0, 1, 0), thickness)
        plt.figure(figsize=(7, 7))
        plt.imshow(arr_with_contours)
    interact(fn, SLICE=(0, arr.shape[0] - 1))

def show_sitk_img_info(img: sitk.Image):
    info = {
        'Pixel Type': img.GetPixelIDTypeAsString(),
        'Dimensions': img.GetSize(),
        'Spacing': img.GetSpacing(),
        'Origin': img.GetOrigin(),
        'Direction': img.GetDirection()
    }
    for k, v in info.items():
        print(f' {k} : {v}')

def custom_brain_extraction(input_path: str, output_path: str, verbose: bool = True):
    """
    使用 ANTs deep learning 模型擷取腦部遮罩，處理單一影像。
    """
    try:
        if verbose:
            print(f"處理: {os.path.basename(input_path)}")

        raw_img = ants.image_read(input_path, reorient='IAL')
        prob_mask = brain_extraction(raw_img, modality='t1', verbose=verbose)
        mask = ants.get_mask(prob_mask, low_thresh=0.4)
        masked = ants.mask_image(raw_img, mask)
        masked.to_file(output_path)

        if verbose:
            print(f"儲存: {output_path}")

    except Exception as e:
        print(f"Error in brain extraction: {e}")
        raise