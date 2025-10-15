import numpy as np
import torch
from scipy import stats

def correct_direction(img):
    return np.fliplr(np.flipud(img.T))

def compute_median_projection(vol: np.ndarray, axis: int, start: int = 40, end: int = 140) -> np.ndarray:
    """計算中位數投影（核心切片範圍）"""
    if axis == 0:
        vol = vol[start:end, :, :]
    elif axis == 1:
        vol = vol[:, start:end, :]
    elif axis == 2:
        vol = vol[:, :, start:end]
    
    median_img = np.median(vol, axis=axis)
    return correct_direction(median_img).copy()