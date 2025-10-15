import numpy as np
import torch
from scipy import stats

def correct_direction(img):
    return np.fliplr(np.flipud(img.T))

def compute_std_projection(vol: np.ndarray, axis: int) -> np.ndarray:
    """計算標準差投影"""
    std_img = np.std(vol, axis=axis)
    return correct_direction(std_img).copy()