import numpy as np
import torch
from scipy import stats

def correct_direction(img):
    return np.fliplr(np.flipud(img.T))

def compute_max_projection(vol: np.ndarray, axis: int) -> np.ndarray:
    """計算最大值投影"""
    max_img = np.max(vol, axis=axis)
    return correct_direction(max_img).copy()