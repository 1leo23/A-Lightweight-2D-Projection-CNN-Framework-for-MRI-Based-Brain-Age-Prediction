import numpy as np
import torch
from scipy import stats

def correct_direction(img):
    return np.fliplr(np.flipud(img.T))

def compute_mean_projection(vol: np.ndarray, axis: int) -> np.ndarray:
    """計算均值投影"""
    mean_img = np.mean(vol, axis=axis)
    return correct_direction(mean_img).copy()