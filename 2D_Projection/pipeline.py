import os
import numpy as np
import nibabel as nib
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
from scipy import stats
from pathlib import Path

from projection_mean import compute_mean_projection
from projection_median import compute_median_projection
from projection_std import compute_std_projection
from projection_max import compute_max_projection
from merge_projections import merge_all_datasets

target_size = 218

def pad_to_218(tensor: torch.Tensor, pad_value: float) -> torch.Tensor:
    c, h, w = tensor.shape
    pad_h = target_size - h
    pad_w = target_size - w
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"影像尺寸 {h}x{w} 大於 target_size={target_size}")
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return TF.pad(tensor, [pad_left, pad_top, pad_right, pad_bottom], fill=pad_value)

def get_background_value(tensor: torch.Tensor) -> float:
    flat = tensor.numpy().flatten()
    return float(stats.mode(flat, keepdims=True).mode[0])

def save_projection(proj: np.ndarray, save_path: str, print_bg: bool = False):
    if os.path.exists(save_path):
        return
    
    tensor = torch.tensor(proj, dtype=torch.float32).unsqueeze(0)
    bg_val = get_background_value(tensor)
    
    if print_bg:
        print(f"Background value: {bg_val:.2f}")
    
    proj_pad = pad_to_218(tensor, pad_value=bg_val)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out_array = np.transpose(proj_pad.numpy(), (1, 2, 0))
    np.save(save_path, out_array)

def process_single_file(nifti_path: str, out_dir: str, image_type: str, print_bg: bool = False):
    vol = nib.load(nifti_path).get_fdata()
    base = Path(nifti_path).stem.replace('.nii', '')
    
    if image_type == 'gray':
        projections = {
            'mean': compute_mean_projection,
            'median': compute_median_projection,
            'std': compute_std_projection
        }
    elif image_type == 'raw':
        projections = {
            'mean': compute_mean_projection,
            'max': compute_max_projection,
            'std': compute_std_projection
        }
    else:
        raise ValueError("image_type 必須是 'gray' 或 'raw'")
    
    axes = {'axi': 1, 'con': 2, 'sal': 0}
    
    for proj_name, proj_func in projections.items():
        proj_dir = os.path.join(out_dir, proj_name)
        os.makedirs(proj_dir, exist_ok=True)
        
        for axis_name, axis_idx in axes.items():
            save_path = os.path.join(proj_dir, f"{base}.{axis_name}.npy")
            
            if proj_name == 'median':
                proj = proj_func(vol, axis_idx, start=40, end=140)
            else:
                proj = proj_func(vol, axis_idx)
            
            save_projection(proj, save_path, print_bg=print_bg)

def process_dataset(input_folder: str, output_root: str, image_type: str):
    dataset_name = os.path.basename(input_folder)
    out_dir = os.path.join(output_root, dataset_name)
    
    nii_files = sorted([f for f in os.listdir(input_folder)
                       if f.endswith('.nii') or f.endswith('.nii.gz')])
    
    for i, fname in enumerate(tqdm(nii_files, desc=f"{dataset_name} ({image_type})")):
        nifti_path = os.path.join(input_folder, fname)
        print_bg = (i < 2)
        process_single_file(nifti_path, out_dir, image_type, print_bg=print_bg)

def process_all_datasets(gray_input_root: str, raw_input_root: str, output_root: str):
    print("="*60)
    print("開始處理灰質機率圖投影")
    print("="*60)
    
    gray_datasets = sorted([d for d in os.listdir(gray_input_root)
                           if os.path.isdir(os.path.join(gray_input_root, d))])
    
    for name in gray_datasets:
        input_path = os.path.join(gray_input_root, name)
        output_path = os.path.join(output_root, 'gray')
        process_dataset(input_path, output_path, image_type='gray')
    
    print("\n" + "="*60)
    print("開始處理原始腦影像投影")
    print("="*60)
    
    raw_datasets = sorted([d for d in os.listdir(raw_input_root)
                          if os.path.isdir(os.path.join(raw_input_root, d))])
    
    for name in raw_datasets:
        input_path = os.path.join(raw_input_root, name)
        output_path = os.path.join(output_root, 'raw')
        process_dataset(input_path, output_path, image_type='raw')
    
    print("\n所有投影處理完成！")

def run_complete_pipeline(gray_input: str, raw_input: str, projection_output: str, merged_output: str):
    """執行完整 pipeline：投影 + 合併"""
    
    print("="*60)
    print("開始完整 Pipeline")
    print("="*60)
    
    print("\nStep 1: 生成投影")
    process_all_datasets(gray_input, raw_input, projection_output)
    
    print("\n" + "="*60)
    print("Step 2: 合併投影")
    print("="*60)
    
    gray_proj_root = os.path.join(projection_output, 'gray')
    raw_proj_root = os.path.join(projection_output, 'raw')
    merge_all_datasets(gray_proj_root, raw_proj_root, merged_output)
    
    print("\n" + "="*60)
    print("Pipeline 完成！")
    print("="*60)
    print(f"投影輸出: {projection_output}")
    print(f"合併輸出: {merged_output}")
    print("合併後每個 .npy 檔案形狀: (218, 218, 6)")
    print("通道順序: [gray_mean, gray_median, gray_std, raw_mean, raw_max, raw_std]")

if __name__ == "__main__":
    gray_input = r"D:\6_30_hsu\7_8_zscore_fsl\gray\8_4_8dataset"
    raw_input = r"D:\6_30_hsu\7_8_processed\zscore\8_4_8dataset"
    projection_output = r"D:\8_4_8dataset_2D_pipeline"
    merged_output = r"D:\8_4_8dataset_2D_merged"
    
    run_complete_pipeline(gray_input, raw_input, projection_output, merged_output)