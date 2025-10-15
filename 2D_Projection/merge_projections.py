import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

def merge_single_file(gray_dir: str, raw_dir: str, output_dir: str, fname: str):
    """合併單一檔案的 6 個投影"""
    paths = {
        'gray_mean': os.path.join(gray_dir, 'mean', fname),
        'gray_median': os.path.join(gray_dir, 'median', fname),
        'gray_std': os.path.join(gray_dir, 'std', fname),
        'raw_mean': os.path.join(raw_dir, 'mean', fname),
        'raw_max': os.path.join(raw_dir, 'max', fname),
        'raw_std': os.path.join(raw_dir, 'std', fname),
    }
    
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        print(f"警告: {fname} 缺少 {missing}")
        return False
    
    channels = []
    for key in ['gray_mean', 'gray_median', 'gray_std', 'raw_mean', 'raw_max', 'raw_std']:
        arr = np.load(paths[key])
        channels.append(arr[:, :, 0])
    
    merged = np.stack(channels, axis=-1)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, fname)
    np.save(output_path, merged)
    return True

def merge_dataset(gray_root: str, raw_root: str, output_root: str, dataset_name: str):
    """合併單一資料集"""
    gray_dir = os.path.join(gray_root, dataset_name)
    raw_dir = os.path.join(raw_root, dataset_name)
    output_dir = os.path.join(output_root, dataset_name)
    
    gray_mean_dir = os.path.join(gray_dir, 'mean')
    if not os.path.exists(gray_mean_dir):
        print(f"找不到資料夾: {gray_mean_dir}")
        return
    
    npy_files = sorted([f for f in os.listdir(gray_mean_dir) if f.endswith('.npy')])
    
    success = 0
    for fname in tqdm(npy_files, desc=f"合併 {dataset_name}"):
        if merge_single_file(gray_dir, raw_dir, output_dir, fname):
            success += 1
    
    print(f"  成功合併 {success}/{len(npy_files)} 個檔案")

def merge_all_datasets(gray_root: str, raw_root: str, output_root: str):
    """批次合併所有資料集"""
    datasets = sorted([d for d in os.listdir(gray_root)
                      if os.path.isdir(os.path.join(gray_root, d))])
    
    for dataset in datasets:
        merge_dataset(gray_root, raw_root, output_root, dataset)