import sys, gc
from pathlib import Path
import time
import logging
import psutil
import multiprocessing as mp
from typing import List, Tuple

import ants
from tqdm import tqdm

from n4_correction import ants_n4_correction
from brain_extraction import custom_brain_extraction
from mni_registration import mni_registration
from normalization import zscore_normalization, adaptive_normal

INPUT_ROOT = Path(r"D:\8_19_pre_2")
OUTPUT_ROOT = Path(r"D:\8_19_pre_2\pre\pre")
MNI_PATH = Path(r"C:\Users\user1\Desktop\Hsu_leo\MNI\MNI152_T1_1mm_brain.nii.gz")
VALID_EXT = (".nii", ".nii.gz")

ENABLE_MULTIPROCESSING = True
NUM_PROCESSES = min(4, mp.cpu_count() - 1)
SKIP_EXISTING = True
MEMORY_LIMIT_GB = 8

def setup_logging():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_ROOT / "preprocessing.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def check_file_exists(file_path: Path) -> bool:
    return file_path.exists() and file_path.stat().st_size > 0

def check_memory_usage():
    memory_gb = psutil.virtual_memory().used / (1024**3)
    if memory_gb > MEMORY_LIMIT_GB:
        gc.collect()
        return False
    return True

def get_all_files() -> List[Tuple[str, Path]]:
    all_files = []
    for dataset_dir in INPUT_ROOT.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        for file_path in dataset_dir.rglob("*"):
            if file_path.is_file() and file_path.name.lower().endswith(VALID_EXT):
                all_files.append((dataset_name, file_path))
    return all_files

def create_output_structure(datasets: List[str]):
    for dataset_name in datasets:
        for step in ("n4", "brain", "mni", "zscore", "min_max"):
            (OUTPUT_ROOT / f"{dataset_name}_{step}").mkdir(parents=True, exist_ok=True)

def process_n4_single(args: Tuple[str, Path]) -> dict:
    dataset_name, src_file = args
    
    if ENABLE_MULTIPROCESSING and mp.current_process().name != 'MainProcess':
        from n4_correction import ants_n4_correction
    
    result = {'dataset': dataset_name, 'filename': src_file.name, 'success': False, 'error': None, 'skipped': False}
    
    try:
        out_n4 = OUTPUT_ROOT / f"{dataset_name}_n4" / src_file.name
        if SKIP_EXISTING and check_file_exists(out_n4):
            result['skipped'] = True
        else:
            ants_n4_correction(str(src_file), str(out_n4))
        result['success'] = True
        gc.collect()
    except Exception as e:
        result['error'] = str(e)
        gc.collect()
    
    return result

def process_brain_extraction_single(args: Tuple[str, Path]) -> dict:
    dataset_name, src_file = args
    result = {'dataset': dataset_name, 'filename': src_file.name, 'success': False, 'error': None, 'skipped': False}
    
    try:
        in_n4 = OUTPUT_ROOT / f"{dataset_name}_n4" / src_file.name
        out_brain = OUTPUT_ROOT / f"{dataset_name}_brain" / src_file.name
        if SKIP_EXISTING and check_file_exists(out_brain):
            result['skipped'] = True
        else:
            custom_brain_extraction(str(in_n4), str(out_brain), verbose=False)
        result['success'] = True
        gc.collect()
    except Exception as e:
        result['error'] = str(e)
        gc.collect()
    
    return result

def process_mni_registration_single(args: Tuple[str, Path, ants.ANTsImage]) -> dict:
    dataset_name, src_file, mni_template = args
    result = {'dataset': dataset_name, 'filename': src_file.name, 'success': False, 'error': None, 'skipped': False}
    
    try:
        in_brain = OUTPUT_ROOT / f"{dataset_name}_brain" / src_file.name
        out_mni = OUTPUT_ROOT / f"{dataset_name}_mni" / src_file.name
        if SKIP_EXISTING and check_file_exists(out_mni):
            result['skipped'] = True
        else:
            mni_registration(str(in_brain), str(out_mni), mni_template)
        result['success'] = True
        gc.collect()
    except Exception as e:
        result['error'] = str(e)
        gc.collect()
    
    return result

def process_zscore_single(args: Tuple[str, Path]) -> dict:
    dataset_name, src_file = args
    result = {'dataset': dataset_name, 'filename': src_file.name, 'success': False, 'error': None, 'skipped': False}
    
    try:
        in_mni = OUTPUT_ROOT / f"{dataset_name}_mni" / src_file.name
        out_zscore = OUTPUT_ROOT / f"{dataset_name}_zscore" / src_file.name
        if SKIP_EXISTING and check_file_exists(out_zscore):
            result['skipped'] = True
        else:
            zscore_normalization(str(in_mni), str(out_zscore))
        result['success'] = True
        gc.collect()
    except Exception as e:
        result['error'] = str(e)
        gc.collect()
    
    return result

def process_minmax_single(args: Tuple[str, Path]) -> dict:
    dataset_name, src_file = args
    
    if ENABLE_MULTIPROCESSING and mp.current_process().name != 'MainProcess':
        from normalization import adaptive_normal
    
    result = {'dataset': dataset_name, 'filename': src_file.name, 'success': False, 'error': None, 'skipped': False}
    
    try:
        in_mni = OUTPUT_ROOT / f"{dataset_name}_mni" / src_file.name
        out_minmax = OUTPUT_ROOT / f"{dataset_name}_min_max" / src_file.name
        if SKIP_EXISTING and check_file_exists(out_minmax):
            result['skipped'] = True
        else:
            adaptive_normal(str(in_mni), str(out_minmax))
        result['success'] = True
        gc.collect()
    except Exception as e:
        result['error'] = str(e)
        gc.collect()
    
    return result

def process_step_batch(step_name: str, process_func, args_list: List, logger, mni_template=None):
    print(f"\n開始步驟: {step_name}")
    logger.info(f"開始處理步驟: {step_name} ({len(args_list)} 個檔案)")
    
    step_start_time = time.time()
    success_count = failed_count = skipped_count = 0
    
    if step_name == "MNI Registration" and mni_template is not None:
        args_list = [(dataset, file_path, mni_template) for dataset, file_path in args_list]
    
    use_multiprocessing = ENABLE_MULTIPROCESSING and NUM_PROCESSES > 1
    if step_name == "Brain Extraction":
        use_multiprocessing = False
    
    if use_multiprocessing:
        with mp.Pool(processes=NUM_PROCESSES) as pool:
            results = pool.map(process_func, args_list)
            for result in results:
                if result['success']:
                    skipped_count += 1 if result['skipped'] else 0
                    success_count += 0 if result['skipped'] else 1
                else:
                    failed_count += 1
    else:
        for args in tqdm(args_list, desc=step_name):
            result = process_func(args)
            if result['success']:
                skipped_count += 1 if result['skipped'] else 0
                success_count += 0 if result['skipped'] else 1
            else:
                failed_count += 1
            if step_name == "Brain Extraction":
                gc.collect()
    
    step_time = time.time() - step_start_time
    print(f"完成: 成功 {success_count}, 失敗 {failed_count}, 跳過 {skipped_count}, 耗時 {step_time/60:.1f} 分鐘")
    
    gc.collect()
    return {'step': step_name, 'success': success_count, 'failed': failed_count, 'skipped': skipped_count}

def run_preprocessing_pipeline():
    print("批量資料集預處理流水線")
    print("="*70)
    
    logger = setup_logging()
    all_files = get_all_files()
    
    if not all_files:
        print("未找到任何需要處理的檔案")
        return
    
    datasets = list(set(dataset for dataset, _ in all_files))
    create_output_structure(datasets)
    
    mni_template = ants.image_read(str(MNI_PATH))
    
    processing_steps = [
        ("N4 Bias Correction", process_n4_single, all_files, None),
        ("Brain Extraction", process_brain_extraction_single, all_files, None),
        ("MNI Registration", process_mni_registration_single, all_files, mni_template),
        ("Z-score Normalization", process_zscore_single, all_files, None),
        ("Min-Max Normalization", process_minmax_single, all_files, None),
    ]
    
    for step_name, process_func, file_list, template in processing_steps:
        process_step_batch(step_name, process_func, file_list, logger, template)
    
    print("\n所有處理完成!")

if __name__ == "__main__":
    if sys.platform.startswith('win') and ENABLE_MULTIPROCESSING:
        mp.set_start_method('spawn', force=True)
    run_preprocessing_pipeline()