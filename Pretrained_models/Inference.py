"""
Brain Age Prediction - Interactive Inference Script

Interactive inference script supporting:
- Single-plane prediction
- Three-plane ensemble (s=1.0)
- Bias-Corrected ensemble (s=1.0 + s=0.3)
"""

import torch
import numpy as np
import os
from pathlib import Path
import torch.nn as nn


# ============================================================
# Model Definition
# ============================================================

class BrainAgeModelWithSex(nn.Module):
    def __init__(self, in_ch=6, hid=128):
        super().__init__()
        # 3-layer CNN: 64 -> 128 -> 256
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 218 -> 109
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 109 -> 54
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()         # Spatial size remains 54x54
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        # Calculate img_feature_dim (will be 256)
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, 218, 218)
            x = self.block1(dummy)
            x = self.block2(x)
            x = self.block3(x)
            x = self.gap(x).flatten(1)
            img_feature_dim = x.shape[1]  # 256

        # +1 for sex (scalar)
        self.fc = nn.Sequential(
            nn.Linear(img_feature_dim + 1, hid),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hid, hid // 2),
            nn.ReLU(),
            nn.Linear(hid // 2, 1)
        )

    def forward(self, x, sex):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).flatten(1)          # (B, 256)
        x = torch.cat([x, sex.unsqueeze(1)], dim=1)  # (B, 257)
        return self.fc(x).squeeze(-1)       # (B,)


# ============================================================
# Core Functions
# ============================================================

def load_model(model_path, device='cuda'):
    """Load pretrained model"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Auto-detect input channels
    state = torch.load(model_path, map_location=device)
    in_ch = state['block1.0.weight'].shape[1]
    
    # Build and load model
    model = BrainAgeModelWithSex(in_ch=in_ch)
    model.load_state_dict(state)
    model.to(device).eval()
    
    return model, device, in_ch


def predict_single(image, sex, model, device):
    """
    Predict brain age for a single subject
    
    Args:
        image: numpy array (C, H, W) or (H, W, C)
        sex: 0 (female) or 1 (male)
        model: loaded model
        device: torch device
    
    Returns:
        predicted age (float)
    """
    # Ensure channel-first format
    if image.shape[0] > image.shape[2]:
        image = image.transpose(2, 0, 1)
    
    # Convert to tensor
    img = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    sex_tensor = torch.tensor([sex], dtype=torch.float32)
    
    # Inference
    with torch.no_grad():
        prediction = model(img.to(device), sex_tensor.to(device))
    
    return prediction.cpu().item()


def predict_ensemble_s1(images_dict, sex, models_dict, device):
    """
    Three-plane ensemble prediction (s=1.0)
    
    Args:
        images_dict: {'axi': array, 'sag': array, 'cor': array}
        sex: 0 (female) or 1 (male)
        models_dict: {'axi': model, 'sag': model, 'cor': model}
        device: torch device
    
    Returns:
        dict with individual and ensemble predictions
    """
    predictions = {}
    
    for direction in ['axi', 'sag', 'cor']:
        if direction in images_dict and direction in models_dict:
            pred = predict_single(images_dict[direction], sex, models_dict[direction], device)
            predictions[direction] = pred
    
    # Calculate ensemble (average)
    if len(predictions) > 0:
        predictions['ensemble'] = np.mean(list(predictions.values()))
    
    return predictions


def predict_bias_corrected(images_dict, sex, models_s1, models_s03, device):
    """
    Bias-Corrected ensemble prediction (s=1.0 + s=0.3)
    
    Uses average of six models: axi_s1, sag_s1, cor_s1, axi_s0.3, sag_s0.3, cor_s0.3
    
    Args:
        images_dict: {'axi': array, 'sag': array, 'cor': array}
        sex: 0 (female) or 1 (male)
        models_s1: {'axi': model, 'sag': model, 'cor': model} for s=1.0
        models_s03: {'axi': model, 'sag': model, 'cor': model} for s=0.3
        device: torch device
    
    Returns:
        dict with all predictions and bias-corrected ensemble
    """
    predictions = {}
    all_preds = []
    
    # s=1.0 model predictions
    for direction in ['axi', 'sag', 'cor']:
        if direction in images_dict and direction in models_s1:
            pred = predict_single(images_dict[direction], sex, models_s1[direction], device)
            predictions[f'{direction}_s1'] = pred
            all_preds.append(pred)
    
    # s=0.3 model predictions
    for direction in ['axi', 'sag', 'cor']:
        if direction in images_dict and direction in models_s03:
            pred = predict_single(images_dict[direction], sex, models_s03[direction], device)
            predictions[f'{direction}_s0.3'] = pred
            all_preds.append(pred)
    
    # Bias-Corrected ensemble (average of six models)
    if len(all_preds) > 0:
        predictions['bias_corrected'] = np.mean(all_preds)
        predictions['s1_ensemble'] = np.mean([predictions.get(f'{d}_s1', np.nan) 
                                              for d in ['axi', 'sag', 'cor'] 
                                              if f'{d}_s1' in predictions])
        predictions['s0.3_ensemble'] = np.mean([predictions.get(f'{d}_s0.3', np.nan) 
                                                for d in ['axi', 'sag', 'cor'] 
                                                if f'{d}_s0.3' in predictions])
    
    return predictions


def predict_batch_single_plane(csv_path, img_dir, model, device, output_csv=None):
    """Batch prediction (single plane)"""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    required_cols = ['NIfTI_ID', 'SEX']
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column")
    
    print(f"\n📊 Found {len(df)} subjects in CSV")
    
    predictions = []
    ids = []
    
    for idx, row in df.iterrows():
        nifti_id = row['NIfTI_ID']
        sex = int(row['SEX'])
        
        img_path = Path(img_dir) / f"{nifti_id}.npy"
        
        if not img_path.exists():
            print(f"⚠️  File not found: {nifti_id}.npy")
            continue
        
        try:
            image = np.load(img_path)
            pred_age = predict_single(image, sex, model, device)
            
            predictions.append(pred_age)
            ids.append(nifti_id)
            
            if (idx + 1) % 10 == 0 or (idx + 1) == len(df):
                print(f"   Progress: {idx + 1}/{len(df)} subjects processed")
        
        except Exception as e:
            print(f"❌ Error processing {nifti_id}: {e}")
            continue
    
    result_df = pd.DataFrame({
        'NIfTI_ID': ids,
        'predicted_age': predictions
    })
    
    result_df = result_df.merge(df[['NIfTI_ID', 'SEX']], on='NIfTI_ID', how='left')
    
    if 'AGE' in df.columns:
        result_df = result_df.merge(df[['NIfTI_ID', 'AGE']], on='NIfTI_ID', how='left')
        result_df['error'] = result_df['predicted_age'] - result_df['AGE']
        result_df['abs_error'] = result_df['error'].abs()
        
        print(f"\n📈 Performance Summary:")
        print(f"   MAE:  {result_df['abs_error'].mean():.2f} years")
        print(f"   RMSE: {np.sqrt((result_df['error']**2).mean()):.2f} years")
        print(f"   Bias: {result_df['error'].mean():+.2f} years")
    
    if output_csv:
        result_df.to_csv(output_csv, index=False)
        print(f"\n💾 Results saved to: {output_csv}")
    
    return result_df


def predict_batch_ensemble_s1(csv_path, img_dirs, models_dict, device, output_csv=None):
    """Batch prediction (three-plane ensemble s=1.0)"""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    required_cols = ['NIfTI_ID', 'SEX']
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column")
    
    print(f"\n📊 Found {len(df)} subjects in CSV")
    
    results = {
        'NIfTI_ID': [],
        'pred_axi': [],
        'pred_sag': [],
        'pred_cor': [],
        'pred_ensemble': []
    }
    
    for idx, row in df.iterrows():
        nifti_id = row['NIfTI_ID']
        sex = int(row['SEX'])
        
        images = {}
        all_exist = True
        
        for direction in ['axi', 'sag', 'cor']:
            img_path = Path(img_dirs[direction]) / f"{nifti_id}.npy"
            if not img_path.exists():
                print(f"⚠️  File not found: {direction}/{nifti_id}.npy")
                all_exist = False
                break
            images[direction] = np.load(img_path)
        
        if not all_exist:
            continue
        
        try:
            preds = predict_ensemble_s1(images, sex, models_dict, device)
            
            results['NIfTI_ID'].append(nifti_id)
            results['pred_axi'].append(preds.get('axi', np.nan))
            results['pred_sag'].append(preds.get('sag', np.nan))
            results['pred_cor'].append(preds.get('cor', np.nan))
            results['pred_ensemble'].append(preds.get('ensemble', np.nan))
            
            if (idx + 1) % 10 == 0 or (idx + 1) == len(df):
                print(f"   Progress: {idx + 1}/{len(df)} subjects processed")
        
        except Exception as e:
            print(f"❌ Error processing {nifti_id}: {e}")
            continue
    
    result_df = pd.DataFrame(results)
    result_df = result_df.merge(df[['NIfTI_ID', 'SEX']], on='NIfTI_ID', how='left')
    
    if 'AGE' in df.columns:
        result_df = result_df.merge(df[['NIfTI_ID', 'AGE']], on='NIfTI_ID', how='left')
        result_df['error'] = result_df['pred_ensemble'] - result_df['AGE']
        result_df['abs_error'] = result_df['error'].abs()
        
        print(f"\n📈 Ensemble Performance Summary:")
        print(f"   MAE:  {result_df['abs_error'].mean():.2f} years")
        print(f"   RMSE: {np.sqrt((result_df['error']**2).mean()):.2f} years")
        print(f"   Bias: {result_df['error'].mean():+.2f} years")
        
        for direction in ['axi', 'sag', 'cor']:
            err = result_df[f'pred_{direction}'] - result_df['AGE']
            mae = err.abs().mean()
            print(f"   {direction.upper()} MAE: {mae:.2f} years")
    
    if output_csv:
        result_df.to_csv(output_csv, index=False)
        print(f"\n💾 Results saved to: {output_csv}")
    
    return result_df


def predict_batch_bias_corrected(csv_path, img_dirs, models_s1, models_s03, device, output_csv=None):
    """Batch prediction (Bias-Corrected ensemble)"""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    required_cols = ['NIfTI_ID', 'SEX']
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column")
    
    print(f"\n📊 Found {len(df)} subjects in CSV")
    
    results = {
        'NIfTI_ID': [],
        'axi_s1': [],
        'sag_s1': [],
        'cor_s1': [],
        'axi_s0.3': [],
        'sag_s0.3': [],
        'cor_s0.3': [],
        's1_ensemble': [],
        's0.3_ensemble': [],
        'bias_corrected': []
    }
    
    for idx, row in df.iterrows():
        nifti_id = row['NIfTI_ID']
        sex = int(row['SEX'])
        
        images = {}
        all_exist = True
        
        for direction in ['axi', 'sag', 'cor']:
            img_path = Path(img_dirs[direction]) / f"{nifti_id}.npy"
            if not img_path.exists():
                print(f"⚠️  File not found: {direction}/{nifti_id}.npy")
                all_exist = False
                break
            images[direction] = np.load(img_path)
        
        if not all_exist:
            continue
        
        try:
            preds = predict_bias_corrected(images, sex, models_s1, models_s03, device)
            
            results['NIfTI_ID'].append(nifti_id)
            results['axi_s1'].append(preds.get('axi_s1', np.nan))
            results['sag_s1'].append(preds.get('sag_s1', np.nan))
            results['cor_s1'].append(preds.get('cor_s1', np.nan))
            results['axi_s0.3'].append(preds.get('axi_s0.3', np.nan))
            results['sag_s0.3'].append(preds.get('sag_s0.3', np.nan))
            results['cor_s0.3'].append(preds.get('cor_s0.3', np.nan))
            results['s1_ensemble'].append(preds.get('s1_ensemble', np.nan))
            results['s0.3_ensemble'].append(preds.get('s0.3_ensemble', np.nan))
            results['bias_corrected'].append(preds.get('bias_corrected', np.nan))
            
            if (idx + 1) % 10 == 0 or (idx + 1) == len(df):
                print(f"   Progress: {idx + 1}/{len(df)} subjects processed")
        
        except Exception as e:
            print(f"❌ Error processing {nifti_id}: {e}")
            continue
    
    result_df = pd.DataFrame(results)
    result_df = result_df.merge(df[['NIfTI_ID', 'SEX']], on='NIfTI_ID', how='left')
    
    if 'AGE' in df.columns:
        result_df = result_df.merge(df[['NIfTI_ID', 'AGE']], on='NIfTI_ID', how='left')
        result_df['error'] = result_df['bias_corrected'] - result_df['AGE']
        result_df['abs_error'] = result_df['error'].abs()
        
        print(f"\n📈 Bias-Corrected Performance Summary:")
        print(f"   Bias-Corrected MAE:  {result_df['abs_error'].mean():.2f} years ⭐")
        print(f"   Bias-Corrected RMSE: {np.sqrt((result_df['error']**2).mean()):.2f} years")
        print(f"   Bias-Corrected Bias: {result_df['error'].mean():+.2f} years")
        
        # Compare strategies
        err_s1 = result_df['s1_ensemble'] - result_df['AGE']
        err_s03 = result_df['s0.3_ensemble'] - result_df['AGE']
        print(f"\n   Comparison:")
        print(f"   s=1.0 ensemble MAE:  {err_s1.abs().mean():.2f} years")
        print(f"   s=0.3 ensemble MAE:  {err_s03.abs().mean():.2f} years")
    
    if output_csv:
        result_df.to_csv(output_csv, index=False)
        print(f"\n💾 Results saved to: {output_csv}")
    
    return result_df


# ============================================================
# Interactive Input Functions
# ============================================================

def get_user_input(prompt, default=None):
    """Get user input"""
    if default:
        user_input = input(f"{prompt} [default: {default}]: ").strip()
        return user_input if user_input else default
    else:
        while True:
            user_input = input(f"{prompt}: ").strip()
            if user_input:
                return user_input
            print("❌ This field cannot be empty. Please enter again.")


def get_yes_no(prompt):
    """Get yes/no answer"""
    while True:
        answer = input(f"{prompt} (y/n): ").strip().lower()
        if answer in ['y', 'yes', 'n', 'no']:
            return answer in ['y', 'yes']
        print("❌ Please enter y or n")


# ============================================================
# Main Program
# ============================================================

def main():
    print("=" * 70)
    print("🧠 Brain Age Prediction - Interactive Inference")
    print("=" * 70)
    
    # ============================================================
    # Step 1: Select inference mode
    # ============================================================
    print("\n【Step 1/4】Select Inference Mode")
    print("-" * 70)
    print("Please select your inference mode:")
    print("  [1] Single subject inference")
    print("  [2] Batch inference (from CSV file)")
    
    while True:
        mode_choice = input("\nEnter option (1 or 2): ").strip()
        if mode_choice in ['1', '2']:
            break
        print("❌ Please enter 1 or 2")
    
    # ============================================================
    # Step 2: Select prediction method
    # ============================================================
    print(f"\n【Step 2/4】Select Prediction Method")
    print("-" * 70)
    print("Please select prediction method:")
    print("  [1] Single-plane prediction (one direction: axi/sag/cor)")
    print("  [2] Baseline three-plane ensemble (s=1.0, highest accuracy) ⭐")
    print("  [3] Bias-Corrected ensemble (s=1.0 + s=0.3)")
    
    while True:
        pred_choice = input("\nEnter option (1/2/3): ").strip()
        if pred_choice in ['1', '2', '3']:
            break
        print("❌ Please enter 1, 2 or 3")
    
    is_single = (pred_choice == '1')
    is_ensemble_s1 = (pred_choice == '2')
    is_bias_corrected = (pred_choice == '3')
    
    # ============================================================
    # Step 3: Load models
    # ============================================================
    print(f"\n【Step 3/4】Load Pretrained Models")
    print("-" * 70)
    
    if is_bias_corrected:
        # Bias-Corrected: need 6 models
        print("Bias-Corrected ensemble requires 6 models:")
        print("  • s=1.0: axi, sag, cor")
        print("  • s=0.3: axi, sag, cor")
        
        models_s1 = {}
        models_s03 = {}
        device = None
        
        # Load s=1.0 models
        print("\n📦 Loading s=1.0 models:")
        for direction in ['axi', 'sag', 'cor']:
            model_path = get_user_input(f"  [{direction.upper()} s=1.0] Model path")
            
            if not os.path.exists(model_path):
                print(f"❌ File not found: {model_path}")
                return
            
            try:
                model, dev, in_ch = load_model(model_path, device='cuda' if device is None else device)
                if device is None:
                    device = dev
                models_s1[direction] = model
                print(f"  ✅ {direction.upper()} s=1.0 loaded (channels: {in_ch})")
            except Exception as e:
                print(f"❌ Error loading {direction} s=1.0 model: {e}")
                return
        
        # Load s=0.3 models
        print("\n📦 Loading s=0.3 models:")
        for direction in ['axi', 'sag', 'cor']:
            model_path = get_user_input(f"  [{direction.upper()} s=0.3] Model path")
            
            if not os.path.exists(model_path):
                print(f"❌ File not found: {model_path}")
                return
            
            try:
                model, dev, in_ch = load_model(model_path, device)
                models_s03[direction] = model
                print(f"  ✅ {direction.upper()} s=0.3 loaded (channels: {in_ch})")
            except Exception as e:
                print(f"❌ Error loading {direction} s=0.3 model: {e}")
                return
        
        print(f"\n✅ All 6 models loaded successfully on {device}")
    
    elif is_ensemble_s1:
        # Three-plane ensemble s=1.0: need 3 models
        print("Three-plane ensemble (s=1.0) requires 3 models (axi, sag, cor)")
        models_dict = {}
        device = None
        
        for direction in ['axi', 'sag', 'cor']:
            print(f"\n📦 [{direction.upper()}] Model:")
            model_path = get_user_input(f"  Enter {direction} model path (.pth file)")
            
            if not os.path.exists(model_path):
                print(f"❌ File not found: {model_path}")
                return
            
            try:
                model, dev, in_ch = load_model(model_path, device='cuda' if device is None else device)
                if device is None:
                    device = dev
                models_dict[direction] = model
                print(f"  ✅ {direction.upper()} model loaded (channels: {in_ch})")
            except Exception as e:
                print(f"❌ Error loading {direction} model: {e}")
                return
        
        print(f"\n✅ All models loaded successfully on {device}")
    
    else:
        # Single plane: need 1 model
        print("Single-plane prediction requires 1 model")
        
        print("\nPlease select direction:")
        print("  [1] Axial")
        print("  [2] Sagittal")
        print("  [3] Coronal")
        
        while True:
            dir_choice = input("\nEnter option (1/2/3): ").strip()
            if dir_choice in ['1', '2', '3']:
                break
            print("❌ Please enter 1, 2 or 3")
        
        direction_map = {'1': 'axi', '2': 'sag', '3': 'cor'}
        selected_direction = direction_map[dir_choice]
        
        model_path = get_user_input(f"\nEnter {selected_direction} model path (.pth file)")
        
        if not os.path.exists(model_path):
            print(f"❌ File not found: {model_path}")
            return
        
        try:
            model, device, in_ch = load_model(model_path)
            print(f"\n✅ Model loaded successfully!")
            print(f"   Direction: {selected_direction.upper()}")
            print(f"   Device: {device}")
            print(f"   Input channels: {in_ch}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return
    
    # ============================================================
    # Step 4: Run inference
    # ============================================================
    print(f"\n【Step 4/4】Run Inference")
    print("-" * 70)
    
    if mode_choice == '1':
        # ========== Single subject ==========
        print("\n📋 Please provide the following information:")
        
        if is_bias_corrected:
            # Bias-Corrected: need 3 images (but use 6 models)
            print("\nProvide 3 directional image files (will use 6 models: s=1.0 and s=0.3)")
            images_dict = {}
            
            for direction in ['axi', 'sag', 'cor']:
                img_path = get_user_input(f"{direction.upper()} image file path (.npy)")
                
                if not os.path.exists(img_path):
                    print(f"❌ File not found: {img_path}")
                    return
                
                images_dict[direction] = np.load(img_path)
                print(f"  ✅ {direction.upper()} image loaded, shape: {images_dict[direction].shape}")
        
        elif is_ensemble_s1:
            # Three-plane ensemble s=1.0: need 3 images
            print("\nThree-plane ensemble requires 3 image files")
            images_dict = {}
            
            for direction in ['axi', 'sag', 'cor']:
                img_path = get_user_input(f"{direction.upper()} image file path (.npy)")
                
                if not os.path.exists(img_path):
                    print(f"❌ File not found: {img_path}")
                    return
                
                images_dict[direction] = np.load(img_path)
                print(f"  ✅ {direction.upper()} image loaded, shape: {images_dict[direction].shape}")
        
        else:
            # Single plane: need 1 image
            img_path = get_user_input(f"\n{selected_direction.upper()} image file path (.npy)")
            
            if not os.path.exists(img_path):
                print(f"❌ File not found: {img_path}")
                return
            
            image = np.load(img_path)
            print(f"   Image shape: {image.shape}")
        
        # Input sex
        while True:
            sex_input = input("\nSex (0=female, 1=male): ").strip()
            if sex_input in ['0', '1']:
                sex = int(sex_input)
                break
            print("❌ Please enter 0 or 1")
        
        # Run inference
        try:
            print(f"\n🔮 Predicting brain age...")
            
            if is_bias_corrected:
                # Bias-Corrected ensemble prediction
                preds = predict_bias_corrected(images_dict, sex, models_s1, models_s03, device)
                
                print("\n" + "=" * 70)
                print("📊 Bias-Corrected Ensemble Prediction Result")
                print("=" * 70)
                print("s=1.0 predictions:")
                print(f"   Axial:    {preds.get('axi_s1', 'N/A'):.2f} years")
                print(f"   Sagittal: {preds.get('sag_s1', 'N/A'):.2f} years")
                print(f"   Coronal:  {preds.get('cor_s1', 'N/A'):.2f} years")
                print(f"   → Average: {preds.get('s1_ensemble', 'N/A'):.2f} years")
                print()
                print("s=0.3 predictions:")
                print(f"   Axial:    {preds.get('axi_s0.3', 'N/A'):.2f} years")
                print(f"   Sagittal: {preds.get('sag_s0.3', 'N/A'):.2f} years")
                print(f"   Coronal:  {preds.get('cor_s0.3', 'N/A'):.2f} years")
                print(f"   → Average: {preds.get('s0.3_ensemble', 'N/A'):.2f} years")
                print("-" * 70)
                print(f"   🎯 Bias-Corrected (6 models avg): {preds.get('bias_corrected', 'N/A'):.2f} years ⭐")
                print(f"   Sex: {'Male' if sex == 1 else 'Female'}")
                print("=" * 70)
            
            elif is_ensemble_s1:
                # Three-plane ensemble prediction
                preds = predict_ensemble_s1(images_dict, sex, models_dict, device)
                
                print("\n" + "=" * 70)
                print("📊 Three-Plane Ensemble Prediction Result (s=1.0)")
                print("=" * 70)
                print(f"   Axial prediction:    {preds.get('axi', 'N/A'):.2f} years")
                print(f"   Sagittal prediction: {preds.get('sag', 'N/A'):.2f} years")
                print(f"   Coronal prediction:  {preds.get('cor', 'N/A'):.2f} years")
                print("-" * 70)
                print(f"   🎯 Ensemble (Average): {preds.get('ensemble', 'N/A'):.2f} years")
                print(f"   Sex: {'Male' if sex == 1 else 'Female'}")
                print("=" * 70)
            
            else:
                # Single-plane prediction
                predicted_age = predict_single(image, sex, model, device)
                
                print("\n" + "=" * 70)
                print("📊 Single-Plane Prediction Result")
                print("=" * 70)
                print(f"   Predicted Age: {predicted_age:.2f} years")
                print(f"   Direction: {selected_direction.upper()}")
                print(f"   Sex: {'Male' if sex == 1 else 'Female'}")
                print(f"   Image: {os.path.basename(img_path)}")
                print("=" * 70)
        
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        # ========== Batch inference ==========
        print("\n📋 CSV File Format Requirements:")
        print("-" * 70)
        print("Your CSV file must contain the following columns:")
        print("  • NIfTI_ID  (required) - Subject ID, matching .npy filenames")
        print("  • SEX       (required) - Sex, 0=female, 1=male")
        print("  • AGE       (optional) - True age, for error calculation")
        print()
        print("Example format:")
        print("  NIfTI_ID              SEX  AGE")
        print("  AnnArbor_sub28433     1    45.5")
        print("  Beijing_sub12345      0    38.2")
        print("-" * 70)
        
        print("\n📋 Please provide the following information:")
        
        # Input CSV path
        csv_path = get_user_input("CSV file path")
        
        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            return
        
        if is_bias_corrected or is_ensemble_s1:
            # Need 3 image directories
            print(f"\n{'Bias-Corrected ensemble' if is_bias_corrected else 'Three-plane ensemble'} requires 3 image directories")
            img_dirs = {}
            
            for direction in ['axi', 'sag', 'cor']:
                img_dir = get_user_input(f"{direction.upper()} image directory path")
                
                if not os.path.isdir(img_dir):
                    print(f"❌ Directory not found: {img_dir}")
                    return
                
                img_dirs[direction] = img_dir
                print(f"  ✅ {direction.upper()} directory verified")
        
        else:
            # Single plane: need 1 image directory
            img_dir = get_user_input(f"\n{selected_direction.upper()} image directory path")
            
            if not os.path.isdir(img_dir):
                print(f"❌ Directory not found: {img_dir}")
                return
        
        # Ask whether to save results
        save_result = get_yes_no("\nSave prediction results to CSV file?")
        
        output_csv = None
        if save_result:
            if is_bias_corrected:
                default_name = "predictions_bias_corrected.csv"
            elif is_ensemble_s1:
                default_name = "predictions_ensemble_s1.csv"
            else:
                default_name = f"predictions_{selected_direction}.csv"
            
            default_output = os.path.join(os.path.dirname(csv_path), default_name)
            output_csv = get_user_input("Output CSV path", default=default_output)
        
        # Run batch inference
        try:
            print(f"\n🔮 Starting batch prediction...")
            
            if is_bias_corrected:
                results = predict_batch_bias_corrected(csv_path, img_dirs, models_s1, models_s03, device, output_csv)
                
                # Display results summary
                print("\n" + "=" * 70)
                print("📊 Bias-Corrected Batch Prediction Summary")
                print("=" * 70)
                print(f"   Total subjects: {len(results)}")
                print(f"   Mean bias-corrected age: {results['bias_corrected'].mean():.2f} years")
                print(f"   Std bias-corrected age: {results['bias_corrected'].std():.2f} years")
                
                # Display first 5 predictions
                print("\n📋 First 5 predictions:")
                display_cols = ['NIfTI_ID', 'axi_s1', 'sag_s1', 'cor_s1', 
                               'axi_s0.3', 'sag_s0.3', 'cor_s0.3', 
                               's1_ensemble', 's0.3_ensemble', 'bias_corrected']
                if 'AGE' in results.columns:
                    display_cols.extend(['AGE', 'error'])
                print(results[display_cols].head().to_string(index=False))
                print("=" * 70)
            
            elif is_ensemble_s1:
                results = predict_batch_ensemble_s1(csv_path, img_dirs, models_dict, device, output_csv)
                
                # Display results summary
                print("\n" + "=" * 70)
                print("📊 Ensemble (s=1.0) Batch Prediction Summary")
                print("=" * 70)
                print(f"   Total subjects: {len(results)}")
                print(f"   Mean ensemble age: {results['pred_ensemble'].mean():.2f} years")
                print(f"   Std ensemble age: {results['pred_ensemble'].std():.2f} years")
                
                # Display first 5 predictions
                print("\n📋 First 5 predictions:")
                display_cols = ['NIfTI_ID', 'pred_axi', 'pred_sag', 'pred_cor', 'pred_ensemble']
                if 'AGE' in results.columns:
                    display_cols.extend(['AGE', 'error'])
                print(results[display_cols].head().to_string(index=False))
                print("=" * 70)
            
            else:
                results = predict_batch_single_plane(csv_path, img_dir, model, device, output_csv)
                
                # Display results summary
                print("\n" + "=" * 70)
                print(f"📊 {selected_direction.upper()} Batch Prediction Summary")
                print("=" * 70)
                print(f"   Total subjects: {len(results)}")
                print(f"   Mean predicted age: {results['predicted_age'].mean():.2f} years")
                print(f"   Std predicted age: {results['predicted_age'].std():.2f} years")
                
                # Display first 5 predictions
                print("\n📋 First 5 predictions:")
                print(results.head().to_string(index=False))
                print("=" * 70)
        
        except Exception as e:
            print(f"❌ Error during batch prediction: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ Inference completed!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()