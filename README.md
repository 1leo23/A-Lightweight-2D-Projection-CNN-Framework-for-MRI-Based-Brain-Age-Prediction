# A-Lightweight-2D-Projection-CNN-Framework-for-MRI-Based-Brain-Age-Prediction

# A Lightweight 2D Projection CNN Framework for MRI-Based Brain Age Prediction

##  Overview
Brain age is an important biomarker that quantifies age-related structural changes in the human brain, with potential for early disease diagnosis and monitoring of healthy aging. We propose a computationally efficient deep learning model based on two-dimensional (2D) projections that balances efficiency and accuracy.

### Key features:
- Multi-channel **2D projection strategy** (T1 MR images + gray matter probability maps)  
- **Lightweight CNN** architecture (approximately 414k parameters, 86% fewer than SFCN)  
- **Age-distribution weighted training** to reduce systematic bias  
- **Grad-CAM visualizations** to confirm biologically plausible attention to brain regions  

---

##  Project Structure

2D_Proj_CNN/
│── 2D_Projection/         # 投影生成程式
│   ├── projection_mean.py
│   ├── projection_std.py
│   ├── projection_max.py
│   ├── projection_median.py
│   ├── merge_projections.py
│   └── pipeline.py
│
│── Preprocessing/         # MRI 前處理流程
│   ├── n4_correction.py
│   ├── brain_extraction.py
│   ├── mni_registration.py
│   ├── normalization.py
│   └── preprocessing_pipeline.py
│
│── Modeling/              # 模型訓練與測試
│   └── Modeling(coronal).ipynb
│
│── images/                # 圖片資料夾
│   ├── 投影片1.JPG
│   ├── 投影片2.JPG
│   ├── 投影片3.JPG
│   └── 投影片4.JPG
│
│── environment.yml        # Conda 環境設定
│── README.md              # 專案文件


---

##  Installation and Environment Setup
1.  Clone this repository:
   ```bash
    git clone https://github.com/<your-username>/A-Lightweight-2D-Projection-CNN.git
cd A-Lightweight-2D-Projection-CNN
2.  Create the conda environment:
    conda env create -f environment.yml
    conda activate ants_brain
Datasets

We aggregated eight publicly available structural MRI datasets, totaling 7,649 healthy participants (aged 5–89 years):

ABIDE, ADNI, BGSP, HBN, CORR, fcon_1000, ICBM, OASIS-3

Preprocessing pipeline:
1.  N4 bias field correction
2.  Skull stripping (ANTsXNet)
3.  Nonlinear registration to MNI152 template
4.  Intensity normalization
5.  Gray matter probability map generation (FSL FAST)
