# A-Lightweight-2D-Projection-CNN-Framework-for-MRI-Based-Brain-Age-Prediction
Brain age is an important biomarker that quantifies age-related structural changes in the human brain, with potential for early disease diagnosis and monitoring of healthy aging. We propose a computationally efficient deep learning model based on two-dimensional (2D) projections that balances efficiency and accuracy.
# A Lightweight 2D Projection CNN Framework for MRI-Based Brain Age Prediction

## 📖 Overview
Brain age is a neuroimaging biomarker that reflects age-related structural changes in the human brain.  
This project proposes a **lightweight 2D projection CNN framework** for MRI-based brain age prediction, balancing high efficiency and accuracy.

### Key features:
- Multi-channel **2D projection strategy** (T1 MR images + gray matter probability maps)  
- **Lightweight CNN** architecture (approximately 414k parameters, 86% fewer than SFCN)  
- **Age-distribution weighted training** to reduce systematic bias  
- **Grad-CAM visualizations** to confirm biologically plausible attention to brain regions  

---

## 🏗️ Project Structure
2D_Proj_CNN/
│── 2D_Projection/ # Scripts for projection generation
│ ├── projection_mean.py
│ ├── projection_std.py
│ ├── projection_max.py
│ ├── projection_median.py
│ ├── merge_projections.py
│ └── pipeline.py
│
│── Preprocessing/ # MRI preprocessing pipeline
│ ├── n4_correction.py
│ ├── brain_extraction.py
│ ├── mni_registration.py
│ ├── normalization.py
│ └── preprocessing_pipeline.py
│
│── Modeling/ # Model training and evaluation
│ └── Modeling(coronal).ipynb
│
│── images/ # Image folder
│ ├── 投影片1.JPG
│ ├── 投影片2.JPG
│ ├── 投影片3.JPG
│ └── 投影片4.JPG
│
│── environment.yml # Conda environment setup
│── README.md # Project documentation

yaml
複製程式碼

---

## ⚙️ Installation and Environment Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/A-Lightweight-2D-Projection-CNN.git
   cd A-Lightweight-2D-Projection-CNN
Create the conda environment:

bash
複製程式碼
conda env create -f environment.yml
conda activate ants_brain
📊 Datasets
We aggregated eight publicly available structural MRI datasets, totaling 7,649 healthy participants (aged 5–89 years):

ABIDE, ADNI, BGSP, HBN, CORR, fcon_1000, ICBM, OASIS-3

Preprocessing pipeline:

N4 bias field correction

Skull stripping (ANTsXNet)

Nonlinear registration to MNI152 template

Intensity normalization (Z-score)

Gray matter probability map generation (FSL FAST)

🔬 Methodology
Projection Strategy
For each MRI scan, compute 6 types of 2D projections:

T1 MR images: Mean, Std, Max

Gray matter maps: Mean, Std, Median

Projections across three planes: Axial, Coronal, Sagittal

Final input tensor: 6×218×218

Lightweight CNN
3 convolution blocks: [64, 128, 256]

Global average pooling + Fully connected layers (257→128→64→1)

Incorporates sex covariate

Total parameters: 414,785

Bias Correction
Age-distribution weighted training

Ensemble strategy (baseline + weighted models)

📈 Results
Single-plane models: MAE ≈ 2.7–2.8 years

Three-plane ensemble: MAE = 2.50 years

Bias-corrected ensemble: MAE = 2.54 years (significantly reduces bias)

Training time: approximately 1.5 hours per model (two orders of magnitude faster than 3D CNNs)

Comparison with previous works:

Jönemo et al. (2023, 2D CNN): MAE ≈ 3.5 years

Ours: MAE = 2.50 years

🔎 Visualization
Grad-CAM highlights the brain regions attended by the model:

Children/Adolescents → Cerebellum

Adults → Cortex

Elderly → Ventricular enlargement & hippocampal atrophy





📌 Limitations
Fewer middle-aged (30–49) and elderly (≥80) samples

Datasets are mostly from Western cohorts → need ethnic diversity

Trained only on healthy controls, not tested on pathological cases

2D projections may miss some subtle 3D details

✨ Citation
If you use this work, please cite:

mathematica
複製程式碼
T.-A. Chang and R.-C. Syu,
"A Lightweight 2D Projection CNN Framework for MRI-Based Brain Age Prediction,"
IEEE Access, 2024. DOI: 10.1109/ACCESS.2024.Doi Number
yaml
複製程式碼

---

### How to Use
1. Place the provided images (`投影片1.JPG`, `投影片2.JPG`, `投影片3.JPG`, `投影片4.JPG`) in the `images/` folder in your project directory, so that they can be displayed correctly in the README.

Would you like me to assist you with any further modifications or provide more specific details for this **README.md** file?




