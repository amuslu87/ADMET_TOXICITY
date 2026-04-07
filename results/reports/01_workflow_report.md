# Workflow Report: Drug Toxicity Prediction Using Molecular Fingerprints and Machine Learning

**Project:** ADMET Toxicity Prediction Portfolio  
**Author:** Matt Muslu  
**Date:** April 2026  
**Dataset:** Tox21 (NIH / MoleculeNet)  
**GitHub:** https://github.com/amuslu87

---

## 1. Project Overview

This project builds a machine learning pipeline to predict drug toxicity from molecular structure. Given a drug compound represented as a SMILES string, the pipeline computes molecular fingerprints and predicts whether the compound is toxic across 12 different biological assay endpoints from the Tox21 dataset.

The workflow is designed to reflect the kind of computational ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) modeling used in early-stage pharmaceutical drug discovery to flag potentially harmful compounds before wet-lab testing.

---

## 2. Dataset

**Source:** Tox21 Challenge dataset, downloaded from the MoleculeNet/DeepChem repository  
**URL:** https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz  
**Size:** 7,831 compounds × 12 toxicity assay endpoints

### Toxicity Assays

The 12 assays cover two biological pathway categories:

**Nuclear Receptor (NR) assays:**
| Assay | Target | Biological Relevance |
|---|---|---|
| NR-AR | Androgen Receptor | Endocrine disruption |
| NR-AR-LBD | Androgen Receptor Ligand Binding Domain | Endocrine disruption |
| NR-AhR | Aryl Hydrocarbon Receptor | Drug metabolism, dioxin response |
| NR-Aromatase | Aromatase enzyme | Estrogen biosynthesis |
| NR-ER | Estrogen Receptor alpha | Endocrine disruption |
| NR-ER-LBD | Estrogen Receptor LBD | Endocrine disruption |
| NR-PPAR-gamma | Peroxisome Proliferator-Activated Receptor gamma | Metabolic regulation |

**Stress Response (SR) assays:**
| Assay | Target | Biological Relevance |
|---|---|---|
| SR-ARE | Antioxidant Response Element | Oxidative stress |
| SR-ATAD5 | ATAD5 genotoxicity reporter | DNA damage / genotoxicity |
| SR-HSE | Heat Shock Element | Cellular stress response |
| SR-MMP | Mitochondrial Membrane Potential | Mitochondrial toxicity |
| SR-p53 | p53 pathway | DNA damage / apoptosis |

### Class Imbalance
All 12 assays are heavily imbalanced — toxic compounds range from 2.9% (NR-PPAR-gamma) to 16.2% (SR-ARE) of tested compounds. This reflects real-world pharmaceutical data, where most tested compounds are non-toxic.

### Missing Values
Labels are missing in 7.2% to 25.8% of compounds per assay (due to experimental data availability), handled by per-assay subsetting prior to model training.

---

## 3. Environment and Dependencies

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.12.3 | Primary language |
| RDKit | 2024.x | Molecular featurization |
| scikit-learn | 1.5.1 | Random Forest model |
| XGBoost | latest | Gradient boosting model |
| pandas | 2.2.2 | Data wrangling |
| numpy | 1.26.4 | Numerical computation |
| matplotlib / seaborn | 3.9.2 | Visualization |
| Streamlit | 1.37.1 | Interactive web app |

---

## 4. Pipeline Steps

### Step 1: Data Download and Exploration (`01_download_and_explore.py`)

- Downloaded the Tox21 CSV dataset programmatically using Python `requests`
- Loaded dataset and identified 12 target columns and the SMILES column
- Computed missing value percentages per assay
- Computed class balance (% positive/toxic) per assay
- Generated two exploratory figures:
  - **Figure 1:** Missing value heatmap across all compounds and assays
  - **Figure 2:** Class imbalance bar chart per assay
- Saved cleaned dataset (rows with valid SMILES) to `data/processed/tox21_clean.csv`

### Step 2: Molecular Featurization (`02_featurize.py`)

Molecular featurization converts a SMILES string into a numerical vector that a machine learning model can learn from.

**Morgan Fingerprints (ECFP4):**
- Algorithm: AllChem.GetMorganFingerprintAsBitVect from RDKit
- Radius: 2 bonds (equivalent to ECFP4 — the industry standard for QSAR modeling)
- Bits: 2,048-dimensional binary vector
- Each bit represents the presence or absence of a specific circular substructure in the molecule
- Computed for all 7,831 valid compounds
- Output: fingerprint matrix of shape (7,831 × 2,048) saved as `fingerprints.npy`

**Physicochemical Descriptors:**
Five Lipinski Rule of Five descriptors were also computed for each compound:
- **MW** — Molecular Weight (Da)
- **LogP** — Octanol-water partition coefficient (lipophilicity)
- **HBD** — Number of hydrogen bond donors
- **HBA** — Number of hydrogen bond acceptors
- **TPSA** — Topological Polar Surface Area (Å²)

These are used for interpretation and visualization, not as model input features.

### Step 3: Model Training (`03_train_models.py`)

One binary classification model was trained per toxicity assay (12 models total).

**Design choices:**
- Each assay treated as an independent binary classification problem (toxic = 1, non-toxic = 0)
- Rows with missing labels for a given assay were excluded from that assay's training
- Stratified 80/20 train/test split to preserve class balance across splits
- Class imbalance handled explicitly:
  - Random Forest: `class_weight='balanced'`
  - XGBoost: `scale_pos_weight = n_negative / n_positive`

**Models trained:**
- **Random Forest:** 200 trees, balanced class weights, all CPU cores
- **XGBoost:** 200 estimators, learning rate 0.05, max depth 6, scale_pos_weight

**Model selection:** For each assay, the model with the higher test ROC-AUC was saved as the final model.

**Evaluation metrics:**
- ROC-AUC (primary): area under the receiver operating characteristic curve
- Average Precision (AP): area under the precision-recall curve, more informative for imbalanced data

**Output:** 12 trained model files saved as `.pkl` in `models/`, plus a performance summary CSV.

### Step 4: Model Interpretation (`04_interpret.py`)

- **Feature importance:** Extracted Mean Decrease Impurity from the best-performing Random Forest model (NR-AhR, AUC = 0.906). Top 30 most important fingerprint bits visualized.
- **Physicochemical distributions:** Compared MW, LogP, HBD, HBA, and TPSA between toxic and non-toxic compounds for the most balanced assay.
- **Performance heatmap:** RF vs XGBoost AUC across all 12 assays in a single heatmap.
- **Property-toxicity correlations:** Pearson correlation between each physicochemical property and each toxicity label.

### Step 5: Interactive Streamlit App (`app/app.py`)

A web application was built using Streamlit with three tabs:

- **Predict Toxicity:** User inputs any SMILES string → molecule is rendered, physicochemical properties computed, Lipinski Rule of Five checked, and predictions made across all 12 assays in real time
- **Model Performance:** Interactive ROC-AUC comparison chart and full results table
- **About:** Project background, methods, and references

**To launch:**
```bash
streamlit run app/app.py
```

---

## 5. Output Files

| File | Location | Description |
|---|---|---|
| `tox21.csv` | data/raw/ | Raw dataset |
| `tox21_clean.csv` | data/processed/ | Cleaned dataset |
| `fingerprints.npy` | data/processed/ | Morgan fingerprint matrix (7831 × 2048) |
| `tox21_features.csv` | data/processed/ | Metadata + targets + descriptors |
| `*_model.pkl` | models/ | 12 trained models (one per assay) |
| `model_performance.csv` | results/tables/ | AUC and AP per assay |
| `physicochemical_summary.csv` | results/tables/ | Descriptor statistics |
| Figures 01–08 | results/figures/ | All analysis visualizations |
| `app.py` | app/ | Streamlit dashboard |

---

## 6. Reproducibility

All scripts are self-contained and run sequentially:

```bash
python scripts/01_download_and_explore.py
python scripts/02_featurize.py
python scripts/03_train_models.py
python scripts/04_interpret.py
streamlit run app/app.py
```

Random seeds are set to 42 throughout. Dataset is downloaded automatically from a public URL.
