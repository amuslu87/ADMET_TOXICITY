# Drug Toxicity Prediction Using Molecular Fingerprints and Machine Learning

A cheminformatics ADMET modeling pipeline that predicts drug compound toxicity across 12 biological assay endpoints using Morgan molecular fingerprints and Random Forest / XGBoost classifiers. Includes an interactive Streamlit app for real-time toxicity prediction of any drug compound.

---

## Background

In pharmaceutical drug discovery, toxicity is one of the leading causes of clinical trial failure and post-market drug withdrawal. Early computational prediction of ADMET (Absorption, Distribution, Metabolism, Excretion, **Toxicity**) properties enables drug discovery teams to flag and deprioritize unsafe compounds before expensive wet-lab experiments.

This project builds a machine learning toxicity classifier using the **Tox21 dataset** — 7,831 compounds tested across 12 nuclear receptor and stress response pathway assays — and demonstrates the full pipeline from raw SMILES strings to interactive web-based predictions.

---

## Key Results

| Metric | Value |
|---|---|
| Compounds analyzed | 7,831 |
| Toxicity assays modeled | 12 |
| Best model | Random Forest (all assays) |
| Mean ROC-AUC | **0.830** |
| Best assay AUC | **0.906** (NR-AhR) |
| Worst assay AUC | **0.715** (NR-ER) |

Random Forest outperformed XGBoost across all 12 assays. Highest accuracy achieved for AhR, ATAD5, and mitochondrial membrane potential assays — three of the most clinically relevant toxicity endpoints in drug development.

---

## Interactive App

The Streamlit app allows real-time toxicity prediction for any drug compound:

- Input any SMILES string → get toxicity predictions across all 12 assays
- Molecule structure rendered automatically
- Lipinski Rule of Five check
- Physicochemical properties computed on the fly

**To launch:**
```bash
streamlit run app/app.py
```

---

## Project Structure

```
ADMET_PROJECT/
├── data/
│   ├── raw/              # Tox21 CSV (excluded from repo)
│   └── processed/        # Fingerprint matrix, features CSV
├── scripts/
│   ├── 01_download_and_explore.py   # Download Tox21, QC, class balance
│   ├── 02_featurize.py              # Morgan fingerprints + descriptors
│   ├── 03_train_models.py           # RF + XGBoost, per-assay training
│   └── 04_interpret.py              # Feature importance, property analysis
├── models/               # 12 trained model .pkl files
├── results/
│   ├── figures/          # 8 analysis figures
│   ├── tables/           # Performance CSV, descriptor stats
│   └── reports/          # Workflow report + Findings report
├── app/
│   └── app.py            # Streamlit dashboard
└── README.md
```

---

## How to Reproduce

### 1. Clone the repository
```bash
git clone https://github.com/amuslu87/admet-toxicity-prediction.git
cd admet-toxicity-prediction
```

### 2. Create conda environment
```bash
conda create -n admet python=3.12
conda activate admet
conda install -c conda-forge rdkit -y
pip install scikit-learn xgboost pandas numpy matplotlib seaborn streamlit requests
```

### 3. Run pipeline in order
```bash
python scripts/01_download_and_explore.py   # Downloads data automatically
python scripts/02_featurize.py
python scripts/03_train_models.py
python scripts/04_interpret.py
```

### 4. Launch the app
```bash
streamlit run app/app.py
```

---

## Methods Summary

| Step | Method |
|---|---|
| Molecular representation | SMILES strings (Tox21 dataset) |
| Featurization | Morgan fingerprints ECFP4 (radius=2, 2048 bits) via RDKit |
| Descriptors | MW, LogP, HBD, HBA, TPSA (Lipinski properties) |
| Model | Random Forest (200 trees) + XGBoost (200 estimators) |
| Class imbalance | `class_weight='balanced'` (RF), `scale_pos_weight` (XGB) |
| Evaluation | ROC-AUC, Average Precision, stratified 80/20 split |
| Interpretation | Feature importance (MDI), physicochemical distributions |

---

## Pharmaceutical Relevance

| Assay | Clinical Relevance |
|---|---|
| SR-MMP | Mitochondrial toxicity → DILI, cardiotoxicity |
| SR-ATAD5 | Genotoxicity → carcinogenicity (ICH S2R1 guideline) |
| NR-AhR | CYP1A1/1A2 induction → drug-drug interactions |
| SR-p53 | DNA damage → genotoxicity risk |
| NR-ER / NR-AR | Endocrine disruption → reproductive toxicity |

---

## Future Directions

- Multi-task neural network across all 12 assays simultaneously
- Graph Neural Networks (GNNs) for improved molecular representation
- SHAP values for substructure-level interpretability
- Integration with osimertinib RNA-seq transcriptomics (Project 1) for structure-to-transcriptome toxicity modeling
- PBPK model integration for in vivo toxicity risk prediction

---

## Author

**Matt Muslu**  
GitHub: [github.com/amuslu87](https://github.com/amuslu87)

*Part of a computational drug discovery portfolio. See also: [Osimertinib RNA-seq Treatment Response Analysis](https://github.com/amuslu87/lung-cancer-rnaseq).*
