# =============================================================================
# Script 01: Download and Explore Tox21 Dataset
# Project: ADMET Toxicity Prediction
# Author: Matt Muslu
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import requests
import gzip
import shutil
import os

# --- Paths -------------------------------------------------------------------
BASE_DIR    = r"C:\Users\amusl\Desktop\DrugDiscov\ADMET_PROJECT"
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
FIG_DIR     = os.path.join(BASE_DIR, "results", "figures")

os.makedirs(RAW_DIR,  exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

# --- Download Tox21 ----------------------------------------------------------
URL      = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
GZ_PATH  = os.path.join(RAW_DIR, "tox21.csv.gz")
CSV_PATH = os.path.join(RAW_DIR, "tox21.csv")

if not os.path.exists(CSV_PATH):
    print("Downloading Tox21 dataset...")
    r = requests.get(URL, stream=True)
    with open(GZ_PATH, "wb") as f:
        shutil.copyfileobj(r.raw, f)
    with gzip.open(GZ_PATH, "rb") as f_in:
        with open(CSV_PATH, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("Download complete.")
else:
    print("Dataset already exists, skipping download.")

# --- Load and inspect --------------------------------------------------------
df = pd.read_csv(CSV_PATH)
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 3 rows:\n{df.head(3)}")

# Identify toxicity target columns (all except smiles and mol_id)
target_cols = [c for c in df.columns if c not in ["smiles", "mol_id"]]
print(f"\nToxicity targets ({len(target_cols)}):\n{target_cols}")

# --- Missing value analysis ---------------------------------------------------
print("\n=== Missing values per target (%) ===")
missing = df[target_cols].isnull().mean() * 100
print(missing.round(1).to_string())

# --- Class balance per target ------------------------------------------------
print("\n=== Class balance per target (% positive) ===")
for col in target_cols:
    pos = df[col].mean() * 100
    print(f"  {col:<35} {pos:.1f}% positive")

# --- Figure 1: Missing value heatmap -----------------------------------------
fig, ax = plt.subplots(figsize=(14, 4))
missing_df = df[target_cols].isnull().astype(int)
sns.heatmap(missing_df.T, cmap="Reds", cbar=False, yticklabels=True,
            xticklabels=False, ax=ax)
ax.set_title("Missing Values in Tox21 Dataset (red = missing)", fontsize=13, fontweight="bold")
ax.set_xlabel("Compounds")
ax.set_ylabel("Toxicity Assay")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_missing_values_heatmap.png"), dpi=150)
plt.close()
print("\nSaved: 01_missing_values_heatmap.png")

# --- Figure 2: Class imbalance bar chart -------------------------------------
pos_rates = df[target_cols].mean() * 100
fig, ax = plt.subplots(figsize=(12, 5))
pos_rates.sort_values().plot(kind="barh", color="#d73027", edgecolor="grey", ax=ax)
ax.axvline(x=50, color="black", linestyle="--", linewidth=0.8)
ax.set_xlabel("% Positive (Toxic) Compounds")
ax.set_title("Class Imbalance Across Tox21 Assays", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_class_imbalance.png"), dpi=150)
plt.close()
print("Saved: 02_class_imbalance.png")

# --- Save cleaned dataset ----------------------------------------------------
# Drop rows with no SMILES
df_clean = df.dropna(subset=["smiles"]).reset_index(drop=True)
df_clean.to_csv(os.path.join(PROC_DIR, "tox21_clean.csv"), index=False)
print(f"\nSaved cleaned dataset: {df_clean.shape[0]} compounds")
print("\nScript 01 complete.")
