# =============================================================================
# Script 04: Model Interpretation — Feature Importance + Physicochemical Analysis
# Project: ADMET Toxicity Prediction
# Author: Matt Muslu
# =============================================================================

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# --- Paths -------------------------------------------------------------------
BASE_DIR  = r"C:\Users\amusl\Desktop\DrugDiscov\ADMET_PROJECT"
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR   = os.path.join(BASE_DIR, "results", "figures")
TAB_DIR   = os.path.join(BASE_DIR, "results", "tables")

# --- Load data ---------------------------------------------------------------
X    = np.load(os.path.join(PROC_DIR, "fingerprints.npy"))
meta = pd.read_csv(os.path.join(PROC_DIR, "tox21_features.csv"))
perf = pd.read_csv(os.path.join(TAB_DIR, "model_performance.csv"))

target_cols = [c for c in meta.columns
               if c not in ["smiles", "mol_id", "MW", "LogP", "HBD", "HBA", "TPSA"]]

# --- Figure 5: Feature importance for best RF model --------------------------
# Use the target with best RF AUC
best_rf_target = perf.sort_values("rf_auc", ascending=False).iloc[0]["target"]
print(f"Extracting feature importance from RF model: {best_rf_target}")

model_path = os.path.join(MODEL_DIR, f"{best_rf_target}_model.pkl")
with open(model_path, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:30]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(30), importances[top_idx], color="#d73027", edgecolor="grey")
    ax.set_xlabel("Fingerprint Bit Index")
    ax.set_ylabel("Importance (Mean Decrease Impurity)")
    ax.set_title(f"Top 30 Morgan Fingerprint Bits\n({best_rf_target}, Random Forest)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(range(30))
    ax.set_xticklabels(top_idx, rotation=90, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "05_feature_importance.png"), dpi=150)
    plt.close()
    print("Saved: 05_feature_importance.png")

# --- Figure 6: Physicochemical property distributions (toxic vs non-toxic) ---
desc_cols = ["MW", "LogP", "HBD", "HBA", "TPSA"]

# Use the target with most balanced classes for this plot
target_for_desc = perf.sort_values("n_pos_test", ascending=False).iloc[0]["target"]
y_labels = meta[target_for_desc].dropna()
desc_subset = meta.loc[y_labels.index, desc_cols + [target_for_desc]].dropna()

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
prop_labels = {
    "MW":   "Molecular Weight (Da)",
    "LogP": "LogP (Lipophilicity)",
    "HBD":  "H-Bond Donors",
    "HBA":  "H-Bond Acceptors",
    "TPSA": "TPSA (Å²)"
}

for ax, col in zip(axes, desc_cols):
    toxic     = desc_subset[desc_subset[target_for_desc] == 1][col]
    non_toxic = desc_subset[desc_subset[target_for_desc] == 0][col]
    ax.hist(non_toxic, bins=40, alpha=0.6, color="#4575b4", label="Non-toxic", density=True)
    ax.hist(toxic,     bins=40, alpha=0.6, color="#d73027", label="Toxic",     density=True)
    ax.set_xlabel(prop_labels[col], fontsize=9)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title(col, fontweight="bold", fontsize=10)
    ax.legend(fontsize=7)

plt.suptitle(f"Physicochemical Properties: Toxic vs Non-Toxic\n({target_for_desc})",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "06_physicochemical_distributions.png"), dpi=150)
plt.close()
print("Saved: 06_physicochemical_distributions.png")

# --- Figure 7: Performance heatmap (RF vs XGB across all targets) ------------
perf_plot = perf[["target", "rf_auc", "xgb_auc"]].set_index("target")
perf_plot.index = [t.replace("NR-", "NR-").replace("SR-", "SR-") for t in perf_plot.index]

fig, ax = plt.subplots(figsize=(5, 7))
sns.heatmap(perf_plot, annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=0.5, vmax=1.0, linewidths=0.5, ax=ax,
            cbar_kws={"label": "ROC-AUC"})
ax.set_title("ROC-AUC: RF vs XGBoost\nAcross All Tox21 Assays",
             fontsize=12, fontweight="bold")
ax.set_xticklabels(["Random Forest", "XGBoost"], rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "07_performance_heatmap.png"), dpi=150)
plt.close()
print("Saved: 07_performance_heatmap.png")

# --- Figure 8: Correlation between physicochemical properties and toxicity ----
corr_rows = []
for target in target_cols:
    y = meta[target].dropna()
    subset = meta.loc[y.index, desc_cols + [target]].dropna()
    if len(subset) < 50:
        continue
    for col in desc_cols:
        corr = subset[col].corr(subset[target])
        corr_rows.append({"target": target, "property": col, "correlation": corr})

corr_df = pd.DataFrame(corr_rows).pivot(index="target", columns="property", values="correlation")

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax,
            cbar_kws={"label": "Pearson r"})
ax.set_title("Correlation: Physicochemical Properties vs Toxicity Labels",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "08_property_toxicity_correlation.png"), dpi=150)
plt.close()
print("Saved: 08_property_toxicity_correlation.png")

# --- Summary stats table -----------------------------------------------------
summary = meta[desc_cols].describe().round(2)
summary.to_csv(os.path.join(TAB_DIR, "physicochemical_summary.csv"))
print("Saved: physicochemical_summary.csv")

print("\nScript 04 complete.")
