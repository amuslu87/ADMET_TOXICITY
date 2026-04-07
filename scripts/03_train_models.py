# =============================================================================
# Script 03: Model Training — Random Forest + XGBoost
# Project: ADMET Toxicity Prediction
# Author: Matt Muslu
# =============================================================================

import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import (roc_auc_score, average_precision_score,
                                     classification_report, roc_curve,
                                     precision_recall_curve)
from sklearn.preprocessing   import label_binarize
from sklearn.utils           import class_weight
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- Paths -------------------------------------------------------------------
BASE_DIR  = r"C:\Users\amusl\Desktop\DrugDiscov\ADMET_PROJECT"
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR   = os.path.join(BASE_DIR, "results", "figures")
TAB_DIR   = os.path.join(BASE_DIR, "results", "tables")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TAB_DIR,   exist_ok=True)

# --- Load data ---------------------------------------------------------------
print("Loading data...")
X = np.load(os.path.join(PROC_DIR, "fingerprints.npy"))
meta = pd.read_csv(os.path.join(PROC_DIR, "tox21_features.csv"))

target_cols = [c for c in meta.columns
               if c not in ["smiles", "mol_id", "MW", "LogP", "HBD", "HBA", "TPSA"]]

print(f"Fingerprint matrix: {X.shape}")
print(f"Targets: {target_cols}\n")

# --- Train one model per target ----------------------------------------------
# Each toxicity assay is treated as a separate binary classification task
# We use the best-performing model (RF or XGB) selected by ROC-AUC

results = []

for target in target_cols:
    y_raw = meta[target]

    # Drop rows where target label is missing
    mask   = y_raw.notnull()
    X_t    = X[mask]
    y_t    = y_raw[mask].astype(int).values
    n_pos  = y_t.sum()
    n_neg  = (y_t == 0).sum()

    if n_pos < 10 or n_neg < 10:
        print(f"  Skipping {target} — insufficient class samples")
        continue

    # Train/test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X_t, y_t, test_size=0.2, random_state=42, stratify=y_t
    )

    # Class weight for imbalanced data
    cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw_dict = dict(enumerate(cw))

    # --- Random Forest -------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_auc   = roc_auc_score(y_test, rf_proba)
    rf_ap    = average_precision_score(y_test, rf_proba)

    # --- XGBoost -------------------------------------------------------------
    scale_pos = n_neg / max(n_pos, 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos, use_label_encoder=False,
        eval_metric="auc", random_state=42, verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc   = roc_auc_score(y_test, xgb_proba)
    xgb_ap    = average_precision_score(y_test, xgb_proba)

    # Pick better model by ROC-AUC
    if rf_auc >= xgb_auc:
        best_model, best_proba, best_auc, best_ap, best_name = rf, rf_proba, rf_auc, rf_ap, "RandomForest"
    else:
        best_model, best_proba, best_auc, best_ap, best_name = xgb_model, xgb_proba, xgb_auc, xgb_ap, "XGBoost"

    print(f"  {target:<35} RF AUC={rf_auc:.3f}  XGB AUC={xgb_auc:.3f}  → {best_name}")

    # Save best model
    with open(os.path.join(MODEL_DIR, f"{target}_model.pkl"), "wb") as f:
        pickle.dump({"model": best_model, "name": best_name, "target": target}, f)

    results.append({
        "target":      target,
        "best_model":  best_name,
        "n_train":     len(y_train),
        "n_test":      len(y_test),
        "n_pos_test":  int(y_test.sum()),
        "rf_auc":      round(rf_auc, 4),
        "xgb_auc":     round(xgb_auc, 4),
        "best_auc":    round(best_auc, 4),
        "best_ap":     round(best_ap, 4),
        "X_test":      X_test,
        "y_test":      y_test,
        "best_proba":  best_proba
    })

# --- Figure 3: Model performance comparison (ROC-AUC per target) -------------
res_df = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ["X_test", "y_test", "best_proba"]}
                        for r in results])

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(res_df))
w = 0.35
ax.bar(x - w/2, res_df["rf_auc"],  w, label="Random Forest", color="#4575b4", edgecolor="grey")
ax.bar(x + w/2, res_df["xgb_auc"], w, label="XGBoost",       color="#d73027", edgecolor="grey")
ax.axhline(y=0.5, color="black", linestyle="--", linewidth=0.8, label="Random (0.5)")
ax.set_xticks(x)
ax.set_xticklabels([t.replace("NR-", "NR\n").replace("SR-", "SR\n")
                    for t in res_df["target"]], fontsize=8)
ax.set_ylabel("ROC-AUC")
ax.set_title("Model Performance per Toxicity Assay", fontsize=13, fontweight="bold")
ax.legend()
ax.set_ylim(0.4, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_model_performance_comparison.png"), dpi=150)
plt.close()
print("\nSaved: 03_model_performance_comparison.png")

# --- Figure 4: ROC curves for all targets (best model) -----------------------
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()

for i, r in enumerate(results):
    fpr, tpr, _ = roc_curve(r["y_test"], r["best_proba"])
    axes[i].plot(fpr, tpr, color="#d73027", linewidth=2,
                 label=f"AUC={r['best_auc']:.3f}")
    axes[i].plot([0,1],[0,1], "k--", linewidth=0.8)
    axes[i].set_title(r["target"], fontsize=8, fontweight="bold")
    axes[i].set_xlabel("FPR", fontsize=7)
    axes[i].set_ylabel("TPR", fontsize=7)
    axes[i].legend(fontsize=7)
    axes[i].tick_params(labelsize=6)

for j in range(len(results), len(axes)):
    axes[j].axis("off")

plt.suptitle("ROC Curves — Best Model per Tox21 Assay", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_roc_curves_all_targets.png"), dpi=150)
plt.close()
print("Saved: 04_roc_curves_all_targets.png")

# --- Save results table ------------------------------------------------------
res_df.to_csv(os.path.join(TAB_DIR, "model_performance.csv"), index=False)
print(f"Saved: model_performance.csv")
print(f"\n=== Overall Mean ROC-AUC: {res_df['best_auc'].mean():.3f} ===")
print("\nScript 03 complete.")
