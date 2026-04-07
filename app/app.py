# =============================================================================
# Streamlit App: Drug Toxicity Predictor
# Project: ADMET Toxicity Prediction
# Author: Matt Muslu
# =============================================================================

import streamlit as st

# must be first Streamlit command
st.set_page_config(page_title="Drug Toxicity Predictor", page_icon="🧬", layout="wide")

import pandas as pd
import numpy as np
import pickle
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Draw
from rdkit import RDLogger
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
RDLogger.DisableLog("rdApp.*")

# --- Paths -------------------------------------------------------------------
BASE_DIR  = r"C:\Users\amusl\Desktop\DrugDiscov\ADMET_PROJECT"
MODEL_DIR = os.path.join(BASE_DIR, "models")
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
TAB_DIR   = os.path.join(BASE_DIR, "results", "tables")

# --- Load models and performance data ----------------------------------------
@st.cache_resource
def load_models():
    models = {}
    for f in os.listdir(MODEL_DIR):
        if f.endswith("_model.pkl"):
            target = f.replace("_model.pkl", "")
            with open(os.path.join(MODEL_DIR, f), "rb") as fh:
                models[target] = pickle.load(fh)
    return models

@st.cache_data
def load_performance():
    return pd.read_csv(os.path.join(TAB_DIR, "model_performance.csv"))

models = load_models()
perf   = load_performance()

# --- Featurize a SMILES ------------------------------------------------------
def featurize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    X  = np.array(list(fp)).reshape(1, -1)
    props = {
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "LogP":             round(Descriptors.MolLogP(mol), 2),
        "H-Bond Donors":    rdMolDescriptors.CalcNumHBD(mol),
        "H-Bond Acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "TPSA (Å²)":        round(rdMolDescriptors.CalcTPSA(mol), 2),
    }
    return mol, X, props

# --- Lipinski Rule of Five check ---------------------------------------------
def lipinski_check(props):
    violations = []
    if props["Molecular Weight"] > 500:  violations.append("MW > 500 Da")
    if props["LogP"] > 5:                violations.append("LogP > 5")
    if props["H-Bond Donors"] > 5:       violations.append("HBD > 5")
    if props["H-Bond Acceptors"] > 10:   violations.append("HBA > 10")
    return violations

# =============================================================================
# APP LAYOUT
# =============================================================================
st.markdown("""
    <style>
    .main { background-color: #f9f9fb; }
    .stAlert { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Molecule_icon.svg/240px-Molecule_icon.svg.png",
             width=80)
    st.title("Drug Toxicity Predictor")
    st.markdown("---")
    st.markdown("**Project:** ADMET Toxicity Prediction  \n**Dataset:** Tox21 (8,014 compounds)  \n**Author:** Matt Muslu")
    st.markdown("---")
    st.markdown("**Models trained:** Random Forest + XGBoost  \n**Assays:** 12 Tox21 endpoints  \n**Fingerprint:** Morgan ECFP4 (2048 bits)")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔬 Predict Toxicity", "📊 Model Performance", "ℹ️ About"])

# -----------------------------------------------------------------------
# TAB 1: PREDICT
# -----------------------------------------------------------------------
with tab1:
    st.header("Predict Drug Toxicity from SMILES")
    st.markdown("Enter a drug molecule as a SMILES string to get toxicity predictions across all 12 Tox21 assays.")

    col1, col2 = st.columns([2, 1])
    with col1:
        smiles_input = st.text_input(
            "SMILES String",
            value="CC(=O)Oc1ccccc1C(=O)O",
            help="Example: CC(=O)Oc1ccccc1C(=O)O = Aspirin"
        )

    example_drugs = {
        "Aspirin":     "CC(=O)Oc1ccccc1C(=O)O",
        "Ibuprofen":   "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "Caffeine":    "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "Tamoxifen":   "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
        "Erlotinib":   "C#Cc1cccc(Nc2ncnc3cc(OCC)c(OCC)cc23)c1",
    }

    with col2:
        selected = st.selectbox("Or pick an example drug:", list(example_drugs.keys()))
        if st.button("Load example"):
            smiles_input = example_drugs[selected]
            st.rerun()

    if smiles_input:
        mol, X, props = featurize_smiles(smiles_input)

        if mol is None:
            st.error("Invalid SMILES string. Please check your input.")
        else:
            st.success("Valid molecule detected.")

            col_mol, col_props = st.columns([1, 1])

            with col_mol:
                st.subheader("Molecule Structure")
                img = Draw.MolToImage(mol, size=(300, 250))
                st.image(img, caption=smiles_input)

            with col_props:
                st.subheader("Physicochemical Properties")
                for k, v in props.items():
                    st.metric(k, v)

                violations = lipinski_check(props)
                if violations:
                    st.warning(f"**Lipinski violations:** {', '.join(violations)}")
                else:
                    st.success("Passes Lipinski Rule of Five (drug-like)")

            st.markdown("---")
            st.subheader("Toxicity Predictions Across 12 Assays")

            pred_rows = []
            for target, model_data in sorted(models.items()):
                model = model_data["model"]
                prob  = model.predict_proba(X)[0][1]
                pred  = "Toxic" if prob >= 0.5 else "Non-toxic"
                auc   = perf[perf["target"] == target]["best_auc"].values
                auc   = auc[0] if len(auc) > 0 else np.nan
                pred_rows.append({
                    "Assay":          target,
                    "Prediction":     pred,
                    "Toxic Prob (%)": round(prob * 100, 1),
                    "Model AUC":      round(auc, 3) if not np.isnan(auc) else "N/A"
                })

            pred_df = pd.DataFrame(pred_rows).sort_values("Toxic Prob (%)", ascending=False)

            def color_pred(val):
                if val == "Toxic":     return "color: #d73027; font-weight: bold"
                if val == "Non-toxic": return "color: #4575b4; font-weight: bold"
                return ""

            st.dataframe(
                pred_df.style.applymap(color_pred, subset=["Prediction"]),
                use_container_width=True, height=420
            )

            n_toxic = (pred_df["Prediction"] == "Toxic").sum()
            st.info(f"**Summary:** Predicted toxic in {n_toxic} / {len(pred_df)} assays")

# -----------------------------------------------------------------------
# TAB 2: MODEL PERFORMANCE
# -----------------------------------------------------------------------
with tab2:
    st.header("Model Performance — All Tox21 Assays")
    st.markdown("ROC-AUC scores for Random Forest and XGBoost across all 12 Tox21 toxicity endpoints.")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Mean RF AUC",  f"{perf['rf_auc'].mean():.3f}")
    col_b.metric("Mean XGB AUC", f"{perf['xgb_auc'].mean():.3f}")
    col_c.metric("Best AUC",     f"{perf['best_auc'].max():.3f}  ({perf.loc[perf['best_auc'].idxmax(), 'target']})")

    st.markdown("---")

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(perf))
    w = 0.35
    ax.bar(x - w/2, perf["rf_auc"],  w, label="Random Forest", color="#4575b4", edgecolor="grey")
    ax.bar(x + w/2, perf["xgb_auc"], w, label="XGBoost",       color="#d73027", edgecolor="grey")
    ax.axhline(y=0.5, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(perf["target"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC per Tox21 Assay", fontweight="bold")
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.dataframe(perf[["target", "best_model", "n_train", "n_test",
                        "n_pos_test", "rf_auc", "xgb_auc", "best_auc", "best_ap"]],
                 use_container_width=True)

# -----------------------------------------------------------------------
# TAB 3: ABOUT
# -----------------------------------------------------------------------
with tab3:
    st.header("About This Project")
    st.markdown("""
    ### Drug Toxicity Prediction Using Molecular Fingerprints and Machine Learning

    This project demonstrates a cheminformatics-based ADMET (Absorption, Distribution,
    Metabolism, Excretion, **Toxicity**) modeling workflow relevant to early-stage drug discovery.

    ---

    ### Dataset
    - **Tox21**: 8,014 compounds × 12 toxicity assays
    - Source: NIH Tox21 Challenge / MoleculeNet
    - Assays cover nuclear receptor (NR) and stress response (SR) pathways

    ### Methods
    - **Featurization**: Morgan fingerprints (ECFP4, radius=2, 2048 bits) via RDKit
    - **Models**: Random Forest (sklearn) + XGBoost — best model selected per assay by ROC-AUC
    - **Class imbalance**: handled via `class_weight='balanced'` (RF) and `scale_pos_weight` (XGB)
    - **Evaluation**: ROC-AUC, Average Precision, stratified 80/20 split

    ### Relevance to Drug Discovery
    Toxicity prediction is a critical step in **lead optimization** — filtering out
    compounds likely to fail in safety studies before expensive wet-lab experiments.
    ADMET models reduce attrition rates in clinical development.

    ---

    **Author:** Matt Muslu
    **GitHub:** [github.com/amuslu87](https://github.com/amuslu87)
    """)
