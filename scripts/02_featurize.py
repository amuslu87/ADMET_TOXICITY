# =============================================================================
# Script 02: Molecular Featurization with RDKit (Morgan Fingerprints)
# Project: ADMET Toxicity Prediction
# Author: Matt Muslu
# =============================================================================

import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")  # suppress RDKit warnings

# --- Paths -------------------------------------------------------------------
BASE_DIR = r"C:\Users\amusl\Desktop\DrugDiscov\ADMET_PROJECT"
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

# --- Load cleaned dataset ----------------------------------------------------
df = pd.read_csv(os.path.join(PROC_DIR, "tox21_clean.csv"))
print(f"Loaded {df.shape[0]} compounds")

target_cols = [c for c in df.columns if c not in ["smiles", "mol_id"]]

# --- Morgan Fingerprints -----------------------------------------------------
# Morgan fingerprints (radius=2, 2048 bits) = ECFP4
# This is the gold standard for QSAR modeling in drug discovery
def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return list(fp)

print("Computing Morgan fingerprints (ECFP4, 2048 bits)...")
fps = df["smiles"].apply(smiles_to_morgan)

# Track invalid SMILES
invalid = fps.isnull().sum()
print(f"Invalid SMILES (could not parse): {invalid}")

# Filter to valid molecules only
valid_mask = fps.notnull()
df_valid   = df[valid_mask].reset_index(drop=True)
fps_valid  = fps[valid_mask].reset_index(drop=True)

print(f"Valid compounds retained: {df_valid.shape[0]}")

# Build fingerprint matrix
fp_matrix = np.array(fps_valid.tolist(), dtype=np.uint8)
print(f"Fingerprint matrix shape: {fp_matrix.shape}")

# --- RDKit Physicochemical Descriptors ---------------------------------------
# Add 5 key physicochemical properties used in Lipinski's Rule of Five
# These are interpretable features for the findings report
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * 5
    return [
        Descriptors.MolWt(mol),                          # Molecular weight
        Descriptors.MolLogP(mol),                         # Lipophilicity
        rdMolDescriptors.CalcNumHBD(mol),                 # H-bond donors
        rdMolDescriptors.CalcNumHBA(mol),                 # H-bond acceptors
        rdMolDescriptors.CalcTPSA(mol)                    # Topological polar surface area
    ]

print("Computing physicochemical descriptors...")
desc_cols = ["MW", "LogP", "HBD", "HBA", "TPSA"]
descriptors = df_valid["smiles"].apply(compute_descriptors)
desc_df = pd.DataFrame(descriptors.tolist(), columns=desc_cols)

# --- Save outputs ------------------------------------------------------------
# Fingerprint matrix as numpy array
np.save(os.path.join(PROC_DIR, "fingerprints.npy"), fp_matrix)

# Metadata + targets + descriptors
meta_df = df_valid[["smiles", "mol_id"] + target_cols].copy()
meta_df = pd.concat([meta_df.reset_index(drop=True), desc_df], axis=1)
meta_df.to_csv(os.path.join(PROC_DIR, "tox21_features.csv"), index=False)

print(f"\nSaved:")
print(f"  - fingerprints.npy  shape: {fp_matrix.shape}")
print(f"  - tox21_features.csv rows: {meta_df.shape[0]}")
print("\nScript 02 complete.")
