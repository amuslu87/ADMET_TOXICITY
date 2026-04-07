# Findings Report: Drug Toxicity Prediction Using Molecular Fingerprints and Machine Learning

**Project:** ADMET Toxicity Prediction Portfolio  
**Author:** Matt Muslu  
**Date:** April 2026  
**Dataset:** Tox21 — 7,831 compounds × 12 toxicity assays

---

## Abstract

A machine learning pipeline was developed to predict drug compound toxicity across 12 biological assay endpoints using Morgan molecular fingerprints derived from SMILES representations. Random Forest classifiers consistently outperformed XGBoost across all assays, achieving a mean ROC-AUC of **0.830** (range: 0.715–0.906). The highest predictive performance was observed for the aryl hydrocarbon receptor (NR-AhR, AUC = 0.906) and the DNA damage genotoxicity assay (SR-ATAD5, AUC = 0.888). Physicochemical analysis revealed that toxic compounds tend toward higher molecular weight and lipophilicity. These models have direct relevance to early-stage computational toxicology screening in pharmaceutical drug discovery.

---

## 1. Dataset Characteristics

### 1.1 Compound Library

The Tox21 dataset contains **7,831 structurally diverse small molecules** including approved drugs, drug candidates, environmental chemicals, and industrial compounds. The mean molecular weight of 276 Da (SD = 166 Da) and mean LogP of 2.37 indicate an overall drug-like collection consistent with Lipinski's Rule of Five:

| Property | Mean | Std | Median | Drug-like Threshold |
|---|---|---|---|---|
| Molecular Weight (Da) | 276.3 | 165.8 | 240.3 | < 500 |
| LogP | 2.37 | 2.37 | 2.37 | < 5 |
| H-Bond Donors | 1.23 | 1.95 | 1.0 | < 5 |
| H-Bond Acceptors | 3.57 | 3.25 | 3.0 | < 10 |
| TPSA (Å²) | 59.6 | 59.0 | 46.5 | < 140 (oral) |

The majority of compounds fall within drug-like property ranges, indicating the dataset is representative of pharmaceutical compound libraries used in safety screening programs.

### 1.2 Class Imbalance

All 12 assays exhibit significant class imbalance, with toxic compounds comprising:
- **Lowest:** NR-PPAR-gamma (2.9% toxic)
- **Highest:** SR-ARE (16.2% toxic)
- **Median:** ~6% toxic

This mirrors real-world pharmaceutical toxicology, where the majority of screened compounds are non-toxic, and correctly identifying the rare toxic compound is the critical challenge.

---

## 2. Model Performance

### 2.1 Overall Results

Random Forest consistently outperformed XGBoost across all 12 assays and was selected as the final model for every endpoint.

| Assay | Biological Target | Best AUC | AP |
|---|---|---|---|
| **NR-AhR** | Aryl Hydrocarbon Receptor | **0.906** | 0.607 |
| **SR-ATAD5** | DNA Damage / Genotoxicity | **0.888** | 0.320 |
| **SR-MMP** | Mitochondrial Membrane Potential | **0.878** | 0.643 |
| **NR-Aromatase** | Aromatase Enzyme | **0.863** | 0.494 |
| **SR-p53** | p53 DNA Damage Pathway | **0.843** | 0.325 |
| **SR-ARE** | Oxidative Stress Response | **0.827** | 0.555 |
| **NR-PPAR-gamma** | Metabolic Receptor | **0.813** | 0.233 |
| **NR-AR-LBD** | Androgen Receptor LBD | **0.807** | 0.506 |
| **NR-ER-LBD** | Estrogen Receptor LBD | **0.790** | 0.404 |
| **SR-HSE** | Heat Shock / Cellular Stress | **0.777** | 0.324 |
| **NR-AR** | Androgen Receptor | **0.756** | 0.470 |
| **NR-ER** | Estrogen Receptor alpha | **0.715** | 0.438 |
| **Mean** | — | **0.830** | 0.444 |

### 2.2 Best Performing Assays

**NR-AhR (AUC = 0.906)** achieved the highest predictive accuracy. The aryl hydrocarbon receptor responds to polycyclic aromatic hydrocarbons and halogenated compounds — structural classes with highly distinctive Morgan fingerprint patterns, explaining the strong model performance. AhR activation is associated with immune suppression, developmental toxicity, and carcinogenesis.

**SR-ATAD5 (AUC = 0.888)** measures DNA damage-induced genotoxicity. Genotoxic compounds often share specific electrophilic structural features (alkylating groups, intercalating aromatic systems) that are well-captured by ECFP4 fingerprints.

**SR-MMP (AUC = 0.878)** measures mitochondrial membrane potential disruption — a key mechanism of drug-induced organ toxicity (DILI, cardiotoxicity). The high predictive accuracy here has direct pharmaceutical relevance, as mitochondrial toxicity is a leading cause of drug withdrawal post-approval.

### 2.3 Most Challenging Assays

**NR-ER (AUC = 0.715)** was the most difficult assay to predict. Estrogen receptor binding involves subtle structural features (specific stereochemistry, hydroxyl positioning) that are difficult to capture with circular fingerprints alone. This is consistent with published literature showing that ER prediction benefits from 3D-structure-based methods.

**NR-AR (AUC = 0.756)** similarly reflects the challenge of predicting nuclear receptor binding from 2D fingerprints, where binding affinity depends on 3D complementarity to the receptor binding pocket.

### 2.4 RF vs XGBoost

Random Forest outperformed XGBoost in all 12 assays. Mean AUC differences:
- Random Forest mean AUC: **0.830**
- XGBoost mean AUC: **0.794**

This is consistent with published benchmarks on Tox21 data. Random Forest handles the sparse, high-dimensional binary fingerprint space (2,048 bits) more efficiently than gradient boosting, which is better suited to dense, continuous feature spaces. Random Forest's bagging approach also provides better variance reduction on imbalanced datasets with relatively small positive class sizes.

---

## 3. Physicochemical Findings

### 3.1 Toxic vs Non-toxic Property Distributions

Physicochemical analysis revealed systematic differences between toxic and non-toxic compounds:

- **Molecular Weight:** Toxic compounds tend to have higher MW. Larger molecules occupy more chemical space and are more likely to interact non-specifically with biological targets.
- **LogP:** Toxic compounds tend toward higher lipophilicity. High LogP compounds penetrate cell membranes more readily, increasing intracellular concentration and the probability of off-target interactions.
- **TPSA:** Lower TPSA in toxic compounds reflects higher membrane permeability — consistent with the LogP observation.
- **HBD/HBA:** Less pronounced differences, though extremely high HBD counts are rare in toxic compounds (very polar molecules tend not to penetrate cell membranes).

### 3.2 Property-Toxicity Correlations

LogP showed the most consistent positive correlation with toxicity across multiple assays, particularly SR-MMP and SR-ATAD5. This is biologically plausible — lipophilic compounds accumulate in mitochondrial membranes and are more likely to intercalate with DNA.

MW showed moderate positive correlation with NR-AhR activity, consistent with the known preference of the AhR for large planar polycyclic molecules.

---

## 4. Drug Discovery Implications

### 4.1 Application in Lead Optimization

In pharmaceutical drug discovery, ADMET models like those developed here are applied during **lead optimization** — the iterative chemical modification of a promising scaffold to improve efficacy while reducing toxicity. Key applications:

- **Virtual screening:** Rank compound libraries by predicted toxicity before synthesis
- **Scaffold hopping:** Identify which structural modifications reduce toxicity signals
- **Go/no-go decisions:** Flag compounds predicted toxic in multiple assays for deprioritization
- **DILI risk assessment:** SR-MMP and SR-p53 predictions are directly relevant to predicting drug-induced liver injury, a major cause of clinical failure

### 4.2 Most Actionable Predictions for Drug Discovery

| Assay | Why it matters in drug development |
|---|---|
| **SR-MMP** (AUC = 0.878) | Mitochondrial toxicity → cardiac/hepatic toxicity risk, leading cause of post-market withdrawal |
| **SR-ATAD5** (AUC = 0.888) | Genotoxicity → carcinogenicity risk, regulatory requirement (ICH S2R1) |
| **NR-AhR** (AUC = 0.906) | Drug metabolism induction → drug-drug interaction risk (CYP1A1/1A2) |
| **SR-p53** (AUC = 0.843) | DNA damage → genotoxicity/carcinogenicity |
| **NR-ER / NR-AR** (AUC ~0.73–0.76) | Endocrine disruption → reproductive toxicity concerns |

### 4.3 Limitations

1. **2D fingerprints only:** Morgan fingerprints encode connectivity but not 3D shape or stereochemistry. For nuclear receptor binding (NR-ER, NR-AR), 3D pharmacophore features or molecular docking scores would improve predictions.

2. **Binary labels:** Tox21 assay data is binary (active/inactive). Dose-response relationships and IC50 values are lost — the model cannot distinguish a weakly toxic from a highly toxic compound within the positive class.

3. **In vitro to in vivo gap:** All Tox21 assays are cell-based biochemical assays. Predictions may not translate directly to in vivo toxicity due to ADME differences (metabolism, protein binding, distribution).

4. **Training data bias:** The Tox21 library overrepresents certain chemical classes. Predictions on highly novel scaffolds (outside the training chemical space) may be unreliable.

5. **Single-task models:** Each assay is modeled independently. Multi-task learning approaches (sharing information across related assays) have been shown to improve performance, particularly for assays with small positive class sizes.

---

## 5. Future Directions

1. **Multi-task neural network:** Train a single deep learning model across all 12 assays simultaneously. Published benchmarks show 5–15% AUC improvement over single-task Random Forest on Tox21.

2. **Graph Neural Networks (GNNs):** Represent molecules as graphs (atoms = nodes, bonds = edges) and use message-passing neural networks. GNNs are the current state-of-the-art for molecular property prediction.

3. **3D structure-based features:** Add RDKit 3D descriptors, USRCAT shape descriptors, or docking scores for improved NR-ER and NR-AR predictions.

4. **SHAP interpretability:** Apply SHAP (SHapley Additive exPlanations) values to identify specific molecular substructures driving toxicity predictions — enabling medicinal chemists to rationally redesign toxic scaffolds.

5. **Integration with RNA-seq data:** Link predicted compound toxicity with transcriptomic pathway activation (from Project 1: osimertinib RNA-seq analysis) to build a structure-to-transcriptome toxicity model — a multi-omics approach directly relevant to mechanistic toxicology.

6. **PBPK integration:** Combine ADMET predictions with physiologically-based pharmacokinetic (PBPK) modeling to predict tissue-level compound concentrations and refine toxicity risk assessment.

---

## 6. Conclusion

This project demonstrates a complete, end-to-end cheminformatics toxicity prediction pipeline — from raw SMILES strings to interactive predictions — achieving a mean ROC-AUC of 0.830 across 12 Tox21 assays. Random Forest with Morgan ECFP4 fingerprints provides a strong, reproducible baseline for computational toxicology that is directly applicable to pharmaceutical drug discovery workflows. The Streamlit application enables real-time toxicity prediction for any drug compound, making the models immediately usable in a lead optimization context.

The strongest predictions (NR-AhR, SR-ATAD5, SR-MMP) cover three of the most critical safety flags in drug development: drug metabolism induction, genotoxicity, and mitochondrial toxicity — providing immediate practical value in a pharmaceutical R&D setting.

---

## References

- Mayr, A., et al. (2016). DeepTox: Toxicity prediction using deep learning. *Frontiers in Environmental Science*, 3, 80.
- Wu, Z., et al. (2018). MoleculeNet: A benchmark for molecular machine learning. *Chemical Science*, 9(2), 513–530.
- Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling*, 50(5), 742–754.
- Huang, R., et al. (2016). Tox21Challenge to build predictive models of nuclear receptor and stress response pathways as mediated by exposure to environmental chemicals and drugs. *Frontiers in Environmental Science*, 3, 85.
- Lipinski, C. A., et al. (2001). Experimental and computational approaches to estimate solubility and permeability in drug discovery. *Advanced Drug Delivery Reviews*, 46(1–3), 3–26.
