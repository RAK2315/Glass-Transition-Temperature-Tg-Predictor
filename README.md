# Glass Transition Temperature (Tg) Predictor

A machine learning-based web application for predicting the glass transition temperature (Tg) of polymers and small molecules using two distinct approaches: physics-informed functional group features and automated SMILES-based predictions.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://glass-transition-temperature-tg-predictor.streamlit.app/)


🌐 **Live Demo**: [https://glass-transition-temperature-tg-predictor.streamlit.app/](https://glass-transition-temperature-tg-predictor.streamlit.app/)

## 🔬 Overview

The Glass Transition Temperature (Tg) is a critical property of polymers that determines their mechanical behavior and thermal stability. This project provides two complementary machine learning approaches for rapid Tg prediction:

1. **Functional Group Mode**: Uses chemically engineered features (functional group counts, molecular weight, melting temperature) for highly accurate and interpretable predictions
2. **SMILES Mode**: Uses automated textual features extracted from SMILES strings for quick, structure-based predictions

The application achieves state-of-the-art performance with **R² ≈ 0.986** and **RMSE ≈ 11 K** for the Functional Group Mode.

## ✨ Features

### Prediction Modes

- **Functional Group Mode**
  - Manual input of 13 molecular descriptors
  - Physics-informed feature engineering
  - Highest accuracy (RMSE ≈ 11 K)
  - Interpretable feature importance analysis
  - Best for: Users with detailed molecular structure knowledge

- **SMILES Mode**
  - Simple SMILES string input
  - Automated feature extraction
  - Good accuracy (RMSE ≈ 21 K)
  - No manual calculations required
  - Best for: Quick predictions and high-throughput screening

### Data Exploration

- **Dataset Viewer**: Browse and filter ~700 polymer compounds
- **Interactive Visualizations**: 
  - Feature distributions
  - Tg vs Tm scatter plots
  - Functional group frequency analysis
- **Functional Group Filtering**: Filter dataset by chemical families (Oxide, Amine, Halide, Aromatic, etc.)

### Model Analysis

- **Performance Comparison**: Side-by-side metrics for all models
- **Feature Importance**: Understand which molecular features drive Tg predictions
- **Interactive Charts**: Explore model performance with Altair visualizations

## 📊 Dataset

The dataset contains approximately **700 unique polymer compounds** with:

- **Functional Group Counts**: #CH3, #CH2, #CH, #C, #OH, #C-O-C, #O=C, #N, #Hal
- **Structural Descriptors**: Double Bond Equivalent (DBA), O:C ratio
- **Molecular Properties**: Molecular weight (M), Melting temperature (Tm)
- **Target Variable**: Glass transition temperature (Tg) in Kelvin
- **SMILES Representations**: For structure-based predictions
- **Literature References**: Traceability to original experimental sources

**Dataset Source**: [Zenodo - Tg Dataset](https://zenodo.org/records/7319485)

### Data Composition

- Small molecules, oligomers, and polymer fragments
- Wide range of functional groups: oxides, amines, aromatics, halides, nitriles
- Experimentally measured Tg and Tm values
- Molecular weights from ~50 to 1000+ g/mol

## 🧪 Methodology

### Functional Group Mode (Physics-Informed)

**Input Features (13 descriptors):**
- Functional group counts: `#CH3`, `#CH2`, `#CH`, `#C`, `#OH`, `#C-O-C`, `#O=C`, `#N`, `#Hal`
- Structural: `DBA` (Double Bond Equivalent), `O:C` ratio
- Global properties: `M` (molecular weight), `Tm` (melting temperature)

**Models Evaluated:**
- Random Forest Regressor
- Extra Trees Regressor
- **Gradient Boosting Regressor** (Best performer)

**Key Advantages:**
- Direct connection to polymer chemistry principles
- Interpretable feature importance
- Captures hydrogen bonding, chain stiffness, and molecular flexibility effects

### SMILES Mode (Data-Driven)

**Input Features:**
- Character n-grams (2-3 characters) extracted from SMILES strings
- Vectorized using CountVectorizer with TF-IDF weighting

**Models Evaluated:**
- Random Forest Regressor
- **Extra Trees Regressor** (Best performer)
- Gradient Boosting Regressor

**Key Advantages:**
- No manual feature engineering required
- Fast predictions from molecular structure
- Scalable for large-scale screening

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/RAK2315/Glass-Transition-Temperature-Tg-Predictor.git
cd Glass-Transition-Temperature-Tg-Predictor
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Requirements

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
altair>=5.0.0
rdkit>=2023.3.1
```

## 💻 Usage

### Running the Application

```bash
streamlit run Home.py
```

The application will open in your browser at `http://localhost:8501`

### Functional Group Mode Example

1. Navigate to **Functional Group Mode** in the sidebar
2. Input molecular descriptors:
   - #CH3 = 2
   - #CH2 = 4
   - #OH = 1
   - M = 150.0 g/mol
   - Tm = 300.0 K
   - (Leave other fields as needed)
3. Click **"Predict Tg (Functional Group Mode)"**
4. View predicted Tg with ±8.01 K error margin

### SMILES Mode Example

1. Navigate to **SMILES Mode** in the sidebar
2. Enter SMILES string (e.g., `CCO` for ethanol)
3. Click **"Predict Tg (SMILES Mode)"**
4. View predicted Tg with ±15.1 K error margin

**SMILES Generation Tool**: [SMILES Generator](https://www.cheminfo.org/flavor/malaria/Utilities/SMILES_generator___checker/index.html)

## 📈 Model Performance

### Functional Group Mode

| Model | R² Score | RMSE (K) | MSE |
|-------|----------|----------|-----|
| Gradient Boosting | **0.9859** | **10.97** | 120.16 |
| Extra Trees | 0.9834 | 11.91 | 141.68 |
| Random Forest | 0.9825 | 12.25 | 150.16 |

**Mean Absolute Error (Cross-Validation)**: ±8.01 K

### SMILES Mode

| Model | R² Score | RMSE (K) | MSE | MAE (K) |
|-------|----------|----------|-----|---------|
| **Extra Trees** | **0.9463** | **21.43** | 459.25 | 12.11 |
| Random Forest | 0.9343 | 23.72 | 562.33 | 13.45 |
| Gradient Boosting | 0.9378 | 23.07 | 532.33 | 12.90 |

**Mean Absolute Error (Cross-Validation)**: ±15.1 K

### Key Findings

- **Functional Group Mode** provides ~2x better accuracy than SMILES Mode
- **#OH count** and **Tm** are the most important predictors (hydrogen bonding and chain stiffness)
- **Molecular Weight (M)** and **#CH3** also contribute significantly
- Physics-informed features enable both high accuracy and interpretability

## 📁 Project Structure

```
Glass-Transition-Temperature-Tg-Predictor/
│
├── .ipynb_checkpoints/              # Jupyter notebook checkpoints
├── .venv/                           # Virtual environment
├── models/                          # Trained ML models
│   ├── functional_group_model.pkl   # Trained Gradient Boosting model
│   ├── functional_group_scaler.pkl  # Feature scaler for FG mode
│   ├── ExtraTrees_SMILES_Model.pkl # Trained Extra Trees model
│   └── smiles_vectorizer.pkl        # SMILES text vectorizer
│
├── pages/                           # Streamlit multi-page app
│   ├── 1_📊About_Dataset.py        # Dataset exploration and visualization
│   ├── 2_🧪ML_Analysis.py          # Model performance analysis
│   └── 3_📝Overview.py             # Project overview and documentation
│
├── .gitignore                       # Git ignore file
├── dataset.csv                      # Training dataset (~700 compounds)
├── EDA and Model creation.ipynb     # Jupyter notebook for model development
├── Home.py                          # Main application entry point
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

## 🔑 Key Features Explained

### Functional Group Inputs

- **#CH3**: Methyl groups (−CH₃)
- **#CH2**: Methylene groups (−CH₂−)
- **#CH**: Methine groups (−CH−)
- **#C**: Quaternary carbons
- **#OH**: Hydroxyl groups (−OH)
- **#C-O-C**: Ether oxygen atoms
- **#O=C**: Carbonyl oxygen atoms
- **DBA**: Double bond equivalent = 1 + Σ[nᵢ(vᵢ - 2)]
- **#N**: Nitrogen atoms
- **#Hal**: Halogen atoms (F, Cl, Br, I)
- **O:C**: Oxygen-to-carbon ratio
- **M**: Molecular weight (g/mol)
- **Tm**: Melting temperature (K)

### SMILES Format

SMILES (Simplified Molecular Input Line Entry System) is a text notation for molecular structures, examples:
- `CCO` = Ethanol (CH₃CH₂OH)
- `CC(=O)C` = Acetone (CH₃COCH₃)
- `C(C1C(C(C(C(O1)O)O)O)O)O` = Glucose

## 📚 Resources

- **Live Application**: [Streamlit App](https://glass-transition-temperature-tg-predictor.streamlit.app/)
- **Dataset Source**: [Zenodo Repository](https://zenodo.org/records/7319485)
- **SMILES Generator**: [Online Tool](https://www.cheminfo.org/flavor/malaria/Utilities/SMILES_generator___checker/index.html)

---

