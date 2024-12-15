# ML Based EEG Frequency Analysis for Alzheimer's Disease Detection

This project implements a machine learning pipeline for analyzing EEG frequency patterns to detect Alzheimer's Disease (AD). It focuses on identifying and analyzing frequency-based differences between AD patients and age-matched healthy controls using EEG data.

Data is based on research data from the [article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0244180) "Resting state EEG biomarkers of cognitive decline associated with Alzheimer's disease and mild cognitive impairment" which is attached to the repository.
## Project Overview

The project analyzes EEG data to:

- Identify characteristic frequencies that distinguish between AD patients and controls
- Develop robust feature selection methods for EEG analysis
- Build and validate machine learning models for AD detection
- Analyze the trade-offs between different classification approaches

## Repository Structure

```
.
├── data/                      # Data directory (not included in repo)
│   ├── processed_data.csv
│   ├── X_ml.csv
│   └── y_ml.csv
├── results/                   # Results and model outputs
│   ├── models/
│   ├── plots/
│   └── metrics/
├── exploration.py            # Initial data exploration
├── preprocessing.py          # Data preprocessing pipeline
├── feature_engineering.py    # Feature selection and engineering
├── model_training.py         # Model training and evaluation
├── report.md                 # Detailed analysis report
└── README.md                 # This file
```

## Main Findings

The analysis identified several key frequency bands that distinguish AD patients from controls:

1. Theta (3-5 Hz) - Global
2. Delta (1-3 Hz) - Temporal regions
3. Alpha (10-13 Hz) - Central regions
4. Beta (13-20 Hz) - Parietal regions

The best performing model (CatBoost) achieved:

- 82% sensitivity
- 85% specificity
- 0.90 ROC-AUC score

## Requirements

- Python 3.8+
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - catboost
  - optuna
  - imbalanced-learn

## Usage

1. Data Preprocessing:

```bash
python preprocessing.py
```

2. Feature Engineering:

```bash
python feature_engineering.py
```

3. Model Training:

```bash
python model_training.py
```

## Results

Detailed results can be found in the `report.md` file, which includes:

- Complete feature importance analysis
- Model performance metrics
- Clinical implications
- Methodological trade-offs
