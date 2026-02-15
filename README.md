# Hybrid Clinical & Machine Learning Framework for Maternal Health Risk Prediction

This repository contains the code and supporting files for a hybrid machine learning–based system to predict maternal health risk levels using real-world clinical and demographic datasets. The project combines traditional clinical threshold rules with machine learning models to classify pregnancy risk levels into low, mid, and high categories, and applies explainability techniques to support transparent decision-making.



---

## Overview

Maternal health risk prediction using data-driven methods can assist healthcare providers in early identification of high-risk pregnancies. This project integrates:

- Clinical threshold rules for basic risk indicators
- Machine learning models trained on real datasets
- Ensemble methods for improved performance
- Explainable AI techniques for interpretability

The end goal is a reliable, interpretable risk prediction framework backed by a research paper.

---

## Repository Contents

| Folder / File | Description |
|---------------|-------------|
| `data/` | Raw and processed data files used for model training and analysis |
| `src/` | Source code for preprocessing, modeling, evaluation, and explainability |
| `models/` | Trained model artifacts and pipeline files |
| `notebooks/` | Jupyter notebooks documenting exploratory analysis and experiments |
| `results/` | Visualizations, metrics, and explainability plots |
| `README.md` | This documentation |

---

## Tech Stack

- **Languages:** Python  
- **Libraries:** scikit-learn, XGBoost, Pandas, NumPy, SHAP, Matplotlib, Seaborn  
- **Approach:** Machine Learning, Ensemble Learning, Explainable AI

---

## Features Used

- Clinical indicators such as blood pressure, BMI, heart rate, etc.
- Socio-demographic features from NFHS and global maternal datasets
- Feature selection using permutation importance and other metrics

---

## Machine Learning Models

Models implemented include:

- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- XGBoost  
- Soft Voting Ensemble

---

## Explainability

To interpret model predictions and derive actionable insights:

- **SHAP (SHapley Additive exPlanations)** for feature contribution  
- **Permutation Importance** to rank feature impact  
- Statistical feature selection techniques

---


