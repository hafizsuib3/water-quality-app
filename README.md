![Water Quality Prediction](Water%20Banner.png)

## Table of Contents
- [About Project](#about-project)  
- [Background Overview](#background-overview)  
- [Problem Statement](#problem-statement)  
- [Objective](#objective)  
- [Built With](#built-with)  
- [Data Source](#data-source)  
- [Methodology](#methodology)  
- [Result and Impact](#result-and-impact)  
- [Challenges and Solutions](#challenges-and-solutions)  
- [Sneak Peek of the App](#sneak-peek-of-the-app-)

---

## About Project

This project trains and deploys a machine learning model to classify **Water Quality Index (WQI)** into categories **A–E** using physicochemical and microbiological parameters.  

It includes:  
- A training pipeline (`train_best_model.py`) that evaluates several classifiers and saves the best model.  
- A **Streamlit web app** (`app.py`) for interactive inference, enabling users to upload data and view predicted WQI classes.  

---

## Background Overview

Water quality monitoring supports public health, environmental stewardship, and resource planning. Traditional assessments can be time-consuming and require expert interpretation.  

**WQI** condenses multiple water parameters into an interpretable score and class, but manual computation and large-scale analysis are not always practical.  

By applying supervised learning to routine monitoring data, this project seeks to **automate WQI classification**, making screening faster and more consistent while preserving transparency on features driving predictions.

---

## Problem Statement

Organizations collect large volumes of water quality measurements, yet turning raw parameters into consistent WQI classes is:  
- Resource-intensive  
- Susceptible to delays  
- Limited by class imbalance and lack of accessible predictive tools  

---

## Objective

This project aims to:  
1. **Develop and evaluate predictive models** using supervised machine learning to classify water quality into WQI categories (A–E).  
2. **Select the most effective algorithm** based on performance on unseen data.  
3. **Deploy the trained model** via a simple, user-friendly **Streamlit web application**.  

Ultimately, the goal is to enable **faster, scalable, and accessible WQI classification** that can support researchers, policymakers, and water resource managers.

---

## Built With

- **Python** (pandas, numpy, scikit-learn, imbalanced-learn)  
- **Streamlit** (for deployment)  
- **Joblib** (for model persistence)  

---

## Data Source

- The dataset used is:  
  `SW_CWC_UttarPradesh2007-2021-SELECTEDPARAMETERS-REMOVEDCOLORODOR-DATACLEANED-OUTLIERSREMOVED-WQI.csv`  

- It includes physicochemical and microbiological parameters such as:  
  - pH  
  - Dissolved Oxygen  
  - Biochemical Oxygen Demand  
  - Conductivity  
  - Nutrients  
  - Coliform counts  

- If **WQI** and **WQI_Classification** are not present in the dataset, they are automatically calculated by the training script using standard permissible and ideal limits.

---

## Methodology

**1. Data Preparation**  
- Load CSV and verify expected columns.  
- Compute **WQI** and **WQI_Classification** if missing.  
- Split dataset (80% training, 20% testing).  
- Apply **SMOTE** to balance minority classes.  

**2. Feature and Target Setup**  
- **Features (X):** All water-quality parameters.  
- **Target (y):** Encoded WQI class (A–E).  

**3. Models Compared**  
- Support Vector Machine (SVM)  
- Decision Tree (DT)  
- K-Nearest Neighbor (KNN)  
- Random Forest (RF)  
- Gradient Boosting (GB)  
- Multi-Layer Perceptron (MLP)  

**4. Model Selection**  
- Evaluate all models on the test split.  
- Select the **best accuracy** model.  
- Save model (`best_wqi_model.pkl`) and feature order (`feature_columns.pkl`).  

**5. Deployment**  
- Deploy the saved model with **Streamlit**.  
- Users can upload CSV files and instantly see predicted WQI classes.

---

## Result and Impact

- **Automated classification** of water quality samples into classes **A–E**.  
- **Balanced training** with SMOTE improved recognition of minority classes.  
- **Best model** (experiment-dependent, e.g., Random Forest or MLP) selected based on test accuracy.  
- **Deployment** allows non-technical users to interact with the model via a web interface.  

> *(You can update this section with actual test accuracy, F1-scores, and confusion matrix images after running your experiments.)*

---

## Challenges and Solutions

**Imbalanced Classes**  
- Challenge: Some WQI classes are underrepresented.  
- Solution: Applied **SMOTE** and used robust evaluation metrics.  

**Feature Variability**  
- Challenge: Column names and units differ across datasets.  
- Solution: Standardized naming and included fallback WQI computation.  

**Generalization**  
- Challenge: Model may overfit to one region.  
- Solution: Apply cross-validation and maintain strict train/test split.  

**Usability**  
- Challenge: Researchers need an easy interface.  
- Solution: **Streamlit app** for interactive use.  

---

## Sneak Peek of the App 

<p align="center">
  <img src="Preview04.gif" alt="Water Quality App Preview" style="width:70%;">
</p>

<p align="center"><em>A lightweight, user-friendly Streamlit app for fast WQI predictions.</em></p>

---
