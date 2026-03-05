# Stroke Risk Prediction – Machine Learning Capstone Project

**Early identification of stroke risk using demographic, clinical, and lifestyle features**

![Stroke Risk Banner](https://via.placeholder.com/1200x400/007BFF/FFFFFF?text=Stroke+Risk+Prediction+Capstone)  

## Project Overview

Stroke remains one of the leading causes of death and long-term disability worldwide. This project develops a **binary classification model** to predict whether a patient is at risk of having a stroke based on easily collectible features (age, hypertension, glucose levels, BMI, smoking status, etc.).

The goal is to:
- Perform exploratory data analysis (EDA) to uncover key risk factors
- Build and compare interpretable and high-performing models
- Provide actionable insights for preventive healthcare strategies

Dataset: [Healthcare Dataset Stroke Data](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (~5,110 records, highly imbalanced: ~5% positive stroke cases)

## Table of Contents

- [Dataset Description](#dataset-description)
- [Project Workflow](#project-workflow)
- [Key Decisions & Improvements](#key-decisions--improvements)
- [Results](#results)
- [Feature Importance](#feature-importance)
- [Limitations & Future Work](#limitations--future-work)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Author & Acknowledgments](#author--acknowledgments)

## Dataset Description

- **Rows**: 5,110 patients
- **Target**: `stroke` (1 = stroke occurred, 0 = no stroke) → ~4.87% positive class
- **Features** (12 total):
  - Demographic: `age`, `gender`, `ever_married`, `work_type`, `Residence_type`
  - Clinical: `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`
  - Lifestyle: `smoking_status`
- Missing values: only `bmi` (~201 missing, 3.9%)

## Project Workflow

1. **Data Loading & Cleaning**
   - Dropped `id` column
   - Removed rare `gender = Other` rows
   - Handled missing BMI values intelligently

2. **Feature Engineering**
   - Created `bmi_missing` indicator flag (strong signal of high-risk patients)
   - Imputed BMI using **age-group-specific medians** (0–17, 18–29, ..., 80+)
   - Added clinically meaningful binary features:
     - `age_over_60`
     - `high_glucose` (≥170 mg/dL)
     - `high_risk_combo` = hypertension + heart_disease + age_over_60

3. **Exploratory Data Analysis**
   - Visualized class imbalance
   - Age, glucose, and BMI distributions by stroke status
   - Stroke rates across categorical variables (hypertension, smoking, work type, etc.)

4. **Preprocessing Pipeline**
   - Numerical features: median imputation + standardization
   - Categorical features: one-hot encoding
   - Used `ColumnTransformer` for clean, reproducible transformations

5. **Handling Imbalance**
   - Applied **SMOTE** (Synthetic Minority Oversampling Technique) **inside cross-validation** to avoid data leakage

6. **Modeling & Evaluation**
   - Compared three models using **Repeated Stratified 5-Fold CV (3 repeats)**:
     - Logistic Regression (class_weight="balanced")
     - Random Forest (200 trees)
     - XGBoost (tuned hyperparameters, AUC-PR focused)
   - Primary metrics: **PR-AUC**, F1-score, Recall, Precision (accuracy ignored due to imbalance)

7. **Final Model**
   - Trained XGBoost on full dataset
   - Extracted and visualized feature importances

## Key Decisions & Improvements

| Decision                              | Rationale                                                                                   |
|---------------------------------------|---------------------------------------------------------------------------------------------|
| Age-group median BMI imputation       | Global median would bias high-risk older patients downward; missingness is MNAR/MAR         |
| `bmi_missing` flag                    | Missing BMI strongly correlates with older age & higher stroke probability (~20% vs 5%)    |
| SMOTE inside CV pipeline              | Prevents target leakage from test set into training                                         |
| Focus on PR-AUC & recall              | ROC-AUC is misleading on highly imbalanced data; recall critical in medical screening       |
| Added clinical binary features        | Improves interpretability and captures known risk synergies                                 |
| XGBoost as final model                | Typically best performance + native handling of missing values & categorical features       |

## Results

*(Insert your actual cross-validation numbers here once you run `evaluate_model()` for all three)*

**Example CV Performance (XGBoost – 15 folds total)**

- PR-AUC:          0.XX ± 0.XX  
- F1-score:        0.XX ± 0.XX  
- Recall:          0.XX ± 0.XX  
- Precision:       0.XX ± 0.XX  
- ROC-AUC:         0.XX ± 0.XX  

**Top 12 Most Important Features** (from final XGBoost model)

1. age                        0.170  
2. work_type_children         0.107  
3. gender_Male                0.087  
4. smoking_status_formerly smoked 0.080  
5. work_type_Private          0.071  
6. smoking_status_smokes      0.069  
7. smoking_status_never smoked 0.059  
8. work_type_Self-employed    0.058  
9. bmi_missing                0.049  
10. ever_married              0.044  
11. Residence_type            0.040  
12. bmi                       0.039  

→ Age dominates, followed by protective child status, smoking history, and the powerful `bmi_missing` signal.

## Limitations & Future Work

- SMOTE may generate unrealistic synthetic samples
- No external validation / prospective data
- Dataset from ~2015–2020 → population shifts possible
- No cost-sensitive learning (false negatives much more expensive than false positives)

**Future improvements**
- SHAP / LIME explanations for individual predictions
- Threshold optimization for high-recall operating point
- Ensemble methods or neural networks (TabNet, simple MLP)
- Integration with real hospital data pipelines

## How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/stroke-risk-prediction.git
   cd stroke-risk-prediction
   ```

2. Install dependenciesBash
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook
   ```bash
   jupyter notebook capstone_project.ipynb
   ```

   ## Technologies Used

   - Python 3.10+
   - pandas, numpy
   - scikit-learn, xgboost, imbalanced-learn
   - matplotlib, seaborn
   - Jupyter Notebook

     ## Authors & Acknowledgments

   - Data Detectives
   - Dataset source: Kaggle – Stroke Prediction Dataset
     Special thanks to open-source ML community and healthcare data providers.
