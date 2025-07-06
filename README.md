# 🏥 Predicting Hospital Readmissions for Diabetic Patients

This project aims to predict whether diabetic patients will be readmitted to the hospital within 30 days using machine learning models. The analysis is based on the [Diabetes 130-US hospitals for years 1999–2008 dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) from the UCI Machine Learning Repository.

---

## 📌 Project Objective

Hospital readmissions are costly and often preventable. The goal of this project is to develop predictive models that identify high-risk diabetic patients who are likely to be readmitted within 30 days, enabling early intervention and improving the quality of care.

---

## 📁 Dataset Description

- **Source**: UCI Machine Learning Repository  
- **Records**: ~100,000 patient encounters  
- **Features**: Demographic, clinical, and hospitalization data  
- **Target Variable**: `readmitted` (whether the patient was readmitted within 30 days)

🔗 [Dataset Link](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

---

## ⚙️ Preprocessing Steps

- Removed features with excessive missing values (e.g., `weight`, `payer_code`)
- Handled missing and placeholder values (`?`)
- Encoded categorical variables
- Performed feature significance testing using ANOVA and Chi-Square
- Balanced class distribution to improve model recall
- Split data using stratified sampling

---

## 🤖 Models Used

- Logistic Regression (LR)  
- Decision Tree (DT)  
- Random Forest (RF)  
- AdaBoost (ADA)  
- XGBoost (XGB)  
- LightGBM (LGB)

✅ **Best model**: LightGBM, with ~63% accuracy and ~59% recall after preprocessing

---

## 📊 Evaluation Metrics

- Accuracy  
- Precision  
- Recall (Focus metric due to class imbalance)  
- F1-Score  
- ROC AUC

---

## 📈 Key Insights

- Significant predictors include number of inpatient visits, number of diagnoses, length of stay, and number of medications.
- Preprocessing steps, especially class balancing and feature selection, significantly improved model performance.
- LightGBM outperformed traditional models like logistic regression and decision trees in recall and AUC.

---
