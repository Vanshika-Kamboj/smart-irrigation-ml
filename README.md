# 🌾 Smart Irrigation Prediction System using Machine Learning

A complete Machine Learning project that predicts irrigation needs for agricultural fields using classification, regression, clustering, and hyperparameter tuning.

---

## 📌 Project Overview

This project solves a real-world agricultural problem — helping farmers decide:
- **When** to irrigate (Classification)
- **How much** water to use (Regression)
- **Which fields** are similar (Clustering)

---

## 🎯 Project Topics Covered

| # | Topic | Description |
|---|-------|-------------|
| 1 | Irrigation Need Classification | Predicts High / Medium / Low irrigation need |
| 2 | Water Usage Regression | Predicts exact water amount in mm |
| 3 | Random Forest Tuning | Best model using RandomizedSearchCV |
| 4 | Field Cluster Discovery | Groups similar fields using KMeans |

---

## 📂 Dataset

- **Source:** Kaggle — Irrigation Water Requirement Prediction Dataset
- **Rows:** 10,000
- **Columns:** 20
- **Target:** Irrigation_Need (High / Medium / Low)

### Features Used:
- Soil Type, Soil pH, Soil Moisture
- Crop Type, Crop Growth Stage
- Temperature, Humidity, Rainfall
- Season, Region, Wind Speed
- Field Area, Previous Irrigation

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python | Programming Language |
| Pandas & NumPy | Data Processing |
| Scikit-learn | ML Models |
| Matplotlib & Seaborn | Visualization |
| Streamlit | Web App |
| Joblib | Save/Load Models |
| Kaggle | Dataset Source |

## 📁 Project Structure

smart-irrigation-ml/
│
├── app.py                    # Streamlit Web App
├── step1_load_data.py        # Data Loading & EDA
├── step2_preprocess.py       # Data Preprocessing
├── step3_classification.py   # Classification Model
├── step4_regression.py       # Regression Model
├── step5_tuning.py           # Random Forest Tuning
├── step6_clustering.py       # KMeans Clustering
├── irrigation_prediction.csv # Dataset
└── README.md                 # Project Description

---

## 🚀 How to Run This Project

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YourUsername/smart-irrigation-ml.git
cd smart-irrigation-ml
```

### Step 2 — Install Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

### Step 3 — Run Each Step in Order
```bash
python step1_load_data.py
python step2_preprocess.py
python step3_classification.py
python step4_regression.py
python step5_tuning.py
python step6_clustering.py
```

### Step 4 — Launch Web App
```bash
streamlit run app.py
```

### Step 5 — Open Browser

---


