# 🧠 Customer Churn Prediction Analysis

## 📌 Overview
This project focuses on predicting customer churn using purchase history data. The analysis includes data preprocessing, feature engineering, handling class imbalance, and evaluating multiple machine learning models to identify the best approach for churn prediction.

---

## 📊 Dataset
The dataset includes synthetic purchase data with key features:
- `PARTY_NAME`: Customer identifier
- `DATE`: Purchase dates

Derived features:
- `tenure_days`: Duration between first and last purchase
- `total_purchases`: Count of purchases
- `expected_interval`: Average days between purchases
- `days_since_last_purchase`: Days since last purchase (as of 2025-06-22)

---

## 🔍 Data Analysis

### 📌 Customer Status Distribution
![alt text](image-1.png)

- **Churned Customers**: 707 (53.2%)  
- **Active Customers**: 622 (46.8%)

> The dataset shows a slight class imbalance, with more churned customers than active ones.

---

## ⚙️ Data Preprocessing

### 🏗 Feature Engineering
- Calculated customer tenure in days
- Derived expected purchase interval
- Computed days since last purchase
- Labeled churn status based on purchase gap

### ⚖ Feature Scaling
- Standardized features using `StandardScaler`
- Encoded churn labels (Churn = 1, Active = 0)

### 🧪 Class Imbalance Handling
- Applied **SMOTE** oversampling
- Result: **584 samples per class**

![alt text](image-2.png)

---

## 🤖 Model Evaluation

We tested four ML models with hyperparameter tuning:

### 1. ✅ Logistic Regression
- **Best Params**:
  ```python
  {'C': 0.01, 'max_iter': 100, 'penalty': None, 'solver': 'lbfgs'}
  ```
- **Cross-val Accuracy**: 99.91%  
- **Test Accuracy**: 99.62%  
- **Classification Report**:
  ```
              precision    recall  f1-score   support
          0       1.00      0.99      1.00       143
          1       0.99      1.00      1.00       123
  ```

---

### 2. 🧭 Support Vector Machine (SVM)
- **Best Params**:
  ```python
  {'C': 10, 'degree': 2, 'gamma': 'scale', 'kernel': 'linear'}
  ```
- **Cross-val Accuracy**: 99.66%  
- **Test Accuracy**: 99.25%  
- **Classification Report**:
  ```
              precision    recall  f1-score   support
          0       1.00      0.99      0.99       143
          1       0.98      1.00      0.99       123
  ```

---

### 3. 🌲 Random Forest
- **Best Params**:
  ```python
  {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
  ```
- **Cross-val Accuracy**: 99.06%  
- **Test Accuracy**: 99.25%  
- **Classification Report**:
  ```
              precision    recall  f1-score   support
          0       1.00      0.99      0.99       143
          1       0.98      1.00      0.99       123
  ```

---

### 4. ⚡ XGBoost
- **Test Accuracy**: 99.25%  
- **Classification Report**:
  ```
              precision    recall  f1-score   support
          0       1.00      0.99      0.99       143
          1       0.98      1.00      0.99       123
  ```

---

## ✅ Conclusion

- All models achieved **>99% accuracy**
- **Logistic Regression** performed best (99.62%)
- Features like **purchase frequency, recency, and tenure** are strong churn predictors
- SMOTE effectively handled class imbalance

---

## 💡 Recommendations

1. Use **Logistic Regression** for deployment:
   - High accuracy, simple, fast to train
2. Monitor customers with:
   - Long inactivity beyond expected interval
   - Declining frequency after short tenure
3. For future improvements:
   - Include purchase amounts, product categories, and demographics

---

## 🔁 Reproducibility Guide

### 📦 Step 1: Install dependencies
```bash
pip install pandas matplotlib scikit-learn imbalanced-learn xgboost
```

### 🧪 Step 2: Run analysis
```python
import pandas as pd

df = pd.read_excel("synthetic_purchase_data.xlsx")
# [Continue with preprocessing, SMOTE, and model training as shown above]
```

### 🧠 Step 3: Use the best model
```python
from sklearn.linear_model import LogisticRegression

best_model = LogisticRegression(C=0.01, max_iter=100, penalty=None, solver='lbfgs')
best_model.fit(x_resampled, y_resampled)
predictions = best_model.predict(new_data)
```

---

## 📎 License
This project is for educational/demo purposes. Please cite appropriately if reused.

---

## 🙌 Acknowledgements
Thanks to the open-source community for providing the tools to enable this analysis!

---

> 📫 For questions or collaboration, feel free to connect via [LinkedIn](#) or raise an issue.
