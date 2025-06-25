# Customer Churn Prediction

A machine learning project that predicts customer churn using purchase history data with 99%+ accuracy across multiple algorithms.

## 📊 Project Overview

This project analyzes customer purchase patterns to predict churn behavior using synthetic purchase data. The analysis includes comprehensive data preprocessing, feature engineering, class imbalance handling, and evaluation of multiple machine learning models.

## 🎯 Key Results

- **Best Model**: Logistic Regression with **99.62% accuracy**
- **Dataset**: 1,329 customers (707 churned, 622 active)
- **Class Balance**: Handled using SMOTE oversampling
- **Features**: Tenure, purchase frequency, recency, and intervals

## 📁 Dataset Features

| Feature | Description |
|---------|-------------|
| `PARTY_NAME` | Customer identifier |
| `DATE` | Purchase dates |
| `tenure_days` | Duration between first and last purchase |
| `total_purchases` | Count of customer purchases |
| `expected_interval` | Average days between purchases |
| `days_since_last_purchase` | Days since last purchase (relative to 2025-06-22) |

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

# Install required packages
pip install pandas matplotlib scikit-learn imbalanced-learn xgboost openpyxl
```

## 🚀 Quick Start

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_excel("synthetic_purchase_data.xlsx")

# Feature engineering and preprocessing
# (See full code in the analysis notebook)

# Best performing model
best_model = LogisticRegression(
    C=0.01, 
    max_iter=100, 
    penalty=None, 
    solver='lbfgs'
)

# Train and predict
best_model.fit(X_resampled, y_resampled)
predictions = best_model.predict(X_test)
```

## 📈 Model Performance

| Model | Cross-Val Accuracy | Test Accuracy | Precision | Recall | F1-Score |
|-------|-------------------|---------------|-----------|--------|----------|
| **Logistic Regression** | **99.91%** | **99.62%** | **99.5%** | **99.5%** | **99.5%** |
| SVM | 99.66% | 99.25% | 99.0% | 99.5% | 99.0% |
| Random Forest | 99.06% | 99.25% | 99.0% | 99.5% | 99.0% |
| XGBoost | - | 99.25% | 99.0% | 99.5% | 99.0% |

## 🛠️ Methodology

### 1. Data Preprocessing
- **Feature Engineering**: Created tenure, purchase frequency, and recency metrics
- **Scaling**: Applied StandardScaler to numerical features
- **Encoding**: Binary encoding for churn status (1=Churn, 0=Active)

### 2. Class Imbalance Handling
![alt text](image-4.png)

- Applied **SMOTE** (Synthetic Minority Oversampling Technique)
- Balanced dataset to 584 samples per class
- Improved model performance on minority class

### 3. Model Selection & Tuning
- **GridSearchCV** for hyperparameter optimization
- **5-fold cross-validation** for robust evaluation
- Tested multiple algorithms: Logistic Regression, SVM, Random Forest, XGBoost

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Classification reports and confusion matrices
- Cross-validation scores for model stability

## 📊 Key Insights
![alt text](image-3.png)

### Customer Distribution
- **53.2%** of customers have churned
- **46.8%** remain active
- Clear patterns in purchase behavior distinguish churned vs. active customers

### Churn Indicators
- Extended periods without purchases (beyond expected interval)
- Short tenure with declining purchase frequency
- Purchase recency is a strong predictor

## 🎯 Business Recommendations

### 1. Model Implementation
- Deploy **Logistic Regression model** for production use
- Advantages: High accuracy, interpretability, fast inference
- Monitor model performance and retrain periodically

### 2. Early Warning System
- Track customers exceeding expected purchase intervals
- Focus retention efforts on high-risk segments
- Implement automated alerts for churn probability > 0.7

### 3. Feature Enhancement
Consider adding these features in future iterations:
- Purchase amounts and transaction values
- Product categories and preferences
- Customer demographics and acquisition channels
- Seasonal purchase patterns

## 📁 Project Structure

```
customer-churn-prediction/
├── README.md
├── requirements.txt
├── data/
│   └── synthetic_purchase_data.xlsx
├── notebooks/
│   └── churn_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
├── models/
│   └── best_model.pkl
└── results/
    ├── model_comparison.png
    └── feature_importance.png
```

## 🔄 Reproduction Steps

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Analysis**
   ```bash
   jupyter notebook notebooks/churn_analysis.ipynb
   ```

3. **Train Models**
   ```python
   python src/model_training.py
   ```

4. **Evaluate Results**
   ```python
   python src/evaluation.py
   ```

## 📋 Requirements

- Python 3.7+
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- imbalanced-learn >= 0.8.0
- xgboost >= 1.5.0
- matplotlib >= 3.4.0
- openpyxl >= 3.0.0

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - [GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Dataset: Synthetic purchase data generated for analysis
- Libraries: scikit-learn, imbalanced-learn, pandas, matplotlib
- Inspiration: Customer retention and churn prediction best practices

---

**Note**: This project uses synthetic data for demonstration purposes. Results may vary with real-world datasets.