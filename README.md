# Diabetes Prediction - XGBoost Model

This project implements an XGBoost Classifier to predict diabetes presence based on medical history and health metrics.

## 📂 Project Structure

```
XGBoost_Model/
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Processed & balanced data
├── models/
│   └── run_YYYY-MM-DD/      # Saved models (timestamped)
├── reports/
│   └── run_YYYY-MM-DD/      # Metrics and Figures
│       ├── figures/
│       ├── evaluation_results.json
│       └── predictions.csv
├── src/
│   └── train_model.py       # Main training script
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## 🚀 Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Training**:
   ```bash
   python src/train_model.py
   ```
   Each run creates a new timestamped folder in `models/` and `reports/` to store the results.

## 📊 Features
- **Data Preprocessing**: Handling duplicates, encoding categorical variables, scaling.
- **Imbalance Handling**: Uses SMOTE and class weighting.
- **Model**: XGBoost Classifier.
- **Evaluation**: ROC-AUC, Precision-Recall, Confusion Matrix.
