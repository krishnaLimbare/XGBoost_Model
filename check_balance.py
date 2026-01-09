import pandas as pd
import os

BASE_DIR = r"c:\Users\Asus\Downloads\XGBoost_Model"
Y_TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'y_train.csv')

y_train = pd.read_csv(Y_TRAIN_PATH)
counts = y_train.iloc[:, 0].value_counts()

print("--- CLASS COUNTS ---")
print(counts)
print(f"Ratio (Neg/Pos): {counts[0]/counts[1]:.2f}")
