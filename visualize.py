# visualize.py
# -------------------------------------------------
# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# -------------------------------------------------
# 2. Load and clean the raw CSV
RAW_CSV = r"C:\Users\user\OneDrive\Desktop\My Work\Data Science Projects\Project\real_estate_model\realtor-data.csv"
df = pd.read_csv(RAW_CSV)

# Keep only rows with the columns we need
df = df.dropna(subset=['price', 'bed', 'bath', 'house_size'])
df['bed']        = df['bed'].astype(int)
df['bath']       = df['bath'].astype(float)
df['house_size'] = df['house_size'].astype(float)
df['acre_lot']   = df['acre_lot'].fillna(0)

# One-hot encode STATE (not city) so we don’t explode the column count
df = pd.get_dummies(df, columns=['state'], drop_first=True)

# -------------------------------------------------
# 3. Build feature matrix X and target y
X = df[['bed', 'bath', 'house_size', 'acre_lot']
       + [c for c in df.columns if c.startswith('state_')]]
y = df['price']

# Train/test split to get y_test for visualisation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# 4. Load model if it exists, otherwise train a quick one
MODEL_PATH = "rf_price_model.joblib"

if os.path.exists(MODEL_PATH):
    print("[INFO] Loading saved model:", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
else:
    print("[INFO] No saved model found – training a small model now ...")
    model = RandomForestRegressor(
        n_estimators=20, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)   # save for next time
    print("[INFO] Model trained and saved to:", MODEL_PATH)

# -------------------------------------------------
# 5. Predict on the test set
y_pred = model.predict(X_test)

# -------------------------------------------------
# 6. Visualisations
## 6-a  Scatter: Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.2, edgecolor='k')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.tight_layout()
plt.show()

## 6-b  Error histogram
errors = y_test - y_pred
plt.figure(figsize=(6,4))
plt.hist(errors, bins=60, color='skyblue', edgecolor='black')
plt.xlabel("Prediction Error (Actual − Predicted)")
plt.title("Prediction Error Distribution")
plt.tight_layout()
plt.show()

## 6-c  Top 15 feature importances
importances = model.feature_importances_
idx = np.argsort(importances)[::-1][:15]
plt.figure(figsize=(8,4))
plt.bar(range(len(idx)), importances[idx])
plt.xticks(range(len(idx)),
           [X.columns[i] for i in idx],
           rotation=45, ha='right')
plt.ylabel("Importance")
plt.title("Top 15 Features")
plt.tight_layout()
plt.show()

## 6-d  Bar chart of price flags  (need flags first)
def flag(row):
    diff = (row['price'] - model.predict(row[X.columns].values.reshape(1,-1))[0]) \
           / model.predict(row[X.columns].values.reshape(1,-1))[0]
    if diff > 0.15:  return "Overpriced"
    if diff < -0.15: return "Underpriced"
    return "Fair"

sample = df.sample(100_000, random_state=42)  # smaller subset for speed
sample['flag'] = sample.apply(flag, axis=1)

sample['flag'].value_counts().plot(kind='bar', figsize=(4,4),
                                   color=['red','green','orange'])
plt.ylabel("Count")
plt.title("Listings by Price Category")
plt.tight_layout()
plt.show()
