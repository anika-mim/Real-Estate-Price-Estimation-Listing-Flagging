import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

print("Current working dir ->", os.getcwd())

# ─────────────────────────────────────────────
# 1. Load data
df = pd.read_csv(
    r"C:\Users\user\OneDrive\Desktop\My Work\Data Science Projects\Project\real_estate_model\realtor-data.csv"
)
print("[OK] Data loaded successfully. Shape:", df.shape)

# quick peek
print("Columns:", df.columns.tolist()[:10])
print(df.head())

# ─────────────────────────────────────────────
# 2. Clean & prepare
df = df.dropna(subset=['price', 'bed', 'bath', 'house_size'])
df['bed']        = df['bed'].astype(int)
df['bath']       = df['bath'].astype(float)
df['house_size'] = df['house_size'].astype(float)
df['acre_lot']   = df['acre_lot'].fillna(0)
print("[CLEANED] Data cleaned and types fixed")

# ─────────────────────────────────────────────
# 3. Feature engineering
df = pd.get_dummies(df, columns=['state'], drop_first=True)
print("[ENGINEERED] One-hot encoding complete")

# ─────────────────────────────────────────────
# 4. Train / test split
X = df[['bed', 'bath', 'house_size', 'acre_lot']
       + [c for c in df.columns if c.startswith('state_')]]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print("[SPLIT] Train/Test created")

# ─────────────────────────────────────────────
# 5. Train model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("[MODEL] Training complete")

# ─────────────────────────────────────────────
# 6. Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"[RESULT] Mean Absolute Error = ${mae:,.0f}")

# ─────────────────────────────────────────────
# 7. Flag listings
df["predicted_price"] = model.predict(X)

def flag(row):
    diff = (row["price"] - row["predicted_price"]) / row["predicted_price"]
    if diff >  0.15: return "Overpriced"
    if diff < -0.15: return "Underpriced"
    return "Fair"

df["price_flag"] = df.apply(flag, axis=1)
print("[FLAGGED] Listings labelled Overpriced / Fair / Underpriced")

print(df[["price", "predicted_price", "price_flag"]].head())
