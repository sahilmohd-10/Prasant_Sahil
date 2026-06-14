

import time
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

DATA_PATH  = "Laptop_Dataset_Prasant.csv"
MODEL_PATH = "model.pkl"
RANDOM_STATE = 42
COLUMNS = ["GPU", "CPU", "RAM", "SSD", "Battery"]
TARGET  = "Buy"


# ── 1. Load ──────────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATA_PATH)
print(f"  Shape : {df.shape}")
print(f"  Buy=1 : {df[TARGET].sum():,}  |  Buy=0 : {(df[TARGET]==0).sum():,}\n")


# ── 2. Encode ─────────────────────────────────────────────────────────────────
# Fit one LabelEncoder per categorical column and keep them for inference.
encoders: dict[str, LabelEncoder] = {}
X = pd.DataFrame()

for col in COLUMNS:
    le = LabelEncoder()
    X[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

y = df[TARGET].values


# ── 3. Split ──────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f"Train : {len(X_train):,}  |  Test : {len(X_test):,}\n")


# ── 4. Train Random Forest ────────────────────────────────────────────────────
print("Training Random Forest …")
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators     = 300,       # enough trees for stable predictions
    max_depth        = 20,        # deep enough to capture combos
    min_samples_leaf = 5,         # avoids overfitting noise
    max_features     = "sqrt",    # standard for RF classifiers
    class_weight     = "balanced",# handles ~74/26 imbalance
    n_jobs           = -1,        # use all CPU cores
    random_state     = RANDOM_STATE,
)
rf.fit(X_train, y_train)
print(f"  Done in {time.time()-t0:.1f}s\n")


# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

acc    = accuracy_score(y_test, y_pred)
auc    = roc_auc_score(y_test, y_prob)
cm     = confusion_matrix(y_test, y_pred)

print("=" * 52)
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"  Confusion Matrix:")
print(f"              Pred=0   Pred=1")
print(f"    Actual=0  {cm[0,0]:>6}   {cm[0,1]:>6}")
print(f"    Actual=1  {cm[1,0]:>6}   {cm[1,1]:>6}")
print("=" * 52)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Don't Buy", "Buy"]))


# ── 6. Feature Importance ─────────────────────────────────────────────────────
fi = pd.Series(rf.feature_importances_, index=COLUMNS).sort_values(ascending=False)
print("Feature Importances:")
for feat, imp in fi.items():
    bar = "█" * int(imp * 60)
    print(f"  {feat:<10} {imp:.4f}  {bar}")
print()


# ── 7. Save ───────────────────────────────────────────────────────────────────
payload = {"model": rf, "encoders": encoders, "columns": COLUMNS}
joblib.dump(payload, MODEL_PATH)
print(f"Model saved → {MODEL_PATH}")