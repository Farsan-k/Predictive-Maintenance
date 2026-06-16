import os
import joblib
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

from preprocessing import preprocessing_pipeline

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("data.csv")

X = df.drop(columns=[
    "Product ID",
    "UDI",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
    "Machine failure"
])

y = df["Machine failure"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# PREPROCESSING
# =====================================================

X_train_processed = preprocessing_pipeline.fit_transform(X_train)
X_test_processed = preprocessing_pipeline.transform(X_test)

# =====================================================
# HYPERPARAMETER TUNING
# =====================================================

best_f1 = -1
best_eps = None
best_min_samples = None

eps_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
min_samples_values = [3, 5, 7, 10, 15]

print("=" * 70)
print("DBSCAN HYPERPARAMETER TUNING")
print("=" * 70)

for eps in eps_values:
    for min_samples in min_samples_values:

        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

        train_clusters = dbscan.fit_predict(X_train_processed)

        anomaly_train = (train_clusters == -1).astype(int)

        precision = precision_score(
            y_train,
            anomaly_train,
            zero_division=0
        )

        recall = recall_score(
            y_train,
            anomaly_train,
            zero_division=0
        )

        f1 = f1_score(
            y_train,
            anomaly_train,
            zero_division=0
        )

        print(
            f"eps={eps:.1f} | "
            f"min_samples={min_samples} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_eps = eps
            best_min_samples = min_samples

print("\n" + "=" * 70)
print("BEST PARAMETERS")
print("=" * 70)

print(f"eps = {best_eps}")
print(f"min_samples = {best_min_samples}")
print(f"Best F1 = {best_f1:.4f}")

# =====================================================
# TRAIN FINAL MODEL
# =====================================================

final_dbscan = DBSCAN(
    eps=best_eps,
    min_samples=best_min_samples
)

final_clusters = final_dbscan.fit_predict(X_train_processed)

anomaly_pred = (final_clusters == -1).astype(int)

# =====================================================
# EVALUATION (TRAIN SET)
# =====================================================

precision = precision_score(
    y_train,
    anomaly_pred,
    zero_division=0
)

recall = recall_score(
    y_train,
    anomaly_pred,
    zero_division=0
)

f1 = f1_score(
    y_train,
    anomaly_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_train,
    anomaly_pred
)

tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 70)
print("FINAL MODEL RESULTS")
print("=" * 70)

print(f"TP = {tp}")
print(f"FP = {fp}")
print(f"FN = {fn}")
print(f"TN = {tn}")

print(f"\nPrecision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")
print(f"F1 Score  = {f1:.4f}")

# =====================================================
# SAVE MODEL
# =====================================================

os.makedirs("model", exist_ok=True)

joblib.dump(
    preprocessing_pipeline,
    "model/preprocessing_pipeline.pkl"
)

joblib.dump(
    final_dbscan,
    "model/dbscan_model.pkl"
)

print("\nSaved:")
print("model/preprocessing_pipeline.pkl")
print("model/dbscan_model.pkl")