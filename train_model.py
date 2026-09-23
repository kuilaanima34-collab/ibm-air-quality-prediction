"""
train_model.py
--------------
Trains a RandomForestRegressor to predict Benzene (C6H6) concentration
and a RandomForestClassifier to predict AQI level.

Outputs saved to models/
  model.pkl              - trained RandomForest pipeline (both reg + clf packed in dict)
  scaler.pkl             - fitted StandardScaler
  metadata.json          - feature list, class labels, train/test scores
  feature_importances.csv - feature importance from the regressor

Run:
    python train_model.py
"""

import os, json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, classification_report

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH   = "Air Quality.csv" if os.path.exists("Air Quality.csv") else os.path.join("data", "Air Quality.csv")
MODELS_DIR  = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

MISSING_SENTINEL = -200
LABEL_MAP        = {0: "Good", 1: "Moderate", 2: "Unhealthy", 3: "Hazardous"}

FEATURE_COLUMNS = [
    "CO(GT)", "PT08.S1(CO)", "NOx(GT)", "NO2(GT)",
    "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
    "Hour", "DayOfWeek", "Month",
]

AQI_POLLUTANTS = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
AQI_WEIGHTS    = [0.30, 0.30, 0.20, 0.20]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def load_and_clean(path):
    df = pd.read_csv(path, sep=",", decimal=".")
    # Drop unnamed trailing columns
    df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], inplace=True)
    # Drop fully empty rows
    df.dropna(how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)
    # Replace -200 sentinel
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace(MISSING_SENTINEL, np.nan)
    # Parse datetime
    df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
    df.drop(columns=["Date", "Time"], inplace=True)
    df["Hour"]      = df["DateTime"].dt.hour
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek
    df["Month"]     = df["DateTime"].dt.month
    # Median impute
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    # Derive AQI score + label
    scores = []
    for col, w in zip(AQI_POLLUTANTS, AQI_WEIGHTS):
        if col in df.columns:
            lo, hi = df[col].min(), df[col].max()
            norm = (df[col] - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=df.index)
            scores.append(norm * w)
    df["AQI_Score"] = sum(scores)
    df["AQI_Level"] = pd.qcut(df["AQI_Score"], q=4, labels=[0, 1, 2, 3]).astype(int)
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train():
    print("=" * 55)
    print("  Air Quality — Model Training")
    print("=" * 55)

    print("\n[1/5] Loading dataset ...")
    df = load_and_clean(DATA_PATH)
    print(f"      {len(df)} rows, {df.shape[1]} columns")

    print("\n[2/5] Splitting features / targets ...")
    X = df[FEATURE_COLUMNS].values
    y_reg = df["C6H6(GT)"].values
    y_cls = df["AQI_Level"].values
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_cls, test_size=0.20, random_state=42
    )
    print(f"      Train: {len(X_train)}  Test: {len(X_test)}")

    print("\n[3/5] Scaling features ...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    print("\n[4/5] Training RandomForestRegressor (C6H6) ...")
    reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg.fit(X_train_sc, yr_train)
    r2 = r2_score(yr_test, reg.predict(X_test_sc))
    print(f"      R2 = {r2:.4f}")

    print("\n[5/5] Training RandomForestClassifier (AQI Level) ...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train_sc, yc_train)
    acc = accuracy_score(yc_test, clf.predict(X_test_sc))
    print(f"      Accuracy = {acc:.4f}")
    print(classification_report(
        yc_test, clf.predict(X_test_sc),
        target_names=list(LABEL_MAP.values())
    ))

    # Save model bundle
    joblib.dump({"regressor": reg, "classifier": clf}, os.path.join(MODELS_DIR, "model.pkl"))

    # Feature importances CSV
    fi = pd.DataFrame({
        "feature":   FEATURE_COLUMNS,
        "importance": reg.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(os.path.join(MODELS_DIR, "feature_importances.csv"), index=False)

    # Metadata JSON
    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "label_map":       {str(k): v for k, v in LABEL_MAP.items()},
        "regression_target": "C6H6(GT)",
        "r2_score":        round(r2, 4),
        "accuracy":        round(acc, 4),
        "n_train":         len(X_train),
        "n_test":          len(X_test),
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n  Models saved to models/")
    print("=" * 55)


if __name__ == "__main__":
    train()
