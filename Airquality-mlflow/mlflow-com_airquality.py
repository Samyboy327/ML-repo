import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
from datetime import datetime

# ---------------- CONFIG ----------------
DATA_PATH = "airquality.csv"
EXPERIMENT_NAME = "mlops-airquality-model-comparison"
# ---------------------------------------


def preprocess_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)

    numeric_cols = [
        "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
        "CO", "SO2", "O3", "Benzene", "Toluene",
        "Xylene", "AQI"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["City", "Date", "AQI_Bucket"])

    for col in numeric_cols:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
            df[col] = df.groupby("City")[col].transform(
                lambda x: x.fillna(x.median())
            )
            df[col] = df[col].fillna(df[col].median())

    le = LabelEncoder()
    df["AQI_Bucket"] = le.fit_transform(df["AQI_Bucket"].astype(str))

    X = df.drop(columns=["City", "Date", "AQI_Bucket"])
    y = df["AQI_Bucket"]

    return train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba, multi_class="ovr")
    }


def main():
    X_train, X_test, y_train, y_test = preprocess_data(DATA_PATH)
    mlflow.set_experiment(EXPERIMENT_NAME)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multiclass",
            random_state=42
        )
    }

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)

            metrics = evaluate_model(model, X_test, y_test)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            mlflow.log_param("model_type", model_name)
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"\n✅ {model_name} Results")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
