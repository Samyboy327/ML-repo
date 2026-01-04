import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from datetime import datetime
import os

# ------------------ CONFIG ------------------
DATA_PATH = "airquality.csv"
EXPERIMENT_NAME = "mlops-airquality"
MODEL_NAME = "mlops-airquality"
RUN_PREFIX = "mlops-airquality"
# --------------------------------------------


def preprocess_data(path):
    df = pd.read_csv(path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Parse and sort dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)

    print("Missing values per column:")
    print(df.isna().sum())

    numeric_cols = [
        "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
        "CO", "SO2", "O3", "Benzene", "Toluene",
        "Xylene", "AQI"
    ]

    # Convert to numeric
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing essentials
    df = df.dropna(subset=["City", "Date", "AQI_Bucket"])

    # Remove negative values and re-impute
    for col in numeric_cols:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
            df[col] = df.groupby("City")[col].transform(
                lambda x: x.fillna(x.median())
            )
            df[col] = df[col].fillna(df[col].median())

    # Encode target
    le = LabelEncoder()
    df["AQI_Bucket"] = le.fit_transform(df["AQI_Bucket"].astype(str))

    # Final NaN safety
    df[numeric_cols] = df[numeric_cols].fillna(
        df[numeric_cols].median()
    )

    # Define features and target
    X = df.drop(columns=["City", "Date", "AQI_Bucket"])
    y = df["AQI_Bucket"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_estimators=100, max_depth=5):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba, multi_class="ovr")
    }
    return metrics


def main():
    X_train, X_test, y_train, y_test = preprocess_data(DATA_PATH)

    mlflow.set_experiment(EXPERIMENT_NAME)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{RUN_PREFIX}-{timestamp}"

    with mlflow.start_run(run_name=run_name) as run:
        n_estimators = 100
        max_depth = 5

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        model = train_model(X_train, y_train, n_estimators, max_depth)

        metrics = evaluate_model(model, X_test, y_test)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(model, artifact_path="model")

        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, MODEL_NAME)

        print(f"\n✅ Run completed: {run.info.run_id}")
        print(f"📊 Metrics: {metrics}")
        print(f"🔖 Registered model: {MODEL_NAME}")
        print("🌐 MLflow UI: http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
