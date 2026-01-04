import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import os
from datetime import datetime


# ------------------ CONFIG ------------------
DATA_PATH = "airquality.csv"         # path to airquality dataset
EXPERIMENT_NAME = "mlops-airquality"
MODEL_NAME = "mlops-airquality"
RUN_PREFIX = "mlops-airquality"
os.makedirs("data", exist_ok=True)
# --------------------------------------------


def preprocess_data(path):
    """
    Read airquality CSV, advanced cleaning, split into train/test.

    Expected columns (at least):
    City, Date, PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3,
    Benzene, Toluene, Xylene, AQI, AQI_Bucket
    """
    df = pd.read_csv(path)

    # Parse date and create simple date features
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    # Drop Date and AQI_Bucket (target is AQI)
    df = df.drop(columns=["Date", "AQI_Bucket"])

    # Target: AQI (numeric)
    y = df["AQI"]
    X = df.drop(columns=["AQI"])

    # Identify categorical and numeric columns
    categorical_cols = ["City"]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    # Preprocessor: One-hot for categorical, imputer for numeric
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_cols)
        ]
    )

    # Fit and transform X
    X_processed = preprocessor.fit_transform(X)

    # Train/test split (no stratify for regression)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test, preprocessor


def train_model(X_train, y_train, n_estimators=100, max_depth=5):
    """Train RandomForestRegressor wrapped in a StandardScaler pipeline."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model, X_test, y_test):
    """Evaluate multi-class model and return metrics."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
    }
    return metrics


def main():
    # Load & preprocess
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(DATA_PATH)

    # Create experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Timestamp for unique run
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{RUN_PREFIX}-{timestamp}"

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        n_estimators = 100
        max_depth = 5
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Train model
        model = train_model(X_train, y_train, n_estimators, max_depth)

        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Register model
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, MODEL_NAME)

        print(f"\n✅ Run completed: {run.info.run_id}")
        print(f"📊 Metrics: {metrics}")
        print(f"🔖 Registered model: {MODEL_NAME}")
        print("View the experiment in MLflow UI at http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
