import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import argparse
import time

DATASET_PATH = "water_potability_preprocessing.csv"
TEST_SIZE = 0.25
MODEL_NAME = "water_potability"

param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [1, 5, 25]
}

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="Default")
args = parser.parse_args()

# mlflow.set_experiment(args.experiment_name)

client = MlflowClient()

# Load data
data = pd.read_csv(DATASET_PATH)
X = data.drop(columns=['Potability'], axis=1)
y = data["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42
)

input_example = X_train.iloc[:5]

rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, param_grid, scoring="accuracy", cv=3)

is_manual_run = not mlflow.active_run()

if is_manual_run:
    mlflow.start_run(run_name="train_tuning")

# Run Training
active_run = mlflow.active_run()
run_id = active_run.info.run_id

print("RUN ID:", run_id)

try:
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    mlflow.log_params(best_params)
    mlflow.log_param("test_size", TEST_SIZE)

    y_pred = best_model.predict(X_test)

    accuracy = best_model.score(X_test, y_test)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="best_model",
        input_example=input_example
    )

    # Regis Model
    model_uri = f"runs:/{run_id}/best_model"
    model_reg = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    version = model_reg.version

    print("MODEL VERSION REGISTERED:", version)

finally:
    if is_manual_run:
        mlflow.end_run()