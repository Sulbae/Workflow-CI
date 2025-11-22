import mlflow
from mlflow.tracking import MlflowClient
import dagshub
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
import random
import numpy as np

# Konfigurasi 
DATASET_PATH = "water_potability_preprocessing.csv"
TEST_SIZE = 0.25
MODEL_NAME = "water_potability"
EXPERIMENT_NAME = "Random Forest_Grid Search"

param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [1, 5, 25]
}

np.random.seed(42)

# Set Tracking
mlflow.set_tracking_uri("https://dagshub.com/Sulbae/Latihan-MLFlow.mlflow")
# Set Nama Eksperimen
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

# Load dataset
data = pd.read_csv(DATASET_PATH)

# Split data
X = data.drop(columns=['Potability'], axis=1)
y = data['Potability']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, 
    random_state=42)

# Menyimpan snippet atau input sample
input_example = X_train.iloc[0:5]

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(rf, param_grid, scoring="accuracy", cv=3)

# Train model
grid_search.fit(X_train, y_train)

# Ambil model terbaik
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Log Param
mlflow.log_param("test_size", TEST_SIZE)
mlflow.log_params(best_params)

# Evaluate model
y_pred = best_model.predict(X_test)

accuracy = best_model.score(X_test, y_test)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# Log Metrics
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1_score", f1)

# Log model
if best_model is not None:
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_model",
        input_example=input_example
)

# Model Regis
active_run = mlflow.active_run()
model_uri = f"runs:/{active_run.info.run_id}/best_model"

model_registered = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)
version = model_registered.version

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=version,
    stage="Production",
    archive_existing_versions=True
)

print(f"REGISTERED_MODEL_VERSION: {version}")