import mlflow.pyfunc
import pandas as pd
import os

# Konfigurasi
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = os.getenv("MODEL_NAME", "potability_model")
MODEL_STAGE = "Production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

# Features
COLUMNS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]

def prediction(data_path="sample_test.csv"):
    df = pd.read_csv(data_path, header=None)
    df.columns = COLUMNS

    result = model.predict(df)
    print("Prediction Result:", result[0])
    
    return result[0]

if __name__ == "__main__":
    prediction()