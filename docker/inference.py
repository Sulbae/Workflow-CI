import mlflow.pyfunc
import pandas as pd
import os

# Konfigurasi
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", None)
MODEL_URI = os.getenv("MODEL_URI", "model/best_model")
TEST_FILE = os.getenv("TEST_FILE", "sample_test.csv")

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

def load_model():
    if TRACKING_URI:
        mlflow.set_tracking_uri(TRACKING_URI)
    
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print(f"[INFO] Model loaded successfuly from: {MODEL_URI}")
    
    return model

def prediction(data_path: str, model):
    df = pd.read_csv(data_path, header=None)
    df.columns = COLUMNS

    result = model.predict(df)
    print("Prediction Result:", result[0])
    
    return result

if __name__ == "__main__":
    model = load_model
    prediction(TEST_FILE, model)