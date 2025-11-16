import mlflow.pyfunc
import pandas as pd
import joblib
import numpy as np
import pandas as pd
import requests
import json

MODEL_PATH = "model/best_model"
TEST_FILE = "sample_test.csv"

model = mlflow.pyfunc.load_model(MODEL_PATH)

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

def prediction(data_path: str):
    df = pd.read_csv(data_path, header=None)
    df.columns = COLUMNS

    result = model.predict(df)
    
    return result[0]

if __name__ == "__main__":
    prediction(TEST_FILE)