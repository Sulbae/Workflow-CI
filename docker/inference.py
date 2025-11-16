import mlflow.pyfunc
import pandas as pd
import joblib
import numpy as np
import pandas as pd
import requests
import json

MODEL_PATH = "model/best_model"

model = mlflow.pyfunc.load_model(MODEL_PATH)

def prediction(data):
    df = pd.DataFrame([data])
    result = model.predict(df)
    return result[0]

if __name__ == "__main__":
    sample = {
        "ph": 1,
        "Hardness": 1,
        "Solids": 1,
        "Chloramines": 1,
        "Sulfate": 1,
        "Conductivity": 1,
        "Organic_carbon": 1,
        "Trihalomethanes": 1,
        "Turbidity":
    }
