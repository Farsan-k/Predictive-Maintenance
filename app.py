from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Machine Failure Detection API",
    version="1.0"
)

# Load saved objects
pipeline = joblib.load(
    "model/preprocessing_pipeline.pkl"
)

dbscan = joblib.load(
    "model/dbscan_model.pkl"
)

# Load original training data
reference_df = pd.read_csv("data.csv")

reference_df = reference_df.drop(columns=[
    "Product ID",
    "UDI",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
    "Machine failure"
])


class MachineData(BaseModel):
    Type: str
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: float
    Torque_Nm: float
    Tool_wear_min: float


@app.get("/")
def home():
    return {
        "message": "DBSCAN Machine Failure Detection API"
    }


@app.post("/predict")
def predict(data: MachineData):

    # New observation
    new_row = pd.DataFrame([{
        "Type": data.Type,
        "Air temperature [K]": data.Air_temperature_K,
        "Process temperature [K]": data.Process_temperature_K,
        "Rotational speed [rpm]": data.Rotational_speed_rpm,
        "Torque [Nm]": data.Torque_Nm,
        "Tool wear [min]": data.Tool_wear_min
    }])

    # Append to reference data
    combined = pd.concat(
        [reference_df, new_row],
        ignore_index=True
    )

    # Apply preprocessing
    X_processed = pipeline.transform(combined)

    # Run DBSCAN again
    clusters = dbscan.fit_predict(X_processed)

    # Last row is the new sample
    prediction = clusters[-1]

    is_anomaly = prediction == -1

    return {
        "cluster": int(prediction),
        "anomaly": bool(is_anomaly),
        "result": (
            "Potential Machine Failure"
            if is_anomaly
            else "Normal Operation"
        )
    }