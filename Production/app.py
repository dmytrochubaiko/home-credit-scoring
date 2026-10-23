import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Union

PROD_PATH = os.path.dirname(os.path.abspath(__file__))
if PROD_PATH not in sys.path:
    sys.path.append(PROD_PATH)

from predict import CreditScoringService

BASE_DIR = os.path.dirname(PROD_PATH)
MODEL_PATH = os.path.join(BASE_DIR, "Models")

app = FastAPI(
    title="Home Credit Scoring API",
    description="API for predicting client default risk using LightGBM, XGBoost, and Stacking ensemble.",
    version="1.0.0"
)

scoring_service = None

@app.on_event("startup")
def load_models():
    global scoring_service
    try:
        scoring_service = CreditScoringService(MODEL_PATH)
        print(" Models successfully loaded into RAM!")
    except Exception as e:
        print(f" Error loading models: {e}")

class ClientRequest(BaseModel):
    data: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(
        ..., description="Client profile dictionary with all aggregated features"
    )

@app.get("/")
def home():
    return {"message": "Credit Scoring API is running! Use /predict endpoint for risk assessment."}

@app.post("/predict")
def predict_credit(request: ClientRequest):
    if scoring_service is None:
        raise HTTPException(status_code=500, detail="Models are not loaded on the server.")
    try:
        client_data = request.data
        is_single = isinstance(client_data, dict)
        
        if is_single:
            client_data = [client_data]
            
        input_df = pd.DataFrame(client_data)
        result_df = scoring_service.predict(input_df)
        result_dict = result_df.to_dict(orient="records")
        prediction_output = result_dict[0] if is_single else result_dict
        
        return {
            "status": "success",
            "prediction": prediction_output
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")
