import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from Production.feature_store import FeatureStore
from Production.predict import CreditScoringService
from Production.schemas import ClientApplication, ClientRequest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Models")
STORE_PATH = os.path.join(BASE_DIR, "FeatureStore", "feature_store_test.parquet")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.scoring_service = CreditScoringService(MODEL_PATH)
        app.state.feature_store = FeatureStore(STORE_PATH)
        print("Models and Feature Store successfully loaded!")
    except Exception as e:
        print(f"Critical error during startup: {e}")
        raise

    yield

    print("Shutting down Credit Scoring API...")


app = FastAPI(
    title="Home Credit Scoring API",
    description="API for predicting client default risk using LightGBM, XGBoost, and Stacking ensemble.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/")
def home():
    return {"message": "Credit Scoring API is running! Use /predict endpoint for risk assessment."}


@app.post("/predict")
def predict_credit(request: ClientRequest):
    try:
        client_data = request.data
        is_single = isinstance(client_data, ClientApplication)

        applications = [client_data] if is_single else client_data
        input_df = pd.DataFrame([app_item.model_dump() for app_item in applications])

        enriched_df = app.state.feature_store.get_features(input_df)
        result_df = app.state.scoring_service.predict(enriched_df)

        result_records = result_df.to_dict(orient="records")
        prediction_output = result_records[0] if is_single else result_records

        return {
            "status": "success",
            "prediction": prediction_output,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {str(e)}")