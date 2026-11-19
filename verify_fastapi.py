import os
import sys
import json
import time
import pandas as pd
from pathlib import Path
from pydantic import ValidationError

ROOT_DIR = Path(r"D:\Pynhon\JP\home_credict_risk")
sys.path.insert(0, str(ROOT_DIR))

from Production.app import app, MODEL_PATH, STORE_PATH, predict_credit
from Production.predict import CreditScoringService
from Production.feature_store import FeatureStore
from Production.schemas import ClientApplication, ClientRequest

def run_verification():
    t_total = time.time()
    print("=" * 60)
    print("FASTAPI PRODUCTION INTEGRATION & INFERENCE VERIFICATION")
    print("=" * 60)

    # 1. Lifespan Loading
    print("\n[Step 1] Initializing Lifespan Components (Models & FeatureStore)...", flush=True)
    t0 = time.time()
    app.state.scoring_service = CreditScoringService(MODEL_PATH)
    app.state.feature_store = FeatureStore(STORE_PATH)
    print(f"-> SUCCESS: Scoring Service & FeatureStore loaded in {time.time()-t0:.2f}s!")
    print(f"-> FeatureStore records: {len(app.state.feature_store.feature_store)} rows")
    print(f"-> LightGBM folds: {len(app.state.scoring_service.lgbm_folds)}")
    print(f"-> XGBoost folds: {len(app.state.scoring_service.xgb_folds)}")
    print(f"-> Meta Model: {type(app.state.scoring_service.meta_model).__name__}")

    # 2. Prepare client data
    print("\n[Step 2] Reading real client data from application_test.csv...", flush=True)
    valid_ids = app.state.feature_store.feature_store["SK_ID_CURR"].head(5).tolist()
    print(f"-> Using SK_ID_CURR from FeatureStore: {valid_ids[:3]}")

    data_path = ROOT_DIR.parent / "Датасети" / "Credit_Risk_Data"
    app_test = pd.read_csv(data_path / "application_test.csv", nrows=50)

    client_1_raw = {k: v for k, v in app_test[app_test["SK_ID_CURR"] == valid_ids[0]].iloc[0].to_dict().items() if pd.notna(v)}
    client_2_raw = {k: v for k, v in app_test[app_test["SK_ID_CURR"] == valid_ids[1]].iloc[0].to_dict().items() if pd.notna(v)}

    client_1 = ClientApplication(**client_1_raw)
    client_2 = ClientApplication(**client_2_raw)

    # 3. Single Client Prediction
    print("\n[Step 3] Testing Single Client POST /predict...", flush=True)
    req_single = ClientRequest(data=client_1)
    t0 = time.time()
    resp_single = predict_credit(req_single)
    single_duration = time.time() - t0
    print(f"-> Single Prediction computed in {single_duration:.4f}s ({single_duration*1000:.1f}ms):")
    print(json.dumps(resp_single, indent=2))
    assert resp_single["status"] == "success"
    assert resp_single["prediction"]["SK_ID_CURR"] == valid_ids[0]

    # 4. Batch Client Prediction
    print("\n[Step 4] Testing Batch Client POST /predict (2 clients)...", flush=True)
    req_batch = ClientRequest(data=[client_1, client_2])
    t0 = time.time()
    resp_batch = predict_credit(req_batch)
    batch_duration = time.time() - t0
    print(f"-> Batch Prediction computed in {batch_duration:.4f}s ({batch_duration*1000:.1f}ms):")
    print(json.dumps(resp_batch, indent=2))
    assert resp_batch["status"] == "success"
    assert len(resp_batch["prediction"]) == 2

    # 5. Schema Validation Error (Pydantic v2 - negative credit)
    print("\n[Step 5] Testing Pydantic v2 Validation on Invalid Data (AMT_CREDIT = -500)...", flush=True)
    try:
        invalid_data = {**client_1_raw, "AMT_CREDIT": -500}
        ClientApplication(**invalid_data)
        print("-> ERROR: Validation failed to raise!")
    except ValidationError as e:
        print("-> SUCCESS: Pydantic v2 caught error:")
        err_info = e.errors()[0]
        print(f"   Loc: {err_info['loc']}, Msg: {err_info['msg']}, Type: {err_info['type']}")

    # 6. Missing Required Field (SK_ID_CURR)
    print("\n[Step 6] Testing Missing Required Field (SK_ID_CURR)...", flush=True)
    try:
        missing_sk_id = {k: v for k, v in client_1_raw.items() if k != "SK_ID_CURR"}
        ClientApplication(**missing_sk_id)
        print("-> ERROR: Validation failed to raise!")
    except ValidationError as e:
        print("-> SUCCESS: Pydantic v2 caught error:")
        err_info = e.errors()[0]
        print(f"   Loc: {err_info['loc']}, Msg: {err_info['msg']}, Type: {err_info['type']}")

    print("\n" + "=" * 60)
    print(f"ALL TESTS COMPLETED SUCCESSFULLY IN {time.time()-t_total:.2f}s!")
    print("=" * 60)

if __name__ == "__main__":
    run_verification()
