import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
import xgboost as xgb
from category_encoders import TargetEncoder

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Strictly import custom transformers from Production.pipeline for clean serialization
from Production.pipeline import (
    fillna_col,
    to_category,
    fillna_category,
    car_age,
    fix_day_employed,
    CustomFeatureBuilder
)

set_config(transform_output="pandas")


def main():
    print("Loading data for training...", flush=True)
    X_train_full = pd.read_parquet(ROOT_DIR / 'X_train_full.parquet')
    y_train = pd.read_parquet(ROOT_DIR / 'y_train.parquet')['TARGET']
    X_test_full = pd.read_parquet(ROOT_DIR / 'X_test_full.parquet')
    test_id = pd.read_parquet(ROOT_DIR / 'test_id.parquet')['SK_ID_CURR']
    print(f"X_train_full shape: {X_train_full.shape}, X_test_full shape: {X_test_full.shape}", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pcb_cols = [col for col in X_train_full if col.startswith("PREV_")]
    bereua_cols = [col for col in X_train_full if col.startswith("BUREAU_") and not col.startswith("BUREAU_BB_")]
    custom_fill_na_cols = pcb_cols + bereua_cols

    cat_cols = set(X_train_full.select_dtypes(include=['string', 'object', 'category']).columns)
    cat_cols_list = list(cat_cols)

    num_zero_transform = FunctionTransformer(fillna_col)
    cat_with_na = Pipeline([
        ("fill_na", FunctionTransformer(fillna_category)),
        ("to_category", FunctionTransformer(to_category))
    ])
    fill_car_age = FunctionTransformer(car_age)
    fix_de = FunctionTransformer(fix_day_employed)

    te_cols = ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE']
    target_encoder_step = TargetEncoder(smoothing=10.0)

    preprocessor = ColumnTransformer(
        transformers=[
            ("tar_encd_col", target_encoder_step, te_cols),
            ("cat_col", cat_with_na, cat_cols_list),
            ("custom_col", num_zero_transform, custom_fill_na_cols),
        ],
        remainder="passthrough"
    )

    # Check CUDA support for XGBoost
    use_cuda = False
    try:
        clf_test = xgb.XGBClassifier(tree_method="hist", device="cuda", n_estimators=1)
        clf_test.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
        use_cuda = True
        print("XGBoost CUDA acceleration is active for fast training.", flush=True)
    except Exception as e:
        print(f"XGBoost CUDA check failed ({e}), using CPU.", flush=True)
        use_cuda = False

    xgb_train_device = "cuda" if use_cuda else "cpu"

    model_lgbm = Pipeline([
        ("custom_feature", CustomFeatureBuilder()),
        ("de_fix", fix_de),
        ("prep", preprocessor),
        ("LGBM", lgb.LGBMClassifier(
            random_state=17,
            n_jobs=4,
            verbose=-1
        ))
    ])

    model_xgb = Pipeline([
        ("custom_feature", CustomFeatureBuilder()),
        ("de_fix", fix_de),
        ("prep", preprocessor),
        ("XGBoost", xgb.XGBClassifier(
            tree_method="hist",
            device=xgb_train_device,
            random_state=17,
            verbosity=0,
            enable_categorical=True,
            eval_metric="auc"
        ))
    ])

    params_path = ROOT_DIR / 'best_params.json'
    with open(params_path, 'r', encoding='utf-8') as f:
        best_params = json.load(f)

    best_params_lgbm = best_params['params']['lgbm']
    best_params_xgb = best_params['params']['xgb']

    # --- Train LightGBM Folds ---
    print("\nStarting LightGBM 5-fold training...", flush=True)
    start_t = time.time()
    oof_result_lgbm = np.zeros(len(X_train_full))
    test_result_lgbm = np.zeros(len(X_test_full))
    model_list_lgbm = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train), 1):
        fold_start = time.time()
        X_tr, y_tr = X_train_full.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train_full.iloc[val_idx], y_train.iloc[val_idx]

        fold_model = clone(model_lgbm)
        fold_model.set_params(**{f"LGBM__{k}": v for k, v in best_params_lgbm.items()})
        fold_model.fit(X_tr, y_tr)

        val_preds = fold_model.predict_proba(X_val)[:, 1]
        oof_result_lgbm[val_idx] = val_preds

        fold_auc = roc_auc_score(y_val, val_preds)
        print(f"LGBM Fold {fold}/5 | ROC-AUC: {fold_auc:.5f} ({time.time() - fold_start:.1f}s)", flush=True)

        test_preds = fold_model.predict_proba(X_test_full)[:, 1]
        test_result_lgbm += test_preds / skf.n_splits
        model_list_lgbm.append(fold_model)

    overall_lgbm_auc = roc_auc_score(y_train, oof_result_lgbm)
    print(f"Overall LGBM OOF ROC-AUC: {overall_lgbm_auc:.5f} (Total: {time.time() - start_t:.1f}s)", flush=True)

    # --- Train XGBoost Folds ---
    print("\nStarting XGBoost 5-fold training...", flush=True)
    start_t = time.time()
    oof_result_xgb = np.zeros(len(X_train_full))
    test_result_xgb = np.zeros(len(X_test_full))
    model_xgb_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train), 1):
        fold_start = time.time()
        X_tr, y_tr = X_train_full.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train_full.iloc[val_idx], y_train.iloc[val_idx]

        fold_model = clone(model_xgb)
        fold_model.set_params(**{f"XGBoost__{k}": v for k, v in best_params_xgb.items()})
        fold_model.fit(X_tr, y_tr)

        val_preds = fold_model.predict_proba(X_val)[:, 1]
        oof_result_xgb[val_idx] = val_preds

        fold_auc = roc_auc_score(y_val, val_preds)
        print(f"XGB Fold {fold}/5 | ROC-AUC: {fold_auc:.5f} ({time.time() - fold_start:.1f}s)", flush=True)

        test_preds = fold_model.predict_proba(X_test_full)[:, 1]
        test_result_xgb += test_preds / skf.n_splits

        # Set device to cpu for portable production inference
        fold_model.set_params(XGBoost__device="cpu")
        model_xgb_list.append(fold_model)

    overall_xgb_auc = roc_auc_score(y_train, oof_result_xgb)
    print(f"Overall XGBoost OOF ROC-AUC: {overall_xgb_auc:.5f} (Total: {time.time() - start_t:.1f}s)", flush=True)

    # --- Train Stacking Meta-Model ---
    print("\nTraining Stacking Meta-Model (Logistic Regression)...", flush=True)
    X_meta_train = np.column_stack([oof_result_lgbm, oof_result_xgb])
    X_meta_test = np.column_stack([test_result_lgbm, test_result_xgb])

    meta_model = LogisticRegression(C=1.0, solver='lbfgs', random_state=17)
    meta_model.fit(X_meta_train, y_train)

    meta_oof_predict = meta_model.predict_proba(X_meta_train)[:, 1]
    meta_test_predict = meta_model.predict_proba(X_meta_test)[:, 1]
    ensemble_auc = roc_auc_score(y_train, meta_oof_predict)
    print(f"Overall Ensemble OOF ROC-AUC: {ensemble_auc:.5f}", flush=True)

    # Risk grading
    conditions = [
        meta_test_predict <= 0.05,
        (meta_test_predict > 0.05) & (meta_test_predict <= 0.15),
        (meta_test_predict > 0.15) & (meta_test_predict <= 0.30)
    ]
    choices = ['A_Low_Risk', 'B_Medium_Risk', 'C_High_Risk']
    risk_grade = np.select(conditions, choices, default='D_Reject')

    # Save models
    models_dir = ROOT_DIR / 'Models'
    models_dir.mkdir(parents=True, exist_ok=True)

    meta_model_path = models_dir / 'meta_model.joblib'
    lgbm_path = models_dir / 'lgbm_folds.joblib'
    xgb_path = models_dir / 'xgb_folds.joblib'

    print(f"\nSaving newly trained artifacts to {models_dir}...", flush=True)
    joblib.dump(meta_model, meta_model_path)
    joblib.dump(model_list_lgbm, lgbm_path, compress=3)
    joblib.dump(model_xgb_list, xgb_path, compress=3)
    print("All models serialized cleanly with Production.pipeline references!", flush=True)

    # Save submission
    submission = pd.DataFrame({
        'SK_ID_CURR': test_id,
        'TARGET': meta_test_predict,
        'RISK_GRADE': risk_grade
    })
    sub_path = ROOT_DIR / 'final_credit_decisions.csv'
    submission.to_csv(sub_path, index=False)
    print(f"Final predictions saved to {sub_path}", flush=True)


if __name__ == "__main__":
    main()
