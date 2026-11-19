import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn import set_config

# Ensure transformers preserve pandas DataFrame structure and categorical dtypes
set_config(transform_output="pandas")

from Production.pipeline import (
    fillna_col,
    to_category,
    fillna_category,
    car_age,
    fix_day_employed,
    CustomFeatureBuilder,
)


class CreditScoringService:
  def __init__(self, model_dir: str):
    set_config(transform_output="pandas")
    self.lgbm_folds = joblib.load(os.path.join(model_dir, "lgbm_folds.joblib"))
    self.xgb_folds = joblib.load(os.path.join(model_dir, "xgb_folds.joblib"))
    self.meta_model = joblib.load(os.path.join(model_dir, "meta_model.joblib"))

  def sanitize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_cols = {
        'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
        'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
        'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'
    }
    for col in df.columns:
        if col not in cat_cols and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
  
  def assign_risk_grade(self, probabilities: np.ndarray) -> np.ndarray:
    conditions = [
        probabilities <= 0.05,
        (probabilities > 0.05) & (probabilities <= 0.15),
        (probabilities > 0.15) & (probabilities <= 0.30)
    ]
    choices = ['A_Low_Risk', 'B_Medium_Risk', 'C_High_Risk']
    return np.select(conditions, choices, default = 'D_Reject')
  def predict(self, df: pd.DataFrame) -> pd.DataFrame:
    df = self.sanitize_dtypes(df)
    lgbm_preds = np.mean(
        [model.predict_proba(df)[:, 1] for model in self.lgbm_folds], axis = 0
    )
    xgb_preds = np.mean(
        [model.predict_proba(df)[:, 1] for model in self.xgb_folds], axis = 0
    )
    X_meta = np.column_stack([lgbm_preds, xgb_preds])
    final_probs = self.meta_model.predict_proba(X_meta)[:, 1]
    risk_grades = self.assign_risk_grade(final_probs)

    result = pd.DataFrame({
            "probability": np.round(final_probs, 4),
            "risk_grade": risk_grades,
            "lgbm_prob": np.round(lgbm_preds, 4),
            "xgb_prob": np.round(xgb_preds, 4),
        })
    if "SK_ID_CURR" in df.columns:
      result.insert(0, "SK_ID_CURR", df["SK_ID_CURR"].values)
    return result
