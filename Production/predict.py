import os
import sys
import joblib
import numpy as np
import pandas as pd

from pipeline import (
    fillna_col,
    to_category,
    fillna_category,
    car_age,
    fix_day_employed,
    custom_feature_full,
)

sys.modules["__main__"].fillna_col = fillna_col
sys.modules["__main__"].to_category = to_category
sys.modules["__main__"].fillna_category = fillna_category
sys.modules["__main__"].car_age = car_age
sys.modules["__main__"].fix_day_employed = fix_day_employed
sys.modules["__main__"].custom_feature_full = custom_feature_full

class CreditScoringService:
  def __init__(self, model_dir: str):
    self.lgbm_folds = joblib.load(os.path.join(model_dir, "lgbm_folds.joblib"))
    self.xgb_folds = joblib.load(os.path.join(model_dir, "xgb_folds.joblib"))
    self.meta_model = joblib.load(os.path.join(model_dir, "meta_model.joblib"))
  
  def assign_risk_grade(self, probabilities: np.ndarray) -> np.ndarray:
    conditions = [
        probabilities <= 0.05,
        (probabilities > 0.05) & (probabilities <= 0.15),
        (probabilities > 0.15) & (probabilities <= 0.30)
    ]
    choices = ['A_Low_Risk', 'B_Medium_Risk', 'C_High_Risk']
    return np.select(conditions, choices, default = 'D_Reject')
  def predict(self, df: pd.DataFrame) -> pd.DataFrame:
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
