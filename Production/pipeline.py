import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
def fillna_col (df):
    return df.fillna(0)
    
def to_category(df):
    return df.astype("category")
    
def fillna_category(df):
    return df.fillna("None")
    
def car_age(df):
    return df.fillna(-1)
    
def fix_day_employed(df):
    df = df.copy()
    if "DAYS_EMPLOYED" in df.columns:
        df["DE_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace({365243: np.nan})
    return df

class CustomFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, ext_cols=None, fallback_value=0.5):
        self.ext_cols = ext_cols or ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        self.fallback_value = fallback_value

    def fit(self, X, y=None):
        self.ext_medians_ = {}
        for col in self.ext_cols:
            if col in X.columns:
                med = X[col].median()
                self.ext_medians_[col] = float(med) if pd.notna(med) else self.fallback_value
            else:
                self.ext_medians_[col] = self.fallback_value
        return self

    def transform(self, X):
        df = X.copy()

        monthly_income = df["AMT_INCOME_TOTAL"] / 12.0
        df["MONTHLY_INCOME_TO_PREV_CREDIT"] = monthly_income / (df["PREV_AMT_CREDIT_MEAN"] + 1.0)
        df["PREV_ANNUITY_TO_MONTHLY_INCOME"] = df["PREV_AMT_ANNUITY_MEAN"] / (monthly_income + 1.0)

        fix_member = df["CNT_FAM_MEMBERS"].fillna(1).replace(0, 1)
        df["INCOME_PER_FAMILY_MEMBER"] = df["AMT_INCOME_TOTAL"] / fix_member

        df["EXT_SOURCE_MIN"] = df[self.ext_cols].min(axis=1)
        df["EXT_SOURCE_MEAN"] = df[self.ext_cols].mean(axis=1)
        df["EXT_SOURCE_MAX"] = df[self.ext_cols].max(axis=1)
        df["EXT_SOURCE_STD"] = df[self.ext_cols].std(axis=1)
        df["EXT_SOURCE_COUNT"] = df[self.ext_cols].notnull().sum(axis=1)

        e1 = df["EXT_SOURCE_1"].fillna(self.ext_medians_["EXT_SOURCE_1"])
        e2 = df["EXT_SOURCE_2"].fillna(self.ext_medians_["EXT_SOURCE_2"])
        e3 = df["EXT_SOURCE_3"].fillna(self.ext_medians_["EXT_SOURCE_3"])
        df["EXT_SOURCE_PROD"] = e1 * e2 * e3

        df["PREV_DAYS_DECISION_SPAN"] = df["PREV_DAYS_DECISION_MAX"] - df["PREV_DAYS_DECISION_MIN"]
        df["BUREAU_DAYS_CREDIT_SPAN"] = df["BUREAU_DAYS_CREDIT_MAX"] - df["BUREAU_DAYS_CREDIT_MIN"]

        return df
