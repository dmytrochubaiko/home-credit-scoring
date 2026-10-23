import numpy as np
import pandas as pd
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

def custom_feature_full(df):
    df = df.copy()

    monthly_income = df["AMT_INCOME_TOTAL"] / 12
    df["MONTHLY_INCOME_TO_PREV_CREDIT"] = monthly_income / (df["PREV_AMT_CREDIT_MEAN"] + 1)
    df["PREV_ANNUITY_TO_MONTHLY_INCOME"] = df["PREV_AMT_ANNUITY_MEAN"] / (monthly_income + 1)

    fix_member = df["CNT_FAM_MEMBERS"].fillna(1).replace(0, 1)
    df["INCOME_PER_FAMILY_MEMBER"] = df["AMT_INCOME_TOTAL"] / fix_member

    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
    df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
    df["EXT_SOURCE_MAX"] = df[ext_cols].max(axis=1)
    df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)

    e1 = df["EXT_SOURCE_1"].fillna(df["EXT_SOURCE_1"].median()).fillna(0.5)
    e2 = df["EXT_SOURCE_2"].fillna(df["EXT_SOURCE_2"].median()).fillna(0.5)
    e3 = df["EXT_SOURCE_3"].fillna(df["EXT_SOURCE_3"].median()).fillna(0.5)
    df["EXT_SOURCE_PROD"] = e1 * e2 * e3
    df["EXT_SOURCE_COUNT"] = df[ext_cols].notnull().sum(axis=1)

    df["PREV_DAYS_DECISION_SPAN"] = df["PREV_DAYS_DECISION_MAX"] - df["PREV_DAYS_DECISION_MIN"]

    df["BUREAU_DAYS_CREDIT_SPAN"] = df["BUREAU_DAYS_CREDIT_MAX"] - df["BUREAU_DAYS_CREDIT_MIN"]

    return df
