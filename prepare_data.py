import os
import gc
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ROOT_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT_DIR / "Data",
    ROOT_DIR.parent / "Датасети" / "Credit_Risk_Data",
    ROOT_DIR / "Credit_Risk_Data"
]
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists() and (p / "application_train.csv").exists()), ROOT_DIR / "Data")
print(f"Using DATA_PATH: {DATA_PATH}")


class BureauBalanceFeature(BaseEstimator, TransformerMixin):
    def __init__(self, merge_column="SK_ID_BUREAU"):
        self.merge_column = merge_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df = pd.get_dummies(df, columns=['STATUS'], prefix='STATUS', dtype=float)
        status_cols = [col for col in df.columns if col.startswith('STATUS_')]
        aggregations = {
            'MONTHS_BALANCE': ['min', 'max', 'size']
        }
        for col in status_cols:
            aggregations[col] = ['mean', 'sum']

        df_agg = df.groupby(self.merge_column).agg(aggregations)
        df_agg.columns = [f'BB_{col[0]}_{col[1].upper()}' for col in df_agg.columns]
        return df_agg.reset_index()


class BureauFeature(BaseEstimator, TransformerMixin):
    def __init__(self, merge_column="SK_ID_CURR"):
        self.merge_column = merge_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        aggregations = {
            'DAYS_CREDIT': ['min', 'max'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max'],
            'SK_ID_BUREAU': ['count'],
            'AMT_CREDIT_SUM': ['mean', 'max'],
            'AMT_CREDIT_MAX_OVERDUE': ['max'],
        }

        status_bb_col = [col for col in df.columns if col.startswith('BB_STATUS_')]
        for col in status_bb_col:
            aggregations[col] = ['mean', 'max']

        months_bal_bb_col = [col for col in df.columns if col.startswith('BB_MONTHS_BALANCE_')]
        for col in months_bal_bb_col:
            if 'MIN' in col:
                aggregations[col] = ['min']
            elif 'MAX' in col:
                aggregations[col] = ['max']
            elif 'SIZE' in col:
                aggregations[col] = ['sum', 'mean']

        df_agg = df.groupby(self.merge_column).agg(aggregations)
        df_agg.columns = [f'BUREAU_{col[0]}_{col[1].upper()}' for col in df_agg.columns]
        return df_agg.reset_index()


class PreviousApplicationFeature(BaseEstimator, TransformerMixin):
    def __init__(self, merge_column="SK_ID_CURR"):
        self.merge_column = merge_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)
        aggregations = {
            'DAYS_DECISION': ['min', 'max'],
            'PAYMENT_RATE': ['mean', 'max', 'min'],
            'SK_ID_PREV': ['count'],
            'AMT_CREDIT': ['mean', 'max', 'sum'],
            'AMT_ANNUITY': ['mean', 'max'],
            'AMT_DOWN_PAYMENT': ['max', 'mean'],
        }
        df_agg = df.groupby(self.merge_column).agg(aggregations)
        df_agg.columns = [f'PREV_{col[0]}_{col[1].upper()}' for col in df_agg.columns]
        return df_agg.reset_index()


class PCBFeature(BaseEstimator, TransformerMixin):
    def __init__(self, merge_column="SK_ID_CURR"):
        self.merge_column = merge_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        aggregations = {
            'MONTHS_BALANCE': ['min', 'max', 'size'],
            'SK_DPD': ['max', 'mean'],
            'SK_DPD_DEF': ['max', 'mean'],
            'CNT_INSTALMENT_FUTURE': ['max', 'mean']
        }
        df = pd.get_dummies(df, columns=['NAME_CONTRACT_STATUS'], prefix='NC_STATUS', dtype=float)
        name_contr_stat_col = [col for col in df.columns if col.startswith('NC_STATUS_')]
        for col in name_contr_stat_col:
            aggregations[col] = ['mean']
        df_agg = df.groupby(self.merge_column).agg(aggregations)
        df_agg.columns = [f'PCB_{col[0]}_{col[1].upper()}' for col in df_agg.columns]
        return df_agg.reset_index()


class CreditCardFeature(BaseEstimator, TransformerMixin):
    def __init__(self, merge_column="SK_ID_CURR"):
        self.merge_column = merge_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["LIMIT_USAGE"] = df["AMT_BALANCE"] / (df["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
        aggregations = {
            'MONTHS_BALANCE': ['min', 'max', 'size'],
            'AMT_BALANCE': ['max', 'mean'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['max', 'mean'],
            'AMT_DRAWINGS_ATM_CURRENT': ['max', 'sum'],
            'AMT_DRAWINGS_CURRENT': ['max', 'sum'],
            'AMT_PAYMENT_CURRENT': ['mean', 'sum'],
            'SK_DPD': ['max', 'mean'],
            'SK_DPD_DEF': ['max', 'mean'],
            'LIMIT_USAGE': ['max', 'mean']
        }
        df = pd.get_dummies(df, columns=['NAME_CONTRACT_STATUS'], prefix='CC_STATUS', dtype=float)
        status_cols = [col for col in df.columns if col.startswith('CC_STATUS_')]
        for col in status_cols:
            aggregations[col] = ['mean']
        df_agg = df.groupby(self.merge_column).agg(aggregations)
        df_agg.columns = [f'CCB_{col[0]}_{col[1].upper()}' for col in df_agg.columns]
        return df_agg.reset_index()


class InstallmentsFeature(BaseEstimator, TransformerMixin):
    def __init__(self, merge_column="SK_ID_CURR"):
        self.merge_column = merge_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        delay = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
        df["DPD"] = delay.clip(lower=0)
        df["PBP"] = (-delay).clip(lower=0)
        df["PAYMENT_DIFF"] = df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]
        df["PAYMENT_RATIO"] = df["AMT_PAYMENT"] / (df["AMT_INSTALMENT"] + 1)

        aggregations = {
            'DAYS_ENTRY_PAYMENT': ['max'],
            'DPD': ['max', 'sum', 'mean'],
            'PAYMENT_DIFF': ['max', 'mean', 'sum', 'min'],
            'PAYMENT_RATIO': ['mean', 'max'],
            'AMT_INSTALMENT': ['sum', 'mean'],
            'AMT_PAYMENT': ['sum', 'mean']
        }
        df_agg = df.groupby(self.merge_column).agg(aggregations)
        df_agg.columns = [f'INS_{col[0]}_{col[1].upper()}' for col in df_agg.columns]
        return df_agg.reset_index()


def main():
    print("Step 1/6: Processing bureau_balance and bureau...")
    bureau_balance = pd.read_csv(DATA_PATH / 'bureau_balance.csv')
    bb_feature = BureauBalanceFeature().transform(bureau_balance)
    del bureau_balance
    gc.collect()

    bureau = pd.read_csv(DATA_PATH / 'bureau.csv')
    bureau_full = bureau.merge(bb_feature, on="SK_ID_BUREAU", how="left")
    del bureau, bb_feature
    gc.collect()

    bureau_feature = BureauFeature().transform(bureau_full)
    del bureau_full
    gc.collect()
    print("Bureau features done.")

    print("Step 2/6: Processing previous applications...")
    previous_application = pd.read_csv(DATA_PATH / 'previous_application.csv')
    previous_application_feature = PreviousApplicationFeature().transform(previous_application)
    del previous_application
    gc.collect()
    print("Previous application features done.")

    print("Step 3/6: Processing POS CASH balance...")
    pos_cash_balance = pd.read_csv(DATA_PATH / 'POS_CASH_balance.csv')
    pcb_feature = PCBFeature().transform(pos_cash_balance)
    del pos_cash_balance
    gc.collect()
    print("POS CASH features done.")

    print("Step 4/6: Processing Credit Card balance...")
    credit_card_balance = pd.read_csv(DATA_PATH / 'credit_card_balance.csv')
    ccb_feature = CreditCardFeature().transform(credit_card_balance)
    del credit_card_balance
    gc.collect()
    print("Credit Card features done.")

    print("Step 5/6: Processing installments payments...")
    instalments_payments = pd.read_csv(DATA_PATH / 'installments_payments.csv')
    ins_feature = InstallmentsFeature().transform(instalments_payments)
    del instalments_payments
    gc.collect()
    print("Installments features done.")

    print("Step 6/6: Merging with application_train and application_test...")
    df_train = pd.read_csv(DATA_PATH / 'application_train.csv')
    y_train = df_train["TARGET"]
    X_train = df_train.drop(columns=["TARGET"])
    del df_train
    gc.collect()

    X_train_full = X_train.merge(bureau_feature, on="SK_ID_CURR", how="left") \
                          .merge(previous_application_feature, on="SK_ID_CURR", how="left") \
                          .merge(pcb_feature, on="SK_ID_CURR", how="left") \
                          .merge(ccb_feature, on="SK_ID_CURR", how="left") \
                          .merge(ins_feature, on="SK_ID_CURR", how="left")
    del X_train
    gc.collect()

    X_test = pd.read_csv(DATA_PATH / 'application_test.csv')
    X_test_full = X_test.merge(bureau_feature, on="SK_ID_CURR", how="left") \
                        .merge(previous_application_feature, on="SK_ID_CURR", how="left") \
                        .merge(pcb_feature, on="SK_ID_CURR", how="left") \
                        .merge(ccb_feature, on="SK_ID_CURR", how="left") \
                        .merge(ins_feature, on="SK_ID_CURR", how="left")
    del X_test, bureau_feature, previous_application_feature, pcb_feature, ccb_feature, ins_feature
    gc.collect()

    print("Saving FeatureStore/feature_store_test.parquet...")
    fs_dir = ROOT_DIR / "FeatureStore"
    fs_dir.mkdir(parents=True, exist_ok=True)
    
    # Use Production.feature_store.FeatureStore to build the exact feature store
    sys.path.insert(0, str(ROOT_DIR))
    from Production.feature_store import FeatureStore
    FeatureStore.build_feature_store(X_test_full, output_path=str(fs_dir), data_sample=500)
    print(f"FeatureStore saved. Rows: 500.")

    print("Saving train and test parquet files...")
    X_train_full = X_train_full.drop(columns=["SK_ID_CURR"])
    test_id = X_test_full.pop('SK_ID_CURR')

    X_train_full.to_parquet(ROOT_DIR / 'X_train_full.parquet', index=False)
    print("X_train_full.parquet saved.")
    X_test_full.to_parquet(ROOT_DIR / 'X_test_full.parquet', index=False)
    print("X_test_full.parquet saved.")
    y_train.to_frame().to_parquet(ROOT_DIR / 'y_train.parquet', index=False)
    print("y_train.parquet saved.")
    test_id.to_frame().to_parquet(ROOT_DIR / 'test_id.parquet', index=False)
    print("test_id.parquet saved.")
    print("Data preparation complete successfully!")


if __name__ == "__main__":
    main()
