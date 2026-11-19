import os
import pandas as pd


class FeatureStore:
    def __init__(self, store_path: str):
        self.feature_store = pd.read_parquet(store_path)

    @staticmethod
    def build_feature_store(df: pd.DataFrame, output_path: str, data_sample: int | None = 500):
        store_cols = ["SK_ID_CURR"] + [
            col for col in df.columns
            if col.startswith(("PREV_", "BUREAU_", "PCB_", "CCB_", "INS_"))
        ]
        subset_df = df.iloc[:data_sample] if data_sample is not None else df
        feature_store_test = subset_df[store_cols].drop_duplicates(subset=["SK_ID_CURR"])

        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, "feature_store_test.parquet")
        feature_store_test.to_parquet(file_path, index=False)

    def get_features(self, application_df: pd.DataFrame) -> pd.DataFrame:
        if "SK_ID_CURR" not in application_df.columns:
            raise ValueError("Missing required column 'SK_ID_CURR' in input data.")
        clean_app_df = application_df.drop(columns=[
            col for col in self.feature_store.columns
            if col != "SK_ID_CURR" and col in application_df.columns
        ])
        full_application_df = clean_app_df.merge(self.feature_store, on="SK_ID_CURR", how="left")
        return full_application_df