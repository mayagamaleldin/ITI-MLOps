import os
import yaml
import argparse

from typing import Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import ExecutorLogger

def read_process_data(
    cfg: Dict[str, Any],
    logger,
) -> None:
    logger.info("Data Processing started")
    df = pd.read_csv(os.path.join(cfg["raw_data_path"], f"{cfg['file_name']}.csv"))
    df.set_index(cfg["id_column"], inplace=True)
    train_df, test_df = train_test_split(
        df,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=df[cfg["target_column"]],
    )
    train_df.to_parquet(
        os.path.join(cfg["processed_data_path"], f"{cfg['file_name']}-train.parquet"),
        engine="pyarrow",
    )
    test_df.to_parquet(
        os.path.join(cfg["processed_data_path"], f"{cfg['file_name']}-test.parquet"),
        engine="pyarrow",
    )

if __name__ == "__main__":
    logger = ExecutorLogger("dvc-training")
    parser = argparse.ArgumentParser(description="ML Training Pipeline Parameters")
    parser.add_argument("--params_path", required=True)
    args = parser.parse_args()
    with open(args.params_path, "r") as file:
        cfg = yaml.safe_load(file)
        read_process_data(cfg["data"], logger)