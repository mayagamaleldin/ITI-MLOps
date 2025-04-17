import os

from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split


def read_process_data(
    cfg: DictConfig,
    logger,
) -> None:
    logger.info("Data Processing started")
    df = pd.read_csv(os.path.join(cfg.raw_data_path, f"{cfg.file_name}.csv"))
    df.set_index(cfg.id_column, inplace=True)
    train_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=df[cfg.target_column],
    )
    train_df.to_parquet(
        os.path.join(cfg.processed_data_path, f"{cfg.file_name}-train.parquet"),
        engine="pyarrow",
    )
    test_df.to_parquet(
        os.path.join(cfg.processed_data_path, f"{cfg.file_name}-test.parquet"),
        engine="pyarrow",
    )
