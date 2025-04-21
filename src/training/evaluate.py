import json
import os
import pickle
import yaml
import pandas as pd
import argparse

from typing import Dict, Any
from skore import EstimatorReport

from src.logger import ExecutorLogger


def evaluate(cfg: Dict[str, Any], logger) -> None:
    logger.info("loading model")
    test_df = pd.read_parquet(
        os.path.join(
            cfg["processed_data_path"], f"{cfg['file_name']}-test.parquet"
        )
    )
    X_test, y_test = (
        test_df.drop(cfg["target_column"], axis=1),
        test_df[cfg["target_column"]],
    )
    with open(
        os.path.join(
            cfg["model_path"], cfg["model_name"], "final_model.pkl"
        ),
        "rb",
    ) as pkl:
        final_model = pickle.load(pkl)
    with open(
        os.path.join(
            cfg["model_path"],
            cfg["model_name"],
            "model_target_translator.pkl",
        ),
        "rb",
    ) as pkl:
        translator = pickle.load(pkl)
    y_test_enc = y_test.apply(lambda x: translator["encoder"][x])
    final_report = EstimatorReport(final_model, X_test=X_test, y_test=y_test_enc)
    logger.info("creating evaluation report")
    evaluation_report = {
        "model_name": cfg["model_name"],
        "estimator_name": final_report.estimator_name_,
        "fitting_time": final_report.fit_time_,
        "accuracy": final_report.metrics.accuracy(),
        "precision": final_report.metrics.precision(),
        "recall": final_report.metrics.recall(),
        "prediction_time": final_report.metrics.timings(),
    }
    logger.info("saving evaluation report")
    if not os.path.exists(
        os.path.join(cfg["reports_path"], cfg["model_name"])
    ):
        os.makedirs(os.path.join(cfg["reports_path"], cfg["model_name"]))
    with open(
        os.path.join(
            cfg["reports_path"], cfg["model_name"], "evaluation_report.json"
        ),
        "w",
    ) as js:
        json.dump(evaluation_report, js, indent=4)

if __name__ == "__main__":
    logger = ExecutorLogger("dvc-training")
    parser = argparse.ArgumentParser(description="ML Training Pipeline Parameters")
    parser.add_argument("--params_path", required=True)
    args = parser.parse_args()
    with open(args.params_path, "r") as file:
        cfg = yaml.safe_load(file)
        evaluate(cfg["evaluate"], logger)