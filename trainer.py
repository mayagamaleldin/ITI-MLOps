import hydra
from omegaconf import DictConfig, OmegaConf

from src.logger import ExecutorLogger
from src.training.evaluate import evaluate
from src.training.process_data import read_process_data
from src.training.train import encode_target_col, trainer


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger = ExecutorLogger("training")
    logger.info("Training started")
    logger.info("Pipeline Parameters: \n" f"{OmegaConf.to_yaml(cfg)}")
    read_process_data(cfg.pipeline.data, logger)
    X, y, X_test, y_test = encode_target_col(cfg.pipeline, logger)
    trainer(X, y, cfg.pipeline, logger)
    evaluate(X_test, y_test, cfg.pipeline, logger)
    logger.info("Training finished")


if __name__ == "__main__":
    main()
