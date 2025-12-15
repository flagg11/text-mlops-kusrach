from src.utils.config import load_config
from src.text_prep.prep_dataset import prep_dataset
from src.ml.train import train_model


def run_pipeline():
    cfg = load_config()

    if cfg["pipeline"]["run_prep"]:
        print("=== DATA PREP ===")
        prep_dataset()

    if cfg["pipeline"]["run_train"]:
        print("=== MODEL TRAINING ===")
        train_model(cfg)


if __name__ == "__main__":
    run_pipeline()
