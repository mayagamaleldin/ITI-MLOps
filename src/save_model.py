# src/save_model.py

import pickle


def save_model(predictor, path):
    with open(path, "wb") as f:
        pickle.dump(predictor, f)