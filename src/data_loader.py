import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

# Example usage
file_path = "data/raw/train.csv"
df = load_data(file_path)