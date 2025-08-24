from sklearn.datasets import fetch_openml
import pandas as pd
import os

def load_data(id :int, location: str) -> pd.DataFrame:
    """
    Load dataset from OpenML
    """
    if not os.path.exists(location) or not os.path.isdir(location):
        raise FileNotFoundError(f"Directory not found: {location}")
    print(f"Loading dataset with id {id} from OpenML...")
    # Fetch the dataset from OpenML using the provided link
    dataset = fetch_openml(data_id=id, as_frame=True)
    # put the dataset into raw folder
    file_path = os.path.join(location, f"{dataset.details['name']}.csv")
    dataset.frame.to_csv(path_or_buf=file_path, index=False)
    return dataset.frame

def load_data_from_file(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a local CSV file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    print(f"Loading dataset from file: {file_path}...")
    return pd.read_csv(file_path)