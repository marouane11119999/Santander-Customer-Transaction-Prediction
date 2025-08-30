import pandas as pd
from sklearn.model_selection import train_test_split

def create_unified_test_split(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    save_path: str = "unified_test_set.csv"
):
    """
    Splits the data into train and unified test set and saves the test set to CSV.

    Parameters
    ----------
    data : pd.DataFrame
        The full dataset including features and target.
    test_size : float
        Fraction of the dataset to reserve as the test set.
    random_state : int
        Random seed to make split reproducible.
    save_path : str
        Path where the unified test set CSV will be saved.

    Returns
    -------
    train_data : pd.DataFrame
        The training set (remaining after test split).
    test_data : pd.DataFrame
        The test set saved to `save_path`.
    """
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state, shuffle=True, stratify= data['target']
    )

    test_data.to_csv(save_path, index=False)
    print(f"Unified test set saved to '{save_path}' ({len(test_data)} rows).")

    return train_data, test_data
