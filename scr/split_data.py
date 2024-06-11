import numpy as np
import pandas as pd
from typing import Tuple, List

def split_data(
    df: pd.DataFrame,
    train_size: int = 4,
    val_size: int = 1,
    target_column: str = 'genre_first',
    drop_columns: List[str] = None,
    random_state: int = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits the data into train, validation, and test sets and separates the features and target.

    Parameters:
        df (pd.DataFrame): The input dataframe containing the data.
        train_size (int): The number of subject IDs to include in the train set. Defaults to 4.
        val_size (int): The number of subject IDs to include in the validation set. Defaults to 1.
        target_column (str): The name of the target column. Defaults to 'genre_first'.
        drop_columns (List[str]): List of columns to drop for features. Defaults to ['subject_id', 'label', 'genre_first'].
        random_state (int): Random seed for reproducibility. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: 
            X_train, y_train, X_val, y_val, X_test, y_test.
    """
    if drop_columns is None:
        drop_columns = ['subject_id', 'label', 'genre_first']
    
    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # Shuffle subject IDs
    subject_ids = df['subject_id'].unique()
    np.random.shuffle(subject_ids)
    
    # Split subject IDs for train, validation, and test sets
    train_ids = subject_ids[:train_size]
    val_ids = subject_ids[train_size:train_size + val_size]
    test_ids = subject_ids[train_size + val_size:]
    
    # Create train, validation, and test sets
    train_df = df[df['subject_id'].isin(train_ids)]
    val_df = df[df['subject_id'].isin(val_ids)]
    test_df = df[df['subject_id'].isin(test_ids)]
    
    # Separate features and target
    X_train = train_df.drop(drop_columns, axis=1)
    y_train = train_df[target_column]
    X_val = val_df.drop(drop_columns, axis=1)
    y_val = val_df[target_column]
    X_test = test_df.drop(drop_columns, axis=1)
    y_test = test_df[target_column]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Example usage
# X_train, y_train, X_val, y_val, X_test, y_test = split_data(df, train_size=4, val_size=1, random_state=42)
