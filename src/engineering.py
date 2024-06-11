from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from typing import List, Callable, Union


def count_above_threshold(series: pd.Series, threshold: float = 1.5) -> int:
    return ((series > threshold) | (series < -threshold)).sum()


def aggregate_data(
    df: pd.DataFrame,
    columns: List[str] = None,
    agg_funcs: List[Union[str, Callable]] = None,
    agg_first: List[str] = None,
    divide_by: int = 1,
    scale: bool = True
) -> pd.DataFrame:
    """
    Aggregates data in a dataframe by subject_id and label, with optional scaling and group division.

    Parameters:
        df (pd.DataFrame): The input dataframe containing the data to be aggregated.
        columns (List[str]): List of columns to aggregate. Defaults to None, which uses specific columns.
        agg_funcs (List[Union[str, Callable]]): List of aggregation functions. Defaults to None, which uses standard functions.
        agg_first (List[str]): List of columns to take the first value from. Defaults to None.
        divide_by (int): Number of groups to divide each subject-label combination into. Must be >= 1. Defaults to 1.
        scale (bool): Whether to scale the data using StandardScaler. Defaults to True.

    Returns:
        pd.DataFrame: The aggregated dataframe.
    """
    if columns is None:
        columns = ['acc_x', 'acc_y', 'acc_z', 'gy_x', 'gy_y', 'gy_z', 'mag_x', 'mag_y', 'mag_z']
    
    if agg_funcs is None:
        agg_funcs = ['max', 'min', 'std', 'mean', 'median', count_above_threshold]
    
    if agg_first is None:
        agg_first = ['genre']
    
    if divide_by < 1:
        raise ValueError("divide_by must be greater than or equal to 1")


    # Initialize dataframe to add rows later
    aggregated_df = pd.DataFrame()

    # Initializing the Scaler (outside the loop for efficiency)
    scaler = StandardScaler()

    # Dictionary for agg_funcs to columns
    agg_dict = {col: agg_funcs for col in columns}
    # "first" aggregations (e.g., "genre", since it is a constant)
    agg_first_dict = {col: 'first' for col in agg_first}
    # Combine the dictionaries
    agg_dict.update(agg_first_dict)

    

    for ID in df["subject_id"].unique():

        # Select rows for the current subject
        subject_rows = df.loc[df["subject_id"] == ID].copy()

        if scale:
            # Scaling the data
            subject_rows[columns] = scaler.fit_transform(subject_rows[columns])

        for label in df["label"].unique():

            subject_label_rows = subject_rows.loc[subject_rows["label"] == label].copy()

            if divide_by > 1:
                length_rows = len(subject_label_rows)
                split_size = length_rows // divide_by

                # Calculate the group assignments
                group_numbers = np.repeat(range(1, divide_by + 1), split_size)
                # Append remaining group numbers if there are any leftover rows
                remaining = length_rows - len(group_numbers)
                group_numbers = np.append(group_numbers, range(1, remaining + 1))
                subject_label_rows['group'] = group_numbers[:length_rows]
            else: 
                subject_label_rows['group'] = 0            

            # Aggregate by the group column
            result = subject_label_rows.groupby('group').agg(agg_dict).reset_index(drop=True)
            
            # Add subject_id and label to the result
            result['subject_id'] = ID
            result['label'] = label

            # Append the result to the aggregated DataFrame
            aggregated_df = pd.concat([aggregated_df, result], ignore_index=True)
    
    # Flatten MultiIndex columns
    aggregated_df.columns = ['_'.join(filter(None, col)).strip() for col in aggregated_df.columns]

    return aggregated_df
