import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from typing import List

def loocv_and_model_evaluation(
    df: pd.DataFrame,
    target_column: str = 'genre_first',
    drop_columns: List[str] = None
) -> float:
    """
    Performs Leave-One-Out Cross-Validation and evaluates the LGBMClassifier model.

    Parameters:
        df (pd.DataFrame): The input dataframe containing the data.
        target_column (str): The name of the target column. Defaults to 'genre_first'.
        drop_columns (List[str]): List of columns to drop for features. Defaults to ['subject_id', 'label', 'genre_first'].

    Returns:
        float: Average accuracy across all LOOCV folds.
    """
    if drop_columns is None:
        drop_columns = ['subject_id', 'label', 'genre_first']
    
    # Verify that target_column exists in the DataFrame
    if target_column not in df.columns:
        raise KeyError(f"The target column '{target_column}' is not found in the DataFrame.")
    
    # Verify that columns to drop exist in the DataFrame
    for col in drop_columns:
        if col not in df.columns:
            raise KeyError(f"The column '{col}' specified to drop is not found in the DataFrame.")
    
    # Separate features and target
    X = df.drop(drop_columns, axis=1)
    y = df[target_column]
    
    loo = LeaveOneOut()
    accuracies = []
    
    for train_index, val_index in loo.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Initialize the LGBMClassifier
        model = lgb.LGBMClassifier(
            boosting_type='gbdt',
            num_leaves=50,
            learning_rate=0.05,
            n_estimators=100
        )
        
        # Fit the model
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        
        # Make predictions on the validation set
        y_pred = model.predict(X_val)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)
    
    # Calculate the average accuracy across all folds
    average_accuracy = np.mean(accuracies)
    return average_accuracy

# Example usage:
# df = pd.read_csv('your_data.csv')
# average_accuracy = loocv_and_model_evaluation(df)
# print(f'Average Accuracy: {average_accuracy:.4f}')
