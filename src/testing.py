import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import lightgbm as lgb
from typing import List
import matplotlib.pyplot as plt

def loocv_testing(
    df: pd.DataFrame,
    target_column: str = 'genre_first',
    drop_columns: List[str] = None
):
    """
    Performs Leave-One-Out Cross-Validation and evaluates the LGBMClassifier model.

    Parameters:
        df (pd.DataFrame): The input dataframe containing the data.
        target_column (str): The name of the target column. Defaults to 'genre_first'.
        drop_columns (List[str]): List of columns to drop for features. Defaults to ['subject_id', 'label', 'genre_first'].

    Returns:
        float: Average accuracy across all LOOCV folds.
        plt.Figure: Matplotlib figure object of the averaged confusion matrix.
        plt.Axes: Matplotlib axes object of the averaged confusion matrix.
    """
    if drop_columns is None:
        drop_columns = ['subject_id', 'label', target_column]
    
    # Verify that target_column exists in the DataFrame
    if target_column not in df.columns:
        raise KeyError(f"The target column '{target_column}' is not found in the DataFrame.")
    
    # Verify that columns to drop exist in the DataFrame
    for col in drop_columns:
        if col not in df.columns:
            raise KeyError(f"The column '{col}' specified to drop is not found in the DataFrame.")
    
    # Initialize list to store cross-validation scores and confusion matrices
    cv_scores = []
    confusion_matrices = []

    # Loop over each unique subject_id
    for val_subject in df["subject_id"].unique():

        # Initialize the LGBMClassifier
        model = lgb.LGBMClassifier(
            boosting_type='gbdt',
            num_leaves=50,
            learning_rate=0.05,
            n_estimators=100
        )

        # Split the data into training and testing sets based on subject_id
        cv_train = df[df['subject_id'] != val_subject]
        cv_test = df[df['subject_id'] == val_subject]

        # Separate features and target
        X_train = cv_train.drop(drop_columns, axis=1)
        X_test = cv_test.drop(drop_columns, axis=1)
        y_train, y_test = cv_train[target_column], cv_test[target_column]

        # Fit the model
        model.fit(X_train, y_train)

        # Make predictions on the test set
        preds = model.predict(X_test)

        # Calculate accuracy
        score = accuracy_score(y_test, preds)
        cv_scores.append(score)
        
        # Calculate confusion matrix for this fold
        conf_mat = confusion_matrix(y_test, preds)
        confusion_matrices.append(conf_mat)

    # Calculate average confusion matrix
    avg_conf_mat = np.mean(confusion_matrices, axis=0)

    # Return the average cross-validation score, the figure object, and the axes object
    return np.mean(cv_scores), avg_conf_mat

# Example usage:
# df = pd.read_csv('your_data.csv')
# average_accuracy, conf_matrix_fig, conf_matrix_ax = loocv_testing(df)
# print(f'Average Accuracy: {average_accuracy:.4f}')
# plt.show()  # This will display the figure returned by the function
