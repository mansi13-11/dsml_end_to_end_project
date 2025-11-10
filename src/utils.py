import os
import sys
import dill
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import pickle
from src.logger import logging
from src.exception import CustomException


def save_object(file_path: str, obj: object) -> None:
    """Saves a Python object to a file using pickle.

    Args:
        file_path (str): The path where the object should be saved.
        obj (object): The Python object to be saved.

    Raises:
        CustomException: If there is an error during the saving process.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error(f"Error saving object at {file_path}: {e}")
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models) -> dict:
    """Evaluates multiple regression models and returns their R2 scores.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing target.
        models (dict): A dictionary of model names and their corresponding instances.

    Returns:
        dict: A dictionary with model names as keys and their R2 scores as values.

    Raises:
        CustomException: If there is an error during model evaluation.
    """
    try:
        r2_report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]

            model.fit(X_train, y_train)

            X_train_pred = model.predict(X_train)
            X_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, X_train_pred)
            test_model_score = r2_score(y_test, X_test_pred)

            r2_report[list(models.keys())[i]] = test_model_score
            logging.info(f"{model_name} - Train R2 Score: {train_model_score}, Test R2 Score: {test_model_score}")

        return r2_report

    except Exception as e:
        logging.error("Error occurred during model evaluation")
        raise CustomException(e, sys)