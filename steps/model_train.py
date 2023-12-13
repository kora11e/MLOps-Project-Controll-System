import logging 
import pandas as pd
from zenml import step

from src.model_dev import LinearRegression
from sklearn.base import RegressorMixin
from .config import MOdelNameConfig

@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: MOdelNameConfig
    ) -> RegressorMixin:
    
    
    try:
        model = None
        if config.model_name == "LinearRegression":
            model = LinearRegression()
            trained_model = model.train(X_train, y_train)
            return train_model
        else: 
            raise ValueError(f"Model {config.model_name} not supported")
    except Exception as e:
        logging(f"Error: {e}")
        raise e