import logging
from abc import ABC, abstractclassmethod
import numpy as np

class Evaluation(ABC):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass

class MSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating MSE')
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f'Error: {e}')
            raise e
        
class R2(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating R2 Score')
            r2 = r2_score(y_true, y_pred)
            logging.info(f'R2 Score: {r2}')
            return r2
        except Exception as e:
            logging.error(f'Error: {e}')
            raise e