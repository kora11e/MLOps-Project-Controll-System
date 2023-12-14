import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression


class Model(ABC):
    @abstractmethod
    def train(self, X_train, X_test):
        pass

class LinearRegression(Model):
    def train(self, X_train, y_train, **kwargs):
        try:    
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error(f"Error: {e}")
            raise(e)
        
#class RandomForest