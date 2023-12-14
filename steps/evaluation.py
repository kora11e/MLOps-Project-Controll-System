from typing import Tuple
from typing_extensions import Annotated
import logging 
import pandas as pd
from zenml import step
from src.evaluate import MSE, R2
from sklearn.base import RegressorMixin

@step
def evaluate_model(model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,   
    ) -> Tuple[
        Annotated[float, 'r2_score'],
        Annotated[float, 'mse']
    ]:
    
    prediction = model.predict(X_test)
    mse_class = MSE()
    mse = mse_class.calculate_scores(y_test, prediction)

    r2_class = R2()
    r2 = r2_class.calculate_scores(y_test, prediction)

    return r2_score, mse

except Exception as e:
    logging.error(f'Error: {e}')
    raise e
