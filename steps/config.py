from zenml.steps import BaseParameters

class MOdelNameConfig(BaseParameters):
    model_name: str = "LinearRegression"
    