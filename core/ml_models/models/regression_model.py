from sklearn.linear_model import LinearRegression, Ridge, Lasso
from .base_model import BaseModel
import pandas as pd

class RegressionModel(BaseModel):
    """
    Regression model wrapper.
    Supports LinearRegression, Ridge, Lasso.
    """

    def __init__(self, model_type: str = "linear", **kwargs):
        model_map = {
            "linear": LinearRegression,
            "ridge": Ridge,
            "lasso": Lasso
        }
        if model_type not in model_map:
            raise ValueError(f"Unsupported model type: {model_type}")
        super().__init__(model=model_map[model_type](**kwargs), name=f"{model_type.capitalize()}Regression")
