from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from .base_model import BaseModel
import pandas as pd

class TreeModel(BaseModel):
    """
    Tree-based regression models.
    Supports RandomForest and GradientBoosting.
    """

    def __init__(self, model_type: str = "random_forest", **kwargs):
        model_map = {
            "random_forest": RandomForestRegressor,
            "gradient_boosting": GradientBoostingRegressor
        }
        if model_type not in model_map:
            raise ValueError(f"Unsupported tree model type: {model_type}")
        super().__init__(model=model_map[model_type](**kwargs), name=model_type.replace("_", " ").title())
