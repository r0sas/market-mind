from typing import Any, Dict
from .regression_model import RegressionModel
from .tree_model import TreeModel
from .neural_model import NeuralModel

class ModelFactory:
    """
    Factory for dynamically creating ML model instances.
    """

    @staticmethod
    def create_model(model_type: str, **kwargs) -> Any:
        """
        Args:
            model_type: 'linear', 'ridge', 'lasso', 'random_forest', 'gradient_boosting', 'mlp'
            kwargs: model-specific parameters
        Returns:
            Initialized model instance
        """
        model_type = model_type.lower()
        if model_type in ["linear", "ridge", "lasso"]:
            return RegressionModel(model_type=model_type, **kwargs)
        elif model_type in ["random_forest", "gradient_boosting"]:
            return TreeModel(model_type=model_type, **kwargs)
        elif model_type in ["mlp", "neural"]:
            return NeuralModel(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
