from sklearn.neural_network import MLPRegressor
from .base_model import BaseModel
import pandas as pd

class NeuralModel(BaseModel):
    """
    Neural network regression model.
    Currently wraps sklearn's MLPRegressor.
    """

    def __init__(self, hidden_layer_sizes=(100,), activation="relu", solver="adam", **kwargs):
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, **kwargs)
        super().__init__(model=model, name="MLPRegressor")
