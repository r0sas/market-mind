"""
Reporting and visualization for ML model results.
"""

from typing import Any, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_actual_vs_predicted(y_true: pd.Series, y_pred: pd.Series, title: str = "Actual vs Predicted") -> None:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_feature_importances(importances_df: pd.DataFrame, top_n: int = 20, title: str = "Feature Importances") -> None:
    top_features = importances_df.head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(title)
    plt.tight_layout()
    plt.show()
