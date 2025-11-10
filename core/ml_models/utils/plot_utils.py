"""
Common plotting utilities for ML visualizations.
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_matrix(df: pd.DataFrame, title: str = "Correlation Matrix") -> None:
    """
    Plot a heatmap of the correlation matrix of a DataFrame.
    """
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_time_series(df: pd.DataFrame, x_col: str, y_cols: List[str], title: str = "Time Series") -> None:
    """
    Plot multiple time series columns.

    Args:
        df: DataFrame
        x_col: Column for x-axis (e.g., dates)
        y_cols: List of y columns to plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    for col in y_cols:
        if col in df.columns:
            plt.plot(df[x_col], df[col], label=col)
    plt.xlabel(x_col)
    plt.ylabel("Values")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
