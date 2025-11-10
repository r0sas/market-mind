from typing import Optional
import pandas as pd

class IVSimplifierWrapper:
    """
    Optional wrapper for your Intrinsic Value simplification logic.
    Allows integration with ML pipeline without modifying original IVSimplifier.
    """

    def __init__(self, simplifier_class, df: pd.DataFrame):
        self.simplifier_class = simplifier_class
        self.df = df
        self.simplified_df: Optional[pd.DataFrame] = None

    def simplify(self) -> pd.DataFrame:
        """Simplify data using underlying IVSimplifier"""
        simplifier = self.simplifier_class(self.df)
        self.simplified_df = simplifier.simplify()
        return self.simplified_df
