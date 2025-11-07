# simplify_data.py

from core.IV_Simplifier.transform_columns import _transform_columns
from core.IV_Simplifier.filter_metrics import _filter_metrics
from core.IV_Simplifier.handle_missing_values import _handle_missing_values
from core.IV_Simplifier.validate_data_quality import validate_data_quality

class IVSimplifier:
    def simplify_data(self):
        self.df = _transform_columns(self.df)
        self.df = _filter_metrics(self.df)
        self.df = _handle_missing_values(self.df)
        validate_data_quality(self.df)
