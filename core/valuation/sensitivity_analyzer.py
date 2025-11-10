from typing import Dict, List
import numpy as np
import logging
from .dcf_model import DCFModel
from .ddm_model import DDMModel

logger = logging.getLogger(__name__)

class SensitivityAnalyzer:
    """
    Performs sensitivity analysis for valuation models.
    """

    def __init__(self, df):
        self.df = df
        self.dcf_model = DCFModel(df)
        self.ddm_model = DDMModel(df)

    def analyze(
        self,
        model: str = 'dcf',
        param: str = 'discount_rate',
        base_value: float = None,
        range_pct: float = 0.2,
        steps: int = 5
    ) -> Dict[str, List[float]]:
        """
        Generate a sensitivity table for a model.

        Args:
            model: Model to analyze ('dcf' or 'ddm_multi_stage')
            param: Parameter to vary ('discount_rate', 'terminal_growth', 'growth_rate')
            base_value: Base value for the parameter (defaults applied if None)
            range_pct: Percentage range (+/-) to vary parameter
            steps: Number of steps in each direction

        Returns:
            Dictionary with parameter values and corresponding valuations
        """
        if base_value is None:
            from core.config import DEFAULT_DISCOUNT_RATE, DEFAULT_TERMINAL_GROWTH
            base_value = DEFAULT_DISCOUNT_RATE if param == 'discount_rate' else DEFAULT_TERMINAL_GROWTH

        param_values = np.linspace(base_value * (1 - range_pct), base_value * (1 + range_pct), steps * 2 + 1)
        valuations = []

        for val in param_values:
            if model == 'dcf':
                if param == 'discount_rate':
                    result = self.dcf_model.calculate(discount_rate=val)
                elif param == 'terminal_growth':
                    result = self.dcf_model.calculate(terminal_growth_rate=val)
                else:
                    result = self.dcf_model.calculate(custom_growth_rate=val)
            elif model == 'ddm_multi_stage':
                if param == 'discount_rate':
                    result = self.ddm_model.calculate(required_rate=val)
                else:
                    result = self.ddm_model.calculate(terminal_growth=val)
            else:
                logger.warning(f"Sensitivity analysis not implemented for model: {model}")
                result = None

            valuations.append(result if result is not None else 0)

        return {'parameter': param, 'values': param_values.tolist(), 'valuations': valuations}
