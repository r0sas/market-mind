from typing import Dict, List, Optional
import logging

from .dcf_model import DCFModel
from .ddm_model import DDMModel
from .pe_model import PEModel
from .base_valuation import BaseValuationModel
from .asset_based_model import AssetBasedModel
from .graham_model import GrahamModel

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ValuationManager:
    """
    Orchestrates all valuation models and manages results, confidence, and warnings.
    """

    def __init__(self, simplified_df):
        if simplified_df is None or simplified_df.empty:
            raise ValueError("simplified_df cannot be None or empty")

        self.df = simplified_df.copy()
        self.models: Dict[str, BaseValuationModel] = {}
        self.results: Dict[str, float] = {}
        self.recommended_models: List[str] = []
        self.model_fit_scores: Dict[str, float] = {}

        self._initialize_models()

        self.current_price: Optional[float] = (
            self.df.loc["Share Price"].iloc[0] if "Share Price" in self.df.index else None
        )

    def _initialize_models(self):
        """
        Instantiate all valuation models with the DataFrame.
        """
        self.models = {
            "dcf": DCFModel(self.df),
            "ddm": DDMModel(self.df),
            "pe_model": PEModel(self.df),
            "asset_based": AssetBasedModel(self.df),
            "graham_value": GrahamModel(self.df)
        }

    def calculate_models(
        self, models_to_calculate: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, float]:
        """
        Calculate specified valuation models (or all if None).
        """
        if models_to_calculate is None:
            models_to_calculate = list(self.models.keys())

        logger.info(f"Calculating models: {models_to_calculate}")
        self.results = {}

        for model_key in models_to_calculate:
            model = self.models.get(model_key)
            if model:
                value = model.calculate(**kwargs)
                if value is not None and value > 0:
                    self.results[model_key] = value
                logger.info(
                    f"{model_key}: {value} (Confidence: {getattr(model, 'confidence', 'N/A')})"
                )

        return self.results.copy()

    def get_confidence_scores(self) -> Dict[str, str]:
        return {k: getattr(m, "confidence", "N/A") for k, m in self.models.items()}

    def get_model_warnings(self) -> Dict[str, List[str]]:
        return {k: getattr(m, "warnings", []) for k, m in self.models.items()}

    def set_recommended_models(self, model_list: List[str], fit_scores: Dict[str, float]):
        self.recommended_models = model_list
        self.model_fit_scores = fit_scores
        logger.info(f"Set recommended models: {model_list}")

    def get_recommended_models(self) -> List[str]:
        return self.recommended_models.copy()

    def get_model_fit_scores(self) -> Dict[str, float]:
        return self.model_fit_scores.copy()

    def get_average_valuation(self, weighted: bool = False) -> Optional[float]:
        if not self.results:
            return None

        valid_results = {k: v for k, v in self.results.items() if v > 0}
        if not valid_results:
            return None

        if not weighted:
            return sum(valid_results.values()) / len(valid_results)

        # Weighted by confidence
        weights_map = {"High": 3, "Medium": 2, "Low": 1}
        total_value, total_weight = 0, 0
        for k, v in valid_results.items():
            conf = getattr(self.models[k], "confidence", "Medium")
            weight = weights_map.get(conf, 2)
            total_value += v * weight
            total_weight += weight

        return total_value / total_weight if total_weight > 0 else None

    def get_margin_of_safety(self, target_margin: float = 0.25) -> Optional[Dict[str, any]]:
        if self.current_price is None or not self.results:
            return None

        analysis = {}
        for k, intrinsic_value in self.results.items():
            margin = (intrinsic_value - self.current_price) / intrinsic_value
            analysis[k] = {
                "intrinsic_value": intrinsic_value,
                "current_price": self.current_price,
                "margin_of_safety": margin,
                "is_undervalued": margin >= target_margin,
                "target_buy_price": intrinsic_value * (1 - target_margin),
                "confidence": getattr(self.models[k], "confidence", "N/A"),
                "warnings": getattr(self.models[k], "warnings", [])
            }
        return analysis

