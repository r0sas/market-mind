"""
valuation/base_valuation.py

Abstract base class and common utilities for valuation models.
Each valuation model should subclass `BaseValuationModel` and
implement the `calculate()` method. Models should accumulate warnings
and optionally set a confidence level.

This file is intentionally small and dependency-free so other modules
can import it without pulling heavy libraries.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ValuationError(Exception):
    """Base exception for valuation-related errors."""
    pass


class BaseValuationModel(ABC):
    """
    Abstract base for valuation models.

    Responsibilities:
    - Hold a reference to the (simplified) financial dataframe or a
      lightweight mapping of required metrics.
    - Collect warnings and set a single confidence label.
    - Enforce that subclasses implement a `calculate()` method that
      returns either a float (intrinsic value per share) or None.
    """

    def __init__(self, data: Any):
        """
        Args:
            data: Simplified financial data structure. Prefer a dict-like
                  object for smaller testable models (e.g. mapping of rows).
        """
        self.data = data
        self.warnings: List[str] = []
        self.confidence: Optional[str] = None

    @abstractmethod
    def calculate(self, **kwargs) -> Optional[float]:
        """
        Run the model calculation and return intrinsic value per share
        or None if calculation is not possible.

        Implementations MUST NOT raise for ordinary validation errors;
        instead they should append to `self.warnings` and return None.
        Unexpected exceptions may bubble up as ValuationError.
        """
        raise NotImplementedError

    def add_warning(self, message: str) -> None:
        """Add a human-readable warning message to the model."""
        if message:
            self.warnings.append(message)
            logger.debug("Model warning added: %s", message)

    def set_confidence(self, level: Optional[str]) -> None:
        """Set a confidence label (e.g. 'High', 'Medium', 'Low')."""
        if level is None:
            return
        level = str(level)
        self.confidence = level

    def get_warnings(self) -> List[str]:
        """Return a copy of warnings collected during calculation."""
        return list(self.warnings)

    def to_result_dict(self) -> Dict[str, Any]:
        """Return a serialisable result dict for this model."""
        return {
            "value": None,  # models should replace this after calculate()
            "confidence": self.confidence,
            "warnings": self.get_warnings(),
        }
