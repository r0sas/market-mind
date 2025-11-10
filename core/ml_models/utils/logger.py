"""
Logging configuration for ml_models.
"""

import logging
from logging import Logger

def get_logger(name: str = __name__, level: int = logging.INFO) -> Logger:
    """
    Returns a configured logger.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # Formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.setLevel(level)
        logger.propagate = False

    return logger
