
# app/logging_utils.py
from __future__ import annotations
import logging

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("trading_system.log"),
            logging.StreamHandler()
        ],
    )

get_logger = logging.getLogger
