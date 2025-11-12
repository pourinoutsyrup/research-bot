"""
workflows/strategy_builder.py
ATHENA v3.0 — Fixed DeepSeek Integration
"""
import logging
from .ai_strategy_extractor import DeepSeekExtractor

logger = logging.getLogger(__name__)

class StrategyBuilder:
    """
    Simple wrapper to provide .ai_extractor
    This is what main.py expects
    """
    def __init__(self, enabled: bool, api_key: str):
        self.enabled = enabled
        self.api_key = api_key
        
        if enabled and api_key:
            self.ai_extractor = DeepSeekExtractor(api_key)
            logger.info("AI Strategy Extraction enabled")
        else:
            self.ai_extractor = None
            logger.warning("AI Strategy Extraction DISABLED — no API key")

    def __bool__(self):
        return self.enabled and self.ai_extractor is not None