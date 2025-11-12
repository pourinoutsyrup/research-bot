"""
ATHENA Configuration per whitepaper specs
"""
from enum import Enum

class ResearchPhase(Enum):
    SCOUT = "concept_acquisition"
    INTERPRET = "theory_extraction" 
    BUILD = "strategy_generation"
    EVALUATE = "backtesting"
    ANALYZE = "performance_analysis"

# ATHENA Performance Thresholds (per whitepaper)
PERFORMANCE_THRESHOLDS = {
    'min_sharpe': 0.3,
    'min_win_rate': 0.5,
    'max_drawdown': 0.20,
    'high_confidence_score': 0.7
}

# Agent Configuration
AGENT_WORKERS = {
    'scout': 4,
    'interpreter': 3, 
    'builder': 2,
    'evaluator': 2,
    'analyst': 2
}