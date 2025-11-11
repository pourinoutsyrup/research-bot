# test_enhanced_detection.py
from tools.backtest_engine import BacktestEngine

backtester = BacktestEngine()

# Test different strategy types
test_strategies = {
    "hf_microstructure": """
def hf_orderbook_strategy(prices):
    # High frequency order book imbalance detection
    # Market microstructure analysis for 5m scalping
    # Real-time spread arbitrage
    return signals
""",
    
    "swing_trend": """
def swing_trend_strategy(prices):
    # Swing trading based on moving averages
    # Position holding for daily trends
    # Fundamental analysis for long term
    return signals
""",
    
    "default_15m": """
def statistical_arbitrage(prices):
    # Statistical arbitrage on 15m timeframe
    # Mean reversion pairs trading
    return signals
"""
}

for name, code in test_strategies.items():
    tf_type = backtester.detect_optimal_timeframe(code)
    optimal_tfs = backtester.get_optimal_timeframes(tf_type)
    print(f"{name}: {tf_type} -> {optimal_tfs}")