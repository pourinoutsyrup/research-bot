class HQMOptimizer:
    """Optimizes High Quality Models per Dollar spent"""
    
    def __init__(self):
        self.hqm_thresholds = {
            'sharpe_ratio': 1.5,      # Must beat market
            'win_rate': 0.60,         # 60%+ win rate
            'max_drawdown': -0.15,    # Limited losses
            'total_trades': 20,       # Statistical significance
            'backtest_period': 90     # Days of testing
        }
        
    def calculate_hqm_score(self, strategy_results: dict, cost: float) -> float:
        """Calculate HQM score (0-100) based on performance vs cost"""
        if cost == 0:
            return 0
            
        performance_score = self._calculate_performance_score(strategy_results)
        cost_efficiency = performance_score / max(cost, 0.01)  # Avoid division by zero
        
        return min(cost_efficiency * 10, 100)  # Normalize to 0-100 scale
    
    def _calculate_performance_score(self, results: dict) -> float:
        """Calculate raw performance score (0-10)"""
        score = 0
        
        # Sharpe ratio contribution (40%)
        sharpe = results.get('sharpe_ratio', 0)
        score += min(sharpe / 2.0, 4.0)  # Cap at 4 points
        
        # Win rate contribution (30%)
        win_rate = results.get('win_rate', 0)
        score += min((win_rate - 0.5) * 6, 3.0)  # 50% = 0, 100% = 3
        
        # Drawdown protection (20%)
        drawdown = abs(results.get('max_drawdown', 0))
        score += max(2.0 - (drawdown * 10), 0)  # -10% = 1, -20% = 0
        
        # Trade frequency (10%)
        trades = results.get('total_trades', 0)
        score += min(trades / 50, 1.0)  # 50+ trades = 1 point
        
        return max(score, 0)