# database/csv_logger.py
import pandas as pd
import logging
import os
from datetime import datetime
from typing import Dict
import sys
import os

# Add the config folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
import settings

class CSVLogger:
    def __init__(self, csv_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.csv_path = csv_path or "data/strategies.csv"
        self._ensure_directory()
        
        # Define comprehensive columns
        self.columns = [
            'timestamp', 'strategy_name', 'strategy_type', 'sharpe_ratio', 
            'edge_score', 'total_return', 'max_drawdown', 'win_rate',
            'profit_factor', 'total_trades', 'best_timeframe', 'source_paper',
            'paper_url', 'published_date', 'backtest_period', 'parameters',
            'indicators', 'description'
        ]

    def _ensure_directory(self):
        """Ensure data directory exists"""
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

    def log_strategy(self, strategy_data: Dict, backtest_results: Dict):
        """Log strategy with comprehensive data"""
        try:
            # Prepare data row
            row = {
                'timestamp': datetime.now().isoformat(),
                'strategy_name': strategy_data.get('name', ''),
                'strategy_type': strategy_data.get('type', 'unknown'),
                'sharpe_ratio': backtest_results.get('sharpe_ratio', 0),
                'edge_score': backtest_results.get('edge_score', 0),
                'total_return': backtest_results.get('total_return', 0),
                'max_drawdown': backtest_results.get('max_drawdown', 0),
                'win_rate': backtest_results.get('win_rate', 0),
                'profit_factor': backtest_results.get('profit_factor', 1),
                'total_trades': backtest_results.get('total_trades', 0),
                'best_timeframe': backtest_results.get('best_timeframe', ''),
                'source_paper': strategy_data.get('source_paper', ''),
                'paper_url': strategy_data.get('paper_url', ''),
                'published_date': strategy_data.get('published_date', ''),
                'backtest_period': f"{settings.TEST_DAYS} days",
                'parameters': str(strategy_data.get('parameters', {})),
                'indicators': str(strategy_data.get('indicators', [])),
                'description': strategy_data.get('description', '')[:500]  # Limit length
            }
            
            # Create or append to CSV
            if os.path.exists(self.csv_path):
                df = pd.read_csv(self.csv_path)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            else:
                df = pd.DataFrame([row], columns=self.columns)
            
            df.to_csv(self.csv_path, index=False)
            self.logger.info(f"✅ Logged strategy to CSV: {strategy_data.get('name', '')}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log strategy to CSV: {e}")

    def get_top_strategies(self, min_sharpe: float = None, limit: int = 10) -> pd.DataFrame:
        """Get top strategies from CSV"""
        if min_sharpe is None:
            min_sharpe = settings.MIN_SHARPE_ALERT
            
        try:
            if not os.path.exists(self.csv_path):
                return pd.DataFrame()
                
            df = pd.read_csv(self.csv_path)
            filtered = df[df['sharpe_ratio'] >= min_sharpe]
            return filtered.nlargest(limit, 'sharpe_ratio')
            
        except Exception as e:
            self.logger.error(f"Error reading CSV: {e}")
            return pd.DataFrame()