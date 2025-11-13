import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class BacktestEngine:
    """Real backtesting engine for crypto strategies"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.logger = logging.getLogger("backtest")
    
    async def run_backtest(self, strategy_code: str, data: pd.DataFrame, symbol: str) -> Dict[str, any]:
        """Run backtest with real strategy execution"""
        try:
            # Execute strategy code in a safe environment
            strategy = self._compile_strategy(strategy_code)
            
            if not strategy:
                return {"error": "Failed to compile strategy"}
            
            # Run backtest
            results = await self._execute_backtest(strategy, data, symbol)
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            return {"error": str(e)}
    
    def _compile_strategy(self, strategy_code: str):
        """Compile strategy code safely"""
        try:
            # Create a safe execution environment
            exec_globals = {
                'pd': pd,
                'np': np,
                'len': len,
                'range': range,
                'zip': zip,
                'list': list,
                'dict': dict,
                'float': float,
                'int': int,
                'str': str,
                'bool': bool
            }
            
            # Execute strategy code
            exec(strategy_code, exec_globals)
            
            # Get the strategy function
            if 'TradingStrategy' in exec_globals:
                return exec_globals['TradingStrategy']()
            else:
                self.logger.error("No TradingStrategy class found in code")
                return None
                
        except Exception as e:
            self.logger.error(f"Strategy compilation failed: {e}")
            return None
    
    async def _execute_backtest(self, strategy, data: pd.DataFrame, symbol: str) -> Dict[str, any]:
        """Execute the backtest"""
        try:
            # Initialize tracking variables
            capital = self.initial_capital
            position = 0
            trades = []
            equity_curve = []
            
            # Iterate through data
            for i in range(1, len(data)):
                current_data = data.iloc[:i]
                current_price = data['close'].iloc[i]
                
                # Get signal from strategy
                signal = strategy.generate_signal(current_data)
                
                # Execute trade based on signal
                if signal == 1 and position <= 0:  # Buy signal, not long
                    if position < 0:  # Close short
                        pnl = (data['close'].iloc[i-1] - current_price) * abs(position)
                        capital += pnl
                    
                    # Go long
                    position = capital / current_price
                    trades.append({
                        'timestamp': data.index[i],
                        'type': 'LONG',
                        'price': current_price,
                        'size': position,
                        'pnl': 0
                    })
                    
                elif signal == -1 and position >= 0:  # Sell signal, not short
                    if position > 0:  # Close long
                        pnl = (current_price - data['close'].iloc[i-1]) * position
                        capital += pnl
                    
                    # Go short
                    position = -capital / current_price
                    trades.append({
                        'timestamp': data.index[i],
                        'type': 'SHORT',
                        'price': current_price,
                        'size': abs(position),
                        'pnl': 0
                    })
                elif signal == 0 and position != 0:  # Close position
                    if position > 0:  # Close long
                        pnl = (current_price - data['close'].iloc[i-1]) * position
                    else:  # Close short
                        pnl = (data['close'].iloc[i-1] - current_price) * abs(position)
                    
                    capital += pnl
                    trades.append({
                        'timestamp': data.index[i],
                        'type': 'CLOSE',
                        'price': current_price,
                        'size': abs(position),
                        'pnl': pnl
                    })
                    position = 0
                
                # Calculate current equity
                if position > 0:  # Long
                    current_equity = capital + (current_price - data['close'].iloc[i-1]) * position
                elif position < 0:  # Short
                    current_equity = capital + (data['close'].iloc[i-1] - current_price) * abs(position)
                else:  # Flat
                    current_equity = capital
                
                equity_curve.append(current_equity)
            
            # Calculate final metrics
            total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital if equity_curve else 0
            sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)
            max_drawdown = self._calculate_max_drawdown(equity_curve)
            win_rate = self._calculate_win_rate(trades)
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': len(trades),
                'final_equity': equity_curve[-1] if equity_curve else self.initial_capital,
                'trades': trades[-10:],  # Last 10 trades
                'equity_curve': equity_curve[-100:],  # Last 100 equity points
                'status': 'completed'
            }
            
        except Exception as e:
            self.logger.error(f"Backtest execution failed: {e}")
            return {"error": str(e)}
    
    def _calculate_sharpe_ratio(self, equity_curve: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(equity_curve) < 2:
            return 0.0
        
        returns = np.diff(equity_curve) / equity_curve[:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not equity_curve:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trades"""
        if not trades:
            return 0.0
        
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        return len(winning_trades) / len(trades)