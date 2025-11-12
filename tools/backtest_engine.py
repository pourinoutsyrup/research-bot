import ccxt
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

class BacktestEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Add ccxt exchange
        import ccxt
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

    def get_optimal_timeframes(self, strategy_complexity: str = "medium"):
        """AI-driven timeframe selection based on strategy type"""
        if strategy_complexity == "high_frequency":
            return ['5m', '15m']  # Primary focus
        elif strategy_complexity == "swing":
            return ['1h', '4h']   # Secondary if AI detects suitability
        else:  # medium/auto-detect
            return ['15m', '1h', '5m']  # 15m as default, 5m for aggressive

    # ADD THIS METHOD TOO - timeframe detection
    def detect_optimal_timeframe(self, strategy_code: str) -> str:
        """Enhanced AI analysis to determine best timeframe for a strategy"""
        
        code_lower = strategy_code.lower()
        
        # High-frequency indicators (5m suitable) - EXPANDED LIST
        hf_indicators = [
            'tick', 'orderbook', 'spread', 'imbalance', 'microstructure', 
            'scalping', 'high frequency', 'hf', 'latency', 'realtime',
            'market making', 'arbitrage', 'short term', 'quick', 'fast',
            'immediate', 'instant', 'rapid', 'momentum', 'burst',
            'volume spike', 'liquidity', 'bid ask', 'quote', 'trade'
        ]
        
        # Swing indicators (15m+ suitable) - EXPANDED LIST  
        swing_indicators = [
            'moving average', 'macd', 'rsi', 'bollinger', 'trend',
            'mean reversion', 'swing', 'position', 'holding', 'overnight',
            'daily', 'weekly', 'monthly', 'long term', 'slow', 'patience',
            'fundamental', 'value', 'investment', 'portfolio'
        ]
        
        # Count matches with weighting
        hf_score = 0
        swing_score = 0
        
        for indicator in hf_indicators:
            if indicator in code_lower:
                hf_score += 2  # Higher weight for HF indicators
                # Extra points for multiple occurrences
                hf_score += code_lower.count(indicator) * 0.5
        
        for indicator in swing_indicators:
            if indicator in code_lower:
                swing_score += 1  # Lower weight for swing indicators
                swing_score += code_lower.count(indicator) * 0.3
        
        # Strategy name analysis (often contains clues)
        strategy_name = self._extract_strategy_name(strategy_code)
        if strategy_name:
            name_lower = strategy_name.lower()
            if any(word in name_lower for word in ['scalp', 'hf', 'fast', 'quick', 'micro']):
                hf_score += 3
            if any(word in name_lower for word in ['swing', 'trend', 'position', 'long']):
                swing_score += 2
        
        self.logger.info(f"🔍 Timeframe detection - HF: {hf_score}, Swing: {swing_score}")
        
        # Determine optimal timeframe with clear thresholds
        if hf_score >= 5:
            return "high_frequency"
        elif swing_score >= 6:
            return "swing"
        else:
            return "medium"  # Default to 15m focus

    def _extract_strategy_name(self, strategy_code: str) -> str:
        """Extract strategy name from code for better analysis"""
        try:
            # Look for strategy name in docstring or function definition
            lines = strategy_code.split('\n')
            for line in lines:
                if 'def strategy' in line or 'def ' in line:
                    # Extract function name
                    return line.split('def ')[1].split('(')[0].strip()
                if 'Research-Based Strategy:' in line or 'ai_' in line:
                    # Extract from docstring
                    return line.split(':')[-1].strip()
            return ""
        except:
            return ""

    # ADD THIS METHOD - optimized backtest for HFT
    def run_high_frequency_backtest(self, strategy_code: str, strategy_type: str = "auto"):
        """Run backtest optimized for 5m/15m timeframes"""
        
        # Auto-detect timeframe if not specified
        if strategy_type == "auto":
            strategy_type = self.detect_optimal_timeframe(strategy_code)
        
        # Select timeframes based on strategy type
        if strategy_type == "high_frequency":
            timeframes = ['5m', '15m']  # Primary focus
            assets = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']  # High liquidity
        elif strategy_type == "swing":
            timeframes = ['15m', '1h']  # 15m primary, 1h secondary
            assets = ['BTC/USDT', 'ETH/USDT']
        else:  # medium/default
            timeframes = ['15m', '5m', '1h']  # 15m as default
            assets = ['BTC/USDT', 'ETH/USDT']
        
        self.logger.info(f"🎯 Running {strategy_type} backtest on {timeframes}")
        
        # Run optimized backtest
        return self.run_comprehensive_backtest(
            strategy_code, 
            assets=assets, 
            timeframes=timeframes
        )

    def fetch_market_data(self, symbol: str, timeframe: str, limit: int = 500):
        """Fetch real market data from exchange"""
        try:
            # Initialize exchange (add this to your __init__ too)
            if not hasattr(self, 'exchange'):
                self.exchange = ccxt.binance({
                    'rateLimit': 1000,
                    'enableRateLimit': True,
                    'timeout': 30000,
                })
            
            # Convert timeframe
            tf_map = {
                '15m': '15m', '1h': '1h', '4h': '4h', '1d': '1d'
            }
            ccxt_tf = tf_map.get(timeframe, '1h')
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, ccxt_tf, limit=limit)
            
            if not ohlcv:
                self.logger.warning(f"No data for {symbol} {timeframe}")
                return []
                
            # Convert to price series (what your strategies expect)
            prices = [candle[4] for candle in ohlcv]  # close prices
            
            self.logger.info(f"✅ Fetched {len(prices)} real candles for {symbol} {timeframe}")
            return prices
            
        except Exception as e:
            self.logger.error(f"❌ Data fetch failed for {symbol} {timeframe}: {e}")
            return []

    def observe_behavior(self, numeric_code: str, asset: str, timeframe: str):
        data = self.get_historical_data(asset, timeframe)
        local = {}
        exec(numeric_code, {"data": data}, local)
        result = local.get('numeric_model', lambda d: d)(data)
        return result

    def run_comprehensive_backtest(self, strategy_input) -> Dict:
        """Run comprehensive backtest across multiple timeframes"""
        try:
            # Handle both dict and string inputs
            if isinstance(strategy_input, dict):
                strategy_code = strategy_input.get('code', '')
                strategy_name = strategy_input.get('name', 'unknown')
                strategy_type = strategy_input.get('type', 'general_math')
            else:
                strategy_code = strategy_input
                strategy_name = 'direct_code'
                strategy_type = 'general_math'
            
            if not strategy_code:
                return {'error': 'No strategy code provided'}
            
            assets, timeframes = self.select_optimal_market_regime(strategy_type)
            
            self.logger.info(f"🧪 Backtesting {strategy_name} on {len(assets)} assets × {len(timeframes)} timeframes")
            
            all_results = []
            
            for asset in assets:
                for timeframe in timeframes:
                    try:
                        result = self._run_single_backtest(strategy_code, asset, timeframe)
                        result['asset'] = asset
                        result['timeframe'] = timeframe
                        all_results.append(result)
                        
                        if 'error' not in result:
                            sharpe = result.get('sharpe_ratio', 0)
                            trades = result.get('num_trades', 0)
                            self.logger.info(f"✅ {asset} {timeframe} | Sharpe: {sharpe:.2f} | Trades: {trades}")
                        else:
                            self.logger.warning(f"❌ {asset} {timeframe} | Error: {result.get('error')}")
                        
                    except Exception as e:
                        self.logger.error(f"❌ Backtest failed for {asset} {timeframe}: {e}")
                        continue
            
            return self._aggregate_results(all_results, strategy_name)
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive backtest failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}

    def select_optimal_market_regime(self, strategy_type: str) -> tuple:
        """Select assets and timeframes for backtesting - MULTIPLE TIMEFRAMES"""
        
        # Available timeframes (1h, 4h, 1d for comprehensive testing)
        TIMEFRAMES = ['15m', '1h', '4h', '1d']
        # Asset selection based on strategy type
        if strategy_type in ['ml_based', 'optimization']:
            assets = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'DOT/USDT']  # More volatile
        elif strategy_type in ['statistical_inference', 'signal_processing']:
            assets = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT']  # More stable
        else:
            assets = ['BTC/USDT', 'ETH/USDT']  # Default
        
        return assets, TIMEFRAMES

    def _run_single_backtest(self, strategy_code: str, asset: str, timeframe: str) -> Dict:
        """Run backtest for a single asset/timeframe combination"""
        try:
            # Get historical data
            data = self._fetch_real_data(asset, timeframe, days=30)
            
            if data is None or len(data) < 20:
                return {'error': 'Insufficient data'}
            
            # Execute strategy - pass the code string directly
            signals = self._execute_strategy_code(strategy_code, data)
            
            # Calculate returns
            returns = self._calculate_strategy_returns(data['close'], signals)
            sharpe = self._calculate_sharpe_ratio(returns)
            
            self.logger.info(f"✅ {asset} {timeframe} | Sharpe: {sharpe:.2f}")
            
            return {
                'asset': asset,
                'timeframe': timeframe,
                'sharpe_ratio': sharpe,
                'total_return': (returns + 1).prod() - 1,
                'win_rate': (returns > 0).sum() / len(returns) if len(returns) > 0 else 0,
                'num_trades': len([s for s in signals if s != 0])
            }
            
        except Exception as e:
            self.logger.error(f"❌ Backtest failed for {asset} {timeframe}: {e}")
            return {'error': str(e)}

    def _fetch_real_data(self, asset: str, timeframe: str, days: int = 30) -> pd.DataFrame:
        """Fetch real historical data from exchange"""
        try:
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            ohlcv = self.exchange.fetch_ohlcv(asset, timeframe, since=since)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # ✅ NEW: DETAILED DEBUG INFO
            self.logger.info(f"✅ Fetched {len(df)} candles for {asset} {timeframe}")
            self.logger.debug(f"📊 Price Range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            self.logger.debug(f"📊 Mean: ${df['close'].mean():.2f} | Std: ${df['close'].std():.2f}")
            self.logger.debug(f"📊 First price: ${df['close'].iloc[0]:.2f} | Last: ${df['close'].iloc[-1]:.2f}")
            self.logger.debug(f"📊 Return: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")
            self.logger.debug(f"📊 Date range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch {asset} data: {e}")
            return None

    def _execute_strategy_code(self, code: str, data: pd.DataFrame) -> List:
        """Safely execute strategy code"""
        try:
            # ✅ DEBUG: Check what type of object we're receiving
            self.logger.info(f"🔍 DEBUG: Code parameter type: {type(code)}")
            self.logger.info(f"🔍 DEBUG: Code parameter preview: {str(code)[:200]}...")
            
            # If it's a dict, extract the actual code
            if isinstance(code, dict):
                self.logger.warning("⚠️ Received dict instead of code string, extracting...")
                if 'code' in code:
                    code = code['code']
                    self.logger.info(f"✅ Extracted code from dict: {len(code)} characters")
                else:
                    self.logger.error(f"❌ Dict has no 'code' key, available keys: {list(code.keys())}")
                    return [0] * len(data)
            
            local_vars = {
                'pd': pd,
                'np': np
            }
            
            # Execute code to define the strategy function
            clean_code = code.strip()
            exec(clean_code, local_vars)
            
            # Call the strategy function
            strategy_func = local_vars.get('strategy')
            if not strategy_func:
                self.logger.error("No 'strategy' function found in code")
                return [0] * len(data)
            
            # ✅ FIX: Pass the price values correctly
            prices_series = data['close'].reset_index(drop=True)
            
            signals = strategy_func(prices_series)
            
            # Debug logging
            self.logger.debug(f"Strategy returned {len(signals)} signals, expected {len(data)}")
            
            if not signals or len(signals) != len(data):
                self.logger.error(f"Signal length mismatch: got {len(signals)}, expected {len(data)}")
                return [0] * len(data)
            
            return signals
            
        except Exception as e:  # ✅ COMPLETE THE EXCEPT BLOCK
            self.logger.error(f"Strategy execution failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return [0] * len(data)

    def _calculate_returns_simple(self, data: pd.DataFrame, signals: List) -> pd.Series:
        """Calculate returns without using pandas advanced features"""
        if len(signals) != len(data):
            signals = signals + [0] * (len(data) - len(signals))
        
        # Simple return calculation
        prices = data['close'].values
        strategy_returns = []
        position = 0
        
        for i in range(len(prices)):
            if i == 0:
                strategy_returns.append(0)
                continue
                
            # Update position
            if signals[i] > 0:
                position = 1
            elif signals[i] < 0:
                position = -1
                
            # Calculate return
            price_return = (prices[i] - prices[i-1]) / prices[i-1]
            period_return = position * price_return
            strategy_returns.append(period_return)
        
        return pd.Series(strategy_returns, index=data.index)
    
    def _calculate_strategy_returns(self, prices: pd.Series, signals: List) -> pd.Series:
        """Calculate strategy returns from signals"""
        try:
            if len(signals) != len(prices):
                self.logger.error(f"Signal length mismatch: {len(signals)} vs price length: {len(prices)}")
                return pd.Series([0] * len(prices))
            
            # Calculate price returns
            price_returns = prices.pct_change().fillna(0)
            
            # Calculate strategy returns (signal * next period return)
            strategy_returns = []
            for i in range(len(signals)):
                if i == 0:
                    strategy_returns.append(0.0)
                else:
                    # Use signal from previous period to determine current position
                    position = signals[i-1]
                    strategy_returns.append(position * price_returns.iloc[i])
            
            return pd.Series(strategy_returns, index=prices.index)
            
        except Exception as e:
            self.logger.error(f"Return calculation failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.Series([0] * len(prices))

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from returns series"""
        try:
            if len(returns) < 2 or returns.std() == 0:
                return 0.0
            
            # Convert to daily returns if we have enough data
            excess_returns = returns - risk_free_rate / 252
            
            # Annualize the Sharpe ratio
            sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252)
            
            return float(sharpe)
            
        except Exception as e:
            self.logger.error(f"Sharpe calculation failed: {e}")
            return 0.0

    def _calculate_performance_metrics_simple(self, returns: pd.Series) -> Dict:
        """Calculate performance metrics without complex pandas operations"""
        if len(returns) < 2 or returns.std() == 0:
            return {'error': 'Insufficient data for metrics'}
        
        returns_values = returns.values
        total_return = np.prod(1 + returns_values) - 1
        sharpe_ratio = np.mean(returns_values) / np.std(returns_values) * np.sqrt(365) if np.std(returns_values) > 0 else 0
        
        cumulative = np.cumprod(1 + returns_values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        win_rate = np.mean(returns_values > 0)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'volatility': np.std(returns_values) * np.sqrt(365),
            'num_trades': np.sum(np.array(returns_values) != 0)
        }

    def _aggregate_results(self, all_results: List[Dict], strategy: Dict) -> Dict:
        """Aggregate results across all tests"""
        valid_results = [r for r in all_results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'All backtests failed'}
        
        # Find best performing combination
        best_result = max(valid_results, key=lambda x: x.get('sharpe_ratio', -999))
        
        sharpe_ratios = [r.get('sharpe_ratio', 0) for r in valid_results]
        win_rate = len([r for r in valid_results if r.get('sharpe_ratio', 0) > 0]) / len(valid_results)
        
        return {
            'best_asset': best_result.get('asset', 'Unknown'),
            'best_timeframe': best_result.get('timeframe', 'Unknown'),
            'best_sharpe': best_result.get('sharpe_ratio', 0),
            'best_total_return': best_result.get('total_return', 0),
            'avg_sharpe': np.mean(sharpe_ratios),
            'win_rate': win_rate,
            'total_tests': len(all_results),
            'successful_tests': len(valid_results)
        }