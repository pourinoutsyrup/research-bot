# crypto_futures_data.py
import ccxt
import pandas as pd
import logging
from typing import Optional, List
import time

class CryptoFuturesData:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.exchange = ccxt.binance({
            'options': {
                'defaultType': 'future',  # Important: use futures
            }
        })
        
    def get_perpetual_data(self, symbol: str, timeframe: str = '1d', 
                          limit: int = 1000) -> Optional[pd.DataFrame]:
        """Get perpetual futures OHLCV data"""
        try:
            self.logger.info(f"📊 Fetching {symbol} perpetual futures data ({timeframe})")
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                self.logger.error(f"No data returned for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('datetime')
            df = df.drop('timestamp', axis=1)
            
            self.logger.info(f"✅ Loaded {len(df)} periods of {symbol} perpetual futures")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch {symbol} futures data: {e}")
            return None
    
    def get_available_pairs(self) -> List[str]:
        """Get available perpetual futures trading pairs"""
        try:
            markets = self.exchange.load_markets()
            perpetual_pairs = [
                symbol for symbol, market in markets.items()
                if market.get('future', False) and market.get('perpetual', False)
            ]
            return perpetual_pairs[:20]  # Return top 20
        except Exception as e:
            self.logger.error(f"Failed to fetch available pairs: {e}")
            return ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT", "SOL/USDT"]