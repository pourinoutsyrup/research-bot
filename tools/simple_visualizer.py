# tools/simple_visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import logging
from typing import Dict, List

class SimpleVisualizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Use a style that works well for Discord
        plt.style.use('default')
    
    def generate_strategy_chart(self, strategy: Dict) -> str:
        """Generate a simple strategy visualization"""
        try:
            # Create sample data
            prices = self._generate_sample_prices()
            signals = self._simulate_strategy(prices, strategy)
            
            # Create the visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            
            # Plot 1: Price and trading signals
            ax1.plot(prices, color='blue', linewidth=1, label='Price')
            
            # Mark buy/sell signals
            buy_indices = [i for i, sig in enumerate(signals) if sig == 1]
            sell_indices = [i for i, sig in enumerate(signals) if sig == -1]
            
            if buy_indices:
                ax1.scatter(buy_indices, [prices[i] for i in buy_indices], 
                           color='green', marker='^', s=50, label='Buy', zorder=5)
            if sell_indices:
                ax1.scatter(sell_indices, [prices[i] for i in sell_indices], 
                           color='red', marker='v', s=50, label='Sell', zorder=5)
            
            ax1.set_title(f"Strategy: {strategy['name']}", fontsize=12, fontweight='bold')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Strategy performance
            returns = self._calculate_returns(prices, signals)
            cumulative_returns = np.cumsum(returns)
            
            ax2.plot(cumulative_returns, color='purple', linewidth=2, label='Cumulative Returns')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_ylabel('Returns')
            ax2.set_xlabel('Time Periods')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            plt.close()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            self.logger.error(f"Chart generation failed: {e}")
            return None
    
    def _generate_sample_prices(self, length=100) -> List[float]:
        """Generate realistic sample price data"""
        # Start at 100
        prices = [100.0]
        
        # Generate with some trend and noise
        for i in range(length - 1):
            # Small upward trend with noise
            change = np.random.normal(0.001, 0.015)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))  # Prevent negative prices
        
        return prices
    
    def _simulate_strategy(self, prices: List[float], strategy: Dict) -> List[int]:
        """Simulate a basic trading strategy"""
        signals = []
        params = strategy.get('parameters', {})
        window = params.get('window', 20)
        threshold = params.get('threshold', 2.0)
        
        for i in range(len(prices)):
            if i < window:
                signals.append(0)
                continue
            
            # Simple mean reversion logic
            recent_prices = prices[i-window:i]
            mean_price = np.mean(recent_prices)
            std_price = np.std(recent_prices) if np.std(recent_prices) > 0 else 1.0
            
            z_score = (prices[i] - mean_price) / std_price
            
            if z_score < -threshold:
                signals.append(1)  # Buy
            elif z_score > threshold:
                signals.append(-1)  # Sell
            else:
                signals.append(0)  # Hold
        
        return signals
    
    def _calculate_returns(self, prices: List[float], signals: List[int]) -> List[float]:
        """Calculate strategy returns"""
        returns = []
        for i in range(1, len(prices)):
            price_return = (prices[i] - prices[i-1]) / prices[i-1]
            # Use previous signal for this period's return
            strategy_return = price_return * signals[i-1] if i > 0 else 0
            returns.append(strategy_return)
        return returns