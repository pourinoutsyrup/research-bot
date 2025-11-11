import requests
import logging
from typing import Dict, Optional

class DiscordAlerter:
    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or self._get_webhook_from_env()
        self.logger = logging.getLogger(__name__)
        
    def _get_webhook_from_env(self):
        """Get webhook URL from environment variables"""
        import os
        return os.getenv('DISCORD_WEBHOOK_URL')
    
    def send_strategy_alert(self, strategy: Dict, backtest_results: Dict):
        """Send Discord alert for new strategy"""
        if not self.webhook_url:
            self.logger.warning("❌ No Discord webhook URL configured")
            return False
            
        try:
            message = self._format_strategy_message(strategy, backtest_results)
            
            response = requests.post(
                self.webhook_url,
                json={"content": message},
                timeout=10
            )
            
            if response.status_code == 204:
                self.logger.info("✅ Discord alert sent successfully")
                return True
            else:
                self.logger.error(f"❌ Discord alert failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Discord alert error: {e}")
            return False
    
    def send_alert(self, message: str):
        """Send simple text alert to Discord"""
        if not self.webhook_url:
            self.logger.warning("❌ No Discord webhook URL configured")
            return False
            
        try:
            response = requests.post(
                self.webhook_url,
                json={"content": message},
                timeout=10
            )
            
            if response.status_code == 204:
                self.logger.info("✅ Discord alert sent successfully")
                return True
            else:
                self.logger.error(f"❌ Discord alert failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Discord alert error: {e}")
            return False
    
    def _format_strategy_message(self, strategy: Dict, backtest_results: Dict) -> str:
        """Format strategy information for Discord"""
        name = strategy.get('name', 'Unknown Strategy')
        strategy_type = strategy.get('type', 'Unknown Type')
        confidence = strategy.get('confidence', 0)
        sharpe = strategy.get('sharpe_ratio', 0)
        
        # Get backtest results if available
        actual_sharpe = backtest_results.get('best_sharpe', sharpe)
        win_rate = backtest_results.get('win_rate', 0)
        total_return = backtest_results.get('best_total_return', 0)
        
        message = f"""
🎯 **NEW STRATEGY DISCOVERED: {name}**

**Mathematical Approach**: {strategy_type}
**Confidence Score**: {confidence:.1%}
**Estimated Sharpe Ratio**: {sharpe:.2f}
**Actual Sharpe Ratio**: {actual_sharpe:.2f}
**Win Rate**: {win_rate:.1%}
**Total Return**: {total_return:.1%}

📊 **Strategy Logic**:
{strategy.get('logic', 'AI-generated mathematical approach')}

💡 **Ready for further analysis and deployment!**
"""
        return message
    
    def send_backtest_alert(self, strategy: Dict, backtest_results: Dict):
        """Send enhanced Discord alert with backtest results"""
        if not self.webhook_url:
            self.logger.warning("❌ No Discord webhook URL configured")
            return False
            
        try:
            message = self._format_backtest_message(strategy, backtest_results)
            
            response = requests.post(
                self.webhook_url,
                json={"content": message},
                timeout=10
            )
            
            if response.status_code == 204:
                self.logger.info("✅ Discord backtest alert sent successfully")
                return True
            else:
                self.logger.error(f"❌ Discord backtest alert failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Discord backtest alert error: {e}")
            return False
    
    def _format_backtest_message(self, strategy: Dict, backtest_results: Dict) -> str:
        """Format backtest results for Discord"""
        results = strategy.get('backtest_results', {})
        
        message = f"""
🎯 **PROVEN STRATEGY: {strategy['name']}**

**Mathematical Approach**: {strategy.get('type', 'Unknown')}
**Actual Sharpe Ratio**: {results.get('best_sharpe', 0):.2f}
**Optimal Asset/Timeframe**: {results.get('best_asset')} {results.get('best_timeframe')}
**Win Rate**: {results.get('win_rate', 0):.1%}
**Total Return**: {results.get('best_total_return', 0):.1%}

📊 **Backtest Summary**:
- Successful Tests: {results.get('successful_tests', 0)}/{results.get('total_tests', 0)}
- Average Sharpe: {results.get('avg_sharpe', 0):.2f}

💡 **Ready for live deployment!**
"""
        return message