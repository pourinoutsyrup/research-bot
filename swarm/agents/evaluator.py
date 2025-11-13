import logging
import pandas as pd
from typing import Dict, Any
from datetime import datetime
from .base import BaseResearchAgent, ResearchTask

class EvaluatorAgent(BaseResearchAgent):
    """ORIGINAL evaluator + x402 paid backtesting"""
    
    def __init__(self, deepseek_client, input_queue: asyncio.Queue, output_queue: asyncio.Queue = None, treasury=None):
        super().__init__("evaluator", deepseek_client, input_queue, output_queue, treasury)
    
    async def execute(self, task_data: Dict) -> Dict[str, Any]:
        """ORIGINAL evaluation + x402 paid backtesting"""
        strategy_data = task_data
        
        # USE x402 paid backtesting with premium data
        if self.treasury:
            evaluation_results = await self._x402_backtest_strategy(strategy_data)
        else:
            evaluation_results = await self._free_backtest_strategy(strategy_data)
        
        return {
            'type': 'evaluated_strategy',
            'original_built_strategy': strategy_data,
            'evaluation_results': evaluation_results,
            'evaluated_at': datetime.now().isoformat(),
            'status': 'completed',
            'agent': self.name
        }
    
    async def _x402_backtest_strategy(self, strategy_data: Dict) -> Dict:
        """PAY for premium backtesting"""
        service_url = "https://api.paid-backtest.com/run"
        cost = Decimal('3.00')
        
        success = await self.treasury.pay_for_service(service_url, cost)
        if success:
            return await self._access_paid_service(service_url, {'strategy': strategy_data})
        
        return await self._free_backtest_strategy(strategy_data)