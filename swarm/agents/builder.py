import logging
from typing import Dict, Any
from datetime import datetime
from .base import BaseResearchAgent, ResearchTask

class BuilderAgent(BaseResearchAgent):
    """ORIGINAL builder + x402 paid strategy generation"""
    
    def __init__(self, deepseek_client, input_queue: asyncio.Queue, output_queue: asyncio.Queue, treasury=None):
        super().__init__("builder", deepseek_client, input_queue, output_queue, treasury)
    
    async def execute(self, task_data: Dict) -> Dict[str, Any]:
        """ORIGINAL building + x402 paid strategy generation"""
        quality_approved_data = task_data
        
        if not quality_approved_data.get('passed_quality', False):
            return None
        
        # USE x402 paid strategy generation for high-quality concepts
        if self.treasury:
            strategy = await self._x402_build_strategy(quality_approved_data)
        else:
            strategy = await self._free_build_strategy(quality_approved_data)
        
        return {
            'type': 'built_strategy',
            'original_quality_check': quality_approved_data,
            'strategy_code': strategy,
            'built_at': datetime.now().isoformat(),
            'agent': self.name
        }
    
    async def _x402_build_strategy(self, research_data: Dict) -> Dict:
        """PAY for professional strategy building"""
        service_url = "https://api.paid-strategy.com/build"
        cost = Decimal('2.00')
        
        success = await self.treasury.pay_for_service(service_url, cost)
        if success:
            return await self._access_paid_service(service_url, {'research': research_data})
        
        return await self._free_build_strategy(research_data)