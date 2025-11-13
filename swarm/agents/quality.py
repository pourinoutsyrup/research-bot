import logging
from typing import Dict, Any
from datetime import datetime
from .base import BaseResearchAgent, ResearchTask

class QualityAgent(BaseResearchAgent):
    """ORIGINAL quality agent + x402 paid validation"""
    
    def __init__(self, deepseek_client, input_queue: asyncio.Queue, output_queue: asyncio.Queue, treasury=None):
        super().__init__("quality", deepseek_client, input_queue, output_queue, treasury)
    
    async def execute(self, task_data: Dict) -> Dict[str, Any]:
        """ORIGINAL quality check + x402 paid validation"""
        interpreted_data = task_data
        
        # USE x402 paid quality assessment for high-potential concepts
        if self.treasury and self._should_pay_for_validation(interpreted_data):
            quality_check = await self._x402_quality_assessment(interpreted_data)
        else:
            quality_check = await self._free_quality_assessment(interpreted_data)
        
        scores = self._parse_scores(quality_check)
        
        return {
            'type': 'quality_checked_concept',
            'original_interpretation': interpreted_data,
            'quality_check': quality_check,
            'scores': scores,
            'passed_quality': scores.get('overall', 0) >= 7.0,
            'quality_checked_at': datetime.now().isoformat(),
            'agent': self.name
        }
    
    async def _x402_quality_assessment(self, interpreted_data: Dict) -> Dict:
        """PAY for professional quality assessment"""
        service_url = "https://api.paid-validation.com/assess"
        cost = Decimal('0.75')
        
        success = await self.treasury.pay_for_service(service_url, cost)
        if success:
            return await self._access_paid_service(service_url, {'concept': interpreted_data})
        
        return await self._free_quality_assessment(interpreted_data)