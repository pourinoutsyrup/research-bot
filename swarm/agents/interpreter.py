import logging
from typing import Dict, Any
from datetime import datetime
from .base import BaseResearchAgent, ResearchTask

class InterpreterAgent(BaseResearchAgent):
    """ORIGINAL interpreter + x402 paid processing"""
    
    def __init__(self, deepseek_client, input_queue: asyncio.Queue, output_queue: asyncio.Queue, treasury=None):
        super().__init__("interpreter", deepseek_client, input_queue, output_queue, treasury)
    
    async def execute(self, task_data: Dict) -> Dict[str, Any]:
        """ORIGINAL interpretation + x402 paid analysis"""
        research_data = task_data
        
        # USE x402 paid interpretation for complex concepts
        if self.treasury and self._requires_paid_interpretation(research_data):
            interpretation = await self._x402_interpret(research_data)
        else:
            interpretation = await self._free_interpret(research_data)
        
        return {
            'type': 'interpreted_concept',
            'original_research': research_data,
            'interpretation': interpretation,
            'interpreted_at': datetime.now().isoformat(),
            'agent': self.name
        }
    
    async def _x402_interpret(self, research_data: Dict) -> Dict:
        """PAY for advanced interpretation"""
        service_url = "https://api.paid-analysis.com/interpret"
        cost = Decimal('1.25')
        
        success = await self.treasury.pay_for_service(service_url, cost)
        if success:
            return await self._access_paid_service(service_url, {'research': research_data})
        
        return await self._free_interpret(research_data)
    
    async def _free_interpret(self, research_data: Dict) -> Dict:
        """ORIGINAL free interpretation"""
        prompt = f"Interpret these research discoveries: {research_data}"
        return await self.deepseek_call(prompt)
    
    def _requires_paid_interpretation(self, research_data: Dict) -> bool:
        """Heuristic for when to pay"""
        concepts = research_data.get('discovered_concepts', [])
        return len(concepts) > 3  # Pay for complex research