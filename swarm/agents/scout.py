import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
from .base import BaseResearchAgent, ResearchTask

class ResearchAgent(BaseResearchAgent):
    """ORIGINAL research agent + x402 paid research"""
    
    def __init__(self, deepseek_client, input_queue: asyncio.Queue, output_queue: asyncio.Queue, treasury=None):
        super().__init__("research_agent", deepseek_client, input_queue, output_queue, treasury)
    
    async def execute(self, task: ResearchTask) -> Dict[str, Any]:
        """ORIGINAL research + x402 paid services"""
        # USE x402 paid research service
        if self.treasury:
            research = await self._x402_research_concept(task.query)
        else:
            research = await self._free_research_concept(task.query)
        
        return {
            'type': 'research_results',
            'original_task': {
                'source': task.source,
                'query': task.query,
                'priority': task.priority,
                'created_at': task.created_at.isoformat()
            },
            'discoveries': research,
            'agent': self.name,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _x402_research_concept(self, query: str) -> Dict:
        """PAY for premium research"""
        service_url = "https://api.paid-research.com/explore"
        cost = Decimal('2.50')
        
        success = await self.treasury.pay_for_service(service_url, cost)
        if success:
            return await self._access_paid_service(service_url, {'concept': query})
        
        return await self._free_research_concept(query)
    
    async def _free_research_concept(self, query: str) -> Dict:
        """ORIGINAL free research"""
        prompt = f"Research this mathematical concept: {query}"
        return await self.deepseek_call(prompt)