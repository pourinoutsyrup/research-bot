import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
from .base import BaseResearchAgent, ResearchTask

class QueryGeneratorAgent(BaseResearchAgent):
    """ORIGINAL agent + x402 paid queries"""
    
    def __init__(self, deepseek_client, output_queue: asyncio.Queue, treasury=None):
        super().__init__("query_generator", deepseek_client, None, output_queue, treasury)
    
    async def run_perpetually(self):
        """ORIGINAL continuous generation + x402 payments"""
        self.logger.info("Starting perpetual query generation")
        
        while True:
            try:
                # USE x402 paid service instead of free generation
                if self.treasury:
                    queries = await self._x402_generate_queries()
                else:
                    queries = await self.generate_pure_math_queries()
                
                for query in queries:
                    task = ResearchTask(
                        source="query_generator",
                        query=query,
                        priority=1
                    )
                    await self.output_queue.put(task)
                    self.logger.info(f"Generated query: {query[:60]}...")
                
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Query generation failed: {e}")
                await asyncio.sleep(10)
    
    async def _x402_generate_queries(self) -> list:
        """PAY for query generation"""
        service_url = "https://api.paid-research.com/generate-queries"
        cost = Decimal('1.00')
        
        success = await self.treasury.pay_for_service(service_url, cost)
        if success:
            paid_result = await self._access_paid_service(service_url, {
                'count': 3,
                'domain': 'mathematics'
            })
            return paid_result.get('queries', [])
        
        # Fallback to original
        return await self.generate_pure_math_queries()
    
    async def execute(self, task: ResearchTask) -> Dict[str, Any]:
        return {}
    
    async def generate_pure_math_queries(self) -> list:
        """ORIGINAL free query generation"""
        return [
            "Apply measure-theoretic probability to model information filtration",
            "Use functional analysis to construct optimal operators", 
            "Implement algebraic topology for persistent homology analysis"
        ]