import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from decimal import Decimal

@dataclass
class ResearchTask:
    source: str
    query: str
    priority: int = 1
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.created_at = datetime.now()

class BaseResearchAgent(ABC):
    """Original base agent + x402 payments"""
    
    def __init__(self, name: str, deepseek_client, input_queue: asyncio.Queue = None, output_queue: asyncio.Queue = None, treasury=None):
        self.name = name
        self.deepseek = deepseek_client
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.treasury = treasury  # ADD x402
        self.logger = logging.getLogger(f"agent.{name}")
        self.results = []
    
    async def run_perpetually(self):
        """ORIGINAL queue-based perpetual runner"""
        self.logger.info(f"Starting perpetual agent: {self.name}")
        
        while True:
            # ORIGINAL: Wait for queue items
            task = await self.input_queue.get()
            self.logger.info(f"{self.name} processing: {task.query[:50]}...")
            
            try:
                # Execute with potential x402 payments
                result = await self.execute(task)
                
                if result and self.output_queue:
                    await self.output_queue.put(result)
                    self.logger.info(f"{self.name} completed task")
                
                self.input_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"{self.name} failed: {e}")
                self.input_queue.task_done()
    
    async def x402_paid_ai_call(self, prompt: str) -> Dict[str, Any]:
        """REPLACE free DeepSeek with paid x402 service"""
        if self.treasury:
            # PAY for AI inference
            service_url = "https://api.paid-ai.com/infer"
            cost = Decimal('0.75')
            
            success = await self.treasury.pay_for_service(service_url, cost)
            if success:
                # Access paid AI service
                return await self._access_paid_service(service_url, {'prompt': prompt})
        
        # Fallback to original DeepSeek if no treasury
        return await self.deepseek_call(prompt)
    
    async def deepseek_call(self, prompt: str) -> Dict[str, Any]:
        """Keep original as fallback"""
        try:
            result = await self.deepseek.extract_tradable_strategy(prompt)
            return result or {}
        except Exception as e:
            self.logger.error(f"DeepSeek call failed: {e}")
            return {"error": str(e)}
    
    @abstractmethod
    async def execute(self, task: ResearchTask) -> Dict[str, Any]:
        pass