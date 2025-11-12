import logging
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ResearchTask:
    source: str
    query: str
    priority: int = 1
    metadata: Dict = None
    
    def __post_init__(self):
        self.metadata = self.metadata or {}
        self.created_at = datetime.now()

class BaseResearchAgent:
    def __init__(self, name: str, deepseek_client):
        self.name = name
        self.deepseek = deepseek_client
        self.logger = logging.getLogger(f"agent.{name}")
        self.results = []
    
    async def execute(self, task: ResearchTask) -> Dict[str, Any]:
        """Execute research task - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement execute method")
    
    async def deepseek_call(self, prompt: str) -> Dict[str, Any]:
        """Make DeepSeek API call using your existing extract_tradable_strategy method"""
        try:
            result = await self.deepseek.extract_tradable_strategy(prompt)
            return result or {}
        except Exception as e:
            self.logger.error(f"DeepSeek call failed: {e}")
            return {"error": str(e)}