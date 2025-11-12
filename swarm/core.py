import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime

class ResearchTask:
    def __init__(self, source: str, query: str, priority: int = 1, metadata: Dict = None):
        self.source = source
        self.query = query
        self.priority = priority
        self.metadata = metadata or {}
        self.created_at = datetime.now()

class ResearchSwarm:
    """Simplified swarm for ATHENA - just task execution"""
    
    def __init__(self, agents: Dict[str, Any], max_workers: int = 8):
        self.agents = agents
        self.max_workers = max_workers
        self.logger = logging.getLogger("swarm")
    
    async def process_tasks(self, tasks: List[ResearchTask]) -> List[Dict[str, Any]]:
        """Process tasks with the given agents"""
        self.logger.info(f"📋 Processing {len(tasks)} tasks with {len(self.agents)} agents")
        
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_with_semaphore(task):
            async with semaphore:
                return await self._process_single_task(task)
        
        results = await asyncio.gather(
            *[process_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
        
        valid_results = [r for r in results if not isinstance(r, Exception)]
        self.logger.info(f"✅ Processed {len(valid_results)} tasks")
        return valid_results
    
    async def _process_single_task(self, task: ResearchTask) -> Dict[str, Any]:
        """Process a single task"""
        try:
            if task.source in self.agents:
                agent = self.agents[task.source]
                result = await agent.execute(task)
                return result
            else:
                return {'error': f'Unknown agent: {task.source}'}
        except Exception as e:
            return {'error': str(e), 'agent': task.source}