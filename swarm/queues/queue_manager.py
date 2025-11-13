import asyncio
from typing import Dict

class ResearchQueueManager:
    """Manages all queues for the perpetual pipeline"""
    
    def __init__(self):
        self.queues = self._initialize_queues()
    
    def _initialize_queues(self) -> Dict[str, asyncio.Queue]:
        """Create all queues with no size limits - let them build up!"""
        return {
            'query': asyncio.Queue(),           # QueryGenerator → ResearchAgent
            'research': asyncio.Queue(),        # ResearchAgent → InterpreterAgent  
            'interpretation': asyncio.Queue(),  # InterpreterAgent → QualityAgent
            'quality': asyncio.Queue(),         # QualityAgent → ResearchAmplifier
            'amplification': asyncio.Queue(),   # ResearchAmplifier → BuilderAgent
            'building': asyncio.Queue(),        # BuilderAgent → EvaluatorAgent
            'evaluation': asyncio.Queue()       # EvaluatorAgent → Results
        }
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get current size of all queues"""
        return {name: queue.qsize() for name, queue in self.queues.items()}
    
    async def wait_until_all_processed(self):
        """Wait for all queues to be empty (for testing/shutdown)"""
        for queue_name, queue in self.queues.items():
            await queue.join()
            print(f"✅ {queue_name} queue processed")