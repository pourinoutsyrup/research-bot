import asyncio
import logging
from typing import List
from swarm.agents.query_generator import QueryGeneratorAgent
from swarm.agents.scout import ResearchAgent
from swarm.agents.interpreter import InterpreterAgent
from swarm.agents.quality import QualityAgent
from swarm.agents.research_amplifier import ResearchAmplifierAgent
from swarm.agents.builder import BuilderAgent
from swarm.agents.evaluator import EvaluatorAgent
from .queue_manager import ResearchQueueManager

class PerpetualCoordinator:
    """Coordinates the perpetual queue-based research pipeline"""
    
    def __init__(self, deepseek_client):
        self.deepseek = deepseek_client
        self.queue_manager = ResearchQueueManager()
        self.agents = []
        self.logger = logging.getLogger("coordinator")
    
    def initialize_agents(self):
        """Initialize all agents with their queues"""
        queues = self.queue_manager.queues
        
        self.agents = [
            # Producer - runs continuously
            QueryGeneratorAgent(
                deepseek_client=self.deepseek,
                output_queue=queues['query']
            ),
            
            # Consumers - only run when queue has items
            ResearchAgent(
                deepseek_client=self.deepseek,
                input_queue=queues['query'],
                output_queue=queues['research']
            ),
            
            InterpreterAgent(
                deepseek_client=self.deepseek, 
                input_queue=queues['research'],
                output_queue=queues['interpretation']
            ),
            
            QualityAgent(
                deepseek_client=self.deepseek,
                input_queue=queues['interpretation'],
                output_queue=queues['quality']
            ),
            
            ResearchAmplifierAgent(
                deepseek_client=self.deepseek,
                input_queue=queues['quality'], 
                output_queue=queues['amplification']
            ),
            
            BuilderAgent(
                deepseek_client=self.deepseek,
                input_queue=queues['amplification'],
                output_queue=queues['building']
            ),
            
            EvaluatorAgent(
                deepseek_client=self.deepseek,
                input_queue=queues['building'],
                output_queue=queues['evaluation'] 
            )
        ]
        
        self.logger.info(f"✅ Initialized {len(self.agents)} perpetual agents")
    
    async def start_perpetual_pipeline(self):
        """Start all agents running perpetually"""
        self.logger.info("🚀 Starting perpetual research pipeline...")
        
        # Start all agents in parallel
        agent_tasks = [
            asyncio.create_task(agent.run_perpetually())
            for agent in self.agents
        ]
        
        # Monitor queue sizes periodically
        monitor_task = asyncio.create_task(self.monitor_queues())
        
        # Wait for all tasks (they run forever)
        await asyncio.gather(*agent_tasks, monitor_task)
    
    async def monitor_queues(self):
        """Monitor queue sizes and log periodically"""
        while True:
            stats = self.queue_manager.get_queue_stats()
            self.logger.info(f"📊 Queue Stats: {stats}")
            await asyncio.sleep(60)  # Log every minute
    
    def scale_agent_type(self, agent_class, count: int):
        """Scale specific agent type by adding more instances"""
        # Implementation for adding more agents to handle queue load
        pass