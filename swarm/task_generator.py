# research_bot/swarm/core.py - UPDATED FOR 6 AGENTS + PARALLEL

import asyncio
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ResearchTask:
    source: str
    query: str
    priority: int = 1
    timestamp: datetime = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

class BaseResearchAgent:
    def __init__(self, name: str, deepseek_client):
        self.name = name
        self.deepseek = deepseek_client
        self.logger = logging.getLogger(f"agent.{name}")
        self.results = []
    
    async def execute(self, task):
        raise NotImplementedError

class ResearchSwarm:
    def __init__(self, deepseek_client, max_workers: int = 15):  # Increased workers
        self.deepseek = deepseek_client
        self.agents = {}
        self.task_queue = asyncio.Queue()
        self.max_workers = max_workers
        self.results = []
        self.logger = logging.getLogger("swarm")
        self.initialize_agents()
    
    def initialize_agents(self):
        """Initialize all 6 core agents"""
        try:
            # Import agents dynamically to avoid circular imports
            from .agents import (
                ArxivMiningAgent, GitHubScoutingAgent, 
                StochasticProcessExpert, OptimizationSpecialist,
                StatisticalInferenceAgent, StrategySynthesisAgent
            )
            
            # SOURCE MINING AGENTS
            self.agents['arxiv_miner'] = ArxivMiningAgent(self.deepseek)
            self.agents['github_scout'] = GitHubScoutingAgent(self.deepseek)
            
            # MATHEMATICAL EXPERT AGENTS  
            self.agents['stochastic_expert'] = StochasticProcessExpert(self.deepseek)
            self.agents['optimization_specialist'] = OptimizationSpecialist(self.deepseek)
            self.agents['statistical_inference'] = StatisticalInferenceAgent(self.deepseek)
            
            # SYNTHESIS AGENT
            self.agents['strategy_synthesis'] = StrategySynthesisAgent(self.deepseek)
            
            self.logger.info(f"🚀 Research Swarm initialized with {len(self.agents)} core agents")
            
        except ImportError as e:
            self.logger.error(f"❌ Failed to import agents: {e}")
            self._create_fallback_agents()
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize agents: {e}")
            self._create_fallback_agents()
    
    def _create_fallback_agents(self):
        """Create fallback agents if imports fail"""
        self.logger.warning("🔄 Creating fallback agents...")
        
        for agent_name in ['arxiv_miner', 'github_scout', 'stochastic_expert', 
                          'optimization_specialist', 'statistical_inference', 'strategy_synthesis']:
            
            class FallbackAgent(BaseResearchAgent):
                async def execute(self, task):
                    self.logger.info(f"🔄 Fallback {self.name} processing: {task.query}")
                    return {
                        'agent': self.name, 
                        'task': task.query,
                        'note': f'Fallback agent - {self.name} not fully implemented'
                    }
            
            self.agents[agent_name] = FallbackAgent(agent_name, self.deepseek)
    
    async def add_tasks(self, tasks):
        """Add tasks to the queue"""
        for task in tasks:
            await self.task_queue.put(task)
        self.logger.info(f"📋 Added {len(tasks)} tasks to queue")
    
    async def process_tasks(self):
        """Process all tasks in parallel with 15 workers"""
        self.logger.info(f"🔄 Processing tasks with {self.max_workers} workers for {len(self.agents)} agents...")
        
        workers = []
        for i in range(self.max_workers):
            worker = asyncio.create_task(self.worker_loop(f"worker-{i}"))
            workers.append(worker)
        
        # Wait for all tasks to complete
        await self.task_queue.join()
        
        # Cancel workers
        for worker in workers:
            worker.cancel()
        
        self.logger.info(f"✅ All tasks processed. Total results: {len(self.results)}")
        return self.results
    
    async def worker_loop(self, worker_name: str):
        """Individual worker processing loop"""
        while True:
            try:
                task = await self.task_queue.get()
                await self.process_task(task)
                self.task_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_name} failed: {e}")
                self.task_queue.task_done()
    
    async def process_task(self, task):
        """Process a single task with the appropriate agent"""
        try:
            if task.source in self.agents:
                result = await self.agents[task.source].execute(task)
                self.results.append(result)
                self.logger.info(f"✅ {task.source} completed: {task.query[:50]}...")
            else:
                self.logger.warning(f"❌ No agent for source: {task.source}")
        except Exception as e:
            self.logger.error(f"❌ Task failed {task.source}: {e}")