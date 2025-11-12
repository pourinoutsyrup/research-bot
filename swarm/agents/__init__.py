# Export all agents for easy importing
from .base import BaseResearchAgent, ResearchTask

# Import your actual agents
from .universal_scout import UniversalScout
from .interpreter import Interpreter
from .builder import Builder
from .evaluator import Evaluator
from .quality import Quality
from .analyst import Analyst

# Agent registry - using your actual agent names
AGENT_REGISTRY = {
    'universal_scout': UniversalScout,
    'interpreter': Interpreter, 
    'builder': Builder,
    'evaluator': Evaluator,
    'quality': Quality,
    'analyst': Analyst,
}