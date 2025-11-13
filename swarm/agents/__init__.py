"""
Perpetual Queue-Based Research Agents
"""

from .base import BaseResearchAgent, ResearchTask
from .query_generator import QueryGeneratorAgent
from .scout import ResearchAgent
from .interpreter import InterpreterAgent
from .quality import QualityAgent
from .research_amplifier import ResearchAmplifierAgent
from .builder import BuilderAgent
from .evaluator import EvaluatorAgent

__all__ = [
    # Base classes
    'BaseResearchAgent',
    'ResearchTask',
    
    # Agent implementations
    'QueryGeneratorAgent',
    'ResearchAgent', 
    'InterpreterAgent',
    'QualityAgent',
    'ResearchAmplifierAgent',
    'BuilderAgent',
    'EvaluatorAgent',
]

# Agent descriptions for reference
AGENT_DESCRIPTIONS = {
    'QueryGeneratorAgent': 'Continuously generates pure mathematical research queries',
    'ResearchAgent': 'Researches mathematical concepts - only runs when queue has items',
    'InterpreterAgent': 'Interprets research discoveries into structured concepts',
    'QualityAgent': 'Quality checks concepts for trading potential',
    'ResearchAmplifierAgent': 'Amplifies research with crypto perpetual futures applications', 
    'BuilderAgent': 'Builds complete crypto trading strategies from research',
    'EvaluatorAgent': 'Evaluates strategies with crypto data and backtesting',
}

# Pipeline order
PIPELINE_ORDER = [
    'QueryGeneratorAgent',
    'ResearchAgent', 
    'InterpreterAgent',
    'QualityAgent',
    'ResearchAmplifierAgent',
    'BuilderAgent',
    'EvaluatorAgent',
]