"""
ATHENA: Autonomous Swarm Architecture for Continuous Quantitative Research
"""
from .coordinator import AthenaCoordinator
from .data.memory import ResearchMemory, PerformanceTracker

__all__ = ['AthenaCoordinator', 'ResearchMemory', 'PerformanceTracker']