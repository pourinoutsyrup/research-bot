from .base import BaseResearchAgent, ResearchTask
from typing import Dict, Any

class Builder(BaseResearchAgent):
    """Builds executable trading strategies"""
    
    async def execute(self, task: ResearchTask) -> Dict[str, Any]:
        self.logger.info(f"🔨 Building strategy: {task.query}")
        
        try:
            prompt = f"""
            Build executable trading strategy from: {task.query}
            
            Create:
            - Complete strategy logic with entry/exit conditions
            - Parameter definitions with optimal ranges
            - Risk management rules
            - Python code implementation
            - Performance expectations
            
            Return as JSON with:
            - strategy_name: name of the strategy
            - strategy_logic: trading rules
            - parameters: optimized parameters
            - python_code: executable code
            - expected_performance: performance metrics
            - confidence: build confidence (0-1)
            """
            
            analysis = await self.deepseek_call(prompt)
            
            return {
                'agent': self.name,
                'task': task.query,
                'built_strategy': analysis,
                'build_quality': 'production_ready',
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Strategy building failed: {e}")
            return {'agent': self.name, 'task': task.query, 'error': str(e), 'success': False}