from .base import BaseResearchAgent, ResearchTask
from typing import Dict, Any

class Interpreter(BaseResearchAgent):
    """Interprets and expands mathematical concepts"""
    
    async def execute(self, task: ResearchTask) -> Dict[str, Any]:
        self.logger.info(f"🧮 Interpreting: {task.query}")
        
        try:
            prompt = f"""
            Interpret this mathematical concept for trading: {task.query}
            
            Provide:
            - Precise mathematical formulation
            - Parameter definitions and ranges
            - Trading strategy applications
            - Risk considerations
            - Implementation requirements
            
            Return as JSON with:
            - mathematical_formulation: exact equations
            - parameters: key parameters with ranges
            - trading_strategy: how to trade this
            - risk_factors: potential risks
            - confidence: interpretation confidence (0-1)
            """
            
            analysis = await self.deepseek_call(prompt)
            
            return {
                'agent': self.name,
                'task': task.query,
                'interpretation': analysis,
                'interpretation_quality': 'high',
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Interpretation failed: {e}")
            return {'agent': self.name, 'task': task.query, 'error': str(e), 'success': False}