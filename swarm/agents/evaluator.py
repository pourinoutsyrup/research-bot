from .base import BaseResearchAgent, ResearchTask
from typing import Dict, Any

class Evaluator(BaseResearchAgent):
    """Evaluates strategy performance and quality"""
    
    async def execute(self, task: ResearchTask) -> Dict[str, Any]:
        self.logger.info(f"📊 Evaluating: {task.query}")
        
        try:
            prompt = f"""
            Evaluate this trading strategy: {task.query}
            
            Assess:
            - Mathematical soundness
            - Market applicability
            - Risk-adjusted returns potential
            - Implementation feasibility
            - Robustness across market conditions
            
            Return as JSON with:
            - mathematical_soundness: score (0-1)
            - market_applicability: score (0-1)
            - risk_adjusted_potential: score (0-1)
            - implementation_feasibility: score (0-1)
            - overall_confidence: overall score (0-1)
            - recommendations: improvement suggestions
            """
            
            analysis = await self.deepseek_call(prompt)
            
            return {
                'agent': self.name,
                'task': task.query,
                'evaluation': analysis,
                'evaluation_quality': 'comprehensive',
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {'agent': self.name, 'task': task.query, 'error': str(e), 'success': False}