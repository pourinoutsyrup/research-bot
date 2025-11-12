from .base import BaseResearchAgent, ResearchTask
from typing import Dict, Any

class Analyst(BaseResearchAgent):
    """Analyzes results and provides insights"""
    
    async def execute(self, task: ResearchTask) -> Dict[str, Any]:
        self.logger.info(f"📈 Analyzing: {task.query}")
        
        try:
            prompt = f"""
            Analyze trading strategy results: {task.query}
            
            Provide insights on:
            - Performance characteristics and metrics
            - Market condition suitability
            - Risk-return profile
            - Improvement opportunities
            - Deployment recommendations
            
            Return as JSON with:
            - performance_metrics: key performance indicators
            - market_suitability: which conditions work best
            - risk_return_profile: risk-adjusted assessment
            - improvement_opportunities: how to enhance
            - deployment_recommendation: yes/no with reasoning
            - confidence: analysis confidence (0-1)
            """
            
            analysis = await self.deepseek_call(prompt)
            
            return {
                'agent': self.name,
                'task': task.query,
                'analysis': analysis,
                'insight_quality': 'actionable',
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {'agent': self.name, 'task': task.query, 'error': str(e), 'success': False}