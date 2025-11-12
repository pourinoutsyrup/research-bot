from .base import BaseResearchAgent, ResearchTask
from typing import Dict, Any

class UniversalScout(BaseResearchAgent):
    """Discovers mathematical concepts from all sources"""
    
    async def execute(self, task: ResearchTask) -> Dict[str, Any]:
        self.logger.info(f"🔍 Universal scout: {task.query}")
        
        try:
            prompt = f"""
            Discover mathematical trading concepts for: {task.query}
            
            Search across:
            - Academic papers and research
            - Technical blogs and forums  
            - GitHub repositories and code
            - Trading communities and discussions
            
            Return as JSON with:
            - technique_name: name of mathematical technique
            - mathematical_formulation: equations/formulas
            - source_types: where this concept appears
            - trading_potential: applicability to crypto trading (0-1)
            - confidence: discovery confidence (0-1)
            """
            
            analysis = await self.deepseek_call(prompt)
            
            return {
                'agent': self.name,
                'task': task.query,
                'discovered_concepts': analysis,
                'concept_count': 1,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Universal scouting failed: {e}")
            return {'agent': self.name, 'task': task.query, 'error': str(e), 'success': False}