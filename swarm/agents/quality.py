from .base import BaseResearchAgent, ResearchTask
from typing import Dict, Any

class Quality(BaseResearchAgent):
    """Market applicability checker - assesses tradability but NEVER filters"""
    
    async def execute(self, task: ResearchTask) -> Dict[str, Any]:
        self.logger.info(f"🔎 Quality check: {task.query[:100]}...")
        
        try:
            # PROMPT FOR TRADABILITY ASSESSMENT (NO FILTERING)
            prompt = f"""
            CONCEPT: {task.query}
            
            ASSESSMENT REQUEST:
            Assess this concept's potential for financial market applications.
            
            IMPORTANT: Assume ALL concepts can be creatively applied to markets somehow.
            Your job is NOT to filter, but to provide scoring for human prioritization.
            
            SCORING:
            - Applicable (YES/NO): Is there ANY conceivable market application? (Assume YES unless mathematically impossible)
            - Ease Score (1-10): How easily can this be implemented as a trading strategy?
            - Tradability Score (1-10): What's the potential profit/alpha generation?
            - Novelty Score (1-10): How unique/unconventional is this approach?
            
            RESPONSE FORMAT (JSON only):
            {{
                "applicable": "YES" or "NO" (ONLY use NO if mathematically impossible),
                "ease_score": number between 1-10,
                "tradability_score": number between 1-10, 
                "novelty_score": number between 1-10,
                "reasoning": "brief explanation of scores"
            }}
            
            REMEMBER: Be EXTREMELY PERMISSIVE with "applicable". Only say NO if there's 
            ZERO mathematical connection to markets (e.g., quantum gravity, pure biology).
            """
            
            response = await self.deepseek_call(prompt)
            response_text = response.get('strategy_code', '{}')
            
            # Parse the JSON response
            try:
                import json
                assessment = json.loads(response_text)
                
                # Extract scores (with defaults that assume applicability)
                applicable = assessment.get('applicable', 'YES').upper() == 'YES'
                ease_score = float(assessment.get('ease_score', 5))
                tradability_score = float(assessment.get('tradability_score', 5))
                novelty_score = float(assessment.get('novelty_score', 5))
                reasoning = assessment.get('reasoning', 'No reasoning provided')
                
            except (json.JSONDecodeError, ValueError):
                # If parsing fails, assume highly applicable with medium scores
                applicable = True
                ease_score = 5.0
                tradability_score = 5.0
                novelty_score = 5.0
                reasoning = "Could not parse AI response - assuming applicable"
            
            return {
                'agent': self.name,
                'task': task.query,
                'quality_assessment': {
                    'applicable': applicable,
                    'ease_score': ease_score,
                    'tradability_score': tradability_score,
                    'novelty_score': novelty_score,
                    'reasoning': reasoning
                },
                'assessment_complete': True,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            return {
                'agent': self.name, 
                'task': task.query,
                'quality_assessment': {
                    'applicable': True,  # Always assume applicable on error
                    'ease_score': 3.0,
                    'tradability_score': 3.0,
                    'novelty_score': 3.0,
                    'reasoning': f'Error occurred: {str(e)}'
                },
                'success': False
            }