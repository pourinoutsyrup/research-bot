import logging
from typing import Dict, Any, Optional
from datetime import datetime

class ResearchTask:
    def __init__(self, source: str, query: str, priority: int = 1, metadata: Dict = None):
        self.source = source
        self.query = query
        self.priority = priority
        self.metadata = metadata or {}
        self.created_at = datetime.now()

class BaseResearchAgent:
    """Base agent for all ATHENA agents"""
    
    def __init__(self, name: str, deepseek_client, **kwargs):
        self.name = name
        self.deepseek = deepseek_client
        self.logger = logging.getLogger(f"agent.{name}")
    
    async def extract_mathematical_concepts(self, content, content_type="research"):
        """Extract math concepts using DeepSeek"""
        try:
            if content_type == "research":
                formatted_content = {
                    'title': content.get('title', 'Research Content'),
                    'summary': content.get('content', content.get('summary', str(content))),
                    'source': content.get('source', 'unknown')
                }
            else:
                formatted_content = {
                    'title': 'Mathematical Concept Search',
                    'summary': str(content),
                    'source': 'general'
                }
            
            if hasattr(self.deepseek, 'extract_tradable_strategy'):
                return self.deepseek.extract_tradable_strategy(formatted_content)
            return None
            
        except Exception as e:
            self.logger.error(f"Concept extraction failed: {e}")
            return None

    async def execute(self, task):
        raise NotImplementedError(f"Execute method not implemented for {self.name}")