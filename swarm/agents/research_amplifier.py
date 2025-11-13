import asyncio
import logging
from typing import Dict, Any, Union
from datetime import datetime
from .base import BaseResearchAgent, ResearchTask

class ResearchAmplifierAgent(BaseResearchAgent):
    """Amplifies research with crypto applications - only runs when queue has items"""
    
    def __init__(self, deepseek_client, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
        super().__init__("research_amplifier", deepseek_client, input_queue, output_queue)
    
    async def execute(self, task_data: Union[ResearchTask, Dict]) -> Dict[str, Any]:
        """Amplify research with crypto perpetual futures applications"""
        
        # Check if this is quality checked concept
        if not isinstance(task_data, dict) or task_data.get('type') != 'quality_checked_concept':
            self.logger.error(f"Invalid input type: {type(task_data)}")
            return None
        
        quality_approved_data = task_data
        
        if not quality_approved_data.get('passed_quality', False):
            self.logger.info("Skipping - failed quality check")
            return None
        
        prompt = f"""
        Find CRYPTO PERPETUAL FUTURES trading applications for this mathematical concept:
        
        MATHEMATICAL CONCEPT: {quality_approved_data['original_interpretation']['original_research']['original_task']['query']}
        QUALITY SCORES: {quality_approved_data['scores']}
        
        Focus specifically on:
        - Crypto perpetual futures markets (BTC, ETH, SOL, etc.)
        - Funding rate arbitrage opportunities
        - Leverage and risk management applications
        - High-frequency or medium-frequency trading
        - Market microstructure applications
        
        Return practical crypto trading applications only.
        """
        
        crypto_applications = await self.deepseek_call(prompt)
        
        return {
            'type': 'amplified_research',
            'original_quality_check': quality_approved_data,
            'crypto_applications': crypto_applications,
            'amplified_at': datetime.now().isoformat(),
            'agent': self.name
        }