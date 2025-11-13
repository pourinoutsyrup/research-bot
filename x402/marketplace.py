import aiohttp
import json
from typing import Dict, List
from decimal import Decimal

class X402ServiceMarketplace:
    """Real marketplace for discoverable paid services"""
    
    def __init__(self):
        self.service_registry = {
            "premium_research": [
                {
                    "name": "MathResearch Pro",
                    "url": "https://api.mathresearch.pro/v1/papers",
                    "cost_per_query": Decimal('3.50'),
                    "quality_score": 0.85,
                    "description": "Premium mathematical research papers"
                },
                {
                    "name": "AcademicAI", 
                    "url": "https://api.academic-ai.com/search",
                    "cost_per_query": Decimal('2.75'),
                    "quality_score": 0.78,
                    "description": "AI-powered academic search"
                }
            ],
            "data_feeds": [
                {
                    "name": "CryptoData Pro",
                    "url": "https://api.cryptodatapro.com/feed",
                    "cost_per_day": Decimal('8.00'),
                    "quality_score": 0.92,
                    "description": "Real-time crypto perpetuals data"
                }
            ],
            "compute_services": [
                {
                    "name": "ComputeGrid",
                    "url": "https://api.computegrid.net/process", 
                    "cost_per_minute": Decimal('0.15'),
                    "quality_score": 0.80,
                    "description": "Distributed computation"
                }
            ]
        }
    
    async def discover_services(self, service_type: str, max_budget: Decimal) -> List[Dict]:
        """Discover available services within budget"""
        available = self.service_registry.get(service_type, [])
        
        # Filter by budget and sort by value (quality/cost)
        affordable = [
            svc for svc in available 
            if svc.get('cost_per_query', Decimal('0')) <= max_budget
        ]
        
        # Sort by value score (quality / cost)
        affordable.sort(key=lambda x: x['quality_score'] / x.get('cost_per_query', Decimal('1')), reverse=True)
        
        return affordable
    
    async def get_service_payment_requirements(self, service_url: str) -> Dict:
        """Get actual payment requirements from service"""
        try:
            async with aiohttp.ClientSession() as session:
                # Services should return HTTP 402 with payment details
                async with session.get(service_url) as response:
                    if response.status == 402:
                        return {
                            'payment_required': True,
                            'amount': Decimal(response.headers.get('X-Payment-Amount')),
                            'currency': response.headers.get('X-Payment-Currency', 'USDC'),
                            'recipient': response.headers.get('X-Payment-Recipient'),
                            'description': response.headers.get('X-Payment-Description', '')
                        }
                    else:
                        return {'payment_required': False}
        except Exception as e:
            return {'error': str(e)}