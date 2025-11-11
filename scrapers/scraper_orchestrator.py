# scrapers/scraper_orchestrator.py
import logging
from typing import List, Dict
import sys
import os

# Add the config folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
import config.settings as settings

from .quant_specific.arxiv_scraper import ArxivScraper
from .quant_specific.ssrn_scraper import SSRNScraper

class EnhancedScraperOrchestrator:
    def __init__(self):
        self.sources = [
            self.search_semantic_scholar,  # Primary - free, good coverage
            self.search_crossref,          # Massive database
            self.search_openalex,          # Modern, comprehensive
            self.search_arxiv              # Keep as backup
        ]
    
    def search_with_mode(self, categories: List[str] = None) -> List[Dict]:
        all_papers = []
        
        # Broader, more effective queries
        queries = [
            'statistical arbitrage financial markets',
            'high frequency trading algorithms',
            'quantitative trading strategies',
            'market microstructure prediction',
            'algorithmic trading machine learning',
            'pairs trading cryptocurrency',
            'volatility forecasting models',
            'liquidity provision algorithms',
            'portfolio optimization techniques',
            'risk management trading'
        ]
        
        for query in queries:
            for source in self.sources:
                try:
                    papers = source(query, max_results=15)
                    all_papers.extend(papers)
                    time.sleep(0.5)  # Gentle rate limiting
                except Exception as e:
                    self.logger.warning(f"Failed to search {source.__name__}: {e}")
        
        return self._deduplicate_papers(all_papers)
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicates based on title similarity"""
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            # Simple title-based deduplication
            title_key = paper['title'].lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)
        
        return unique_papers