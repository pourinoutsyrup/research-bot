# scrapers/quant_specific/arxiv_scraper.py
import arxiv
import logging
from typing import List, Dict
from ..base_scraper import BaseScraper

class ArxivScraper(BaseScraper):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.client = arxiv.Client()

    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search arXiv for papers and return as dictionaries"""
        try:
            # Build search query with relevant categories
            search_query = f"({query}) AND (cat:q-fin.CP OR cat:q-fin.TR OR cat:q-fin.ST OR cat:cs.LG OR cat:cs.AI OR cat:stat.ML)"
            self.logger.info(f"Searching arXiv with query: {search_query}")
            
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            papers = []
            for result in self.client.results(search):
                paper_dict = {
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'summary': result.summary,
                    'pdf_url': result.pdf_url,
                    'published': result.published.isoformat() if result.published else None,
                    'categories': result.categories,
                    'source': 'arxiv'
                }
                papers.append(paper_dict)
            
            self.logger.info(f"Successfully processed {len(papers)} papers from arXiv")
            return papers
            
        except Exception as e:
            self.logger.error(f"Error searching arXiv: {e}")
            return []

    def get_paper_details(self, paper_id: str) -> Dict:
        """Get detailed information for a specific paper"""
        try:
            search = arxiv.Search(id_list=[paper_id])
            results = list(self.client.results(search))
            if results:
                result = results[0]
                return {
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'summary': result.summary,
                    'pdf_url': result.pdf_url,
                    'published': result.published,
                    'categories': result.categories
                }
            return {}
        except Exception as e:
            self.logger.error(f"Error getting paper details: {e}")
            return {}