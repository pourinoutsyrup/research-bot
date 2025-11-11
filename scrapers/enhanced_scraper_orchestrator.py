import requests
import time
import logging
from typing import List, Dict
import arxiv
import socket

class EnhancedScraperOrchestrator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.timeout = 10
        
        # Only use working sources - remove semantic_scholar
        self.sources = [
            ('crossref', self.search_crossref),  # Highest quality
            ('arxiv', self.search_arxiv),        # Latest research
        ]
        
        self.working_sources = set()
    
    def _is_network_available(self, host: str) -> bool:
        """Check if network host is reachable"""
        try:
            socket.create_connection((host, 443), timeout=5)
            return True
        except OSError:
            return False
    
    def search_semantic_scholar(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Semantic Scholar API with quick timeout"""
        # Skip if we know it's not working
        if 'semantic_scholar' not in self.working_sources and len(self.working_sources) > 0:
            return []
            
        try:
            # Quick network check first
            if not self._is_network_available('api.semantic-scholar.org'):
                return []
                
            clean_query = query
            url = "https://api.semantic-scholar.org/graph/v1/paper/search"
            
            params = {
                'query': clean_query,
                'limit': max_results,
                'fields': 'title,abstract,authors,url,year'
            }
            
            # Very short timeout for this problematic API
            response = self.session.get(url, params=params, timeout=8)
            if response.status_code != 200:
                return []
                
            papers = []
            data = response.json()
            
            for paper in data.get('data', []):
                papers.append({
                    'title': paper.get('title', ''),
                    'summary': paper.get('abstract', ''),
                    'authors': [author.get('name') for author in paper.get('authors', [])],
                    'pdf_url': paper.get('url', ''),
                    'published': paper.get('year', ''),
                    'source': 'semantic_scholar'
                })
            
            if papers:
                self.working_sources.add('semantic_scholar')
                self.logger.info(f"✅ Found {len(papers)} papers from Semantic Scholar for '{clean_query}'")
            return papers
            
        except Exception as e:
            # Silent fail - this API is often problematic
            return []
    
    def search_crossref(self, query: str, max_results: int = 12) -> List[Dict]:
        """CrossRef with quality filtering"""
        try:
            url = "https://api.crossref.org/works"
            
            params = {
                'query': query,
                'rows': max_results * 2,  # Get more to filter
                'select': 'title,abstract,author,URL,created,subject,reference-count'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            if response.status_code != 200:
                return []
                
            papers = []
            data = response.json()
            
            for item in data.get('message', {}).get('items', []):
                title = item.get('title', [''])[0] if item.get('title') else ''
                
                # Quality filters
                if not title or len(title) < 15:  # Skip short/empty titles
                    continue
                    
                if 'conference' in str(item.get('subject', [])).lower():  # Prefer journals over conferences
                    continue
                    
                authors = []
                for author in item.get('author', []):
                    given = author.get('given', '')
                    family = author.get('family', '')
                    author_name = f"{given} {family}".strip()
                    if author_name:
                        authors.append(author_name)
                
                papers.append({
                    'title': title,
                    'summary': item.get('abstract', ''),
                    'authors': authors,
                    'pdf_url': item.get('URL', ''),
                    'published': item.get('created', {}).get('date-parts', [[None]])[0][0],
                    'subjects': item.get('subject', []),
                    'reference_count': item.get('reference-count', 0),
                    'source': 'crossref'
                })
            
            # Return top N by quality
            quality_papers = papers[:max_results]
            if quality_papers:
                self.working_sources.add('crossref')
                self.logger.info(f"✅ Found {len(quality_papers)} quality papers from CrossRef for '{query}'")
            return quality_papers
            
        except Exception as e:
            self.logger.error(f"❌ CrossRef search failed: {e}")
            return []
    
    def search_arxiv(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search arXiv as backup"""
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            papers = []
            for result in client.results(search):
                papers.append({
                    'title': result.title,
                    'summary': result.summary,
                    'authors': [author.name for author in result.authors],
                    'pdf_url': result.pdf_url,
                    'published': result.published.year if result.published else None,
                    'categories': result.categories,
                    'source': 'arxiv'
                })
            
            if papers:
                self.working_sources.add('arxiv')
                self.logger.info(f"✅ Found {len(papers)} papers from arXiv for '{query}'")
            else:
                self.logger.info(f"⚠️  No papers found from arXiv for '{query}'")
            return papers
            
        except Exception as e:
            self.logger.error(f"❌ arXiv search failed: {e}")
            return []
    
    def search_with_mode(self, categories: List[str] = None) -> List[Dict]:
        """Enhanced search using queries from settings"""
        all_papers = []
        
        # IMPORTANT: Import settings to get the queries
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
        import settings
        
        # Use queries from settings based on categories
        queries = self._get_queries_from_categories(categories, settings)
        
        self.logger.info(f"🔍 Starting enhanced search with {len(self.sources)} sources")
        self.logger.info(f"🎯 Using {len(queries)} queries from categories: {categories}")
        
        successful_queries = 0
        total_attempts = 0
        
        for i, query in enumerate(queries):
            query_papers = []
            self.logger.info(f"📚 Query {i+1}/{len(queries)}: '{query}'")
            
            for source_name, source_func in self.sources:
                total_attempts += 1
                try:
                    # Quick check if we should skip this source
                    if self.working_sources and source_name not in self.working_sources and len(self.working_sources) > 0:
                        self.logger.debug(f"   Skipping {source_name} - not working")
                        continue
                        
                    papers = source_func(query, max_results=8)
                    if papers:
                        query_papers.extend(papers)
                        successful_queries += 1
                        self.logger.info(f"   ✅ {source_name}: {len(papers)} papers")
                    
                    time.sleep(0.5)  # Gentle rate limiting
                    
                except Exception as e:
                    self.logger.debug(f"   ❌ {source_name} failed: {e}")
                    continue
            
            if query_papers:
                all_papers.extend(query_papers)
                self.logger.info(f"   📊 Total for this query: {len(query_papers)} papers")
            else:
                self.logger.info(f"   ⚠️  No papers found for this query")
        
        # Deduplicate and return
        unique_papers = self._deduplicate_papers(all_papers)
        self.logger.info(f"🎯 Total unique papers collected: {len(unique_papers)}")
        self.logger.info(f"📊 Success rate: {successful_queries}/{total_attempts} source queries")
        
        # Log sources breakdown
        source_counts = {}
        for paper in unique_papers:
            source = paper.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        for source, count in source_counts.items():
            self.logger.info(f"   📊 {source}: {count} papers")
        
        return unique_papers

    def _get_queries_from_categories(self, categories, settings):
        """Get appropriate queries based on requested categories"""
        all_queries = []
        
        if not categories:
            categories = ['quant', 'broad']  # Default
        
        for category in categories:
            if category == 'quant' and hasattr(settings, 'QUANT_QUERIES'):
                all_queries.extend(settings.QUANT_QUERIES)
            elif category == 'broad' and hasattr(settings, 'BROAD_QUERIES'):
                all_queries.extend(settings.BROAD_QUERIES)
            elif category == 'novel' and hasattr(settings, 'NOVEL_QUERIES'):
                all_queries.extend(settings.NOVEL_QUERIES)
            elif category == 'fringe' and hasattr(settings, 'FRINGE_QUERIES'):
                all_queries.extend(settings.FRINGE_QUERIES)
            elif category == 'crypto' and hasattr(settings, 'SCRAPING_QUERIES'):
                all_queries.extend(settings.SCRAPING_QUERIES)
        
        # Remove duplicates and limit to reasonable number
        unique_queries = list(dict.fromkeys(all_queries))  # Preserves order
        return unique_queries[:15]  # Limit to 15 queries max
    
    def _deduplicate_papers(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicates based on title similarity"""
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            if not paper.get('title'):
                continue
                
            # Simple title-based deduplication
            title_key = paper['title'].lower().strip()[:80]
            if title_key and title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_papers.append(paper)
        
        return unique_papers