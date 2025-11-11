import requests
import logging
from typing import List
from datetime import datetime
from ..base_scraper import BaseScraper, ScrapedContent

class GitHubScraper(BaseScraper):
    def __init__(self):
        super().__init__()
        self.base_url = "https://api.github.com"
        self.source_name = "github"
    
    def search(self, query: str, max_results: int = 20) -> List[ScrapedContent]:
        """Search GitHub for trading repositories"""
        
        search_url = f"{self.base_url}/search/repositories"
        params = {
            'q': f'{query} trading cryptocurrency algorithm',
            'sort': 'updated',
            'order': 'desc',
            'per_page': min(max_results, 100)
        }
        
        try:
            response = self.safe_request(search_url, params=params)
            if not response:
                return []
            
            data = response.json()
            repos = data.get('items', [])
            
            results = []
            for repo in repos[:max_results]:
                content = self._process_repository(repo)
                if content:
                    results.append(content)
            
            return results
            
        except Exception as e:
            logging.error(f"GitHub search error: {e}")
            return []
    
    def _process_repository(self, repo: dict) -> ScrapedContent:
        """Process GitHub repository into our format"""
        
        # Get README content
        readme_content = self._get_readme_content(repo['full_name'])
        
        full_content = f"""
        Repository: {repo['name']}
        Description: {repo.get('description', '')}
        README: {readme_content}
        Stars: {repo['stargazers_count']}
        Language: {repo.get('language', '')}
        """
        
        strategy_mentions = self.extract_strategy_mentions(full_content)
        
        return ScrapedContent(
            source=self.source_name,
            url=repo['html_url'],
            title=repo['name'],
            content=full_content,
            authors=[repo['owner']['login']],
            published_date=repo['created_at'][:10],
            tags=[repo.get('language', ''), f"stars:{repo['stargazers_count']}"],
            strategy_mentions=strategy_mentions
        )
    
    def _get_readme_content(self, full_name: str) -> str:
        """Get README content from repository"""
        try:
            readme_url = f"{self.base_url}/repos/{full_name}/readme"
            response = self.safe_request(readme_url)
            if response:
                data = response.json()
                # Could decode base64 content here if needed
                return data.get('name', 'README.md')
        except:
            pass
        return ""