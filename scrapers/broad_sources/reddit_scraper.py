import requests
import logging
from typing import List
from ..base_scraper import BaseScraper, ScrapedContent

class RedditScraper(BaseScraper):
    def __init__(self):
        super().__init__()
        self.base_url = "https://www.reddit.com"
        self.source_name = "reddit"
        self.session.headers.update({'User-Agent': 'TradingResearchBot/1.0'})
    
    def search(self, query: str, max_results: int = 20) -> List[ScrapedContent]:
        """Search Reddit for trading discussions"""
        
        subreddits = ['algotrading', 'quant', 'wallstreetbets', 'cryptocurrency']
        all_results = []
        
        for subreddit in subreddits:
            try:
                url = f"{self.base_url}/r/{subreddit}/search.json"
                params = {
                    'q': query,
                    'restrict_sr': 'on',
                    'sort': 'relevance',
                    'limit': min(10, max_results)
                }
                
                response = self.safe_request(url, params=params)
                if not response:
                    continue
                
                data = response.json()
                posts = data.get('data', {}).get('children', [])
                
                for post in posts[:5]:
                    post_data = post['data']
                    content = self._process_post(post_data)
                    if content:
                        all_results.append(content)
                        
            except Exception as e:
                logging.error(f"Reddit search error for r/{subreddit}: {e}")
        
        return all_results[:max_results]
    
    def _process_post(self, post_data: dict) -> ScrapedContent:
        """Process Reddit post into our format"""
        
        full_content = f"""
        Title: {post_data['title']}
        Content: {post_data.get('selftext', '')}
        Score: {post_data['score']}
        Comments: {post_data['num_comments']}
        """
        
        strategy_mentions = self.extract_strategy_mentions(full_content)
        
        return ScrapedContent(
            source=self.source_name,
            url=f"https://reddit.com{post_data['permalink']}",
            title=post_data['title'],
            content=full_content,
            authors=[post_data['author']],
            published_date=datetime.fromtimestamp(post_data['created_utc']).strftime('%Y-%m-%d'),
            tags=[f"r/{post_data['subreddit']}", f"score:{post_data['score']}"],
            strategy_mentions=strategy_mentions
        )