# scrapers/base_scraper.py
import requests
import logging
import time
from typing import Optional

class BaseScraper:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def safe_request(self, url: str, method: str = "GET", **kwargs):
        """Make a safe HTTP request with error handling"""
        try:
            response = self.session.request(method, url, timeout=30, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for {url}: {e}")
            return None

    def delay_request(self, delay_seconds: float = 2.0):
        """Add delay between requests to be respectful"""
        time.sleep(delay_seconds)