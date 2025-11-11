import requests
from bs4 import BeautifulSoup
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_ssrn_with_headers():
    """Test SSRN with proper headers to avoid blocking"""
    logging.info("Testing SSRN with enhanced headers...")
    
    url = "https://papers.ssrn.com/sol3/research?q=cryptocurrency+trading"
    
    # Enhanced headers to mimic real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        session = requests.Session()
        session.headers.update(headers)
        
        response = session.get(url, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        
        if response.status_code == 403:
            logging.error("Still getting 403. Let's try a different approach.")
            return False
            
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        paper_links = soup.find_all('a', href=lambda x: x and 'abstract_id' in x)
        
        logging.info(f"Found {len(paper_links)} paper links")
        
        for link in paper_links[:3]:
            title = link.get_text().strip()
            href = link.get('href')
            print(f"Title: {title}")
            print(f"URL: https://papers.ssrn.com{href}")
            print("-" * 50)
            
        return True
        
    except Exception as e:
        logging.error(f"Error: {e}")
        return False

def test_arxiv_as_fallback():
    """Test arXiv as a fallback - they have a good API"""
    logging.info("Testing arXiv API as reliable fallback...")
    
    try:
        import arxiv
        client = arxiv.Client()
        
        search = arxiv.Search(
            query="cryptocurrency AND trading",
            max_results=5,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        results = list(client.results(search))
        
        logging.info(f"Found {len(results)} arXiv papers")
        
        for result in results:
            print(f"Title: {result.title}")
            print(f"Authors: {', '.join(a.name for a in result.authors)}")
            print(f"PDF: {result.pdf_url}")
            print(f"Published: {result.published}")
            print(f"Summary: {result.summary[:200]}...")
            print("-" * 50)
            
        return True
        
    except Exception as e:
        logging.error(f"arXiv error: {e}")
        return False

if __name__ == "__main__":
    logging.info("Starting scraper tests...")
    
    # Test SSRN first
    ssrn_success = test_ssrn_with_headers()
    
    if not ssrn_success:
        logging.info("SSRN failed, trying arXiv...")
        test_arxiv_as_fallback()