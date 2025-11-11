import logging
from scrapers.quant_specific.arxiv_scraper import ArxivScraper
from workflows.strategy_builder import StrategyBuilder

def test_basic_flow():
    """Test the basic scrape -> build flow"""
    logging.basicConfig(level=logging.INFO)
    
    # Test with just one query and one paper
    scraper = ArxivScraper()
    builder = StrategyBuilder()
    
    print("🔍 Testing basic pipeline...")
    
    # Get one paper
    papers = scraper.search("crypto arbitrage", max_results=1)
    
    if papers:
        paper = papers[0]
        print(f"📄 Paper: {paper.title}")
        print(f"📝 Content preview: {paper.content[:200]}...")
        
        # Build strategy
        strategy = builder.extract_strategy_from_paper(
            paper.content, paper.title, paper.url
        )
        
        if strategy:
            print(f"✅ Strategy built: {strategy.name}")
            print(f"📊 Type: {strategy.logic}")
            print(f"🔧 Parameters: {strategy.parameters}")
            print(f"📈 Code generated: {len(strategy.code_snippet)} chars")
        else:
            print("❌ No strategy extracted")
    else:
        print("❌ No papers found")

if __name__ == "__main__":
    test_basic_flow()