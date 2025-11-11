# test_deepseek.py
import os
from dotenv import load_dotenv
from workflows.ai_strategy_extractor import DeepSeekStrategyExtractor

load_dotenv()  # ← This was missing

def test_deepseek():
    extractor = DeepSeekStrategyExtractor()
    
    test_paper = {
        'title': 'Statistical Arbitrage in Cryptocurrency Markets',
        'summary': 'We develop a pairs trading strategy for major cryptocurrencies. We identify cointegrated pairs using the Engle-Granger test and generate trading signals when the price spread deviates by more than 2 standard deviations from the mean. Positions are entered when the z-score exceeds 2.0 and exited when it returns to zero.',
        'authors': ['John Smith', 'Jane Doe'],
        'categories': ['q-fin.TR', 'q-fin.ST'],
        'pdf_url': 'http://example.com/paper.pdf',
        'published': '2024-01-01'
    }
    
    print("Testing DeepSeek extraction...")
    strategy = extractor.extract_tradable_strategy(test_paper)
    
    if strategy:
        print("✅ SUCCESS: Found tradable strategy!")
        print(f"Name: {strategy['name']}")
        print(f"Type: {strategy['type']}")
        print(f"Logic: {strategy['logic']}")
        print(f"Confidence: {strategy['extraction_confidence']}")
    else:
        print("❌ No tradable strategy found")

if __name__ == "__main__":
    test_deepseek()