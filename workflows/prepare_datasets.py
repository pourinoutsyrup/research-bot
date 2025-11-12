"""
Script to pre-download and cache papers from multiple sources
Run this once to build your local dataset
"""
import arxiv
import asyncio
import pickle
from pathlib import Path

async def download_arxiv_dataset():
    """Download a broad set of papers from arXiv for caching"""
    categories = [
        'q-fin.*',  # Quantitative finance
        'stat.ML',  # Machine learning
        'cs.LG',    # Machine learning
        'math.OC',  # Optimization and control
        'math.ST',  # Statistics
        'physics.data-an'  # Data analysis
    ]
    
    all_papers = []
    
    for category in categories:
        print(f"📥 Downloading papers from {category}...")
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=100,  # Conservative limit
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        try:
            papers = []
            for result in client.results(search):
                papers.append({
                    'title': result.title,
                    'summary': result.summary,
                    'published': result.published,
                    'authors': [author.name for author in result.authors],
                    'categories': result.categories,
                    'pdf_url': result.pdf_url,
                    'source': 'arxiv',
                    'category': category
                })
            
            all_papers.extend(papers)
            print(f"✅ Downloaded {len(papers)} papers from {category}")
            
            # Be nice to arXiv - delay between categories
            await asyncio.sleep(10)
            
        except Exception as e:
            print(f"❌ Failed to download {category}: {e}")
            continue
    
    # Save the dataset
    output_path = Path("data/datasets/arxiv_dataset.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(all_papers, f)
    
    print(f"💾 Saved {len(all_papers)} papers to {output_path}")

if __name__ == "__main__":
    asyncio.run(download_arxiv_dataset())