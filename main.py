import asyncio
import logging
import os
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    logger.info("🚀 Starting Athena Research Pipeline...")
    
    try:
        # Use your actual DeepSeek class
        from workflows.ai_strategy_extractor import DeepSeekExtractor
        deepseek = DeepSeekExtractor(api_key=os.getenv('DEEPSEEK_API_KEY'))
        logger.info("✅ DeepSeek Extractor loaded")
    except ImportError as e:
        logger.error(f"❌ Failed to import DeepSeekExtractor: {e}")
        return
    except Exception as e:
        logger.error(f"❌ DeepSeek initialization failed: {e}")
        return
    
    try:
        from coordinator import AthenaCoordinator
        # Use SWARM_CONFIG from settings
        coordinator = AthenaCoordinator(deepseek, settings.SWARM_CONFIG)
        logger.info("✅ Athena Coordinator initialized")
    except Exception as e:
        logger.error(f"❌ Coordinator initialization failed: {e}")
        return
    
    # Use your ACTIVE_QUERIES from settings
    test_queries = settings.ACTIVE_QUERIES[:3]  # Test with first 3 queries
    
    logger.info(f"🧪 Running research cycle with {len(test_queries)} queries...")
    results = await coordinator.run_research_cycle(test_queries)
    
    logger.info(f"✅ Research complete. Found {len(results)} validated concepts")
    
    for i, result in enumerate(results):
        logger.info(f"📊 Result {i+1}: {result.concept}")
        logger.info(f"   Confidence: {result.confidence_score:.2f}")
        logger.info(f"   Source: {result.source}")

if __name__ == "__main__":
    asyncio.run(main())