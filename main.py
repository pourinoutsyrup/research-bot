import asyncio
import logging
import os
from decimal import Decimal
from coordinator.x402_coordinator import X402Coordinator

async def main():
    """Main entry point for Athena-X with x402 integration"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("athena_x")
    
    try:
        logger.info("🚀 Starting Athena-X: Autonomous Quant Research Swarm with x402")
        
        # Initialize x402 coordinator with treasury wallet
        wallet_address = os.getenv('X402_WALLET_ADDRESS', '0xYourTreasuryWalletAddress')
        coordinator = X402Coordinator(wallet_address)
        
        # Initialize autonomous swarm
        coordinator.initialize_autonomous_swarm()
        
        logger.info("✅ Athena-X initialized with x402 treasury")
        logger.info(f"💰 Daily budget: {coordinator.treasury.daily_limit}")
        logger.info(f"👥 Active agents: {len(coordinator.agents)}")
        
        # Start autonomous research cycle
        await coordinator.run_autonomous_research_cycle()
        
    except Exception as e:
        logger.error(f"Athena-X failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())