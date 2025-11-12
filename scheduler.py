# scheduler.py
import schedule
import time
import logging
import traceback
from datetime import datetime
import settings
from main import ResearchPipeline
from tools.discord import DiscordAlerter

class PipelineScheduler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.discord_alerter = DiscordAlerter()
        self.pipeline = ResearchPipeline()
        self.failure_count = 0
        self.max_failures = 3

    def run_pipeline_safely(self, categories=None):
        """Run pipeline with comprehensive error handling"""
        try:
            self.logger.info(f"🚀 Starting scheduled pipeline run for categories: {categories}")
            
            results = self.pipeline.run(categories)
            
            # Reset failure count on success
            self.failure_count = 0
            
            self.logger.info("✅ Pipeline completed successfully")
            
            # Send success summary
            if results and len(results) > 0:
                exceptional = [r for r in results if r.get('sharpe_ratio', 0) >= settings.MIN_SHARPE_ALERT]
                summary = f"📊 Pipeline Summary - {len(exceptional)} exceptional strategies found\n"
                
                for i, strategy in enumerate(exceptional[:3], 1):
                    summary += f"{i}. {strategy['name']} (Sharpe: {strategy['sharpe_ratio']:.2f})\n"
                
                self.discord_alerter.send_alert(summary)
                
        except Exception as e:
            self.failure_count += 1
            error_msg = f"❌ Pipeline failed (attempt {self.failure_count}): {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            # Send failure alert
            self.discord_alerter.send_alert(error_msg)
            
            # Stop if too many failures
            if self.failure_count >= self.max_failures:
                critical_msg = "🚨 CRITICAL: Pipeline failed 3 times in a row. Stopping scheduler."
                self.logger.error(critical_msg)
                self.discord_alerter.send_alert(critical_msg)
                return False
            
        return True

    def start_scheduler(self):
        """Start the scheduled pipeline using your search strategy"""
        self.logger.info("🕒 Starting pipeline scheduler...")
        
        # Use your search strategy from settings
        # Daily - quant + broad
        schedule.every().day.at("09:00").do(
            lambda: self.run_pipeline_safely(['quant', 'broad'])
        )
        
        # Weekly - all categories
        schedule.every().monday.at("10:00").do(
            lambda: self.run_pipeline_safely(['quant', 'broad', 'novel'])
        )
        
        # Weekend - novel/fringe
        schedule.every().saturday.at("11:00").do(
            lambda: self.run_pipeline_safely(['novel', 'fringe'])
        )
        
        # Initial run
        self.run_pipeline_safely(['quant', 'broad'])
        
        # Main loop
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                self.logger.info("⏹️ Scheduler stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes on error

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scheduler = PipelineScheduler()
    scheduler.start_scheduler()