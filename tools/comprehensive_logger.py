import logging
import json
import pandas as pd
from datetime import datetime
import os
from typing import Dict, List

class ComprehensiveLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup main logger
        self.logger = logging.getLogger('research_bot')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler for detailed logs
        log_file = os.path.join(log_dir, f"research_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Results storage
        self.results_file = os.path.join(log_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.all_results = []
    
    def log_strategy_extraction(self, paper: Dict, strategy: Dict = None):
        """Log paper processing results"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'paper_title': paper.get('title'),
            'paper_categories': paper.get('categories', []),
            'extraction_success': strategy is not None,
            'strategy_name': strategy.get('name') if strategy else None,
            'strategy_type': strategy.get('type') if strategy else None,
            'parameters': strategy.get('parameters') if strategy else None
        }
        self.all_results.append(entry)
        
        if strategy:
            self.logger.info(f"📄 EXTRACTED: {paper['title'][:60]}... → {strategy['name']}")
        else:
            self.logger.debug(f"📄 SKIPPED: {paper['title'][:60]}... - No tradable strategy")
    
    def log_backtest_results(self, strategy_name: str, results: Dict):
        """Log backtest results with full details"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'strategy_name': strategy_name,
            'backtest_results': results,
            'best_sharpe': results.get('best_sharpe', 0),
            'best_asset': results.get('best_asset'),
            'best_timeframe': results.get('best_timeframe'),
            'total_tests': len(results.get('all_results', []))
        }
        self.all_results.append(entry)
        
        self.logger.info(f"🎯 BACKTEST: {strategy_name} | Sharpe: {results.get('best_sharpe', 0):.2f} | Asset: {results.get('best_asset')} | TF: {results.get('best_timeframe')}")
    
    def save_final_report(self):
        """Save comprehensive results report"""
        # Save JSON results
        with open(self.results_file, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        
        # Generate summary CSV
        summary_data = []
        for result in self.all_results:
            if 'backtest_results' in result:
                summary_data.append({
                    'strategy_name': result['strategy_name'],
                    'best_sharpe': result['best_sharpe'],
                    'best_asset': result['best_asset'],
                    'best_timeframe': result['best_timeframe'],
                    'total_tests': result['total_tests'],
                    'timestamp': result['timestamp']
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_file = self.results_file.replace('.json', '_summary.csv')
            df.to_csv(csv_file, index=False)
            self.logger.info(f"💾 Saved results: {csv_file}")
        
        return self.results_file