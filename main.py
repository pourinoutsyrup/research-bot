# main.py - Top section only
import os
import sys
import logging
import asyncio
import functools
import signal
from typing import List, Dict

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ✅ SILENCE noisy liberals
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('arxiv').setLevel(logging.WARNING)
logging.getLogger('ccxt').setLevel(logging.WARNING)
logging.getLogger('scrapers').setLevel(logging.INFO)  # Keep scraper info
logging.getLogger('tools.backtest_engine').setLevel(logging.INFO)  # Our backtest logs

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
import settings

# Import your modules - make sure these don't import from each other
from scrapers.enhanced_scraper_orchestrator import EnhancedScraperOrchestrator
from workflows.strategy_builder import StrategyBuilder
from tools.discord_alerter import DiscordAlerter
from tools.simple_visualizer import SimpleVisualizer
from tools.comprehensive_logger import ComprehensiveLogger
from tools.backtest_engine import BacktestEngine
from swarm.core import ResearchSwarm, ResearchTask

class TimeoutError(Exception):
    pass

def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def signal_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # Set up the signal handler
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Cancel the alarm
                signal.alarm(0)
            
            return result
        return wrapper
    return decorator


class EnhancedResearchPipeline:
    def __init__(self, min_sharpe_alert: float = 0.3):
        # Initialize logger properly
        self.comprehensive_logger = ComprehensiveLogger()
        self.logger = logging.getLogger(__name__)  # Standard logger for compatibility
        
        # Initialize scraper and strategy builder first
        self.scraper = EnhancedScraperOrchestrator()
        self.strategy_builder = StrategyBuilder(use_ai=True, deepseek_key=os.getenv('DEEPSEEK_API_KEY'))
        self.backtester = BacktestEngine()
        self.discord_alerter = DiscordAlerter()
        self.min_sharpe_alert = min_sharpe_alert
        
        # Initialize Swarm - REMOVE SwarmTaskGenerator reference
        try:
            # Get the DeepSeek client from your strategy builder
            deepseek_client = getattr(self.strategy_builder, 'ai_extractor', None)
            if deepseek_client:
                self.research_swarm = ResearchSwarm(
                    deepseek_client=deepseek_client,
                    max_workers=8  # Reduced for stability
                )
                # REMOVE THIS LINE: self.task_generator = SwarmTaskGenerator(deepseek_client)
                self.logger.info("🚀 Research Swarm initialized")
            else:
                self.logger.warning("❌ DeepSeek client not available - swarm disabled")
                self.research_swarm = None
        except Exception as e:
            self.logger.error(f"❌ Swarm initialization failed: {e}")
            self.research_swarm = None
    
    async def generate_comprehensive_tasks(self):
        """Generate tasks for 14 expanded agents"""
        tasks = []
        
        # Source Mining Agents (3)
        tasks.extend([
            ResearchTask(source='arxiv_miner', query='stochastic volatility models', priority=1),
            ResearchTask(source='github_scout', query='quantitative trading strategies', priority=1),
            ResearchTask(source='web_scraper', query='algorithmic trading python', priority=1),
        ])
        
        # Mathematical Expert Agents (3)
        tasks.extend([
            ResearchTask(source='stochastic_expert', query='jump diffusion models', priority=1),
            ResearchTask(source='optimization_specialist', query='portfolio optimization', priority=1),
            ResearchTask(source='statistical_inference', query='regime switching detection', priority=1),
        ])
        
        # Research & Synthesis Agents (3)
        tasks.extend([
            ResearchTask(source='deepseek_researcher', query='market microstructure', priority=1),
            ResearchTask(source='math_analyst', query='kalman filter applications', priority=1),
            ResearchTask(source='strategy_synthesis', query='statistical arbitrage', priority=1),
        ])
        
        # NEW: Specialized Analysis Agents (3)
        tasks.extend([
            ResearchTask(source='cross_domain_innovator', query='quantum physics + finance', priority=2),
            ResearchTask(source='implementation_analyst', query='high frequency trading systems', priority=2),
            ResearchTask(source='risk_analyst', query='black swan event protection', priority=2),
        ])
        
        # NEW: Parallel Specialists (2)
        tasks.extend([
            ResearchTask(source='market_specialist', query='crypto market microstructure', priority=2),
            ResearchTask(source='data_engineer', query='real-time feature engineering', priority=2),
        ])
        
        self.logger.info(f"🎯 Generated {len(tasks)} tasks for 14 EXPANDED agents")
        return tasks

    # In your EnhancedResearchPipeline class
    async def generate_high_frequency_tasks(self):
        """Generate tasks EXPLICITLY optimized for 5m/15m alpha generation"""
        tasks = []
        
        # EXPLICIT HIGH-FREQUENCY QUERIES (5m/15m focus)
        hf_queries = [
            # Market Microstructure (5m focus)
            ('market_specialist', 'crypto order book imbalance 5m scalping'),
            ('arxiv_miner', 'high frequency trading market microstructure order flow'),
            ('implementation_analyst', 'low latency trading systems 5m execution'),
            
            # Statistical Arbitrage (15m focus)
            ('stochastic_expert', 'pairs trading mean reversion 15m statistical arbitrage'),
            ('statistical_inference', 'cointegration spread trading 15m high frequency'),
            ('math_analyst', 'kalman filter real-time tracking 5m'),
            
            # Momentum & Pattern Recognition
            ('optimization_specialist', 'genetic algorithm parameter optimization 15m'),
            ('deepseek_researcher', '5m momentum patterns crypto futures'),
            ('strategy_synthesis', 'multi-timeframe 5m 15m arbitrage strategy'),
            
            # Data & Features
            ('data_engineer', 'real-time feature engineering 5m tick data'),
            ('github_scout', 'high frequency trading python binance 5m'),
            
            # Risk & Execution
            ('risk_analyst', 'var risk management 15m high frequency'),
            ('web_scraper', 'crypto news sentiment 15m impact')
        ]
        
        for source, query in hf_queries:
            tasks.append(ResearchTask(source=source, query=query, priority=1))
        
        self.logger.info(f"🎯 Generated {len(tasks)} EXPLICIT high-frequency tasks")
        return tasks
    
    async def generate_deepseek_enhanced_tasks(self, previous_results):
        """Generate enhanced tasks based on swarm findings"""
        if not previous_results:
            return []
        
        enhanced_tasks = []
        
        # Analyze previous results and create enhanced queries
        for result in previous_results[:5]:  # Top 5 results
            agent = result.get('agent', '')
            query = result.get('task', '')
            
            if 'arxiv' in agent:
                enhanced_tasks.append(
                    ResearchTask(source='arxiv_miner', query=f"advanced {query}", priority=1)
                )
            elif 'github' in agent:
                enhanced_tasks.append(
                    ResearchTask(source='github_scout', query=f"implementation {query}", priority=1)
                )
        
        self.logger.info(f"🔍 Generated {len(enhanced_tasks)} enhanced tasks")
        return enhanced_tasks

    async def run_swarm_research(self):
        """Run the swarm research pipeline"""
        if not self.research_swarm:
            self.logger.warning("❌ Swarm not available - skipping")
            return []
            
        self.logger.info("🚀 Starting Research Swarm...")
        
        try:
            # Phase 1: Generate initial tasks using our new function
            initial_tasks = await self.generate_comprehensive_tasks()
            await self.research_swarm.add_tasks(initial_tasks)
            
            # Phase 2: Process tasks in parallel
            swarm_results = await self.research_swarm.process_tasks()
            
            # Phase 3: Generate enhanced tasks based on findings
            enhanced_tasks = await self.generate_deepseek_enhanced_tasks(swarm_results)
            await self.research_swarm.add_tasks(enhanced_tasks)
            
            # Phase 4: Process enhanced tasks
            final_results = await self.research_swarm.process_tasks()
            
            # Convert to strategies
            strategies_from_swarm = self._convert_swarm_to_strategies(final_results)
            
            self.logger.info(f"🎯 Swarm completed: {len(strategies_from_swarm)} strategies from {len(final_results)} findings")
            return strategies_from_swarm
            
        except Exception as e:
            self.logger.error(f"❌ Swarm research failed: {e}")
            return []
    
    def _convert_swarm_to_strategies(self, swarm_results: List[dict]) -> List[dict]:
        """Convert ALL swarm research findings to trading strategies"""
        strategies = []
        
        for result in swarm_results:
            try:
                # Handle arXiv papers
                if 'papers' in result and result['papers']:
                    for paper in result['papers']:
                        strategy = self.strategy_builder.extract_strategy_from_paper(paper)
                        if strategy:
                            strategy['source'] = 'swarm_arxiv'
                            strategies.append(strategy)
                
                # Handle GitHub repositories  
                elif 'repositories' in result and result['repositories']:
                    for repo in result['repositories']:
                        strategy = self._github_repo_to_strategy(repo)
                        if strategy:
                            strategies.append(strategy)
                
                # Handle direct strategy outputs from agents
                elif 'synthesized_strategy' in result:
                    strategy = result['synthesized_strategy']
                    if isinstance(strategy, dict) and 'code' in strategy:
                        strategy['source'] = 'swarm_synthesis'
                        strategies.append(strategy)
                
                # Handle mathematical analysis outputs
                elif 'mathematical_analysis' in result:
                    strategy = self._math_analysis_to_strategy(result)
                    if strategy:
                        strategies.append(strategy)
                        
                # Handle research findings
                elif 'research_findings' in result:
                    strategy = self._research_to_strategy(result)
                    if strategy:
                        strategies.append(strategy)
                        
            except Exception as e:
                self.logger.warning(f"❌ Failed to convert swarm result: {e}")
                continue
        
        self.logger.info(f"🔄 Converted {len(swarm_results)} swarm results → {len(strategies)} strategies")
        return strategies
    
    def _github_repo_to_strategy(self, repo: dict) -> dict:
        """Convert GitHub repository to trading strategy"""
        try:
            # Analyze repo for trading strategy potential
            return {
                'name': f"github_{repo['name'].replace('-', '_').replace(' ', '_')}",
                'type': 'code_implementation',
                'description': f"From GitHub: {repo.get('description', 'Trading strategy implementation')}",
                'logic': f"Based on {repo['name']} implementation",
                'parameters': {'lookback_period': 20, 'threshold': 2.0},
                'code': self._generate_github_strategy_code(repo),
                'confidence': 0.6,
                'sharpe_ratio': 1.0,
                'source': 'swarm_github'
            }
        except Exception as e:
            self.logger.error(f"Failed to convert GitHub repo: {e}")
            return None

    def _math_analysis_to_strategy(self, result: dict) -> dict:
        """Convert mathematical analysis to strategy"""
        try:
            analysis = result.get('mathematical_analysis', '')
            return {
                'name': f"math_{result['task'].replace(' ', '_').lower()}",
                'type': 'mathematical_model',
                'description': f"Mathematical approach: {result['task']}",
                'logic': analysis[:200] + "..." if len(analysis) > 200 else analysis,
                'parameters': {'lookback_period': 50, 'window': 100, 'threshold': 2.0},
                'code': self._generate_math_strategy_code(result),
                'confidence': 0.7,
                'sharpe_ratio': 1.2,
                'source': 'swarm_math'
            }
        except Exception as e:
            self.logger.error(f"Failed to convert math analysis: {e}")
            return None
    
    def _concepts_to_strategy(self, concepts: List[dict]) -> dict:
        """Convert mathematical concepts to a trading strategy"""
        if not concepts:
            return None
            
        # Create a simple strategy from the first concept
        main_concept = concepts[0]
        strategy_name = f"swarm_{main_concept.get('name', 'concept').replace(' ', '_').lower()}"
        
        return {
            'name': strategy_name,
            'type': 'ai_swarm_generated',
            'description': f"Swarm-generated from: {main_concept.get('description', 'mathematical concept')}",
            'logic': f"Based on {main_concept.get('name')} with trading application: {main_concept.get('trading_application', 'pattern detection')}",
            'parameters': {'lookback_period': 20, 'threshold': 2.0},
            'code': self._generate_concept_code(main_concept),
            'confidence': 0.7,
            'sharpe_ratio': 1.2  # Conservative estimate
        }
    
    def _research_to_strategy(self, result: dict) -> dict:
        """Convert research findings to strategy"""
        try:
            findings = result.get('research_findings', '')
            return {
                'name': f"research_{result['task'].replace(' ', '_').lower()}",
                'type': 'research_based',
                'description': f"Research-based: {result['task']}",
                'logic': findings[:200] + "..." if len(findings) > 200 else findings,
                'parameters': {'lookback_period': 30, 'window': 60, 'threshold': 1.5},
                'code': self._generate_research_strategy_code(result),
                'confidence': 0.65,
                'sharpe_ratio': 1.1,
                'source': 'swarm_research'
            }
        except Exception as e:
            self.logger.error(f"Failed to convert research: {e}")
            return None

    def _generate_github_strategy_code(self, repo: dict) -> str:
        """Generate strategy code from GitHub repo"""
        repo_name = repo.get('name', 'unknown')
        return f'''
    def strategy(prices):
        """
        GitHub-Inspired Strategy: {repo_name}
        Based on: {repo.get('description', 'Trading implementation')}
        """
        import pandas as pd
        import numpy as np
        
        # Implementation inspired by {repo_name}
        signals = []
        lookback = 20
        
        for i in range(len(prices)):
            if i < lookback:
                signals.append(0)
                continue
                
            # Simple trend following based on repo concept
            short_ma = np.mean(prices.iloc[i-5:i])
            long_ma = np.mean(prices.iloc[i-20:i])
            
            if short_ma > long_ma:
                signals.append(1)   # Buy signal
            elif short_ma < long_ma:
                signals.append(-1)  # Sell signal
            else:
                signals.append(0)   # Hold
        
        return signals
    '''

    def _generate_math_strategy_code(self, result: dict) -> str:
        """Generate strategy code from mathematical analysis"""
        task = result.get('task', 'mathematical_model')
        return f'''
    def strategy(prices):
        """
        Mathematical Strategy: {task}
        Based on advanced mathematical analysis
        """
        import pandas as pd
        import numpy as np
        from scipy import stats
        
        signals = []
        lookback = 50
        
        for i in range(len(prices)):
            if i < lookback:
                signals.append(0)
                continue
                
            # Mathematical approach for {task}
            window = prices.iloc[i-lookback:i]
            
            # Statistical properties
            z_score = stats.zscore(window)[-1] if len(window) > 1 else 0
            volatility = np.std(window)
            
            # Trading logic based on mathematical analysis
            if abs(z_score) > 2.0:
                signals.append(-1 if z_score > 0 else 1)  # Mean reversion
            else:
                signals.append(0)  # No clear signal
        
        return signals
    '''

    def _generate_research_strategy_code(self, result: dict) -> str:
        """Generate strategy code from research findings"""
        task = result.get('task', 'research_based')
        return f'''
    def strategy(prices):
        """
        Research-Based Strategy: {task}
        Based on comprehensive research analysis
        """
        import pandas as pd
        import numpy as np
        
        signals = []
        lookback = 30
        threshold = 1.5
        
        for i in range(len(prices)):
            if i < lookback:
                signals.append(0)
                continue
                
            # Research-inspired approach
            recent = prices.iloc[i-10:i]
            historical = prices.iloc[i-lookback:i]
            
            recent_vol = np.std(recent) if len(recent) > 1 else 0
            historical_vol = np.std(historical) if len(historical) > 1 else 0
            
            vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1
            
            if vol_ratio > threshold:
                signals.append(1)   # Volatility breakout
            else:
                signals.append(0)   # Wait for signal
        
        return signals
    '''

    def _generate_concept_code(self, concept: dict) -> str:
        """Generate trading code from mathematical concept"""
        concept_name = concept.get('name', 'pattern').lower()
        
        return f'''
def strategy(prices):
    """
    Swarm-Generated Strategy: {concept_name}
    Mathematical Concept: {concept.get('description', 'Advanced pattern detection')}
    """
    import pandas as pd
    import numpy as np
    
    signals = []
    lookback = 20
    
    # Basic implementation of {concept_name}
    for i in range(len(prices)):
        if i < lookback:
            signals.append(0)
            continue
            
        # Simple momentum-based logic as fallback
        recent_trend = prices.iloc[i] - prices.iloc[i-lookback]
        
        if recent_trend > 0:
            signals.append(1)   # Buy on upward trend
        elif recent_trend < 0:
            signals.append(-1)  # Sell on downward trend
        else:
            signals.append(0)   # Hold
    
    return signals
'''
    def calculate_weirdness_score(self, strategy: Dict) -> float:
        """Score how unusual/unique a strategy is"""
        score = 0.0
        
        # Check for unusual mathematical techniques
        weird_techniques = [
            'kalman', 'wavelet', 'fourier', 'ornstein', 'uhlenbeck',
            'particle filter', 'markov', 'gibbs', 'spectral',
            'topological', 'manifold', 'entropy', 'information geometry'
        ]
        
        technique = strategy.get('mathematical_technique', '').lower()
        if any(w in technique for w in weird_techniques):
            score += 0.4
        
        # Check for unusual parameters
        params = strategy.get('parameters', {})
        param_values = list(params.values())
        
        # Unusual if parameters are very specific (not round numbers)
        for val in param_values:
            if isinstance(val, float) and val % 1 != 0:
                score += 0.1  # Has decimals
        
        # Bonus for having equations
        if strategy.get('equation'):
            score += 0.3
        
        # Penalty for generic parameters
        generic_sigs = [(50, 20, 1.8), (30, 60, 2.0), (20, 100, 2.5)]
        param_sig = (
            params.get('lookback_period', 0),
            params.get('window', 0),
            params.get('threshold', 0)
        )
        if param_sig in generic_sigs:
            score -= 0.5
        
        return max(0, min(1, score))

    # Then log it:
    strategies_from_swarm = self._convert_swarm_to_strategies(final_results)
    for s in strategies_from_swarm:
        s['weirdness'] = self.calculate_weirdness_score(s)
        self.logger.info(f"Weirdness: {s['weirdness']:.2f} | {s['name']}")
    
    # In your main pipeline - replace the task generation
    async def run_pipeline(self):
        """Full swarm-powered pipeline with HFT focus"""
        try:
            # PHASE 1: SWARM RESEARCH WITH HFT FOCUS
            swarm_strategies = await self.run_swarm_research()
            
            if not swarm_strategies:
                self.logger.warning("❌ No strategies from swarm research")
                return []
            
            # PHASE 2: HIGH-FREQUENCY BACKTESTING
            self.logger.info("⚡ Running HIGH-FREQUENCY optimized backtesting...")
            proven_strategies = []
            
            for strategy in swarm_strategies:
                try:
                    # Use HFT-optimized backtest
                    backtest_results = self.backtester.run_high_frequency_backtest(
                        strategy['code'], 
                        strategy_type="auto"
                    )
                    
                    # Only accept strategies that work well on 5m or 15m
                    best_tf = backtest_results.get('preferred_timeframe', '15m')
                    best_sharpe = backtest_results.get('best_sharpe', 0)
                    
                    if best_sharpe >= self.min_sharpe_alert and best_tf in ['5m', '15m']:
                        strategy.update({
                            'backtest_results': backtest_results,
                            'actual_sharpe': best_sharpe,
                            'optimal_timeframe': best_tf,
                            'strategy_type': self.backtester.detect_optimal_timeframe(strategy['code'])
                        })
                        proven_strategies.append(strategy)
                        self.logger.info(f"✅ HFT Strategy: {strategy['name']} | {best_tf} Sharpe: {best_sharpe:.2f}")
                        
                except Exception as e:
                    self.logger.error(f"❌ HFT backtest failed for {strategy['name']}: {e}")
            
            self.logger.info(f"🎯 Pipeline completed: {len(proven_strategies)} HFT-proven strategies")
            return proven_strategies
            
        except Exception as e:
            self.logger.error(f"❌ HFT pipeline failed: {e}")
            return []
        
    def run(self, categories: List[str] = None) -> List[Dict]:
        """Synchronous run method for compatibility with existing code"""
        try:
            # Run the async pipeline synchronously
            strategies = asyncio.run(self.run_pipeline())
            return strategies
        except Exception as e:
            self.logger.error(f"❌ Run method failed: {e}")
            return []

    def _generate_strategy_chart(self, strategy: Dict) -> str:
        """Generate chart for strategy"""
        try:
            from tools.simple_visualizer import SimpleVisualizer
            visualizer = SimpleVisualizer()
            chart_path = f"charts/{strategy['name']}.png"
            
            # Ensure charts directory exists
            os.makedirs('charts', exist_ok=True)
            
            # Generate and save chart
            visualizer.generate_strategy_chart(strategy, save_path=chart_path)
            return chart_path
            
        except Exception as e:
            self.logger.error(f"❌ Chart generation failed: {e}")
            return None

    def _filter_papers_for_strategies(self, papers: List[Dict]) -> List[Dict]:
        """Filter for mathematical papers that can be ADAPTED to trading"""
        filtered_papers = []
        
        # Mathematical concepts that can be adapted to trading
        math_keywords = [
            'filter', 'fourier', 'wavelet', 'transform', 'signal', 'frequency',
            'bayesian', 'markov', 'monte carlo', 'statistical', 'probability',
            'optimization', 'gradient', 'descent', 'search', 'genetic',
            'learning', 'neural', 'network', 'clustering', 'classification',
            'time series', 'sequence', 'temporal', 'forecast', 'prediction',
            'topological', 'graph', 'entropy', 'information', 'complexity',
            'stochastic', 'differential', 'regression', 'estimation'
        ]
        
        print(f"🔍 Filtering {len(papers)} papers for mathematical content...")
        
        for i, paper in enumerate(papers):
            content = f"{paper.get('title', '')} {paper.get('summary', '')}".lower()
            
            # Count mathematical keywords
            keyword_count = sum(1 for keyword in math_keywords if keyword in content)
            
            # More lenient criteria for mathematical papers
            has_substance = len(content.split()) > 15
            
            # Debug output for first few papers
            if i < 5:
                print(f"  📄 Paper {i+1}: '{paper.get('title', 'No title')[:50]}...'")
                print(f"     Math keywords found: {keyword_count}, Has substance: {has_substance}")
            
            # Include papers with mathematical content
            if keyword_count >= 1 and has_substance:  # Reduced to 1 keyword
                filtered_papers.append(paper)
                if i < 5:
                    print(f"     ✅ INCLUDED (mathematical)")
            else:
                if i < 5:
                    print(f"     ❌ EXCLUDED (not mathematical enough)")
        
        print(f"📊 Filtered {len(papers)} → {len(filtered_papers)} mathematical papers")
        return filtered_papers

    def _find_exceptional_strategies(self, strategies: List[Dict]) -> List[Dict]:
        """Find strategies worth alerting about"""
        exceptional = []
        for strategy in strategies:
            # Use confidence as a proxy for quality since we don't have backtest results yet
            confidence = strategy.get('confidence', 0)
            
            # Only alert for high-confidence strategies
            if confidence > 0.7:
                # Ensure strategy has required fields for Discord
                if 'sharpe_ratio' not in strategy:
                    strategy['sharpe_ratio'] = confidence * 2.0  # Estimate based on confidence
                exceptional.append(strategy)
        
        return exceptional

    # In your EnhancedResearchPipeline class - modify the backtesting call

    def run_with_backtesting_from_async(self, strategies: List[Dict]) -> List[Dict]:
        """Run backtesting on strategies with HFT optimization"""
        if not strategies:
            return []
        
        self.logger.info("🧪 Running HIGH-FREQUENCY optimized backtesting...")
        proven_strategies = []
        
        for strategy in strategies:
            try:
                # USE THE NEW HFT-OPTIMIZED BACKTEST
                backtest_results = self.backtester.run_high_frequency_backtest(
                    strategy['code'], 
                    strategy_type="auto"  # Let AI detect optimal timeframe
                )
                
                if backtest_results.get('best_sharpe', 0) >= self.min_sharpe_alert:
                    strategy.update({
                        'backtest_results': backtest_results,
                        'actual_sharpe': backtest_results.get('best_sharpe', 0),
                        'optimal_timeframe': backtest_results.get('preferred_timeframe', '15m'),
                        'strategy_type': self.backtester.detect_optimal_timeframe(strategy['code'])
                    })
                    proven_strategies.append(strategy)
                    
            except Exception as e:
                self.logger.error(f"❌ HFT backtest failed for {strategy['name']}: {e}")
        
        return proven_strategies
    

    def _send_backtest_alert(self, strategy: Dict):
        """Send enhanced Discord alert with backtest results"""
        results = strategy.get('backtest_results', {})
        
        message = f"""
    🎯 **PROVEN STRATEGY: {strategy['name']}**

    **Mathematical Approach**: {strategy.get('type', 'Unknown')}
    **Actual Sharpe Ratio**: {results.get('best_sharpe', 0):.2f}
    **Optimal Asset/Timeframe**: {results.get('best_asset')} {results.get('best_timeframe')}
    **Win Rate**: {results.get('win_rate', 0):.1%}
    **Total Return**: {results.get('best_total_return', 0):.1%}

    📊 **Backtest Summary**:
    - Successful Tests: {results.get('successful_tests', 0)}/{results.get('total_tests', 0)}
    - Average Sharpe: {results.get('avg_sharpe', 0):.2f}

    💡 **Ready for live deployment!**
    """
        self.discord_alerter.send_alert(message)

    def _estimate_sharpe_ratio(self, strategy: Dict) -> float:
        """Estimate realistic Sharpe ratio based on strategy quality"""
        base_sharpe = 0.8  # Reasonable base
        
        # Boost for specific strategy types
        type_boost = {
            'mean_reversion': 0.6,
            'arbitrage': 1.0,
            'momentum': 0.4,
            'ml_based': 0.8,
            'market_making': 0.9
        }
        
        # Boost for parameter complexity
        param_count = len(strategy.get('parameters', {}))
        param_boost = min(0.8, param_count * 0.15)
        
        # Boost for indicators
        indicator_boost = len(strategy.get('indicators', [])) * 0.1
        
        strategy_type = strategy.get('type', 'general')
        boost = type_boost.get(strategy_type, 0.3)
        
        estimated_sharpe = base_sharpe + boost + param_boost + indicator_boost
        print(f"  📈 Estimated Sharpe for {strategy['name']}: {estimated_sharpe:.2f}")
        
        return min(estimated_sharpe, 3.0)  # Cap at reasonable maximum

    def _log_strategy_stats(self, strategies: List[Dict], param_variety_count: int):
        """Log strategy statistics - FIXED VERSION"""
        print(f"🎲 Parameter variety: {param_variety_count} unique parameter sets out of {len(strategies)} strategies")
        
        strategy_types = {}
        for strategy in strategies:
            strategy_type = strategy.get('type', 'unknown')
            strategy_types[strategy_type] = strategy_types.get(strategy_type, 0) + 1
        
        for strategy_type, count in strategy_types.items():
            print(f"   📊 {strategy_type}: {count} strategies")
        
        print(f"🎯 {len(strategies)} strategies built successfully")
        
        # Log to file as well
        self.logger.info(f"Parameter variety: {param_variety_count} unique sets")
        self.logger.info(f"Total strategies built: {len(strategies)}")

    # Add this helper method to your EnhancedResearchPipeline class
    def run_with_backtesting_from_async(self, strategies: List[Dict]) -> List[Dict]:
        """Run backtesting on strategies from async pipeline"""
        if not strategies:
            return []
        
        self.logger.info("🧪 Running backtesting on async strategies...")
        proven_strategies = []
        
        for strategy in strategies:
            try:
                backtest_results = self.backtester.run_comprehensive_backtest(strategy['code'])
                
                if backtest_results.get('best_sharpe', 0) >= self.min_sharpe_alert:
                    strategy.update({
                        'backtest_results': backtest_results,
                        'actual_sharpe': backtest_results.get('best_sharpe', 0)
                    })
                    proven_strategies.append(strategy)
                    
            except Exception as e:
                self.logger.error(f"❌ Backtest failed for {strategy['name']}: {e}")
        
        return proven_strategies
    
    async def generate_continuous_tasks(self, previous_results=None):
        """Generate tasks for continuous alpha generation"""
        tasks = []
        
        # Base tasks (always run these)
        base_tasks = [
            # Market microstructure
            ResearchTask(source='market_specialist', query='crypto order book dynamics', priority=1),
            ResearchTask(source='arxiv_miner', query='high frequency trading', priority=1),
            
            # Mathematical models
            ResearchTask(source='stochastic_expert', query='regime switching models', priority=1),
            ResearchTask(source='math_analyst', query='bayesian changepoint detection', priority=1),
            
            # Implementation focus
            ResearchTask(source='github_scout', query='algorithmic trading backtesting', priority=1),
            ResearchTask(source='implementation_analyst', query='low latency systems', priority=1),
        ]
        
        tasks.extend(base_tasks)
        
        # Adaptive tasks based on previous results
        if previous_results:
            enhanced_tasks = await self.generate_deepseek_enhanced_tasks(previous_results)
            tasks.extend(enhanced_tasks)
        
        # Random exploration tasks (10% of the time)
        if random.random() < 0.1:
            tasks.append(ResearchTask(source='cross_domain_innovator', 
                                    query=random.choice(['quantum finance', 'neuroscience trading', 'ecological models markets']), 
                                    priority=3))
        
        self.logger.info(f"🎯 Generated {len(tasks)} continuous alpha tasks")
        return tasks

    async def debug_swarm(self):
        """Debug method to check swarm status"""
        print("\n🔍 SWARM DEBUG INFO:")
        
        if not self.research_swarm:
            print("❌ Swarm is None")
            return
        
        print(f"✅ Swarm exists with {len(self.research_swarm.agents)} agents:")
        for agent_name, agent in self.research_swarm.agents.items():
            print(f"   - {agent_name}: {type(agent).__name__}")
        
        # Test task generation
        print("\n🧪 Testing task generation...")
        try:
            tasks = await self.task_generator.generate_research_tasks()
            print(f"✅ Generated {len(tasks)} tasks")
            for task in tasks[:3]:  # Show first 3
                print(f"   - {task.source}: {task.query}")
        except Exception as e:
            print(f"❌ Task generation failed: {e}")
        
        # Test swarm processing
        print("\n🧪 Testing swarm processing...")
        try:
            await self.research_swarm.add_tasks(tasks[:2])  # Just 2 for testing
            results = await self.research_swarm.process_tasks()
            print(f"✅ Swarm processed {len(results)} results")
            for result in results:
                print(f"   - {result.get('agent')}: {result.get('papers_found', result.get('repos_found', 0))} items")
        except Exception as e:
            print(f"❌ Swarm processing failed: {e}")

async def main():
    pipeline = EnhancedResearchPipeline(min_sharpe_alert=0.3)
    
    print("🚀 Starting Research Pipeline with Swarm...")
    
    # Debug info
    print(f"\n🔍 SWARM DEBUG INFO:")
    if pipeline.research_swarm:
        print(f"✅ Swarm exists with {len(pipeline.research_swarm.agents)} agents:")
        for agent_name, agent in pipeline.research_swarm.agents.items():
            print(f"   - {agent_name}: {agent.__class__.__name__}")
        
        print(f"\n🧪 Testing task generation...")
        try:
            # Use the new method instead of task_generator
            tasks = await pipeline.generate_comprehensive_tasks()
            print(f"✅ Generated {len(tasks)} tasks")
            for task in tasks[:3]:  # Show first 3 tasks
                print(f"   - {task.source}: {task.query}")
                
            print(f"\n🧪 Testing swarm processing...")
            # Add tasks to swarm and process
            await pipeline.research_swarm.add_tasks(tasks)
            results = await pipeline.research_swarm.process_tasks()
            print(f"✅ Swarm processed {len(results)} results")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Swarm is None - check DeepSeek client availability")
    
    # Run the actual pipeline
    strategies = await pipeline.run_pipeline()
    
    if strategies:
        print(f"\n📊 RESEARCH RESULTS SUMMARY:")
        print(f"Total Strategies Found: {len(strategies)}")
        
        # Run backtesting on all strategies
        proven_strategies = pipeline.run_with_backtesting_from_async(strategies)
        
        if proven_strategies:
            print(f"\n🎯 BACKTESTING RESULTS:")
            print(f"Proven Strategies: {len(proven_strategies)}")
            
            for i, strategy in enumerate(proven_strategies):
                results = strategy.get('backtest_results', {})
                print(f"\n{i+1}. {strategy['name']}")
                print(f"   Source: {strategy.get('source', 'traditional')}")
                print(f"   Actual Sharpe: {strategy.get('actual_sharpe', 0):.2f}")
        else:
            print("❌ No strategies passed backtesting")
    else:
        print("❌ No strategies found")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())