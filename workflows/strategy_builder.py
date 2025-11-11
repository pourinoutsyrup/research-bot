# workflows/strategy_builder.py
import logging
from typing import Dict, Optional, List

class StrategyBuilder:
    def __init__(self, use_ai: bool = True, deepseek_key: str = None):
        self.logger = logging.getLogger(__name__)
        self.use_ai = use_ai
        
        if use_ai and deepseek_key:
            from workflows.ai_strategy_extractor import DeepSeekStrategyExtractor
            self.ai_extractor = DeepSeekStrategyExtractor(deepseek_key)
            self.logger.info("🤖 AI Strategy Extraction enabled")
        else:
            self.ai_extractor = None
            self.logger.info("🔍 Using pattern-based strategy extraction")
            
        # Keep existing pattern matching as fallback
        self.STRATEGY_PATTERNS = {
            'mean_reversion': ['z-score', 'bollinger', 'mean reversion', 'oversold', 'overbought', 'stationary'],
            'momentum': ['momentum', 'trend', 'breakout', 'moving average crossover', 'rsi', 'macd'],
            'arbitrage': ['arbitrage', 'spread', 'price difference', 'cross-exchange', 'statistical arbitrage'],
            'market_making': ['market making', 'bid-ask', 'spread capture', 'inventory management'],
            'ml_based': ['random forest', 'neural network', 'LSTM', 'gradient boosting', 'SVM', 'machine learning']
        }

    def extract_strategy_from_paper(self, paper: Dict) -> Optional[Dict]:
        """Extract trading strategy - AI first, template fallback"""
        try:
            content = f"{paper.get('title', '')} {paper.get('summary', '')}"
            
            # Quick filter for mathematical content
            math_keywords = [
                'filter', 'fourier', 'wavelet', 'transform', 'signal', 'frequency',
                'bayesian', 'markov', 'monte carlo', 'statistical', 'probability',
                'optimization', 'gradient', 'descent', 'search', 'genetic',
                'learning', 'neural', 'network', 'clustering', 'classification',
                'time series', 'sequence', 'temporal', 'forecast', 'prediction',
                'topological', 'graph', 'entropy', 'information', 'complexity',
                'stochastic', 'differential', 'regression', 'estimation'
            ]
            
            content_lower = content.lower()
            math_keyword_count = sum(1 for keyword in math_keywords if keyword in content_lower)
            
            if math_keyword_count < 2:
                return None

            # TRY AI EXTRACTION FIRST
            if self.use_ai and self.ai_extractor:
                ai_strategy = self.ai_extractor.extract_tradable_strategy(paper)
                if ai_strategy:
                    self.logger.info(f"✅ AI extracted: {ai_strategy['name']}")
                    return ai_strategy
                else:
                    self.logger.debug(f"AI rejected, trying template fallback...")

            # FALLBACK: Template-based extraction
            math_approach = self._classify_mathematical_approach(content)
            if not math_approach:
                return None

            strategy = self._adapt_math_to_trading(math_approach, paper, content)
            self._debug_strategy_code(strategy)
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"Strategy extraction error: {e}")
            return None
        
    def _debug_strategy_code(self, strategy: Dict):
        """Debug method to check generated code"""
        print(f"\n🔍 DEBUG Strategy: {strategy['name']}")
        print(f"Type: {strategy['type']}")
        print(f"Parameters: {strategy.get('parameters', {})}")
        print(f"Code length: {len(strategy['code'])} characters")
        
        # Show first few lines of code
        code_lines = strategy['code'].split('\n')
        print("Code preview (first 5 lines):")
        for i, line in enumerate(code_lines[:5]):
            print(f"  {i+1}: {line}")
        
        # Test if code compiles
        try:
            # Create a minimal environment for testing
            test_env = {'pd': None, 'np': None, 'fft': None, 'stats': None}
            exec(strategy['code'], test_env)
            print("✅ Code compiles successfully")
            
            # Test if strategy function exists
            if 'strategy' in test_env:
                print("✅ Strategy function found")
            else:
                print("❌ Strategy function NOT found")
                
        except Exception as e:
            print(f"❌ Code compilation failed: {e}")
            # Show the exact line that failed if possible
            import traceback
            print(f"Error details: {traceback.format_exc()}")

    def _classify_mathematical_approach(self, content: str) -> str:
        """Classify the main mathematical approach in the paper"""
        content_lower = content.lower()
        
        math_categories = {
            'signal_processing': ['fourier', 'wavelet', 'filter', 'signal', 'frequency', 'fft', 'spectral'],
            'optimization': ['optimization', 'gradient', 'descent', 'search', 'genetic', 'evolutionary', 'parameter'],
            'statistical': ['bayesian', 'markov', 'statistical', 'probability', 'regression', 'hypothesis', 'confidence'],
            'time_series': ['time series', 'sequence', 'temporal', 'forecast', 'arima', 'autocorrelation'],
            'ml_based': ['neural', 'learning', 'network', 'clustering', 'classification', 'machine learning', 'ai'],
            'advanced_math': ['topological', 'graph', 'entropy', 'information', 'stochastic', 'manifold']
        }
        
        for category, keywords in math_categories.items():
            if any(keyword in content_lower for keyword in keywords):
                return category
        
        return 'general_math'

    def _adapt_math_to_trading(self, math_approach: str, paper: Dict, content: str) -> Dict:
        """Adapt mathematical concept to trading strategy - MAKE EACH UNIQUE"""
        
        # Generate unique parameters based on paper content
        unique_params = self._generate_unique_parameters(math_approach, content)
        
        # Base strategy template
        strategy = {
            'name': self._generate_strategy_name(paper.get('title', 'Unknown'), math_approach),
            'type': math_approach,
            'source_paper': paper.get('title', 'Unknown'),
            'paper_url': paper.get('pdf_url', ''),
            'published_date': paper.get('published', ''),
            'source': paper.get('source', 'unknown'),
            'confidence': 0.8,
            'description': f"Trading strategy adapted from {math_approach} mathematical approach",
            'adaptation_notes': self._generate_adaptation_notes(math_approach, content),
            'parameters': unique_params,
            'indicators': self._get_indicators_for_approach(math_approach)
        }
        
        # Generate unique code for each strategy
        strategy['code'] = self._generate_unique_code(math_approach, unique_params, content)
        
        return strategy
    
    def _generate_adaptation_notes(self, math_approach: str, content: str) -> str:
        """Generate notes on how to adapt the mathematical concept to trading"""
        adaptations = {
            'signal_processing': 'Convert price series to frequency domain, identify dominant cycles for entry/exit signals',
            'optimization': 'Use optimization algorithms to find optimal trading parameters across historical data',
            'statistical': 'Apply hypothesis testing to detect regime changes, use confidence intervals for risk management',
            'time_series': 'Forecast price movements using time series models, identify seasonal patterns',
            'ml_based': 'Train models on market features to predict price direction or volatility',
            'general_math': 'Mathematical framework applied to financial time series analysis'
        }
        return adaptations.get(math_approach, 'Mathematical framework applied to financial time series analysis')

    def _get_indicators_for_approach(self, math_approach: str) -> List[str]:
        """Get appropriate indicators for each mathematical approach"""
        indicator_map = {
            'signal_processing': ['Spectral Analysis', 'Frequency Components', 'FFT'],
            'optimization': ['Fitness Function', 'Parameter Space', 'Evolutionary Search'],
            'statistical': ['Statistical Tests', 'Confidence Intervals', 'Hypothesis Testing'],
            'time_series': ['Autocorrelation', 'Seasonal Decomposition', 'Trend Analysis'],
            'ml_based': ['Feature Engineering', 'Pattern Recognition', 'Prediction Models'],
            'general_math': ['Mathematical Transform', 'Feature Extraction', 'Analytical Framework']
        }
        return indicator_map.get(math_approach, ['Mathematical Analysis'])

    def _generate_strategy_name(self, title: str, strategy_type: str) -> str:
        """Generate clean strategy name"""
        import re
        # Clean the title
        clean_title = re.sub(r'[^a-zA-Z0-9\s]', ' ', title)
        clean_title = ' '.join(clean_title.split()[:3])  # Use first 3 words
        clean_title = clean_title.lower().replace(' ', '_')
        
        # Combine with strategy type
        return f"{strategy_type}_{clean_title}"[:50]

    def _generate_unique_parameters(self, math_approach: str, content: str) -> Dict:
        """Generate unique parameters for each strategy based on paper content"""
        import hashlib
        
        content_hash = hashlib.md5(content.encode()).hexdigest()
        hash_int = int(content_hash[:8], 16)
        
        base_params = {
            'signal_processing': {'window': 50, 'frequency_bands': 5, 'threshold': 0.6},  # CHANGED from 1.5 to 0.6
            'optimization': {'population_size': 100, 'generations': 50, 'mutation_rate': 0.1, 'threshold': 0.5, 'window': 20},  # ADDED threshold and window
            'statistical': {'confidence_level': 0.80, 'sample_size': 100, 'test_statistic': 'zscore', 'window': 30},  # CHANGED from 0.95 to 0.80, ADDED window
            'time_series': {'lags': 10, 'seasonality': 24, 'smoothing_factor': 0.3, 'window': 20},  # ADDED window
            'ml_based': {'window': 30, 'prediction_horizon': 5, 'feature_count': 8},
            'general_math': {'window': 30, 'complexity': 1.0}
        }
        
        params = base_params.get(math_approach, {'window': 30, 'threshold': 0.5}).copy()  # CHANGED from 2.0 to 0.5
        
        # Make parameters unique with SMALLER variations
        if 'window' in params:
            params['window'] = max(10, params['window'] + (hash_int % 11) - 5)  # ±5 variation (was ±10)
        
        if 'threshold' in params:
            params['threshold'] = max(0.3, params['threshold'] + ((hash_int % 7) - 3) * 0.1)  # ±0.3 variation (was ±0.5)
        
        params['unique_id'] = content_hash[:8]
        
        return params

    def _generate_unique_code(self, math_approach: str, parameters: Dict, content: str) -> str:
        """Generate unique code for each strategy with actual parameter variations"""
        
        base_code_generators = {
            'signal_processing': self._generate_signal_processing_code,
            'optimization': self._generate_optimization_code,
            'statistical': self._generate_statistical_code,
            'time_series': self._generate_time_series_code,
            'ml_based': self._generate_ml_based_code
        }
        
        if math_approach in base_code_generators:
            base_code = base_code_generators[math_approach]()
            # Inject unique parameters into the code
            return self._inject_parameters_into_code(base_code, parameters, math_approach)
        else:
            return self._generate_general_math_code(math_approach)

    def _inject_parameters_into_code(self, code: str, parameters: Dict, math_approach: str) -> str:
        """Inject unique parameters into the strategy code - FIXED VERSION"""
        
        # For signal processing - inject window size
        if math_approach == 'signal_processing' and 'window' in parameters:
            code = code.replace('window_size = 50', f'window_size = {parameters["window"]}')
            if 'threshold' in parameters:
                code = code.replace('* 2.0', f'* {parameters["threshold"]}')  # ADDED threshold injection
        
        # For optimization - inject threshold AND window
        elif math_approach == 'optimization':
            if 'threshold' in parameters:
                code = code.replace('best_threshold = 1.5', f'best_threshold = {parameters["threshold"]}')  # FIXED: was looking for 0.5
            if 'window' in parameters:
                code = code.replace('best_window = 20', f'best_window = {parameters["window"]}')
        
        # For statistical - inject confidence level AND window
        elif math_approach == 'statistical':
            if 'window' in parameters:
                code = code.replace('window = 30', f'window = {parameters["window"]}')
            if 'confidence_level' in parameters:
                threshold = 1 - parameters['confidence_level']
                code = code.replace('p_value < 0.05', f'p_value < {threshold:.3f}')  # FIXED: use .3f for precision
        
        # For time series - inject window
        elif math_approach == 'time_series' and 'window' in parameters:
            code = code.replace('window = 20', f'window = {parameters["window"]}')
        
        # For ml_based - inject window
        elif math_approach == 'ml_based' and 'window' in parameters:
            code = code.replace('window = 30', f'window = {parameters["window"]}')
        
        return code
    
    # FIXED CODE GENERATION METHODS - NO INDENTATION IN STRINGS
    def _generate_signal_processing_code(self) -> str:
        return '''import pandas as pd
import numpy as np
from scipy import fft

def strategy(prices):
    signals = []
    window_size = 50
    
    for i in range(len(prices)):
        if i < window_size:
            signals.append(0)
            continue
        price_window = prices.iloc[i-window_size:i]
        fft_result = fft.fft(price_window.values)
        magnitude = np.abs(fft_result[1:len(fft_result)//2])
        if len(magnitude) > 0:
            dominant_idx = np.argmax(magnitude) + 1
            if magnitude[dominant_idx-1] > np.mean(magnitude) * 2.0:
                signals.append(1)
            else:
                signals.append(0)
        else:
            signals.append(0)
    return signals
'''

    def _generate_optimization_code(self) -> str:
        return '''import pandas as pd
import numpy as np

def strategy(prices):
    signals = []
    best_window = 20
    best_threshold = 1.5
    
    returns = prices.pct_change().rolling(best_window).mean()
    volatility = prices.pct_change().rolling(best_window).std()
    z_scores = returns / volatility.replace(0, np.nan)
    
    for z in z_scores:
        if pd.isna(z):
            signals.append(0)
        elif z > best_threshold:
            signals.append(-1)
        elif z < -best_threshold:
            signals.append(1)
        else:
            signals.append(0)
    return signals
'''

    def _generate_statistical_code(self) -> str:
        return '''import pandas as pd
import numpy as np
from scipy import stats

def strategy(prices):
    signals = []
    window = 30
    
    for i in range(len(prices)):
        if i < window * 2:
            signals.append(0)
            continue
        recent = prices.iloc[i-window:i].values
        historical = prices.iloc[i-window*2:i-window].values
        if len(recent) < 2 or len(historical) < 2:
            signals.append(0)
            continue
        try:
            t_stat, p_value = stats.ttest_ind(recent, historical, equal_var=False)
        except:
            p_value = 0.5
        if p_value < 0.05:
            if np.mean(recent) > np.mean(historical):
                signals.append(1)
            else:
                signals.append(-1)
        else:
            signals.append(0)
    return signals
'''

    def _generate_time_series_code(self) -> str:
        return '''import pandas as pd
import numpy as np

def strategy(prices):
    signals = []
    window = 20
    
    for i in range(len(prices)):
        if i < window:
            signals.append(0)
            continue
        price_window = prices.iloc[i-window:i]
        returns = price_window.pct_change().dropna()
        if len(returns) > 1:
            autocorr = returns.autocorr(lag=1)
            if pd.isna(autocorr):
                autocorr = 0
        else:
            autocorr = 0
        x = np.arange(len(price_window))
        try:
            slope = np.polyfit(x, price_window.values, 1)[0]
        except:
            slope = 0
        if autocorr > 0.1 and slope > 0:
            signals.append(1)
        elif autocorr < -0.1 and slope < 0:
            signals.append(-1)
        else:
            signals.append(0)
    return signals
'''

    def _generate_ml_based_code(self) -> str:
        return '''import pandas as pd
import numpy as np

def strategy(prices):
    signals = []
    window = 30
    
    for i in range(len(prices)):
        if i < window:
            signals.append(0)
            continue
        current_price = prices.iloc[i]
        ma_5 = prices.iloc[i-5:i].mean()
        ma_10 = prices.iloc[i-10:i].mean()
        if ma_5 > ma_10 and current_price > ma_5:
            signals.append(1)
        elif ma_5 < ma_10 and current_price < ma_5:
            signals.append(-1)
        else:
            signals.append(0)
    return signals
'''

    def _generate_general_math_code(self, math_approach: str) -> str:
        return '''import pandas as pd
import numpy as np

def strategy(prices):
    signals = []
    window = 30
    
    for i in range(len(prices)):
        if i < window:
            signals.append(0)
            continue
        price_window = prices.iloc[i-window:i]
        returns = price_window.pct_change().dropna()
        if len(returns) < 5:
            signals.append(0)
            continue
        volatility = returns.std()
        skewness = returns.skew()
        if pd.isna(skewness):
            skewness = 0
        if volatility > returns.std() * 1.5 and abs(skewness) > 0.5:
            if skewness > 0:
                signals.append(1)
            else:
                signals.append(-1)
        else:
            signals.append(0)
    return signals
'''

    # Keep your existing pattern-based methods for fallback
    def _pattern_based_extraction(self, paper: Dict) -> Optional[Dict]:
        """Improved pattern-based extraction with better fallbacks"""
        try:
            content = f"{paper.get('title', '')} {paper.get('summary', '')}"
            
            # Skip papers without trading relevance
            trading_keywords = ['trading', 'arbitrage', 'portfolio', 'investment', 'market', 'price', 'volatility']
            if not any(keyword in content.lower() for keyword in trading_keywords):
                self.logger.debug(f"Skipping non-trading paper: {paper.get('title', '')[:50]}")
                return None

            strategy_type = self._classify_strategy_type(content)
            confidence = self._calculate_confidence(content, strategy_type)
            
            # Extract basic strategy components
            strategy = {
                'name': self._generate_strategy_name(paper['title'], strategy_type),
                'type': strategy_type,
                'description': self._extract_strategy_description(content),
                'logic': self._extract_trading_logic(content),
                'parameters': self._extract_parameters(content),
                'indicators': self._extract_indicators(content),
                'source_paper': paper['title'],
                'paper_url': paper.get('pdf_url', ''),
                'published_date': paper.get('published', ''),
                'confidence': confidence,
                'sharpe_ratio': self._estimate_sharpe_ratio(confidence, strategy_type)
            }
            
            # Generate realistic code
            strategy['code'] = self._generate_realistic_code(strategy)
            
            # Relaxed validation - always return something if we got this far
            if self._validate_strategy_quality(strategy):
                return strategy
            else:
                # Return a minimal valid strategy even if validation fails
                self.logger.debug(f"Validation failed for {strategy['name']}, returning minimal strategy")
                return self._create_minimal_strategy(paper, strategy_type)
                
        except Exception as e:
            self.logger.error(f"Pattern extraction error: {e}")
            return None

    def _estimate_sharpe_ratio(self, confidence: float, strategy_type: str) -> float:
        """Estimate sharpe ratio based on confidence and strategy type"""
        base_sharpe = confidence * 2.0  # Base estimate
        
        # Adjust based on strategy type
        type_multipliers = {
            'arbitrage': 1.5,
            'market_making': 1.3,
            'mean_reversion': 1.2,
            'momentum': 1.1,
            'ml_based': 1.4,
            'general': 1.0
        }
        
        multiplier = type_multipliers.get(strategy_type, 1.0)
        return base_sharpe * multiplier

    def _calculate_confidence(self, content: str, strategy_type: str) -> float:
        """Calculate confidence score based on content quality"""
        confidence = 0.3  # Base confidence
        
        # Boost confidence for specific strategy types
        if strategy_type != 'general':
            confidence += 0.2
        
        # Boost for having parameters
        if len(self._extract_parameters(content)) > 0:
            confidence += 0.2
        
        # Boost for having indicators
        if len(self._extract_indicators(content)) > 0:
            confidence += 0.2
        
        # Boost for longer content
        if len(content) > 500:
            confidence += 0.1
        
        return min(confidence, 0.9)  # Cap at 0.9

    def _create_minimal_strategy(self, paper: Dict, strategy_type: str) -> Dict:
        """Create a minimal valid strategy when extraction fails"""
        return {
            'name': f"minimal_{strategy_type}_{paper['title'][:30].replace(' ', '_')}",
            'type': strategy_type,
            'description': f"Minimal strategy based on: {paper['title']}",
            'logic': "Basic pattern-based approach",
            'parameters': {'window': 20, 'threshold': 2.0},
            'indicators': [],
            'source_paper': paper['title'],
            'paper_url': paper.get('pdf_url', ''),
            'published_date': paper.get('published', ''),
            'confidence': 0.3,
            'code': self._generate_realistic_code({'type': strategy_type, 'parameters': {'window': 20, 'threshold': 2.0}})
        }

    def _classify_strategy_type(self, content: str) -> str:
        """Classify strategy type based on keywords"""
        content_lower = content.lower()
        
        for strategy_type, keywords in self.STRATEGY_PATTERNS.items():
            if any(keyword in content_lower for keyword in keywords):
                return strategy_type
        
        return 'general'

    def _extract_strategy_description(self, content: str) -> str:
        """Extract brief description"""
        return content[:200] + "..." if len(content) > 200 else content

    def _extract_trading_logic(self, content: str) -> str:
        """Extract trading logic"""
        logic_keywords = ['when', 'if', 'signal', 'trigger', 'condition']
        sentences = content.split('.')
        logic_sentences = [s for s in sentences if any(kw in s.lower() for kw in logic_keywords)]
        return ' '.join(logic_sentences[:2]) if logic_sentences else "Pattern-based strategy"

    def _extract_parameters(self, content: str) -> Dict:
        """Extract parameters from text"""
        import re
        params = {}
        
        # Look for common parameters
        window_match = re.search(r'(\d+)[\s-]*(day|period|window)', content.lower())
        if window_match:
            params['window'] = int(window_match.group(1))
        
        threshold_match = re.search(r'(\d+\.?\d*)\s*(standard deviation|threshold)', content.lower())
        if threshold_match:
            params['threshold'] = float(threshold_match.group(1))
        
        return params if params else {'window': 20, 'threshold': 2.0}

    def _extract_indicators(self, content: str) -> list:
        """Extract indicators mentioned"""
        indicators = []
        indicator_keywords = {
            'RSI': 'rsi',
            'MACD': 'macd',
            'Moving Average': 'moving average|ma ',
            'Bollinger': 'bollinger',
            'Volume': 'volume',
            'Volatility': 'volatility'
        }
        
        content_lower = content.lower()
        for indicator, pattern in indicator_keywords.items():
            if any(p in content_lower for p in pattern.split('|')):
                indicators.append(indicator)
        
        return indicators

    def _generate_realistic_code(self, strategy: Dict) -> str:
        """Generate executable code"""
        params = strategy.get('parameters', {})
        window = params.get('window', 20)
        threshold = params.get('threshold', 2.0)
        
        return f'''def strategy(prices):
        """
        {strategy['name']}
        Type: {strategy['type']}
        """
        import pandas as pd
        import numpy as np
        
        signals = []
        ma = prices.rolling(window={window}).mean()
        std = prices.rolling(window={window}).std()
        
        for i in range(len(prices)):
            if i < {window}:
                signals.append(0)
                continue
                
            z_score = (prices.iloc[i] - ma.iloc[i]) / std.iloc[i] if std.iloc[i] > 0 else 0
            
            if z_score > {threshold}:
                signals.append(-1)
            elif z_score < -{threshold}:
                signals.append(1)
            else:
                signals.append(0)
        
        return signals'''

    def _validate_strategy_quality(self, strategy: Dict) -> bool:
        """Relaxed validation - ensure basic structure exists"""
        required_fields = ['name', 'type', 'code', 'confidence']
        return all(field in strategy for field in required_fields)