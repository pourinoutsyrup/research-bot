# workflows/ai_strategy_extractor.py
import logging
import requests
import json
import os
from typing import Dict, Optional, List
import re

class DeepSeekStrategyExtractor:
    def __init__(self, api_key: str = None):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.enabled = self.api_key is not None
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        
        if not self.enabled:
            self.logger.warning("🚫 DeepSeek AI Extractor disabled - no API key provided")
            self.logger.info("💡 Set DEEPSEEK_API_KEY environment variable or pass api_key parameter")
        else:
            self.logger.info("✅ DeepSeek AI Strategy Extractor enabled")

    def extract_tradable_strategy(self, paper: Dict) -> Optional[Dict]:
        """Use DeepSeek AI to find actual tradable strategies in papers"""
        if not self.enabled:
            self.logger.warning("AI extraction disabled - skipping paper")
            return None
            
        try:
            content = self._prepare_paper_content(paper)
            response = self._call_deepseek_api(content)
            
            if response and response.get('tradable', False):
                strategy = self._build_strategy_from_ai(paper, response)
                self.logger.info(f"🎯 AI extracted tradable strategy: {strategy['name']}")
                return strategy
            else:
                reason = response.get('reason', 'No tradable strategy found') if response else 'API call failed'
                self.logger.info(f"❌ AI rejected: {paper.get('title', '')[:60]}... - {reason}")
                return None
                
        except Exception as e:
            self.logger.error(f"DeepSeek extraction failed: {e}")
            return None

    def _prepare_paper_content(self, paper: Dict) -> str:
        """Prepare paper content for AI analysis"""
        title = paper.get('title', 'No title')
        summary = paper.get('summary', 'No abstract')
        authors = ', '.join(paper.get('authors', []))
        categories = paper.get('categories', [])
        
        return f"""
PAPER TITLE: {title}
ABSTRACT: {summary}
AUTHORS: {authors}
CATEGORIES: {categories}
"""

    def _call_deepseek_api(self, content: str) -> Optional[Dict]:
        """Call DeepSeek API for strategy extraction"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": """Extract trading parameters from research papers. Return JSON with:

    CRITICAL: paper_specific_parameters must contain ONLY numeric values!

    {
        "tradable": true/false,
        "paper_specific_parameters": {
            "lookback_period": 20,  // ✅ NUMBER not string
            "threshold": 2.5,        // ✅ NUMBER not string
            "window": 30             // ✅ NUMBER not string
        },
        "strategy_type": "signal_processing/optimization/statistical_inference/ml_based",
        "confidence": 0.0-1.0
    }

    EXAMPLES:
    Paper: "We use 20-day windows with 2σ thresholds"
    → {"paper_specific_parameters": {"lookback_period": 20, "threshold": 2.0}, "tradable": true}

    Paper: "Training for 50 epochs with 0.1 learning rate"  
    → {"paper_specific_parameters": {"epochs": 50, "learning_rate": 0.1}, "tradable": true}

    Paper: "Adaptive RL-learned threshold"
    → {"paper_specific_parameters": {"lookback_period": 30, "threshold": 1.5}, "tradable": true}
    // ✅ Convert to reasonable defaults if no explicit numbers

    IF NO EXPLICIT NUMBERS: Use reasonable trading defaults:
    - lookback_period: 20-50 (based on context)
    - threshold: 1.5-2.5 (standard deviations)
    - window: 10-30 (days)

    NEVER return strings in parameters!"""
                },
                {
                    "role": "user", 
                    "content": f"Extract NUMERIC parameters from this paper:\n\n{content}"
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        # ... rest of the method stays the same
        
        try:
            self.logger.info(f"🔧 Making API call with content length: {len(content)}")
            response = requests.post(
                self.base_url, 
                headers=headers, 
                json=payload, 
                timeout=60
            )
            
            self.logger.info(f"🔧 API Response Status: {response.status_code}")
            
            if response.status_code != 200:
                self.logger.error(f"❌ API HTTP Error {response.status_code}: {response.text[:500]}")
                return None
                
            result = response.json()
            self.logger.info(f"🔧 API Response received, parsing content...")
            
            content = result['choices'][0]['message']['content']
            
            # Debug: Log the raw response
            self.logger.info(f"🔧 Raw AI Response: {content[:500]}...")
            
            # Simple JSON extraction
            content = content.strip()
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            # Try to find JSON object
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                parsed_result = json.loads(json_str)
                self.logger.info(f"✅ Successfully parsed API response: {parsed_result}")
                return parsed_result
            else:
                self.logger.error(f"❌ No JSON found in response: {content[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.error("❌ API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ JSON decode error: {e}")
            self.logger.error(f"❌ Raw content: {content[:500]}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Unexpected error in API call: {e}")
            return None
    
    def _build_strategy_from_ai(self, paper: Dict, ai_result: Dict) -> Dict:
        """Build strategy object from AI extraction"""
        # DEBUG: Log what AI returned
        self.logger.info(f"🔍 AI Result Keys: {list(ai_result.keys())}")
        if 'paper_specific_parameters' in ai_result:
            self.logger.info(f"📊 AI Paper Params: {ai_result['paper_specific_parameters']}")
        
        strategy_type = ai_result.get('strategy_type', 'ai_extracted')
        
        # DEBUG: Call _extract_parameters and log the result
        extracted_params = self._extract_parameters(ai_result)
        self.logger.info(f"📊 EXTRACTED PARAMETERS: {extracted_params}")
        
        strategy = {
            'name': self._generate_strategy_name(paper['title'], strategy_type),
            'type': strategy_type,
            'description': f"AI-extracted from: {paper['title']}",
            'logic': self._build_trading_logic(ai_result),
            'parameters': extracted_params,  # Use the extracted params directly
            'indicators': ai_result.get('indicators_used', []),
            'timeframe': ai_result.get('timeframe', '1h'),
            'entry_rules': ai_result.get('entry_conditions', []),
            'exit_rules': ai_result.get('exit_conditions', []),
            'source_paper': paper['title'],
            'paper_url': paper.get('pdf_url', ''),
            'published_date': paper.get('published', ''),
            'extraction_confidence': ai_result.get('confidence', 0.5),
            'extraction_reason': ai_result.get('reason', '')
        }
        
        # DEBUG: Log the final strategy parameters
        self.logger.info(f"📊 FINAL STRATEGY PARAMS: {strategy['parameters']}")
        
        # ✅ CRITICAL: Generate code based on AI extraction
        strategy['code'] = self._generate_ai_based_code(strategy)
        
        # ✅ DEBUG: Verify code was generated
        if strategy['code']:
            self.logger.info(f"✅ Generated strategy code: {len(strategy['code'])} characters")
        else:
            self.logger.error("❌ Failed to generate strategy code!")
            # Fallback to ensure we always have code
            strategy['code'] = self._generate_fallback_code(strategy)
        
        return strategy

    def _build_trading_logic(self, ai_result: Dict) -> str:
        """Build human-readable trading logic from AI extraction"""
        entries = ai_result.get('entry_conditions', [])
        exits = ai_result.get('exit_conditions', [])
        indicators = ai_result.get('indicators_used', [])
        
        logic_parts = []
        if entries:
            logic_parts.append(f"ENTRY: {' AND '.join(entries[:2])}")  # Limit to 2 main conditions
        if exits:
            logic_parts.append(f"EXIT: {' OR '.join(exits[:2])}")  # Limit to 2 main conditions
        if indicators:
            logic_parts.append(f"INDICATORS: {', '.join(indicators)}")
            
        return ' | '.join(logic_parts) if logic_parts else "AI-extracted trading strategy"

    def _extract_parameters(self, ai_result: Dict) -> Dict:
        """Extract parameters from AI result - ENSURE WINDOWS ARE INTEGERS"""
        # Get AI-extracted parameters
        paper_params = ai_result.get('paper_specific_parameters', {})
        
        # ✅ NEW: VALIDATE AND CONVERT TO NUMBERS WITH INTEGER WINDOWS
        validated_params = {}
        for key, value in paper_params.items():
            if isinstance(value, (int, float)):
                # Convert to integer if it's a window/lookback parameter
                if any(window_key in key.lower() for window_key in ['window', 'lookback', 'period', 'lag']):
                    validated_params[key] = max(1, int(value))  # Ensure minimum window of 1
                else:
                    validated_params[key] = value
            elif isinstance(value, str):
                # Try to extract number from string
                import re
                numbers = re.findall(r'\d+\.?\d*', value)
                if numbers:
                    num_value = float(numbers[0])
                    # Convert to integer if it's a window parameter
                    if any(window_key in key.lower() for window_key in ['window', 'lookback', 'period', 'lag']):
                        validated_params[key] = max(1, int(num_value))  # Ensure integer and min 1
                    else:
                        validated_params[key] = num_value
                    self.logger.warning(f"⚠️ Converted string param {key}='{value}' to {validated_params[key]}")
                else:
                    self.logger.warning(f"⚠️ Skipping non-numeric param {key}='{value}'")
        
        if validated_params:
            self.logger.info(f"✅ Validated numeric parameters: {validated_params}")
            return validated_params
        
        # Fallback to defaults with INTEGER windows
        self.logger.warning("⚠️ No valid numeric parameters found, using defaults")
        return {'lookback_period': 20, 'threshold': 2.0}  # Default window is integer

    def _call_deepseek_api(self, content: str) -> Optional[Dict]:
        """Call DeepSeek API for strategy extraction"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": """You extract MATHEMATICAL TECHNIQUES from research papers that could be adapted for trading.

                    YOUR JOB:
                    1. Find the CORE MATHEMATICAL EQUATION or ALGORITHM
                    2. Identify what makes it UNUSUAL/NOVEL
                    3. Translate the math into tradable parameters

                    LOOK FOR:
                    - Differential equations (SDEs, PDEs, ODEs)
                    - Transform methods (Fourier, Wavelet, Laplace)
                    - Filter designs (Kalman, Particle, Wiener)
                    - Statistical tests (cointegration, stationarity, regime detection)
                    - Optimization algorithms (gradient methods, genetic algorithms)
                    - Network/graph algorithms (clustering, centrality measures)

                    EXAMPLES:

                    Paper: "We use Ornstein-Uhlenbeck process: dX_t = θ(μ - X_t)dt + σdW_t with θ=0.15"
                    → {
                        "tradable": true,
                        "mathematical_technique": "Ornstein-Uhlenbeck mean reversion",
                        "equation": "dX_t = θ(μ - X_t)dt + σdW_t",
                        "paper_specific_parameters": {
                            "mean_reversion_speed": 0.15,
                            "lookback_period": 20
                        },
                        "uniqueness": "Fast mean reversion speed"
                    }

                    Paper: "Kalman filter with measurement noise R=0.01, process noise Q=0.001"
                    → {
                        "tradable": true,
                        "mathematical_technique": "Kalman filtering",
                        "equation": "x_k = Ax_{k-1} + w_k, y_k = Hx_k + v_k",
                        "paper_specific_parameters": {
                            "measurement_noise": 0.01,
                            "process_noise": 0.001,
                            "state_dimension": 2
                        },
                        "uniqueness": "Low noise ratio for high-frequency data"
                    }

                    Paper: "FFT with 256-point window, extract frequencies 0.1-0.5 Hz"
                    → {
                        "tradable": true,
                        "mathematical_technique": "Fourier spectral analysis",
                        "equation": "X(f) = ∫x(t)e^{-2πift}dt",
                        "paper_specific_parameters": {
                            "fft_window": 256,
                            "freq_min": 0.1,
                            "freq_max": 0.5
                        },
                        "uniqueness": "Specific frequency band isolation"
                    }

                    Paper: "We propose a novel architecture with attention mechanisms"
                    → {
                        "tradable": false,
                        "reason": "No concrete mathematical formulation or parameters"
                    }

                    RETURN FORMAT:
                    {
                        "tradable": true/false,
                        "mathematical_technique": "Name of the core math",
                        "equation": "The actual equation/algorithm",
                        "paper_specific_parameters": {
                            "param1": NUMBER,
                            "param2": NUMBER
                        },
                        "uniqueness": "What makes this technique different",
                        "confidence": 0.0-1.0
                    }

                    If paper has NO explicit equations/algorithms → tradable: false
                    If paper just describes concepts → tradable: false
                    ONLY extract when you find CONCRETE MATH"""
        
        try:
            self.logger.info(f"🔧 Making API call with content length: {len(content)}")
            response = requests.post(
                self.base_url, 
                headers=headers, 
                json=payload, 
                timeout=60
            )
            
            self.logger.info(f"🔧 API Response Status: {response.status_code}")
            
            if response.status_code != 200:
                self.logger.error(f"❌ API HTTP Error {response.status_code}: {response.text[:500]}")
                return None
                
            result = response.json()
            self.logger.info(f"🔧 API Response received, parsing content...")
            
            content = result['choices'][0]['message']['content']
            
            # Simple JSON extraction - remove any markdown wrappers
            content = content.strip()
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            # Try to find JSON object
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                parsed_result = json.loads(json_str)
                self.logger.info(f"✅ Successfully parsed API response")
                return parsed_result
            else:
                self.logger.error(f"❌ No JSON found in response: {content[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.error("❌ API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ JSON decode error: {e}")
            self.logger.error(f"❌ Raw content: {content[:500]}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Unexpected error in API call: {e}")
            return None

    def _generate_mean_reversion_code(self, strategy: Dict) -> str:
        """Generate mean reversion code"""
        window = strategy['parameters'].get('window', 20)
        threshold = strategy['parameters'].get('threshold', 2.0)
        
        return f'''
def mean_reversion_strategy(prices):
    """
    AI Mean Reversion: {strategy['name']}
    Logic: {strategy['logic']}
    """
    import pandas as pd
    import numpy as np
    
    signals = []
    rolling_mean = prices.rolling(window={window}).mean()
    rolling_std = prices.rolling(window={window}).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan).fillna(method='bfill').fillna(1.0)
    
    z_score = (prices - rolling_mean) / rolling_std
    
    for i in range(len(prices)):
        if pd.isna(z_score.iloc[i]):
            signals.append(0)
        elif z_score.iloc[i] > {threshold}:
            signals.append(-1)  # Sell (overbought)
        elif z_score.iloc[i] < -{threshold}:
            signals.append(1)   # Buy (oversold)
        else:
            signals.append(0)   # Hold
    
    return signals
'''

    def _generate_momentum_code(self, strategy: Dict) -> str:
        """Generate momentum strategy code"""
        short_window = strategy['parameters'].get('short_window', 10)
        long_window = strategy['parameters'].get('long_window', 30)
        
        return f'''
def momentum_strategy(prices):
    """
    AI Momentum: {strategy['name']}
    Logic: {strategy['logic']}
    """
    import pandas as pd
    
    signals = []
    short_ma = prices.rolling(window={short_window}).mean()
    long_ma = prices.rolling(window={long_window}).mean()
    
    for i in range(len(prices)):
        if i < {long_window}:
            signals.append(0)
            continue
            
        if short_ma.iloc[i] > long_ma.iloc[i] and short_ma.iloc[i-1] <= long_ma.iloc[i-1]:
            signals.append(1)   # Buy (golden cross)
        elif short_ma.iloc[i] < long_ma.iloc[i] and short_ma.iloc[i-1] >= long_ma.iloc[i-1]:
            signals.append(-1)  # Sell (death cross)
        else:
            signals.append(0)   # Hold
    
    return signals
'''

    def _generate_arbitrage_code(self, strategy: Dict) -> str:
        """Generate arbitrage strategy code (placeholder)"""
        return '''
def arbitrage_strategy(prices):
    """
    AI Arbitrage Strategy - Requires multiple data sources
    This is a placeholder - real arbitrage needs multiple price feeds
    """
    # Arbitrage strategies typically need multiple assets/exchanges
    # Returning neutral signals for now
    return [0] * len(prices)
'''

    def _generate_generic_ai_code(self, strategy: Dict) -> str:
        """Generate generic code for AI-extracted strategies"""
        return f'''
def ai_strategy(prices):
    """
    AI-Extracted Strategy: {strategy['name']}
    Type: {strategy['type']}
    Logic: {strategy['logic']}
    Confidence: {strategy.get("extraction_confidence", 0.5)}
    """
    import pandas as pd
    import numpy as np
    
    # This is a generic implementation of the AI-extracted strategy
    # Based on: {strategy['logic']}
    
    signals = []
    lookback = 14  # Default lookback
    
    # Simple momentum implementation as fallback
    for i in range(len(prices)):
        if i < lookback:
            signals.append(0)
        elif prices.iloc[i] > prices.iloc[i-lookback]:
            signals.append(1)   # Buy on upward momentum
        else:
            signals.append(-1)  # Sell on downward momentum
    
    return signals
'''
    def _generate_ai_based_code(self, strategy: Dict) -> str:
        """Generate code from AI-extracted strategy - ENSURE INTEGER WINDOWS"""
        params = strategy.get('parameters', {})
        
        # ✅ ENSURE lookback_period is an integer
        lookback = int(params.get('lookback_period', params.get('window', 20)))
        threshold = params.get('threshold', 2.0)
        
        return f'''
    def strategy(prices):
        """
        {strategy['name']}
        Type: {strategy.get('type', 'ai_extracted')}
        Logic: {strategy.get('logic', 'AI-extracted trading strategy')}
        """
        import pandas as pd
        import numpy as np
        
        signals = []
        
        # Convert to pandas Series for rolling operations
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        
        # ✅ lookback is now guaranteed to be an integer
        rolling_mean = prices.rolling(window={lookback}).mean()
        rolling_std = prices.rolling(window={lookback}).std()
        
        for i in range(len(prices)):
            if i < {lookback}:
                signals.append(0)
                continue
                
            if rolling_std.iloc[i] == 0:
                signals.append(0)
                continue
                
            z_score = (prices.iloc[i] - rolling_mean.iloc[i]) / rolling_std.iloc[i]
            
            if z_score > {threshold}:
                signals.append(-1)  # Overbought - sell
            elif z_score < -{threshold}:
                signals.append(1)   # Oversold - buy
            else:
                signals.append(0)   # Hold
        
        return signals
    '''
    def _generate_ai_based_code(self, strategy: Dict) -> str:
    """Generate code from extracted mathematical technique"""
    technique = strategy.get('mathematical_technique', '').lower()
    equation = strategy.get('equation', '')
    params = strategy.get('parameters', {})
    
    # Route to technique-specific generators
    if 'kalman' in technique:
        return self._generate_kalman_code(params, equation)
    elif 'fourier' in technique or 'fft' in technique:
        return self._generate_fft_code(params, equation)
    elif 'ornstein' in technique or 'mean reversion' in technique:
        return self._generate_ou_code(params, equation)
    elif 'wavelet' in technique:
        return self._generate_wavelet_code(params, equation)
    else:
        return self._generate_generic_equation_code(strategy)

def _generate_kalman_code(self, params: Dict, equation: str) -> str:
    """Generate Kalman filter implementation"""
    measurement_noise = params.get('measurement_noise', 0.01)
    process_noise = params.get('process_noise', 0.001)
    
    return f'''
def strategy(prices):
    """
    Kalman Filter Strategy
    Equation: {equation}
    """
    import numpy as np
    import pandas as pd
    
    signals = []
    
    # Kalman filter initialization
    x = prices.iloc[0]  # Initial state
    P = 1.0  # Initial covariance
    R = {measurement_noise}  # Measurement noise
    Q = {process_noise}  # Process noise
    
    for price in prices:
        # Prediction
        x_pred = x
        P_pred = P + Q
        
        # Update
        K = P_pred / (P_pred + R)  # Kalman gain
        x = x_pred + K * (price - x_pred)
        P = (1 - K) * P_pred
        
        # Trading signal
        innovation = price - x
        if innovation > 2 * np.sqrt(P):
            signals.append(-1)  # Price too high
        elif innovation < -2 * np.sqrt(P):
            signals.append(1)   # Price too low
        else:
            signals.append(0)
    
    return signals
'''

    def _generate_fft_code(self, params: Dict, equation: str) -> str:
        """Generate FFT-based strategy"""
        window = params.get('fft_window', 256)
        freq_min = params.get('freq_min', 0.1)
        freq_max = params.get('freq_max', 0.5)
        
        return f'''
    def strategy(prices):
        """
        FFT Spectral Analysis
        Equation: {equation}
        """
        import numpy as np
        from scipy import fft
        
        signals = []
        window_size = {window}
        
        for i in range(len(prices)):
            if i < window_size:
                signals.append(0)
                continue
            
            # Extract window
            price_window = prices.iloc[i-window_size:i].values
            
            # Compute FFT
            fft_vals = fft.fft(price_window)
            freqs = fft.fftfreq(len(price_window))
            
            # Filter frequency band
            mask = (freqs >= {freq_min}) & (freqs <= {freq_max})
            power = np.abs(fft_vals[mask])
            
            # Signal based on dominant frequency power
            if len(power) > 0 and np.max(power) > np.mean(power) * 2:
                signals.append(1)
            else:
                signals.append(0)
        
        return signals
    '''

    def _generate_ou_code(self, params: Dict, equation: str) -> str:
        """Generate Ornstein-Uhlenbeck mean reversion strategy"""
        theta = params.get('mean_reversion_speed', 0.15)
        lookback = params.get('lookback_period', 20)
        
        return f'''
    def strategy(prices):
        """
        Ornstein-Uhlenbeck Mean Reversion
        Equation: {equation}
        """
        import numpy as np
        
        signals = []
        theta = {theta}
        
        for i in range(len(prices)):
            if i < {lookback}:
                signals.append(0)
                continue
            
            # Estimate mean from recent prices
            mu = prices.iloc[i-{lookback}:i].mean()
            
            # Current deviation from mean
            deviation = prices.iloc[i] - mu
            
            # Expected mean reversion: E[dX] = -theta * deviation * dt
            expected_change = -theta * deviation
            
            # Trading signal
            if expected_change > 0.5:
                signals.append(1)   # Expect upward reversion
            elif expected_change < -0.5:
                signals.append(-1)  # Expect downward reversion
            else:
                signals.append(0)
        
        return signals
    '''

    def _generate_fallback_code(self, strategy: Dict) -> str:
        """Generate fallback code if AI code generation fails"""
        params = strategy.get('parameters', {})
        lookback = params.get('lookback_period', params.get('window', 20))
        threshold = params.get('threshold', 2.0)
        
        return f'''
    def strategy(prices):
        """
        Fallback AI Strategy: {strategy['name']}
        Type: {strategy.get('type', 'ai_extracted')}
        Logic: {strategy.get('logic', 'AI-extracted trading strategy')}
        """
        import pandas as pd
        import numpy as np
        
        signals = []
        
        # Simple mean reversion as fallback
        rolling_mean = prices.rolling(window={lookback}).mean()
        rolling_std = prices.rolling(window={lookback}).std()
        
        for i in range(len(prices)):
            if i < {lookback}:
                signals.append(0)
                continue
                
            if rolling_std.iloc[i] == 0:
                signals.append(0)
                continue
                
            z_score = (prices.iloc[i] - rolling_mean.iloc[i]) / rolling_std.iloc[i]
            
            if z_score > {threshold}:
                signals.append(-1)  # Overbought - sell
            elif z_score < -{threshold}:
                signals.append(1)   # Oversold - buy
            else:
                signals.append(0)   # Hold
        
        return signals
    '''

    def _generate_strategy_name(self, paper_title: str, strategy_type: str) -> str:
        """Generate a clean strategy name"""
        # Clean the paper title
        clean_title = re.sub(r'[^a-zA-Z0-9]', ' ', paper_title)
        clean_title = ' '.join(clean_title.split()[:4])  # First 4 words
        clean_title = clean_title.lower().replace(' ', '_')
        
        return f"ai_{strategy_type}_{clean_title}"[:50]