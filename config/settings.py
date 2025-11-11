import os
from dotenv import load_dotenv

load_dotenv()

# config/settings.py - ENHANCED MATHEMATICAL FOCUS WITH SWARM INTEGRATION

# API Keys
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # Optional, for enhanced GitHub access

# Alert Preferences
MIN_SHARPE_ALERT = 0.3  # Lowered for more strategy discovery
MIN_CONFIDENCE_ALERT = 0.7

# Swarm Configuration
SWARM_CONFIG = {
    'max_workers': 8,
    'arxiv_max_results': 25,
    'github_max_results': 15,
    'timeout_seconds': 30,
    'retry_attempts': 2
}

# CORE MATHEMATICAL DOMAINS - Expanded for Swarm Research
MATHEMATICAL_DOMAINS = {
    'stochastic_processes': [
        'brownian motion', 'levy processes', 'ornstein-uhlenbeck',
        'jump diffusion', 'stochastic calculus', 'ito calculus',
        'stochastic differential equations', 'martingale theory',
        'markov processes', 'point processes', 'regime switching'
    ],
    
    'optimization_control': [
        'convex optimization', 'non-convex optimization', 'global optimization',
        'stochastic optimization', 'dynamic programming', 'optimal control',
        'reinforcement learning', 'multi-armed bandits', 'bayesian optimization',
        'evolutionary algorithms', 'swarm intelligence', 'metaheuristics'
    ],
    
    'time_series_analysis': [
        'state space models', 'kalman filtering', 'particle filtering',
        'spectral analysis', 'wavelet analysis', 'fourier transforms',
        'arima models', 'garch models', 'cointegration', 'vector autoregression',
        'structural breaks', 'regime detection', 'anomaly detection'
    ],
    
    'information_theory': [
        'entropy', 'mutual information', 'transfer entropy',
        'information geometry', 'fisher information', 'kullback-leibler divergence',
        'minimum description length', 'kolmogorov complexity', 'information bottleneck'
    ],
    
    'graph_network_theory': [
        'graph neural networks', 'network analysis', 'community detection',
        'random graph models', 'small-world networks', 'scale-free networks',
        'graph signal processing', 'topological data analysis', 'persistent homology'
    ],
    
    'statistical_methods': [
        'bayesian inference', 'monte carlo methods', 'markov chain monte carlo',
        'hierarchical models', 'gaussian processes', 'dirichlet processes',
        'nonparametric statistics', 'copula models', 'extreme value theory'
    ]
}

# PURE MATHEMATICAL QUERIES - Enhanced for Comprehensive Coverage
QUANT_QUERIES = [
    # Core Mathematical Finance
    'stochastic calculus financial applications',
    'partial differential equations finance',
    'measure theory probability finance',
    'optimal stopping theory',
    'portfolio optimization mathematics',
    
    # Advanced Stochastic Methods
    'levy processes financial modeling',
    'fractional brownian motion',
    'regime switching models',
    'stochastic volatility models',
    'jump diffusion processes',
    
    # Quantitative Methods
    'numerical methods finance',
    'finite difference methods',
    'monte carlo methods finance',
    'finite element methods financial',
    'computational finance algorithms'
]

BROAD_QUERIES = [
    # Signal Processing & Time Series
    'kalman filter time series',
    'fourier analysis signal processing', 
    'wavelet transform signal decomposition',
    'hidden markov model state estimation',
    'particle filter sequential monte carlo',
    'bayesian inference probabilistic models',
    
    # Optimization & Search
    'gradient descent optimization',
    'monte carlo methods',
    'simulated annealing optimization',
    'genetic algorithms optimization',
    'convex optimization algorithms',
    
    # Statistical Methods
    'change point detection statistics',
    'anomaly detection algorithms',
    'clustering high dimensional data',
    'dimensionality reduction techniques',
    'manifold learning geometry'
]

NOVEL_QUERIES = [
    # Modern Mathematical Approaches
    "neural ordinary differential equations",
    "attention mechanism mathematics", 
    "transformer architecture theory",
    "geometric deep learning",
    "causal inference methods",
    
    # Emerging Mathematical Fields
    "federated learning mathematics",
    "meta learning theory",
    "neural architecture search",
    "explainable AI mathematics",
    "AI safety theory",
    
    # Advanced ML Mathematics
    "information bottleneck deep learning",
    "neural tangent kernel theory",
    "mean field theory neural networks",
    "category theory machine learning",
    "topological deep learning"
]

# SWARM-SPECIFIC QUERIES - For Parallel Research Agents
SWARM_QUERIES = {
    'arxiv_focused': [
        "stochastic process financial modeling",
        "reinforcement learning algorithmic trading", 
        "time series forecasting machine learning",
        "bayesian inference finance",
        "optimization portfolio management",
        "graph neural networks financial networks",
        "anomaly detection market regimes",
        "signal processing financial time series"
    ],
    
    'github_technical': [
        "algorithmic-trading python",
        "quantitative-finance machine-learning",
        "time-series-forecasting deep-learning",
        "stochastic-processes implementation",
        "optimization-methods financial",
        "bayesian-methods trading"
    ],
    
    'cross_domain': [
        "physics-inspired financial models",
        "biological systems financial networks", 
        "neuroscience market prediction",
        "ecological models portfolio",
        "social network analysis finance",
        "complex systems financial markets"
    ]
}

# FRINGE & CUTTING-EDGE MATHEMATICS
FRINGE_QUERIES = [
    # Quantum Mathematics
    "quantum machine learning",
    "quantum finance applications",
    "quantum annealing optimization",
    
    # Advanced Topological Methods
    "topological data analysis finance",
    "persistent homology applications",
    "algebraic topology data analysis",
    
    # Information-Theoretic Approaches
    "information geometry finance",
    "fisher information markets",
    "minimum description length trading",
    
    # Unconventional Mathematical Approaches
    "reservoir computing theory",
    "liquid state machines finance",
    "hyperdimensional computing",
    "kolmogorov complexity applications"
]

# Source Configuration for Swarm
SOURCES = {
    'arxiv': {
        'enabled': True,
        'categories': ['q-fin', 'stat.ML', 'cs.LG', 'math.OC', 'math.ST', 'math.PR'],
        'max_results': 25
    },
    'github': {
        'enabled': True,
        'topics': ['algorithmic-trading', 'quantitative-finance', 'machine-learning'],
        'languages': ['Python', 'Julia', 'R']
    },
    'paperswithcode': {
        'enabled': True,
        'domains': ['Finance', 'Time Series', 'Machine Learning']
    }
}

# Search Strategy with Swarm Integration
SEARCH_STRATEGY = {
    'daily': {
        'sources': ['arxiv', 'github'],
        'domains': ['stochastic_processes', 'optimization_control'],
        'queries': QUANT_QUERIES[:5] + BROAD_QUERIES[:3]
    },
    'weekly': {
        'sources': ['arxiv', 'github', 'paperswithcode'],
        'domains': list(MATHEMATICAL_DOMAINS.keys()),
        'queries': QUANT_QUERIES + BROAD_QUERIES + NOVEL_QUERIES[:3]
    },
    'exploratory': {
        'sources': ['arxiv'],
        'domains': ['all'],
        'queries': FRINGE_QUERIES + NOVEL_QUERIES
    }
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    'timeframes': ['1h', '4h', '1d'],
    'assets': ['BTC/USDT', 'ETH/USDT', 'SPY', 'GLD'],
    'lookback_periods': [50, 100, 200],
    'test_periods': 500
}

# Strategy Evaluation
STRATEGY_EVALUATION = {
    'min_sharpe': 0.3,
    'min_confidence': 0.6,
    'max_drawdown': 0.15,
    'min_success_rate': 0.55,
    'parameter_variety_threshold': 0.3
}

# Performance Metrics
PERFORMANCE_METRICS = {
    'primary': ['sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor'],
    'secondary': ['calmar_ratio', 'sortino_ratio', 'omega_ratio', 'tail_ratio'],
    'risk': ['var_95', 'cvar_95', 'volatility', 'beta']
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'research_bot.log',
    'max_size_mb': 100,
    'backup_count': 5
}

# Swarm Agent Configuration
AGENT_CONFIG = {
    'arxiv_miner': {
        'enabled': True,
        'workers': 3,
        'rate_limit': 1.0  # seconds between requests
    },
    'github_scout': {
        'enabled': True,
        'workers': 2,
        'rate_limit': 2.0
    },
    'math_extractor': {
        'enabled': True,
        'workers': 2
    },
    'cross_domain_bridge': {
        'enabled': True,
        'workers': 1
    }
}

# Cross-Domain Bridge Sources
CROSS_DOMAIN_SOURCES = {
    'physics': ['statistical mechanics', 'fluid dynamics', 'thermodynamics', 'quantum physics'],
    'biology': ['evolutionary biology', 'neuroscience', 'ecology', 'systems biology'],
    'computer_science': ['distributed systems', 'database theory', 'compiler design', 'operating systems'],
    'social_sciences': ['game theory', 'network science', 'behavioral economics', 'social networks']
}