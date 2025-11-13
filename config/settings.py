import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

# ==================== ATHENA SWARM CONFIGURATION ====================

SWARM_CONFIG = {
    'max_concurrent_agents': 12,
    'max_workers_per_agent': 2,
    'task_timeout_seconds': 120,
    'retry_attempts': 2,
    'rate_limit_delay': 0.8,
    
    # Pipeline control
    'max_queries_per_cycle': 8,
    'min_confidence_threshold': 0.6,
    'max_bullshit_score': 0.3,
    'quality_threshold': 0.5,
    
    # Resource management
    'max_concepts_per_agent': 10,
    'max_strategies_per_run': 40,
    'research_cycle_interval': 1800,  # 1 hour

    
}

# ==================== AGENT SPECIFIC CONFIGURATION ====================

AGENT_CONFIG = {
    'universal_scout': {
        'enabled': True,
        'workers': 3,
        'rate_limit': 1.5,
        'max_concepts': 20,
        'discovery_depth': 2
    },
    'interpreter': {
        'enabled': True, 
        'workers': 2,
        'rate_limit': 1.0,
        'interpretation_depth': 3
    },
    'builder': {
        'enabled': True,
        'workers': 3,
        'rate_limit': 1.0,
        'max_strategies': 15
    },
    'evaluator': {
        'enabled': True,
        'workers': 2,
        'rate_limit': 1.0,
        'evaluation_metrics': ['sharpe', 'win_rate', 'max_drawdown']
    },
    'quality': {
        'enabled': True,
        'workers': 3,
        'rate_limit': 1.0,
        'min_quality_score': 0.6
    },
    'analyst': {
        'enabled': True,
        'workers': 1,
        'rate_limit': 1.0,
        'analysis_depth': 2
    }
}

# ==================== STRATEGY EVALUATION ====================

STRATEGY_EVALUATION = {
    'min_sharpe': 0.3,
    'min_confidence': 0.6,
    'max_drawdown': 0.20,
    'min_success_rate': 0.50,
    'min_trades': 10,
    
    # Performance thresholds
    'threshold_sharpe': 0.7,
    'threshold_win_rate': 0.6,
    
    # Bonus scoring
    'parameter_variety_bonus': 0.3,
    'weirdness_bonus': 0.1,
    'novelty_bonus': 0.15,
    
    # Quality thresholds
    'min_applicability_score': 0.4,
    'max_bullshit_score': 0.3,
    'min_mathematical_rigor': 0.6,
    'min_implementation_feasibility': 0.5,
    'min_novelty_score': 0.3
}

# ==================== BACKTESTING CONFIGURATION ====================

BACKTEST_CONFIG = {
    'timeframes': ['5m', '15m', '1h', '4h', '1d'],
    'assets': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT'],
    'lookback_periods': [100, 200, 500],
    'test_periods': 1000,
    'initial_capital': 10000,
    'commission': 0.001,  # 0.1% for crypto
    'slippage': 0.0005,  # 0.05% slippage
    
    # Crypto-specific
    'crypto_assets': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT'],
    'funding_rate_consideration': True,
    'liquidation_risk_assessment': True
}

# ==================== PERFORMANCE METRICS ====================

PERFORMANCE_METRICS = {
    'primary': ['sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor', 'total_return'],
    'secondary': ['calmar_ratio', 'sortino_ratio', 'omega_ratio', 'tail_ratio', 'skewness'],
    'risk': ['var_95', 'cvar_95', 'volatility', 'beta', 'ulcer_index'],
    'crypto_specific': ['volatility_adjusted_sharpe', 'regime_adaptability', 'liquidity_efficiency']
}

# ==================== KEEP ALL YOUR EXISTING SEARCH TERMS ====================

# Seed concepts for diamond mining (KEEP ALL)
CRYPTO_FOCUSED_SEEDS = [
    "volatility modeling in incomplete markets",
    "regime detection in high-frequency data", 
    "liquidity provision mathematical models",
    "network effects in cryptocurrency markets",
    "multi-scale analysis of price series",
    "stochastic control for portfolio optimization",
    "information asymmetry in decentralized markets",
    "market microstructure crypto adaptations",
    "cross-exchange arbitrage mathematics",
    "on-chain analytics predictive models",
    "physics chaos theory market prediction",
    "biological neural networks trading signals",
    "engineering control theory portfolio management", 
    "quantum computing optimization finance",
    "thermodynamics market entropy models",
    "fluid dynamics price movement analogs",
    "statistical mechanics risk distribution",
    "signal processing noise filtering markets",
    "graph theory market network analysis",
    "machine learning mathematical foundations",
    "sentiment analysis mathematical foundations",
    "volatility surface modeling derivatives",
    "liquidation cascade prediction models",
    "whale transaction pattern recognition",
    "market manipulation detection algorithms",
    "flash crash prediction mathematics", 
    "exchange arbitrage network analysis",
    "stablecoin mechanism design",
    "yield farming optimization math",
    "liquidity pool mathematical models"
]

# All your existing query categories (KEEP ALL)
SIGNAL_PROCESSING_QUERIES = [
    'kalman filter state estimation time series',
    'extended kalman filter nonlinear systems',
    'particle filter sequential monte carlo',
    'unscented kalman filter',
    'adaptive filtering recursive least squares',
    'fourier transform spectral analysis',
    'fast fourier transform signal processing',
    'wavelet transform multi-resolution',
    'short-time fourier transform',
    'hilbert transform instantaneous frequency',
    'empirical mode decomposition',
    'wiener filter optimal filtering',
    'savitzky-golay filter smoothing',
    'butterworth filter design',
    'hodrick prescott filter detrending',
    'bandpass filter frequency selection'
]

STOCHASTIC_PROCESS_QUERIES = [
    'ornstein uhlenbeck process mean reversion',
    'vasicek model interest rates',
    'cox ingersoll ross process',
    'jump diffusion merton model',
    'regime switching markov',
    'levy process stable distributions',
    'poisson process jumps',
    'compound poisson jump',
    'heston stochastic volatility',
    'sabr volatility model',
    'garch volatility clustering',
    'egarch asymmetric volatility',
    'gjr-garch leverage effects',
    'hawkes process self-exciting',
    'hawkes process clustering',
    'multivariate hawkes process'
]

STATISTICAL_INFERENCE_QUERIES = [
    'engle granger cointegration test',
    'johansen cointegration multiple series',
    'cointegration pairs trading',
    'error correction model',
    'hidden markov model regime switching',
    'bayesian change point detection',
    'cusum change point algorithm',
    'sequential probability ratio test',
    'structural break detection',
    'granger causality test',
    'transfer entropy causality',
    'copula dependence modeling',
    'vine copula multivariate',
    'augmented dickey fuller test',
    'phillips perron stationarity',
    'kpss test trend stationarity',
    'ljung box autocorrelation test',
    'jarque bera normality test'
]

TIME_SERIES_ANALYSIS_QUERIES = [
    'arima autoregressive integrated',
    'arma model selection',
    'arimax exogenous variables',
    'seasonal arima sarima',
    'vector autoregression var',
    'state space model kalman',
    'dynamic linear model',
    'unobserved components model',
    'seasonal decomposition time series',
    'seasonal trend decomposition loess',
    'autocorrelation function analysis',
    'partial autocorrelation pacf',
    'spectral density periodogram'
]

# TIER 2: PROVEN BUT UNDERUSED
OPTIMIZATION_QUERIES = [
    'genetic algorithm optimization',
    'differential evolution algorithm',
    'evolution strategies cma-es',
    'genetic programming symbolic regression',
    'particle swarm optimization',
    'ant colony optimization',
    'bee algorithm optimization',
    'firefly algorithm',
    'bayesian optimization gaussian process',
    'hyperparameter tuning bayesian',
    'surrogate optimization expensive functions',
    'expected improvement acquisition',
    'nelder mead simplex optimization',
    'simulated annealing global optimization',
    'coordinate descent optimization',
    'powell method optimization',
    'pareto optimization multi-objective',
    'nsga-ii multi-objective genetic',
    'scalarization multi-objective optimization'
]

INFORMATION_THEORY_QUERIES = [
    'shannon entropy information content',
    'differential entropy continuous',
    'mutual information dependence',
    'conditional mutual information',
    'joint entropy multivariate',
    'transfer entropy information flow',
    'effective transfer entropy',
    'directed information theory',
    'kullback leibler divergence',
    'jensen shannon divergence',
    'bhattacharyya distance',
    'hellinger distance',
    'kolmogorov complexity algorithmic',
    'lempel ziv complexity',
    'approximate entropy',
    'sample entropy time series',
    'permutation entropy'
]

# TIER 3: WEIRD BUT VALUABLE
PHYSICS_INSPIRED_QUERIES = [
    'statistical mechanics equilibrium',
    'thermodynamic entropy disorder',
    'boltzmann distribution energy',
    'partition function statistical',
    'gibbs distribution temperature',
    'ising model phase transition',
    'phase transition critical points',
    'critical phenomena scaling',
    'percolation theory threshold',
    'criticality self-organized',
    'diffusion equation spreading',
    'reaction diffusion systems',
    'anomalous diffusion fractional',
    'advection diffusion transport',
    'non-equilibrium dynamics',
    'relaxation dynamics systems',
    'metastable states transitions'
]

NETWORK_GRAPH_QUERIES = [
    'correlation network analysis',
    'partial correlation network',
    'dynamic correlation network',
    'hierarchical clustering dendrogram',
    'minimum spanning tree clustering',
    'maximum spanning tree network',
    'planar maximally filtered graph',
    'community detection algorithms',
    'modularity optimization louvain',
    'betweenness centrality network',
    'eigenvector centrality measure',
    'pagerank centrality algorithm',
    'dynamic network topology',
    'temporal network analysis',
    'network stability measures',
    'network entropy information'
]

BIOLOGY_INSPIRED_QUERIES = [
    'lotka volterra predator prey',
    'predator prey dynamics cycles',
    'competition models ecology',
    'sir model epidemic spreading',
    'seir epidemic model',
    'epidemic threshold network',
    'information spreading cascade',
    'evolutionary game theory',
    'replicator dynamics evolution',
    'adaptive dynamics evolution',
    'fitness landscape optimization',
    'swarm intelligence algorithms',
    'collective decision making',
    'synchronization coupled oscillators'
]

CONTROL_THEORY_QUERIES = [
    'pid control feedback systems',
    'linear quadratic regulator lqr',
    'model predictive control mpc',
    'optimal control theory',
    'adaptive control systems',
    'self-tuning regulator',
    'gain scheduling control',
    'robust control uncertainty',
    'observer design state estimation',
    'luenberger observer',
    'sliding mode control',
    'backstepping control nonlinear'
]

# TIER 4: ADVANCED/EXPERIMENTAL
MACHINE_LEARNING_MATH_QUERIES = [
    'attention mechanism neural networks',
    'self-attention transformer',
    'memory augmented neural networks',
    'lstm long short-term memory',
    'gru gated recurrent unit',
    'bidirectional recurrent networks',
    'echo state networks reservoir',
    'reinforcement learning control',
    'q-learning temporal difference',
    'policy gradient methods',
    'actor-critic algorithms',
    'deep reinforcement learning',
    'online learning streaming data',
    'incremental learning algorithms',
    'continual learning catastrophic forgetting',
    'meta-learning few-shot',
    'learning to learn algorithms',
    'neural architecture search',
    'adversarial training robustness',
    'adversarial examples defense',
    'robust optimization worst-case'
]

CHAOS_NONLINEAR_QUERIES = [
    'lyapunov exponent chaos',
    'strange attractor dynamics',
    'bifurcation theory nonlinear',
    'chaos control synchronization',
    'fractal dimension time series',
    'multifractal analysis',
    'hurst exponent long memory',
    'detrended fluctuation analysis',
    'nonlinear time series analysis',
    'phase space reconstruction',
    'recurrence plot analysis',
    'nonlinear prediction methods'
]

ADVANCED_STATISTICS_QUERIES = [
    'extreme value theory tail risk',
    'generalized pareto distribution',
    'peaks over threshold method',
    'block maxima extreme events',
    'robust statistics outliers',
    'm-estimators robust regression',
    'trimmed mean robust estimation',
    'bootstrap confidence intervals',
    'jackknife variance estimation',
    'permutation test hypothesis',
    'monte carlo simulation methods'
]

# Keep all your existing query sets
TIER_1_QUERIES = (
    SIGNAL_PROCESSING_QUERIES +
    STOCHASTIC_PROCESS_QUERIES +
    STATISTICAL_INFERENCE_QUERIES +
    TIME_SERIES_ANALYSIS_QUERIES
)

TIER_2_QUERIES = (
    OPTIMIZATION_QUERIES +
    INFORMATION_THEORY_QUERIES
)

TIER_3_QUERIES = (
    PHYSICS_INSPIRED_QUERIES +
    NETWORK_GRAPH_QUERIES +
    BIOLOGY_INSPIRED_QUERIES +
    CONTROL_THEORY_QUERIES
)

TIER_4_QUERIES = (
    MACHINE_LEARNING_MATH_QUERIES +
    CHAOS_NONLINEAR_QUERIES +
    ADVANCED_STATISTICS_QUERIES
)

ALL_MATH_QUERIES = TIER_1_QUERIES + TIER_2_QUERIES + TIER_3_QUERIES + TIER_4_QUERIES

# CRYPTO-SPECIFIC QUERIES
CRYPTO_SPECIFIC_QUERIES = [
    'order book imbalance prediction',
    'market making optimal strategies',
    'high frequency trading microstructure',
    'limit order book dynamics',
    'price impact models trading',
    'adverse selection market making',
    'inventory risk market makers',
    'funding rate arbitrage perpetual',
    'triangular arbitrage cryptocurrency',
    'cross-exchange arbitrage',
    'cex dex arbitrage opportunities',
    'basis trading futures spot',
    'liquidation cascade prediction',
    'whale wallet flow analysis',
    'exchange netflow indicators',
    'mempool transaction analysis',
    'mev maximal extractable value',
    'sandwich attack detection',
    'realized volatility forecasting',
    'implied volatility smile crypto',
    'volatility clustering garch crypto',
    'jump detection cryptocurrency',
    'tail risk extreme events crypto'
]

# ==================== SEARCH STRATEGIES (KEEP ALL) ====================

SEARCH_STRATEGY = {
    'diamond_mining': {
        'description': 'Maximum scope diamond mining - all sources and techniques',
        'queries': CRYPTO_FOCUSED_SEEDS + ALL_MATH_QUERIES + CRYPTO_SPECIFIC_QUERIES,
        'sources': ['academic', 'github', 'cross_domain'],
        'focus': 'Find novel mathematical concepts everywhere'
    },
    
    'aggressive_discovery': {
        'description': 'Maximum scope - all mathematical techniques',
        'queries': ALL_MATH_QUERIES + CRYPTO_SPECIFIC_QUERIES,
        'sources': ['academic', 'github'],
        'focus': 'Quantity - find everything applicable'
    },
    
    'proven_edges': {
        'description': 'Deploy-ready strategies only',
        'queries': TIER_1_QUERIES + CRYPTO_SPECIFIC_QUERIES,
        'sources': ['academic', 'github'],
        'focus': 'Quality - high probability winners'
    },
    
    'experimental': {
        'description': 'Explore weird cross-domain ideas',
        'queries': TIER_3_QUERIES + TIER_4_QUERIES + CRYPTO_FOCUSED_SEEDS,
        'sources': ['academic', 'cross_domain'],
        'focus': 'Innovation - untested techniques'
    }
}

# ==================== ACTIVE CONFIGURATION ====================

# Default strategy
DEFAULT_STRATEGY = 'diamond_mining'

# Active queries based on strategy
ACTIVE_QUERIES = SEARCH_STRATEGY[DEFAULT_STRATEGY]['queries']

# Active configuration
ACTIVE_CONFIG = SWARM_CONFIG

# ==================== MATHEMATICAL TECHNIQUE PRIORITIES ====================

TECHNIQUE_PRIORITIES = {
    # High Priority (Established finance math)
    'kalman_filter': 10,
    'ornstein_uhlenbeck': 10,
    'hawkes_process': 9,
    'cointegration': 9,
    'garch': 9,
    
    # Medium Priority (Proven techniques)
    'wavelet_transform': 8,
    'transfer_entropy': 8,
    'genetic_algorithm': 7,
    'particle_filter': 7,
    'hidden_markov_model': 7,
    
    # Experimental Priority (High potential)
    'network_analysis': 6,
    'phase_transition': 6,
    'chaos_theory': 5,
    'quantum_mechanics': 4,
    'biological_models': 4
}

# ==================== LOGGING CONFIGURATION ====================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'athena_research.log',
    'max_size_mb': 100,
    'backup_count': 5
}

# Alert thresholds
MIN_SHARPE_ALERT = 0.3
MIN_CONFIDENCE_ALERT = 0.7