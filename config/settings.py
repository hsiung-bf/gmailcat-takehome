"""Central configuration: thresholds, model names, and paths."""

# Data paths
DATA_DIR = "data"
INBOX_DIR = "inbox"
OUTPUTS_DIR = "outputs"
CACHE_DIR = "cache"

# Embedding model
EMBEDDING_MODEL = "all-mpnet-base-v2" 

# Cross-encoder configuration
CROSS_ENCODER_ENABLED = True
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Candidate generation configuration
CANDIDATE_LIMIT_BM25 = 50  # Max candidates from BM25 search
CANDIDATE_LIMIT_SEMANTIC = 50  # Max candidates from semantic search
CANDIDATE_LIMIT_TOTAL = 100  # Max total candidates before cross-encoder

# Threshold partitioning configuration
THRESHOLDING_STRATEGY = "fixed"  # "fixed" or "statistical"

# Fixed theshold partitioning configs
FIXED_HIGH_PERCENTILE = 85.0
FIXED_LOW_PERCENTILE = 15.0
APPLY_SIGMOID = True

# Statistical threshold partitioning configs
STATISTICAL_MAD_MULTIPLIER = 0.75
STATISTICAL_TARGET_GREY_FRAC = 0.10
STATISTICAL_MIN_CANDIDATES = 5