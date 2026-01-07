"""Project-wide numeric constants for stable comparisons and epsilons.

Keep this module minimal and import-safe to avoid circular imports.
"""

# Numerical epsilon for vector normalization (used for cosine-norm stability)
NORM_EPS: float = 1e-12

# Epsilon used to avoid division by zero when summing attention masks
# (used in mean pooling denominator safeguards)
MASK_SUM_EPS: float = 1e-9

# Large negative value used when masking before max-pooling to ensure
# masked positions do not affect the max operation.
MASK_NEG_INF: float = -1e9

# Deterministic random seed for projection matrix generation
RANDOM_SEED: int = 42
