# =============================================================================
# File: test_projection_matrix_cache.py
# Date: 2026-01-09
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================

import numpy as np

from app.services.cache_registry import clear_projection_matrix_cache, get_projection_matrix_cache
from app.services.embedder_service import SentenceTransformer


def test_projection_matrix_cache_recreation():
    # Use a fixed-size input vector so input_dim is deterministic
    input_dim = 128
    proj_dim = 64
    emb = np.arange(input_dim, dtype=np.float32)

    # Ensure cache starts empty
    clear_projection_matrix_cache()
    cache = get_projection_matrix_cache()
    assert cache.size() == 0

    # First projection: creates and caches the random matrix
    out1 = SentenceTransformer._project_embedding(emb, proj_dim)
    assert cache.size() >= 1

    # Clear cache and recompute; deterministic RNG seed should recreate identical matrix
    clear_projection_matrix_cache()
    assert cache.size() == 0

    out2 = SentenceTransformer._project_embedding(emb, proj_dim)

    # Results should be identical after recreation
    assert np.allclose(out1, out2, atol=1e-6)
