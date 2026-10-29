"""Reproducible read-subset sampling used to calibrate fusion weights."""

from __future__ import annotations

import numpy as np


def subsample_depths(meth_full, cov_full, depths, mean_full, seed):
    """Draw each requested depth independently from the original reads."""
    cov = np.asarray(cov_full, dtype=np.int64)
    meth = np.asarray(meth_full, dtype=np.int64)
    if np.any(cov < 0) or np.any(meth < 0) or np.any(meth > cov):
        raise ValueError("thinning requires 0 <= methylated reads <= coverage")
    rng_cov = np.random.default_rng(seed)
    rng_meth = np.random.default_rng(seed + 555)
    result = {}
    for depth in sorted(depths, reverse=True):
        if not np.isfinite(depth) or depth <= 0:
            raise ValueError("thinning depths must be positive and finite")
        keep = min(1.0, depth / max(float(mean_full), 1e-9))
        new_cov = rng_cov.binomial(cov, keep)
        new_meth = np.zeros_like(new_cov)
        ok = new_cov > 0
        new_meth[ok] = rng_meth.hypergeometric(meth[ok], (cov - meth)[ok], new_cov[ok])
        result[depth] = (new_meth.astype(float), new_cov.astype(float))
    return result
