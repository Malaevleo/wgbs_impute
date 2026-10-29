"""Shared argument validation for both Python and command-line callers."""

from __future__ import annotations

import math
from numbers import Integral


def validate_options(**values):
    nonnegative = {"min_ref_mean_cov", "window", "sd_passthrough", "noise_floor", "n_refs", "seed"}
    positive = {
        "min_ref_site_cov",
        "neighbors",
        "max_cov",
        "neighbor_min_cov",
        "threads",
        "local_iterations",
        "local_depth",
        "max_train",
        "refnn_k",
        "kappa_sites",
    }
    for name, value in values.items():
        if name in {
            "window",
            "n_refs",
            "seed",
            "min_ref_site_cov",
            "neighbors",
            "neighbor_min_cov",
            "threads",
            "local_iterations",
            "local_depth",
            "max_train",
            "refnn_k",
            "kappa_sites",
        }:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if name in nonnegative | positive | {"shrink"}:
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name in nonnegative and value < 0:
                raise ValueError(f"{name} must be nonnegative")
            if name in positive and value <= 0:
                raise ValueError(f"{name} must be positive")
            if name == "shrink" and not 0 <= value <= 1:
                raise ValueError("shrink must be between 0 and 1")
    if "preset" in values and values["preset"] not in {"pairwise", "research"}:
        raise ValueError("preset must be pairwise or research")


def validate_chrom(chrom):
    if not chrom or chrom in {".", ".."} or any(c in chrom for c in "/\\\t\r\n\x00"):
        raise ValueError(f"invalid chromosome name: {chrom!r}")
