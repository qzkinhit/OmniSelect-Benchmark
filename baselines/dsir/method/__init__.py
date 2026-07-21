"""DSIR baseline — Data Selection with Importance Resampling (CPU-light).

Reference: Xie et al., "Data Selection for Language Models via Importance
Resampling (DSIR)", NeurIPS 2023.
"""
from .dsir_select import dsir_select, hashed_ngram_counts

__all__ = ["dsir_select", "hashed_ngram_counts"]
