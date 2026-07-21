"""ZIP / Entropy-Law baseline — model-free redundancy-minimizing selection (CPU).

Reference: Yin et al., "Entropy Law: The Story Behind Data Compression and LLM
Performance" (ZIP), USTC-StarTeam/ZIP.
"""
from .zip_select import compression_ratio, select, zip_select

__all__ = ["select", "zip_select", "compression_ratio"]
