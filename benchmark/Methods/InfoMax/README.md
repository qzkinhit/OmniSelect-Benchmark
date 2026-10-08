# InfoMax

Fidelity label: **local implementation**.

Source: Tan et al., Data Pruning by Information Maximization, ICLR 2025.

Strategy name: `infomax`. Implementation: [`method.py`](method.py). The authors' repository carries no license, so the code is written from the objective in the paper and copies nothing.

What is transferred: the quadratic objective importance minus pairwise redundancy, max s^T x - alpha x^T K x over x in [0, 1]^n with sum(x) = k, with a sparsified similarity (cosine of L2-normalized features kept for the 10 nearest neighbours of each record, symmetrized) and a gradient solver on the relaxed problem (projected gradient ascent onto the capped simplex, 300 steps, step 1 / (2 alpha ||K||_2)). The k largest coordinates are selected.

Inputs on this benchmark: importance is the min-max scaled influence channel of the track, features are the track representation (CLIP features, normalized windows, standardized TEP or Electricity rows, native penultimate features). alpha = beta mean(s) / (rho mean row sum of K) with beta = 1 and rho = k / n.

Deviations: the paper's importance scores come from a trained model's per-sample information measures. Here the influence channel plays that role. The balance hyperparameter is set by the normalization above, not tuned per dataset. Excluded on text, where LESS is the gradient-based member of the unified text portfolio.

Standalone run: `benchmark/MethodsRunScript/run_infomax/`.
