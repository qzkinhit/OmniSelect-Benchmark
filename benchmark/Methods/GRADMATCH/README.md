# GRADMATCH

Fidelity label: **published-core transfer**.

Source: Killamsetty, Durga, Ramakrishnan, De and Iyer, GRAD-MATCH: Gradient Matching based Data Subset Selection for Efficient Deep Model Training, ICML 2021.

Strategy name: `gradmatch`. Implementation: [`method.py`](method.py), shared last-layer model in [`../_gradients.py`](../_gradients.py).

What is transferred: per-class orthogonal matching pursuit with non-negative weights and ridge regularization (lam 0.5) that approximates the summed training gradient of each class by a weighted sum of member gradients (`PerClass` with `valid=False`, the paper's default, OrthogonalMP_REG of the authors' CORDS code). Class budgets are proportional to class sizes (largest remainder, exactly k in total). Unfilled slots are drawn at random, as in the authors' code. `target="val"` matches the summed V_con gradient instead. The OMP runs in kernel form on <g_i, g_j> = (phi_i . phi_j)(e_i . e_j), which equals OMP on the explicit last-layer gradients.

Where it runs: the same last layers as GLISTER (vision probe, TEP21 MLP, forecasting DLinear, one group for regression). Excluded on tabular, text and native for the reasons listed in the membership table.

Deviations: one-shot selection instead of periodic re-selection during training. The OMP weights are computed (returned by `gradmatch`) but the downstream learner trains without them, because the learners of this benchmark take unweighted subsets. Groups of more than 2,000 records (forecasting) add ceil(budget / 50) atoms per OMP iteration.

Standalone run: `benchmark/MethodsRunScript/run_gradmatch/`.
