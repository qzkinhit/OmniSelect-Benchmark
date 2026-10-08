# GLISTER

Fidelity label: **published-core transfer**.

Source: Killamsetty, Sivasubramanian, Ramakrishnan and Iyer, GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning, AAAI 2021.

Strategy name: `glister`. Implementation: [`method.py`](method.py), shared last-layer model in [`../_gradients.py`](../_gradients.py).

What is transferred: the one-step Taylor estimate of the validation log-likelihood, gain_i = g_i . G_V, with last-layer gradients g_i = phi_i e_i^T fixed at the warm-start parameters and the validation gradient G_V recomputed at Theta_S = Theta_0 - eta sum_{s in S} g_s after every greedy round (the R-greedy variant of the authors' CORDS code, `glisterstrategy.py`). The validation set is V_con.

Where it runs (`extras['last_layer']`): the frozen-feature vision probe (logistic head on CLIP or DINOv2 features, warm-started with `last_layer_warm` = 100 iterations on the whole pool), TEP21 with the MLP (output layer on the second hidden layer, MLP warm-started 30 iterations), forecasting with DLinear (DLinear is linear in [trend, 1, seasonal, 1], warm-started 10 epochs, half squared error). Excluded on tabular (TabPFN has no training gradient), text (LESS is the text gradient method) and native (needs gradients during the ResNet-18 run).

Deviations: one-shot selection before training instead of re-selection every few epochs during training. The step is eta = 1 / (k mean ||phi||^2) on the summed gradients, which equals one gradient step of length 1 / L on the mean gradient (the authors use the model learning rate on the sum). 20 greedy rounds of k / 20 records. No regularization term and no random share. The downstream learner is the track's own learner.

Standalone run: `benchmark/MethodsRunScript/run_glister/`. Inside the controller the strategy is a portfolio member when the membership table of the track admits it (`omniselect/core/portfolio/membership.py`).
