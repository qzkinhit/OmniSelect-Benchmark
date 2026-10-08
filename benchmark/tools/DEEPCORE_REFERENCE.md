# DeepCore reference implementations

`deepcore_reference.py` holds standalone numpy copies of four selection rules collected by the
DeepCore library (Guo, Zhao and Bai, DEXA 2022). `omniselect/tests/test_baseline_deepcore_consistency.py`
checks that they return the same selections as the portfolio implementations under
`benchmark/Methods/`.

| Name | Family | Rule | Reference |
|---|---|---|---|
| `herding` | geometric | greedily moves the mean of the selected set toward the mean of the full set | Welling, ICML 2009 |
| `kcenter` | geometric | farthest-point (greedy k-center) coverage | Sener and Savarese, ICLR 2018 |
| `el2n` | score-based | keeps the k records with the largest norm of softmax(logits) minus the one-hot label | Paul et al., NeurIPS 2021 |
| `grand` | score-based | last-layer gradient-norm proxy, EL2N times the feature norm, top k | Paul et al., NeurIPS 2021 |

`kcenter` is a portfolio candidate of the controller and is not reported as an external baseline.

```bash
python -m benchmark.tools.deepcore_fidelity --dataset uoft-cs/cifar10 --pool 1500 --test 1000
```

The check encodes a small CIFAR subset with frozen CLIP, runs each rule at a 30% budget on a clean
pool and on a pool with 40% label flips, fits a logistic probe on each selection and prints the
top-1 test accuracy. On the clean pool the four rules stay close to random selection, which agrees
with the DeepCore finding that random selection is a strong baseline with strong features. On the
noisy pool EL2N and GraNd select the flipped labels first and fall below random selection, which is
the documented noise sensitivity of error-based scores.

## References

- Guo, Zhao and Bai. DeepCore: A Comprehensive Library for Coreset Selection in Deep Learning. DEXA 2022. https://github.com/PatrickZH/DeepCore
- Welling. Herding Dynamical Weights to Learn. ICML 2009.
- Sener and Savarese. Active Learning for Convolutional Neural Networks: A Core-Set Approach. ICLR 2018.
- Paul, Ganguli and Dziugaite. Deep Learning on a Data Diet: Finding Important Examples Early in Training. NeurIPS 2021.
