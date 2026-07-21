import numpy as np

from mmdataselect.selectors.external_baselines import (
    dmf_published_update,
    quadmix_expected_counts,
    quadmix_published_core,
)


def test_quadmix_expected_count_direction_and_domain_rank():
    quality = np.array([0.95, 0.80, 0.20, 0.10, 0.90, 0.70, 0.30, 0.05])
    domains = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    counts = quadmix_expected_counts(
        quality, domains, lambdas=100.0, omegas=0.5,
        etas=1.0, epsilons=0.001,
    )
    assert counts[0] > counts[2]
    assert counts[4] > counts[6]
    assert counts[2] == counts[3] == 0.001
    assert counts[6] == counts[7] == 0.001


def test_quadmix_fixed_budget_is_reproducible_and_domain_aware():
    rng = np.random.default_rng(3)
    x0 = rng.normal(-3, 0.2, size=(60, 6))
    x1 = rng.normal(3, 0.2, size=(60, 6))
    X = np.vstack([x0, x1])
    domains = np.repeat([0, 1], 60)
    quality = np.tile(np.linspace(1.0, 0.0, 60), 2)
    a = quadmix_published_core(quality, X, 30, domains=domains, seed=9, omegas=0.1)
    b = quadmix_published_core(quality, X, 30, domains=domains, seed=9, omegas=0.1)
    assert a == b
    assert len(a) == len(set(a)) == 30
    assert 0 < np.sum(domains[a] == 0) < 30


def test_dmf_published_update_uses_equation_8_before_projection():
    rng = np.random.default_rng(4)
    good = rng.random(240)
    S = np.stack([good, rng.random(240), rng.random(240)])
    truth = set(np.argsort(-good)[:80])

    def reward(sel):
        return len(set(sel) & truth) / 80

    initial_reward = reward(np.argsort(-S.mean(axis=0))[:80])
    selected, trace = dmf_published_update(S, 80, reward, rounds=6, eta=0.2, return_trace=True)
    first = trace[0]
    expected = first["theta"] + 0.2 * (first["actor_rewards"] - first["actor_rewards"].mean())
    np.testing.assert_allclose(first["raw_next"], expected, atol=1e-12)
    assert reward(selected) > initial_reward
    assert reward(selected) > 0.9
