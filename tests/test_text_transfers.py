import numpy as np

from mmdataselect.selectors.text_transfers import (
    coverage_token_order,
    herding_token_order,
)


def _fixture():
    features = np.eye(6, dtype=np.float32)
    domains = ["a", "a", "a", "b", "b", "b"]
    tokens = [4, 4, 4, 4, 4, 4]
    budgets = {"a": 8, "b": 8}
    return features, domains, tokens, budgets


def test_text_transfer_orders_are_complete_unique_and_deterministic():
    features, domains, tokens, budgets = _fixture()
    orders = [
        coverage_token_order(features, domains, tokens, budgets),
        herding_token_order(features, domains, tokens, budgets),
    ]
    for order in orders:
        assert sorted(order) == list(range(6))
        assert len(order) == len(set(order))
    assert orders[0] == coverage_token_order(features, domains, tokens, budgets)
    assert orders[1] == herding_token_order(features, domains, tokens, budgets)
