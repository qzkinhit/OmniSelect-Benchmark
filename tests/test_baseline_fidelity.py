"""Baseline fidelity tests: each reproduced baseline must exhibit the defining
mechanism of its original paper on controlled data. These are the checks that
certify the reproduction is RIGHT, not just runnable.

- herding (Welling 2009): selected set matches the data mean better than random.
- k-center (Sener & Savarese 2018): selected set has smaller covering radius than random.
- SemDeDup (Abbas 2023): on a pool with injected near-duplicates, removes duplicates
  at a much higher rate than random.
- Density (Sachdeva 2024): selected set covers the space (max-min distance smaller
  than random's).
- EL2N vs GraNd (Paul et al. 2021): scores are computed by different rules and give
  genuinely different rankings (regression guard against the old aliasing bug).
- CCS (Zheng ICLR 2023): keeps a stratified difficulty profile, not only the hardest.
- DMF dynamic (Yang 2025): multiplicative-weights reweighting concentrates weight on
  the channel that the validation reward favours.
- QuaDMix (2024): joint objective yields both higher mean quality than random and
  wider feature dispersion than pure quality Top-K.
"""
import numpy as np
import pytest

from mmdataselect.selectors.external_baselines import (
    ccs, density_select, dmf_dynamic, el2n, grand_expected, herding,
    kcenter_greedy, quadmix, semdedup)


@pytest.fixture(scope="module")
def blobs():
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((8, 16)) * 4.0
    X = np.concatenate([c + rng.standard_normal((40, 16)) for c in centers])
    y = np.repeat(np.arange(8), 40)
    return X, y


def test_herding_matches_mean_better_than_random(blobs):
    X, _ = blobs
    k = 40
    sel = herding(X, k)
    rng = np.random.default_rng(1)
    rand = rng.permutation(len(X))[:k]
    err_h = np.linalg.norm(X[sel].mean(0) - X.mean(0))
    err_r = np.linalg.norm(X[rand].mean(0) - X.mean(0))
    assert err_h < err_r, f"herding 未逼近全体均值 ({err_h:.3f} vs {err_r:.3f})"


def test_kcenter_covering_radius_smaller_than_random(blobs):
    X, _ = blobs
    k = 24
    sel = kcenter_greedy(X, k, seed=0)
    rng = np.random.default_rng(1)
    rand = rng.permutation(len(X))[:k]

    def cover_radius(idx):
        D = np.linalg.norm(X[:, None, :] - X[idx][None, :, :], axis=2)
        return D.min(axis=1).max()

    assert cover_radius(sel) < cover_radius(rand), "k-center 未缩小覆盖半径"


def test_semdedup_removes_injected_duplicates(blobs):
    X, _ = blobs
    rng = np.random.default_rng(2)
    n_dup = 80
    seeds = rng.integers(0, len(X), n_dup)
    dups = X[seeds] + 0.01 * rng.standard_normal((n_dup, X.shape[1]))
    pool = np.concatenate([X, dups])
    is_dup = np.zeros(len(pool), bool); is_dup[len(X):] = True
    k = len(pool) // 2
    kept = semdedup(pool, k, seed=0)
    dup_kept = is_dup[kept].mean()
    dup_base = is_dup.mean()
    assert dup_kept < dup_base * 0.75, \
        f"SemDeDup 未优先去掉近重复 (保留集中重复率 {dup_kept:.2f} vs 池 {dup_base:.2f})"


def test_density_inverse_propensity_thins_dense_regions():
    # 原文机制:密集区被按密度反比稀释。构造一团超密集点 + 均匀散布点,
    # 多种子下密集团的人均选中频率应明显低于散布点。
    rng = np.random.default_rng(0)
    dense = rng.standard_normal((150, 8)) * 0.05          # 超密集团
    spread = rng.standard_normal((150, 8)) * 3.0           # 散布点
    X = np.concatenate([dense, spread])
    k = 100
    counts = np.zeros(len(X))
    for s in range(10):
        for i in density_select(X, k, seed=s):
            counts[i] += 1
    dense_rate = counts[:150].mean()
    spread_rate = counts[150:].mean()
    assert dense_rate < spread_rate * 0.7, \
        f"density 采样未稀释密集区 (密集团均频 {dense_rate:.2f} vs 散布 {spread_rate:.2f})"


def test_el2n_and_grand_rank_differently(blobs):
    X, y = blobs
    n, C = len(y), int(y.max()) + 1
    rng = np.random.default_rng(4)
    logits = rng.standard_normal((n, C))
    k = n // 2
    s_el2n = el2n(logits, y, k, is_logits=True)
    s_grand = grand_expected(X, y, k, seed=0)
    overlap = len(set(s_el2n) & set(s_grand)) / k
    assert overlap < 0.95, f"EL2N 与 GraNd 选择几乎相同 (overlap={overlap:.2f}),疑似退化为别名"


def test_ccs_stratified_not_only_hardest(blobs):
    X, y = blobs
    n, C = len(y), int(y.max()) + 1
    rng = np.random.default_rng(5)
    # 构造双峰难度: 一半样本近正确 one-hot, 一半近均匀(难)
    P = np.zeros((n, C)); P[np.arange(n), y] = 1.0
    hard = rng.permutation(n)[: n // 2]
    P[hard] = 1.0 / C
    k = n // 4
    sel_ccs = set(ccs(P, y, k, is_logits=False))
    sel_hard = set(el2n(P, y, k, is_logits=False))
    frac_hard_ccs = len(sel_ccs & set(hard)) / k
    frac_hard_el2n = len(sel_hard & set(hard)) / k
    assert frac_hard_el2n > 0.9, "EL2N 应几乎全取难样本"
    assert frac_hard_ccs < 0.8, f"CCS 应跨难度分层取样 (难样本占比 {frac_hard_ccs:.2f})"


def test_dmf_concentrates_weight_on_rewarded_channel():
    rng = np.random.default_rng(6)
    n = 300
    good = rng.random(n)              # 通道0: 与真效用一致
    noise1, noise2 = rng.random(n), rng.random(n)
    S = np.stack([good, noise1, noise2])
    k = 100
    truth = set(np.argsort(-good)[:k])

    def reward(sel):
        return len(set(sel) & truth) / k

    sel = dmf_dynamic(S, k, val_reward=reward, rounds=5, seed=0)
    assert reward(sel) > 0.9, f"DMF 未收敛到有效通道 (reward={reward(sel):.2f})"


def test_quadmix_balances_quality_and_dispersion(blobs):
    X, _ = blobs
    rng = np.random.default_rng(7)
    q = rng.random(len(X))
    k = 80
    sel_qm = quadmix(q, X, k, seed=0)
    sel_topq = list(np.argsort(-q)[:k])
    rand = list(rng.permutation(len(X))[:k])
    assert q[sel_qm].mean() > q[rand].mean(), "QuaDMix 平均质量未高于随机"
    disp = lambda idx: np.linalg.norm(X[idx] - X[idx].mean(0), axis=1).mean()
    assert disp(sel_qm) >= disp(sel_topq) * 0.95, "QuaDMix 多样性不应明显差于纯质量 Top-K"


@pytest.mark.parametrize("quality_kind", ["tied", "constant"])
def test_quadmix_is_deterministic_without_replacement(quality_kind):
    rng = np.random.default_rng(11)
    n, k = 240, 173
    X = np.zeros((n, 8)) if quality_kind == "constant" else rng.standard_normal((n, 8))
    q = np.zeros(n) if quality_kind == "constant" else np.repeat(np.arange(12), n // 12)
    first = quadmix(q, X, k, bins=20, seed=19)
    second = quadmix(q, X, k, bins=20, seed=19)
    assert first == second
    assert len(first) == k
    assert len(set(first)) == k
    assert min(first) >= 0 and max(first) < n
