"""Rebuild robustness, validation-size and signal-removal figures from reported cells."""
import collections
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path.cwd()
SELECTION = json.loads((Path(__file__).resolve().parents[1] / "seeds_reported.json").read_text())
REL = {"C-100": "vision/cifar100/clip_vitb32", "C-100N": "vision/cifar100n/clip_vitb32",
       "TEP21": "process/tep21/mlp", "Elec": "tabular/electricity/tabpfn", "ETTm1": "timeseries/ETTm1/dlinear",
       "ETTh1": "timeseries/ETTh1/dlinear", "ETTh2": "timeseries/ETTh2/dlinear", "CSTR": "timeseries/daisy_cstr/dlinear",
       "Steam": "timeseries/daisy_steamgen/dlinear", "C-10-CLIP": "vision/cifar10_clip/clip_vitb32", "Text": "text/five_domain/smollm2_135m"}
BY_DS = {v.split('/')[1]: k for k, v in REL.items()}
DISPLAYED = {"random", "influence_only", "coreset", "mmdataselect", "auth_only", "herding", "el2n", "grand", "ccs", "density", "quadmix_pub", "dmf_pub"}
TEXT_ALIAS = {"coverage_text": "coreset", "fixed_fusion": "mmdataselect", "herding_text": "herding", "density_text": "density"}


def read(path, name):
    return json.loads((path / name).read_text())


def save(name, data):
    Path(name).write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def agg(xs):
    return {"mean": mean(xs), "n": len(xs), "sample_sd": float(np.std(xs, ddof=1)), "values": xs}


def cell(path, batch, cond, track, ds, learner, seed):
    dec, lb, met = (read(path, name) for name in ("decision.json", "leaderboard.json", "metrics.json"))
    by_name = {r['name']: r for r in lb['rows']}
    rows = met['rows']
    return {"batch": batch, "cond": cond, "track": track, "ds": ds, "learner": learner, "seed": seed,
            "utility": met['utility'], "elected": dec['elected'], "test": by_name[dec['elected']]['u_test'],
            "rank": dec.get('u_rank_elected'), "random": rows['random']['u_test'], "random_rank": by_name['random']['u_rank'],
            "full": rows.get('full', {}).get('u_test'),
            "best_disp": max(v['u_test'] for k, v in rows.items() if TEXT_ALIAS.get(k, k) in DISPLAYED and v.get('u_test') is not None),
            "n_cand": len(lb['rows']), "adopted": dec['adopted']}


def robustness():
    rows = []
    for batch in ('robustness',):
        for path in sorted((ROOT / batch).glob('*/*/*/*/seed_*/decision.json')):
            cond, track, ds, learner, seed_dir = path.parent.relative_to(ROOT / batch).parts
            seed = int(seed_dir.removeprefix('seed_'))
            assert seed in SELECTION['seeds'][BY_DS[ds]], (batch, ds, seed)
            rows.append(cell(path.parent, batch, cond, track, ds, learner, seed))
    for task in ('C-100', 'TEP21', 'ETTh1'):
        track, ds, learner = REL[task].split('/')
        original_batch = 'main' if task == 'ETTh1' else 'main'
        for seed in SELECTION['seeds'][task]:
            rows.append(cell(ROOT / 'main' / REL[task] / f'seed_{seed}', original_batch, 'main', track, ds, learner, seed))
    rows.sort(key=lambda r: (r['cond'], r['track'], r['ds'], r['learner'], r['seed']))
    save('robust_cells_view.json', {'note': 'One row per reported robustness cell; main cells are the 40% injection and clean-validation controls.', 'rows': rows})
    cells = collections.defaultdict(list)
    for r in rows:
        cond = 'ratio_0.4' if r['cond'] == 'main' else r['cond']
        ds = 'text' if r['ds'] == 'five_domain' else r['ds']
        cells[(cond, f"{ds}/{r['learner']}")].append(r)
    summ, source = collections.defaultdict(dict), {}
    for (cond, key), rs in sorted(cells.items()):
        rs.sort(key=lambda r: r['seed'])
        assert len(rs) == 3 and len({r['batch'] for r in rs}) == 1, (cond, key)
        e, rd = mean([r['test'] for r in rs]), mean([r['random'] for r in rs])
        summ[cond][key] = {'n': 3, 'seeds': [r['seed'] for r in rs], 'utility': rs[0]['utility'], 'elected': e, 'random': rd,
                           'full': mean([r['full'] for r in rs]), 'best_displayed_per_run_mean': mean([r['best_disp'] for r in rs]),
                           'gain_pct': 100 * (e - rd) / abs(rd), 'runs_above_random': sum(r['test'] > r['random'] for r in rs),
                           'elected_names': [r['elected'] for r in rs]}
        source[f'{cond}/{key}'] = rs[0]['batch']
    summ['_source'] = source
    save('robustness_summary.json', summ)
    pts = []
    for track, ds, learner, metric in (('vision', 'cifar100', 'clip_vitb32', 'accuracy'), ('process', 'tep21', 'mlp', 'macro_f1')):
        control = {}
        for rate in ('0.0', '0.1', '0.2', '0.3', '0.4', '0.5'):
            cond = 'ratio_0.4' if rate == '0.0' else f'val_sym_{rate}'
            for r in cells[(cond, f'{ds}/{learner}')]:
                if rate == '0.0':
                    control[r['seed']] = r['test']
                where = f"{r['batch']}/{track}/{ds}/{learner}" if r['cond'] == 'main' else f"{r['batch']}/{cond}/{track}/{ds}/{learner}"
                pts.append({'track': track, 'dataset': ds, 'learner': learner, 'metric': metric, 'noise_rate': float(rate), 'seed': r['seed'],
                            'elected': r['elected'], 'test': r['test'], 'rank': r['rank'], 'random_test': r['random'],
                            'random_rank': r['random_rank'], 'candidate_count': r['n_cand'], 'delta_control': r['test'] - control[r['seed']],
                            'source': f"<repo>/results_and_logs/{where}/seed_{r['seed']}"})
    summary = []
    for track in ('vision', 'process'):
        for rate in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            rs = [r for r in pts if r['track'] == track and r['noise_rate'] == rate]
            summary.append({'track': track, 'noise_rate': rate, 'seeds': [r['seed'] for r in rs],
                            **{k: agg([r[k] for r in rs]) for k in ('test', 'rank', 'random_test', 'delta_control')}})
    save('validation_noise_plot_data.json', {'statistics': 'mean and sample standard deviation (ddof=1); delta = noisy-validation cell minus the clean-validation cell of the same seed',
                                           'validation_noise': {'points': pts, 'summary': summary}})
    print('robustness cells', len(rows), 'validation-noise points', len(pts))


def elected(path):
    d, lb = read(path, 'decision.json'), read(path, 'leaderboard.json')
    return next(r['u_test'] for r in lb['rows'] if r['name'] == d['elected'])


def ablation():
    out = {}
    for task, seeds in SELECTION['signal_ablation_seeds'].items():
        task_rows = {}
        for cond in ('drop_A', 'drop_I', 'drop_C'):
            diffs, relative = [], []
            for seed in seeds:
                full = elected(ROOT / 'main' / REL[task] / f'seed_{seed}')
                removed = elected(ROOT / 'signal_drop' / cond / REL[task] / f'seed_{seed}')
                diffs.append(removed - full); relative.append((removed - full) / abs(full) * 100)
            n = len(diffs); m = sum(diffs) / n
            sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1)); half = float(stats.t.ppf(.975, n - 1)) * sd / math.sqrt(n)
            task_rows[cond] = {'n': n, 'seeds': seeds, 'mean_rel': sum(relative) / n, 'mean_abs': m, 'lo': m - half, 'hi': m + half,
                               'verdict': 'worse' if m + half < 0 else 'better' if m - half > 0 else 'tie',
                               'neg': sum(x < 0 for x in diffs), 'pos': sum(x > 0 for x in diffs), 'rel_per_seed': relative}
        out[task] = task_rows
    save('abl_v23_summary.json', out)
    print('signal-removal cells', sum(sum(r['n'] for r in task.values()) for task in out.values()))
    return out


def validation_size(signal_drop):
    pts = []
    for n, seeds in SELECTION['validation_size_seeds'].items():
        for seed in seeds:
            batch = 'main' if n == '800' else f'validation_size/n_{n}'
            path = ROOT / batch / REL['C-100'] / f'seed_{seed}'
            m, d = read(path, 'metrics.json')['rows'], read(path, 'decision.json')
            pts.append({'val_n': int(n), 'seed': seed, 'test': m['mmds_adapt']['u_test'], 'rank': d.get('u_rank_elected'),
                        'elected': d['elected'], 'random_test': m['random']['u_test'], 'infl_test': m['influence_only']['u_test'],
                        'auth_test': m['auth_only']['u_test'], 'condition': 'full' if n == '800' else f'val_n={n}',
                        'track': 'vision', 'dataset': 'cifar100', 'source': 'main' if n == '800' else batch})
    summary = []
    for n in map(int, SELECTION['validation_size_seeds']):
        rs = [p for p in pts if p['val_n'] == n]
        summary.append({'val_n': n, 'seeds': [p['seed'] for p in rs],
                        **{k: agg([p[k] for p in rs]) for k in ('test', 'rank', 'random_test', 'infl_test', 'auth_test')}})
    out = {'points': pts, 'summary': summary}
    if 'validation_size_selection' in SELECTION:
        out['note'] = SELECTION['validation_size_selection']
    save('validation_size.json', out)
    print('validation-size points', len(pts))


robustness()
validation_size(ablation())
