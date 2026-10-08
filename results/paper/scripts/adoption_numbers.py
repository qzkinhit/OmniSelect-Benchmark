"""Adoption statistics of the reported main runs from decision.json and leaderboard.json."""
import json, glob, os
import numpy as np
V = 'view/main'
PATHS = json.load(open('paper_seeds.json'))['rule']['seeds']
LOW = ('neg_mase', 'neg_gmean_ppl')
MAP = {"C-100": "vision/cifar100/clip_vitb32", "C-100N": "vision/cifar100n/clip_vitb32", "C-10": "native/cifar10/resnet18_scratch",
       "ETTh1": "timeseries/ETTh1/dlinear", "ETTm1": "timeseries/ETTm1/dlinear", "ETTh2": "timeseries/ETTh2/dlinear",
       "CSTR": "timeseries/daisy_cstr/dlinear", "Steam": "timeseries/daisy_steamgen/dlinear", "TEP21": "process/tep21/mlp",
       "Elec": "tabular/electricity/tabpfn", "Text": "text/five_domain/smollm2_135m"}
M = json.load(open('paper_main_stats.json'))['tasks']
res = {'adopt': 0, 'better': 0, 'worse': 0, 'equal': 0, 'runs': 0}
ret_gain = {}; el_gain = {}; ret_sm = {}; el_sm = {}
for t, rel in MAP.items():
    rnd = M[t]['rows']['random']['per_seed']
    rg = []; eg = []; rv = []; ev = []; r0v = []
    for s in PATHS[t]:
        d = f'{V}/{rel}/seed_{s}'
        dec = json.load(open(f'{d}/decision.json')); lb = json.load(open(f'{d}/leaderboard.json'))
        test = {r['name']: r.get('u_test') for r in lb['rows']}
        res['runs'] += 1
        ref_t = test[dec['reference']]; el_t = dec.get('elected_test_utility')
        if el_t is None: el_t = test[dec['elected']]
        r0 = test['random']
        sign = 1  # stored utilities are already oriented larger-is-better
        rg.append((ref_t - r0) / abs(r0)); eg.append((el_t - r0) / abs(r0)); rv.append(ref_t); ev.append(el_t); r0v.append(r0)
        if dec['adopted']:
            res['adopt'] += 1
            res['better' if el_t > ref_t else 'worse' if el_t < ref_t else 'equal'] += 1
    ret_gain[t] = float(np.mean([x for x in rg])); el_gain[t] = float(np.mean(eg))
    ret_sm[t] = float((np.mean(rv) - np.mean(r0v)) / abs(np.mean(r0v))); el_sm[t] = float((np.mean(ev) - np.mean(r0v)) / abs(np.mean(r0v)))
res['retain_ref_mean_gain_pct'] = 100 * float(np.mean(list(ret_gain.values())))
res['elected_mean_gain_pct_per_run_avg'] = 100 * float(np.mean(list(el_gain.values())))
# same definition as the main table: gain of the seed means, averaged over tasks
res['retain_ref_seedmean_gain_pct'] = 100 * float(np.mean(list(ret_sm.values())))
res['elected_seedmean_gain_pct'] = 100 * float(np.mean(list(el_sm.values())))
print(json.dumps(res, indent=0))
json.dump({'summary': res, 'retain_ref_gain_by_task': ret_gain, 'elected_gain_by_task': el_gain}, open('adoption_numbers.json', 'w'), indent=1)
