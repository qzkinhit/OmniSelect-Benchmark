"""Summary of the robustness batch: per condition and task, elected test utility against Random,
full data and the best displayed strategy, over the recorded seeds."""
import json, glob, os, collections
import numpy as np
import sys
R = sys.argv[1] if len(sys.argv) > 1 else 'robustness'
DISP = ["random","influence_only","coreset","mmdataselect","auth_only","herding","el2n","grand","ccs","density","quadmix_pub","dmf_pub"]
out = collections.defaultdict(dict)
for dec_path in sorted(glob.glob(f'{R}/*/*/*/*/seed_*/decision.json')):
    cell = os.path.dirname(dec_path); parts = cell[len(R)+1:].split('/')
    cond, track, ds, learner, seed = parts[0], parts[1], parts[2], parts[3], parts[4]
    dec = json.load(open(dec_path)); met = json.load(open(f'{cell}/metrics.json'))
    lb = {r['name']: r.get('u_test') for r in json.load(open(f'{cell}/leaderboard.json'))['rows'] if r.get('u_test') is not None}
    rows = {k: v.get('u_test') for k, v in met['rows'].items() if v.get('u_test') is not None}
    el = dec.get('elected_test_utility'); el = lb.get(dec['elected']) if el is None else el
    key = f'{ds}/{learner}'
    d = out[cond].setdefault(key, {'seeds': [], 'elected': [], 'random': [], 'full': [], 'best_disp': [], 'utility': met['utility'], 'names': []})
    d['seeds'].append(seed); d['elected'].append(el); d['names'].append(dec['elected'])
    d['random'].append(rows.get('random', lb.get('random'))); d['full'].append(rows.get('full', lb.get('full')))
    disp = [rows.get(k, lb.get(k)) for k in DISP if rows.get(k, lb.get(k)) is not None]
    d['best_disp'].append(max(disp) if disp else None)
summ = {}
for cond, tasks in out.items():
    summ[cond] = {}
    for key, d in tasks.items():
        def m(x):
            x = [v for v in x if v is not None]; return float(np.mean(x)) if x else None
        r, e = m(d['random']), m(d['elected'])
        summ[cond][key] = {'n': len(d['seeds']), 'utility': d['utility'], 'elected': e, 'random': r, 'full': m(d['full']),
                           'best_displayed_per_run_mean': m(d['best_disp']),
                           'gain_pct': 100 * (e - r) / abs(r) if (e is not None and r) else None,
                           'runs_above_random': sum(1 for a, b in zip(d['elected'], d['random']) if a is not None and b is not None and a > b),
                           'elected_names': d['names']}
json.dump(summ, open(sys.argv[2] if len(sys.argv) > 2 else 'robustness_summary.json', 'w'), indent=1)
for cond in sorted(summ):
    print('==', cond)
    for key, s in sorted(summ[cond].items()):
        print('  %-40s n=%d gain=%s above=%d/%d el=%.4f rnd=%.4f full=%s' % (key, s['n'], None if s['gain_pct'] is None else round(s['gain_pct'], 2), s['runs_above_random'], s['n'], s['elected'] or 0, s['random'] or 0, None if s['full'] is None else round(s['full'], 4)))
