"""Every number the text cites from the all-seed main batches. Output: paper_numbers.json."""
import json, math
import numpy as np
from scipy import stats
M = json.load(open('paper_main_stats.json'))['tasks']; N = json.load(open('paper_nocoop_stats.json'))['tasks']
A = json.load(open('paper_main_agg.json')); R = json.load(open('table_method_ranks.json'))
COLS = ["C-100","C-100N","C-10","ETTh1","ETTm1","ETTh2","CSTR","Steam","TEP21","Elec","Text"]
LOW = ('neg_mase','neg_gmean_ppl')
out = {}
def sgn(t): return -1 if M[t]['utility'] in LOW else 1
def tint(d):
    m=float(np.mean(d)); 
    if np.std(d)==0: return m,m,m
    h=float(stats.t.ppf(.975,len(d)-1)*np.std(d,ddof=1)/math.sqrt(len(d))); return m,m-h,m+h
runs=0; above=0
for t in COLS:
    ps=M[t]['rows']['mmds_adapt']['per_seed']; rs=M[t]['rows']['random']['per_seed']
    for s in ps:
        runs+=1; above += sgn(t)*(ps[s]-rs[s])>0
out['runs']=runs; out['runs_above_random']=int(above)
out['vs_strongest']=A['vs_strongest']; out['vs_strongest_counts']=A['vs_strongest_counts']
for kind in ('external','signal'):
    rows={t:{'baseline':M[t]['strongest_'+kind],**M[t]['verdicts'][M[t]['strongest_'+kind]]} for t in COLS}
    out['vs_'+kind]=rows
    out['vs_'+kind+'_counts']={v:sum(r['verdict']==v for r in rows.values()) for v in ('W','T','L')}
out['cells']=A['verdicts_all_cells']
out['gain']={b:v['mean_gain']*100 for b,v in A['gains'].items()}
out['oracle_gain']=A['displayed_oracle_gain']*100
out['friedman']=A['friedman_original_methods']; out['wilcoxon_holm_max']=max(v['holm'] for v in A['wilcoxon'].values())
out['mean_rank14']={r['method'] if isinstance(r,dict) and 'method' in r else str(r):r for r in []}
out['circle']=R['method_to_circle_rank']; out['rows_sorted_by_rank']=R['rows_sorted_by_rank']
# score-only best count
SO=['random','coreset','auth_only','herding','density','quadmix_pub']
best={t:max(SO,key=lambda k:sgn(t)*M[t]['rows'][k]['mean']) for t in COLS}
out['score_only_best']=best
out['displayed_below_random']=A['displayed_below_random_tasks']
out['means']={t:{k:M[t]['rows'][k]['mean'] for k in M[t]['rows']} for t in COLS}
out['nc_means']={t:N[t]['rows']['mmds_adapt']['mean'] for t in COLS}
# vs full
out['vs_full']={t:M[t]['vs_full'] for t in COLS}
# cooperation vs NC
coop={}
for t in COLS:
    a=M[t]['rows']['mmds_adapt']['per_seed']; b=N[t]['rows']['mmds_adapt']['per_seed']
    d=[sgn(t)*(a[s]-b[s]) for s in a]; m,lo,hi=tint(d)
    coop[t]={'mean':m,'lo':lo,'hi':hi,'n':len(d),'nonzero':sum(x!=0 for x in d),'pos':sum(x>0 for x in d),'neg':sum(x<0 for x in d)}
out['coop_vs_nc']=coop
# elections, adoptions
el={}; tot_ad=0; ad_better=0; ad_worse=0; ad_equal=0; ret_gain=[]; n_runs=0; coop_el=0; clean_el=0; tasks_multi=0
for t in COLS:
    ps=M[t]['per_seed']; lbt={}
    names=[ps[s]['elected'] for s in ps]
    el[t]={'distinct':len(set(names)),'names':names}
    tasks_multi += len(set(names))>1
    coop_el += sum(n.startswith('coop') for n in names); clean_el += names.count('clean_top')
    for s,p in ps.items():
        n_runs+=1
        if p['adopted']:
            tot_ad+=1
            ref=p['reference']; ref_t=M[t]['rows'].get(ref,{}).get('per_seed',{}).get(s)
out['elections']=el; out['n_runs']=n_runs; out['adoptions']=tot_ad; out['coop_elected']=coop_el; out['clean_top_elected']=clean_el; out['tasks_multi_elected']=tasks_multi
out['fde']=A['fde']
json.dump(out,open('paper_numbers.json','w'),indent=1,default=float)
print(json.dumps({k:out[k] for k in ['runs','runs_above_random','vs_strongest_counts','vs_external_counts','vs_signal_counts','oracle_gain','adoptions','coop_elected','clean_top_elected','tasks_multi_elected','score_only_best']},indent=0,default=float))
print({k:round(v,2) for k,v in out['gain'].items()})
