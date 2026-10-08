"""Signal dependency-closure ablations with persistent prediction reuse.

Port of the v2.3 signal-removal driver used for the reported experiments. Provide a
control run with config.json and splits.json; complete candidate predictions are
reused when present. --verify-full compares a full replay with a complete control.

python -m run_omniselect.signal_drop --track vision --seed 0 --condition drop_A \
    --source-cell results_and_logs/main/vision/cifar100/clip_vitb32/seed_0
"""
from __future__ import annotations
import argparse, copy, dataclasses, hashlib, importlib.metadata, json, os
from pathlib import Path
import sys, types
import numpy as np

WORKTREE = Path(__file__).resolve().parents[1]
SOURCE_ROOT = WORKTREE / 'results_and_logs'
OUT_ROOT = WORKTREE / 'results_and_logs'
CACHE_ROOT = WORKTREE / '.cache' / 'signal_drop'
DATA_ROOT = WORKTREE / 'data'
SOURCE_CELL = None
sys.path.insert(0, str(WORKTREE))
CONDITIONS = ('drop_A','drop_I','drop_C')
AXIS = {'drop_A':0,'drop_I':1,'drop_C':2}
REMOVED = {
    'full': set(),
    'drop_A': set('auth_only auth2_only auth3_only auth_bottom clean_top coop_herding mmdataselect dmf_pub quadmix_pub'.split()),
    'drop_I': set('influence_only el2n ccs grand glister gradmatch dsdm infomax mmdataselect dmf_pub'.split()),
    'drop_C': set('coreset kcenter herding density semdedup d4 quadmix_pub infomax mmdataselect dmf_pub auth3_only clean_top coop_herding'.split()),
}
BOUNDARIES = {
    'drop_I':'Pool-only out-of-fold label-quality estimates in A and learned cleanliness remain. V_con influence probabilities and last-layer gradients are unavailable to candidates.',
    'drop_C':'Nearest-neighbour label agreement defining A remains. Coverage, novelty, diversity penalties, and representative-selection consumers are removed.',
    'drop_A':'All cooperative candidates are removed, including ungated candidates, because their duplicate representatives and refill order use authenticity.',
}

def canonical(x): return json.loads(json.dumps(x,sort_keys=True,default=str))
def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''): h.update(block)
    return h.hexdigest()
def digest(x): return hashlib.sha256(json.dumps(canonical(x),sort_keys=True).encode()).hexdigest()
def array_sha(a):
    if a is None: return 'none'
    a=np.asarray(a)
    return hashlib.sha256(str((a.dtype.str,a.shape)).encode()+np.ascontiguousarray(a).tobytes()).hexdigest()
def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,default=str)+'\n');os.replace(tmp,path)
# The control cells are the reported main runs of the published record archive.
TASKS={'vision':('main','vision/cifar100/clip_vitb32'),'process':('main','process/tep21/mlp'),
       'cifar100n':('main','vision/cifar100n/clip_vitb32'),'electricity':('main','tabular/electricity/tabpfn'),
       'ETTm1':('main','timeseries/ETTm1/dlinear'),'ETTh1':('main','timeseries/ETTh1/dlinear')}
TRACK_OF={'vision':'vision','process':'process','cifar100n':'vision','electricity':'tabular','ETTm1':'timeseries','ETTh1':'timeseries'}
def source_cell(track,seed):
    b,rel=TASKS[track]
    return SOURCE_CELL if SOURCE_CELL is not None else SOURCE_ROOT/b/rel/f'seed_{seed}'
def cell_path(track,seed,batch):
    return OUT_ROOT/batch/TASKS[track][1]/f'seed_{seed}'
def flatten(d,prefix=''):
    result={}
    for k,v in d.items():
        name=f'{prefix}.{k}' if prefix else k
        if isinstance(v,dict): result.update(flatten(v,name))
        else: result[name]=v
    return result

def configs(track,seed):
    from tracks.common.experiment import load_track
    from omniselect.config.config import OmniSelectConfig
    src=source_cell(track,seed); raw=json.loads((src/'config.json').read_text())
    assert raw['track']['dataset']==TASKS[track][1].split('/')[1], 'source dataset differs from requested task'
    v22='influence_within_class' not in raw['track']
    assert raw['track']['seed']==seed, 'source seed differs from requested seed'
    track=TRACK_OF[track]
    assert raw['track']['track']==track, 'source track differs from requested task'
    for pkg in ('numpy','scipy','scikit-learn'):
        assert importlib.metadata.version(pkg)==raw['environment']['packages'][pkg],pkg
    obj=load_track(track)
    t=obj.config_cls(**raw['track'])
    # JSON arrays are restored as tuple fields, including cooperative tuple concatenation.
    defaults=obj.config(raw['track']['dataset'],learner=raw['track']['learner'],seed=seed)
    for f in dataclasses.fields(t):
        if isinstance(getattr(defaults,f.name),tuple): setattr(t,f.name,tuple(getattr(t,f.name)))
    expected_omni=copy.deepcopy(raw['omniselect'])
    # Archived controls carry two fields of the disabled, unpublished Bayes stub.
    assert not expected_omni['synthesis'].pop('bayes',False)
    expected_omni['synthesis'].pop('bayes_probes',None)
    o=OmniSelectConfig.preset(expected_omni['protocol'],**flatten({k:v for k,v in expected_omni.items() if k!='protocol'}))
    td=canonical(t.to_dict())
    if v22:
        assert td.pop('influence_within_class') is False
    assert td==raw['track']
    assert canonical(o.to_dict())==expected_omni
    assert o.cache.by_subset_hash and o.fidelity.scoring_equals_reported
    assert not o.signals.alignment and not o.synthesis.policy_search
    return obj,t,o,raw

def split_ids(data):
    return {'pool':data.pool_ids,**{s:[data.val_ids[int(i)] for i in data.splits.get(s)] for s in ('con','rank','conf')},'test':data.test_ids}
def data_fingerprint(data):
    return {'arrays':{k:array_sha(v) for k,v in sorted(data.arrays.items()) if isinstance(v,np.ndarray)},
            'labels':array_sha(data.labels),'labels_clean':array_sha(data.labels_clean),
            'ids':canonical(split_ids(data)),'tags':list(map(str,data.tags)),
            'budget':data.budget,'utility':data.utility}

def load_verified(obj,t,o,raw,src):
    t.data_root=str(DATA_ROOT)
    if t.track=='process':
        t.manifest=str(DATA_ROOT/'tep/SHA256SUMS.txt')
    after=obj.load(t,o)
    b=data_fingerprint(after)
    split=json.loads((src/'splits.json').read_text())
    assert canonical(split_ids(after))==split['ids']
    assert list(map(str,after.tags))==split['pool_tags']
    if t.track=='vision':
        assert after.provenance['encoding']['features_sha256']==raw['encoding']['features_sha256']
    context={'data':b,'original_track':raw['track'],'original_omniselect':raw['omniselect'],
             'commit':raw.get('git',{}).get('sha'),'packages':{p:raw['environment']['packages'][p] for p in ('numpy','scipy','scikit-learn')}}
    return after,context

def project_config(t,o,condition):
    if condition=='full': return
    from omniselect.core.selection.fusion_grid import default_grid
    axis=AXIS[condition]
    weights=t.grid_weights if t.grid_weights is not None else default_grid(3)
    projected=[]
    for w in weights:
        v=np.array(w,dtype=float);v[axis]=0
        if v.sum()==0: continue
        v/=v.sum(); item=tuple(float(x) for x in v)
        if item not in projected: projected.append(item)
    assert projected and all(w[axis]==0 for w in projected)
    t.grid_weights=tuple(projected)
    t.methods=tuple(n for n in t.methods if n not in REMOVED[condition])
    if condition=='drop_A': t.grid_q=(0.0,);t.auth_q=0.0
    if condition in ('drop_A','drop_C'): o.cooperative.enabled=False
    if condition=='drop_C':
        t.grid_lam=(0.0,);t.lam=0.0
        o.synthesis.consensus_diversity_lam=0.0;o.synthesis.infomax_solver=False

def sanitize(sig,condition,perturb=False):
    if condition=='full': return sig
    axis=AXIS[condition]; name=('auth','influence','redundancy')[axis]
    if perturb:
        # Deliberately adversarial input at the adapter boundary; consumers must never see it.
        setattr(sig,name,np.linspace(-1e90,1e90,len(sig.auth)))
        if condition=='drop_I': sig.proba=np.full((len(sig.auth),2),np.nan)
        sig.extras['imp_dyn']=np.full(len(sig.auth),np.nan)
    setattr(sig,name,np.zeros_like(sig.auth,dtype=float))
    sig.extras.pop('imp_dyn',None)  # the removed FixedFusion is its sole active consumer
    if condition=='drop_A': sig.extras.pop('auth_features',None)
    if condition=='drop_I':
        sig.proba=None
        for key in ('last_layer','gradient_factors'): sig.extras.pop(key,None)
        sig.reference_ids=[];sig.reference_source='removed_influence_module'
    if condition=='drop_C': sig.extras.pop('coverage_features',None)
    sig.saved={k:v for k,v in sig.saved.items() if k not in ('authenticity','influence','redundancy','cleanliness','imp_dyn')}
    assert np.all(getattr(sig,name)==0)
    return sig

def cloned_module(name,rel,replacements=()):
    path=WORKTREE/rel; source=path.read_text()
    for old,new in replacements:
        assert source.count(old)==1,(rel,old,source.count(old))
        source=source.replace(old,new)
    module=types.ModuleType(name);module.__file__=str(path);sys.modules[name]=module
    exec(compile(source,str(path),'exec'),module.__dict__)
    return module

def adapted_experiment(condition):
    synth_changes=[];controller_changes=[]
    if condition!='full':
        active=tuple(c for c in range(3) if c!=AXIS[condition])
        synth_changes=[('for c in range(len(cur_w)):',f'for c in {active!r}:')]
    if condition=='drop_A': controller_changes=[('votes + 1e-9 * S[0]','votes')]
    synth=cloned_module('_signal_drop_synthesis','omniselect/core/adjudication/synthesis.py',synth_changes)
    controller=cloned_module('_signal_drop_controller','omniselect/core/adjudication/controller.py',controller_changes)
    controller.coordinate_ascent=synth.coordinate_ascent
    experiment=cloned_module('_signal_drop_experiment','tracks/common/experiment.py')
    experiment.adjudicate=controller.adjudicate
    original_membership=experiment.membership
    def membership(track,mode):
        rows=original_membership(track,mode)
        for r in rows:
            if r.name in REMOVED[condition]:
                r.included=False;r.reason=f'{condition}: removed by declared signal dependency closure'
        return rows
    experiment.membership=membership
    experiment.reference_members=lambda track,mode:[r.name for r in membership(track,mode) if r.included and r.role=='reference']
    experiment.challenger_members=lambda track,mode:[r.name for r in membership(track,mode) if r.included and r.role=='challenger']
    original_adjudicate=experiment.adjudicate
    def audited_adjudicate(**kwargs):
        names=[n for n,_ in kwargs['references']+kwargs['challengers']]
        assert 'random' in names and 'full' not in names
        assert not REMOVED[condition].intersection(names)
        if condition in ('drop_A','drop_C'): assert not kwargs['cooperative']
        if condition!='full':
            assert np.all(kwargs['scores'][AXIS[condition]]==0)
            assert all(w[AXIS[condition]]==0 for w in kwargs['grid'].weights)
        if condition=='drop_A': assert tuple(kwargs['grid'].q_grid)==(0.0,)
        if condition=='drop_C': assert tuple(kwargs['grid'].lam_grid)==(0.0,)
        result=original_adjudicate(**kwargs)
        assert not result.construction_errors,result.construction_errors
        return result
    experiment.adjudicate=audited_adjudicate
    return experiment

def persistent_cache_class(src,context,data,cache_dir,instances):
    from omniselect.core.adjudication.cache import FitCache,FitEntry
    from omniselect.utils.hashing import sel_sha12
    class ExactCache(FitCache):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            assert self.enabled and self.sort_subsets
            self.eager_splits=('con','rank','conf','test')
            self.exact={};self.short={};self.reused=[];self.persisted=[]
            self.context_sha=digest(context);self.root=cache_dir/self.context_sha
            self.root.mkdir(parents=True,exist_ok=True)
            self.source_index={}
            instances.append(self)
            for path in sorted((src/'candidates').glob('*/scores.json')):
                score=json.loads(path.read_text());candidate=path.parent
                if not (candidate/'selection.npz').is_file() or not all(
                    (candidate/'per_unit'/f'{split}.npz').is_file() for split in ('con','rank','conf','test')
                ):
                    continue
                with np.load(candidate/'selection.npz',allow_pickle=False) as z:
                    ids=np.asarray(z['idx'],dtype=np.int64).copy()
                    assert int(z['n'])==data.n
                assert np.array_equal(ids,np.sort(np.unique(ids)))
                assert len(ids)==score['n_selected'] and sel_sha12(ids)==score['sel_sha12']
                key=(array_sha(ids),self.fidelity_key(score['fidelity']))
                self.source_index.setdefault(key,(candidate,ids,score))

        def validate_units(self,per_unit):
            for split,pu in per_unit.items():
                target=self.learner.split_arrays[split][1]
                assert np.array_equal(pu['target'],target),(split,'target/order mismatch')
                for k in ('prediction','correct','proba'):
                    if k in pu: assert len(pu[k])==len(target)

        def evaluate(self,subset,stage,splits,eager=True):
            ids=np.sort(np.asarray(subset,dtype=np.int64))
            fid=dict(self.learner.fidelity(stage)); short=(sel_sha12(ids),self.fidelity_key(fid))
            full=(array_sha(ids),short[1])
            if short in self.short: assert self.short[short]==full,'short subset hash collision'
            self.short[short]=full
            path=self.root/digest(full)
            if short not in self._store:
                found=self.source_index.get(full)
                pu=None;origin=None
                if found is not None:
                    candidate,saved,score=found
                    assert np.array_equal(ids,saved) and canonical(fid)==score['fidelity']
                    pu={s:dict(np.load(candidate/'per_unit'/f'{s}.npz',allow_pickle=False)) for s in ('con','rank','conf','test')}
                    origin=str(candidate)
                elif (path/'complete.json').exists():
                    meta=json.loads((path/'complete.json').read_text())
                    assert meta['context_sha256']==self.context_sha and meta['fidelity']==canonical(fid)
                    for filename,expected_sha in meta['files'].items(): assert sha_file(path/filename)==expected_sha
                    with np.load(path/'selection.npz',allow_pickle=False) as z: assert np.array_equal(z['idx'],ids)
                    pu={s:dict(np.load(path/f'{s}.npz',allow_pickle=False)) for s in ('con','rank','conf','test')}
                    origin=str(path)
                if pu is not None:
                    self.validate_units(pu)
                    self._store[short]=FitEntry(short[0],fid,len(ids),0.0,None,pu,{s:0.0 for s in pu})
                    self.reused.append({'subset_sha256':full[0],'fidelity':fid,'source':origin,'n':len(ids)})
            before=len(self.events)
            result=super().evaluate(ids,stage,splits,eager=True)
            self.validate_units(result.per_unit)
            if any(e.fitted for e in self.events[before:]) and set(result.per_unit)=={'con','rank','conf','test'} and not (path/'complete.json').exists():
                path.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path/'selection.npz',idx=ids)
                for s,pu in result.per_unit.items(): np.savez_compressed(path/f'{s}.npz',**pu)
                files={p.name:sha_file(p) for p in path.iterdir() if p.is_file()}
                write_json(path/'complete.json',{'context_sha256':self.context_sha,'fidelity':fid,'subset_sha256':full[0],'files':files})
                self.persisted.append(str(path))
            return result
    return ExactCache

def compare_records(expected,actual):
    def candidates(root):
        out={}
        for p in (root/'candidates').glob('*/scores.json'):
            s=json.loads(p.read_text())
            with np.load(p.parent/'selection.npz',allow_pickle=False) as z: ids=z['idx'].tolist()
            out[s['name']]={'ids':ids,'role':s['role'],'stage':s['stage'],'fidelity':s['fidelity'],
                            'scoring':s['scoring'],'utility':s['utility']}
        return out
    a,b=candidates(expected),candidates(actual)
    assert a.keys()==b.keys(),{'missing':sorted(a.keys()-b.keys()),'extra':sorted(b.keys()-a.keys())}
    for name in a: assert a[name]==b[name],f'candidate replay mismatch: {name}'
    da=json.loads((expected/'decision.json').read_text());db=json.loads((actual/'decision.json').read_text())
    assert da==db,'decision/gate replay mismatch'
    assert json.loads((expected/'splits.json').read_text())['ids']==json.loads((actual/'splits.json').read_text())['ids']
    return {'equal':True,'candidates':len(a),'decision_sha256':digest(da)}

def run(track,seed,condition,*,batch=None,perturb=False,verify_full=False):
    src=source_cell(track,seed);obj,t,o,raw=configs(track,seed)
    data,context=load_verified(obj,t,o,raw,src)
    batch=batch or f'signal_drop/{condition}'
    dest=cell_path(track,seed,batch)
    if (dest/'decision.json').exists():
        assert (dest/'ablation_audit.json').exists(),'incomplete adapter audit'
        if verify_full:
            write_json(dest/'equivalence.json',compare_records(src,dest))
        return dest
    original_t=copy.deepcopy(t)
    project_config(t,o,condition)
    allowed={'methods','grid_weights','grid_q','grid_lam','auth_q','lam'}
    assert all(canonical(getattr(t,f.name))==canonical(getattr(original_t,f.name))
               for f in dataclasses.fields(t) if f.name not in allowed)
    active_t=canonical(t.to_dict());active_o=canonical(o.to_dict())
    original_signals=obj.signals
    obj.load=lambda *_:data
    obj.signals=lambda d,tc,oc:sanitize(original_signals(d,original_t,oc),condition,perturb)
    experiment=adapted_experiment(condition)
    instances=[]
    experiment.FitCache=persistent_cache_class(src,context,data,CACHE_ROOT,instances)
    result=experiment.run_cell(obj,t,o,out_root=OUT_ROOT,batch=batch,
                               cli=[str(Path(__file__)),'--track',track,'--seed',str(seed),'--condition',condition])
    audit={'condition':condition,'source':str(src),'context_sha256':digest(context),'data_fingerprint':context['data'],
           'source_config_sha256':sha_file(src/'config.json'),'source_decision_sha256':sha_file(src/'decision.json'),
           'removed_candidates':sorted(REMOVED[condition]),'boundary':BOUNDARIES.get(condition),
           'active_track':active_t,'active_omniselect':active_o,'perturb_removed_signal':perturb,
           'cache_reuse':[row for c in instances for row in c.reused],
           'new_fit_predictions':[p for c in instances for p in c.persisted],
           'cache_stats':[c.stats() for c in instances],
           'model_note':'Cached predictions do not materialize model objects; model_path is null for a cached elected model.',
           'result':result}
    write_json(dest/'ablation_audit.json',audit)
    if verify_full:
        assert condition=='full', '--verify-full requires --condition full'
        report=compare_records(src,dest);write_json(dest/'equivalence.json',report)
    return dest

def main(argv=None):
    global SOURCE_ROOT, OUT_ROOT, CACHE_ROOT, DATA_ROOT, SOURCE_CELL
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track',choices=list(TASKS),required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--condition',choices=['full',*CONDITIONS],required=True)
    parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT)
    parser.add_argument('--source-cell',type=Path,help='Explicit control cell, including its seed directory')
    parser.add_argument('--out',type=Path,default=OUT_ROOT)
    parser.add_argument('--cache-root',type=Path,default=CACHE_ROOT)
    parser.add_argument('--data-root',type=Path,default=DATA_ROOT)
    parser.add_argument('--batch',help='Output batch; default signal_drop/<condition>')
    parser.add_argument('--perturb',action='store_true',help='Perturb the removed channel before sanitizing it')
    parser.add_argument('--verify-full',action='store_true',help='Check exact candidate and decision replay against a complete control')
    args=parser.parse_args(argv)
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    if args.verify_full and args.condition!='full':
        parser.error('--verify-full requires --condition full')
    SOURCE_ROOT=args.source_root.resolve();OUT_ROOT=args.out.resolve()
    CACHE_ROOT=args.cache_root.resolve();DATA_ROOT=args.data_root.resolve()
    SOURCE_CELL=args.source_cell.resolve() if args.source_cell else None
    print(run(args.track,args.seed,args.condition,batch=args.batch,
              perturb=args.perturb,verify_full=args.verify_full))


if __name__=='__main__':main()
