"""One controlled normal-only experiment; official labels read after selection.

No event accuracy, theft probability, or camera throughput is implied.
"""
import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from pose_data import load_tracks, window_ends, HISTORY
from pose_model import PosePredictor
from baseline_metrics import aggregate_frames, frame_metrics

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')

class Windows(Dataset):
    def __init__(self, root, records):
        self.tracks=[]; self.index=[]; self.lengths={}; self.source_files=[]
        for rec in records:
            p=root/'data/poselift/original/pose'/rec['directory']/rec['source']
            if hashlib.sha256(p.read_bytes()).hexdigest()!=rec['sha256']:
                raise ValueError('Source changed after audit: '+p.name)
            source=p.name.split('_alphapose')[0]; self.source_files.append(p.name)
            for person,frames,_,features,target,valid in load_tracks(p):
                if len(frames): self.lengths[source]=max(self.lengths.get(source,0), int(frames.max())+1)
                track=len(self.tracks)
                self.tracks.append((source,person,frames,features,target,valid))
                for end in window_ends(frames,valid):
                    # Score model and persistence on the same observable joints.
                    if (valid[end]&valid[end-1]).sum()>=4:
                        self.index.append((track,int(end)))
        if not self.index: raise ValueError('No usable windows')

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        track,end=self.index[i]; _,_,_,features,target,valid=self.tracks[track]
        mask=np.repeat(valid[end]&valid[end-1],2).astype(np.float32)
        return features[end-HISTORY:end], target[end], mask

    def location(self, i):
        track,end=self.index[i]; source,_,frames,_,_,_=self.tracks[track]
        return source,int(frames[end])

def errors(prediction,target,mask):
    return (F.smooth_l1_loss(prediction,target,reduction='none')*mask).sum(1)/mask.sum(1).clamp_min(1)

def score(model, dataset, batch_size):
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    model.eval(); model_scores=[]; persistence_scores=[]
    with torch.inference_mode():
        for x,y,mask in loader:
            model_scores.extend(errors(model(x),y,mask).tolist())
            previous=x[:,-1].reshape(-1,17,4)[:,:,:2].reshape(-1,34)
            persistence_scores.extend(errors(previous,y,mask).tolist())
    return np.asarray(model_scores),np.asarray(persistence_scores)

def scenes(dataset,scores,lengths=None):
    return aggregate_frames([(dataset.location(i)[0],dataset.location(i)[1],float(s)) for i,s in enumerate(scores)], lengths or dataset.lengths)

def main(args):
    root=Path(__file__).resolve().parents[1]
    out=root/'artifacts/baseline'/args.run
    if out.exists(): raise ValueError('Choose a new run name; existing experiments are immutable')
    out.mkdir(parents=True)
    config={'seed':17,'threads':2,'batch':256,'max_epochs':15,'patience':4,'learning_rate':0.001,'history':HISTORY,'minimum_common_joints':4,'normal_validation_quantile':0.99,'scene_aggregation':'max across eligible person tracks at current frame','score':'masked next-pose smooth-L1 error','symmetric_smoothing':False}
    audit=json.loads((root/'artifacts/data_audit/audit.json').read_text())
    records={g:[f for f in audit['files'] if f['group']==g] for g in ('fit','validation','official_test')}
    assert not ({f['source'] for f in records['fit']}&{f['source'] for f in records['validation']})
    assert not ({f['source'] for f in records['official_test']}&{f['source'] for f in records['fit']+records['validation']})
    provenance={'config':config,'sources':records,'audit_sha256':hashlib.sha256((root/'artifacts/data_audit/audit.json').read_bytes()).hexdigest(),'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('pose_data.py'),Path(__file__).with_name('pose_model.py'),Path(__file__).with_name('baseline_metrics.py'))},'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform()}
    write_json(out/'run_manifest.json',provenance)
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    torch.set_num_threads(config['threads']);torch.use_deterministic_algorithms(True)
    fit=Windows(root,records['fit']);validation=Windows(root,records['validation'])
    print(f'CPU normal-only fit: {len(fit)} windows; validation: {len(validation)} windows',flush=True)
    loader=DataLoader(fit,batch_size=config['batch'],shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(config['seed']))
    model=PosePredictor();optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    best=float('inf');bad=0;history=[];started=time.perf_counter()
    for epoch in range(1,config['max_epochs']+1):
        model.train();total=0.;n=0
        for x,y,mask in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=errors(model(x),y,mask).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            total+=float(loss.detach())*len(x);n+=len(x)
        vs,_=score(model,validation,config['batch']);val=float(vs.mean())
        history.append({'epoch':epoch,'fit_loss':total/n,'validation_loss':val})
        print(f'Epoch {epoch:02d}: fit={total/n:.6f} validation={val:.6f}',flush=True)
        if val<best:
            best=val;best_epoch=epoch;bad=0
            torch.save({'state_dict':model.state_dict(),'config':config,'epoch':epoch},out/'best.pt')
        else:bad+=1
        write_json(out/'history.json',history)
        if bad>=config['patience']:break
    training_seconds=time.perf_counter()-started
    checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['state_dict']);model.eval()
    val_model,val_persistence=score(model,validation,config['batch'])
    val_scenes=scenes(validation,val_model);p_scenes=scenes(validation,val_persistence)
    threshold=float(np.quantile(np.concatenate([a[np.isfinite(a)] for a in val_scenes.values()]),config['normal_validation_quantile'],method='higher'))
    p_threshold=float(np.quantile(np.concatenate([a[np.isfinite(a)] for a in p_scenes.values()]),config['normal_validation_quantile'],method='higher'))
    selection={'best_epoch':best_epoch,'validation_loss':best,'persistence_validation_loss':float(val_persistence.mean()),'threshold':threshold,'persistence_threshold':p_threshold,'threshold_rule':'99th percentile of covered normal validation scene scores; strict greater-than alert','training_seconds':training_seconds,'parameters':sum(p.numel() for p in model.parameters()),'fitting_windows':len(fit),'validation_windows':len(validation),'checkpoint_sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),'official_labels_used_for_selection':False}
    # Persist chosen checkpoint and thresholds BEFORE loading any official labels.
    write_json(out/'selection.json',selection)
    test=Windows(root,records['official_test'])
    labels={}
    for rec in records['official_test']:
        source=rec['source'].split('_alphapose')[0]
        labels[source]=np.load(root/'data/poselift/original/gt/test_frame_mask'/(source+'.npy'),allow_pickle=False)
    lengths={k:len(v) for k,v in labels.items()}
    test_model,test_persistence=score(model,test,config['batch'])
    ts=scenes(test,test_model,lengths);ps=scenes(test,test_persistence,lengths)
    all_labels=np.concatenate([labels[k] for k in sorted(labels)])
    metrics=frame_metrics(all_labels,np.concatenate([ts[k] for k in sorted(labels)]),threshold)
    comparator=frame_metrics(all_labels,np.concatenate([ps[k] for k in sorted(labels)]),p_threshold)
    per_source={k:frame_metrics(labels[k],ts[k],threshold) for k in sorted(labels)}
    np.savez_compressed(out/'test_scene_scores.npz',**{k:v for k,v in ts.items()})
    # Model-only microbenchmark, not a camera or whole-pipeline benchmark.
    example=torch.from_numpy(fit[0][0][None]);durations=[]
    with torch.inference_mode():
        for _ in range(50):model(example)
        for _ in range(300):
            t=time.perf_counter();model(example);durations.append((time.perf_counter()-t)*1000)
    report={'selection':selection,'test':metrics,'persistence_comparator':comparator,'per_test_source':per_source,'eligible_test_windows':len(test),'cpu_model_only_batch1_ms':{'median':float(np.median(durations)),'p95':float(np.quantile(durations,.95)),'threads':2,'warmup':50,'iterations':300},'limits':['Pose prediction anomaly scores are not concealment labels or theft probabilities.','No continuous-video event precision, false alerts per hour, or source FPS claim is supported.','Metrics are conditional on scored frames; overall recall treats abstention as no alert.','Scene labels cannot identify which person acted; no per-person classification accuracy is claimed.','Producer interpolation/smoothing causality, joint semantics, and actor/session independence remain unverified.','No YOLO/video extractor or low-spec shop hardware was evaluated.']}
    write_json(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='per_test_source'},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='pose_baseline_v1');main(parser.parse_args())
