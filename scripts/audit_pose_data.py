"""Audit all downloaded numeric data, retaining missing observations as missing."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from pose_data import load_tracks, window_ends, validation_sources, CONFIDENCE, HISTORY

def run(root):
    base=root/'data/poselift/original'
    paths=sorted((base/'pose/train').glob('*.json'))
    assert len(paths)==104
    held=validation_sources(paths)
    summary={}; files=[]; fingerprints={}; duplicated=[]
    for split in ('train','test'):
        stats=collections.Counter(); camera_stats=collections.Counter()
        for p in sorted((base/'pose'/split).glob('*.json')):
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in fingerprints: duplicated.append([fingerprints[digest],split+'/'+p.name])
            fingerprints[digest]=split+'/'+p.name
            stats['files']+=1;camera_stats[p.name.split('_')[0]]+=1
            label=None
            if split=='test':
                label=np.load(base/'gt/test_frame_mask'/(p.name.split('_alphapose')[0]+'.npy'),allow_pickle=False)
                assert label.ndim==1 and np.isin(label,[0,1]).all()
                stats['label_frames']+=len(label);stats['positive_label_frames']+=int(label.sum())
            seen=set(); file_windows=0
            for person,t,raw,x,y,valid in load_tracks(p):
                stats['tracks']+=1; stats['observations']+=len(t)
                stats['observations_with_nonfinite_keypoints']+=int((~np.isfinite(raw).all(axis=(1,2))).sum())
                stats['nonfinite_joint_triplets']+=int((~np.isfinite(raw).all(axis=2)).sum())
                stats['insufficient_quality_frames']+=int((valid.sum(axis=1)<8).sum())
                stats['frame_gaps']+=int((np.diff(t)!=1).sum())
                stats['tracks_shorter_than_window']+=int(len(t)<=HISTORY)
                assert np.isfinite(x).all() and np.isfinite(y).all()
                ends=window_ends(t,valid);file_windows+=len(ends)
                if label is not None:
                    stats['out_of_range_observations']+=int(((t<0)|(t>=len(label))).sum())
                    assert ((t>=0)&(t<len(label))).all()
                seen.update(t.tolist())
            if label is not None: stats['label_frames_without_pose']+=len(label)-len(seen)
            group=('validation' if p.name in held else 'fit') if split=='train' else 'official_test'
            stats['eligible_windows']+=file_windows
            files.append({'source':p.name,'directory':split,'group':group,'camera':p.name.split('_')[0],'eligible_windows':file_windows,'sha256':digest})
        summary[split]={'counts':dict(stats),'cameras':dict(camera_stats)}
    assert len([f for f in files if f['directory']=='test'])==47
    assert not duplicated, 'Duplicate source bytes found; inspect before fitting'
    report={'schema_version':1,'summary':summary,'files':files,'exact_duplicate_files':duplicated,
      'contract':{'joints':17,'values_per_joint':['centered_scaled_x','centered_scaled_y','confidence','validity'],'history_frames':HISTORY,'confidence_min':CONFIDENCE,'target':'next observed normalized pose, masked loss','cross_gap_windows':False},
      'limitations':['Timing is frame-index based; true sampling cadence needs validation on video.','Test labels are frame/scene-level; do not assign to every person.','Source-file grouping is not proof of actor/session independence.','Joint semantic ordering and producer smoothing causality remain unverified.','Normal-only training is publisher-described, not independently verified from raw video.']}
    out=root/'artifacts/data_audit';out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'summary':summary,'groups':dict(collections.Counter(f['group'] for f in files)), 'windows_by_group':{g:sum(f['eligible_windows'] for f in files if f['group']==g) for g in ('fit','validation','official_test')}},indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args();run(args.root)
