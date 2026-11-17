"""Numeric, gap-aware PoseLift input contract; no test labels used for fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np

CONFIDENCE = 0.2
HISTORY = 8
MIN_JOINTS = 8

def load_tracks(path):
    people = json.loads(Path(path).read_text(encoding='utf-8'))
    for person, observations in people.items():
        ordered = sorted(observations, key=int)
        frames = np.asarray([int(t) for t in ordered], dtype=np.int64)
        raw = np.asarray([observations[t]['keypoints'] for t in ordered], dtype=np.float32).reshape(-1,17,3)
        valid = np.isfinite(raw).all(axis=-1) & (raw[:,:,2] >= CONFIDENCE)
        clean = np.where(valid[:,:,None], raw, 0)
        count = valid.sum(axis=1)
        center = clean[:,:,:2].sum(axis=1) / np.maximum(count[:,None],1)
        centered = np.where(valid[:,:,None],clean[:,:,:2]-center[:,None,:],0)
        scale = np.sqrt((centered**2).sum(axis=(1,2))/np.maximum(count,1))
        usable = (count >= MIN_JOINTS) & (scale > 1e-6)
        valid = valid & usable[:,None]
        xy = np.where(valid[:,:,None],centered/np.maximum(scale[:,None,None],1e-6),0)
        confidence = np.where(valid,clean[:,:,2],0)
        features = np.concatenate((xy,confidence[:,:,None],valid[:,:,None]),axis=-1).reshape(-1,68).astype(np.float32)
        yield str(person),frames,raw,features,xy.reshape(-1,34).astype(np.float32),valid

def window_ends(frames, valid, history=HISTORY):
    """Indices of next-frame targets; history and target must be observed/usable."""
    if len(frames) <= history:
        return np.empty(0,dtype=np.int64)
    no_gap = np.convolve((np.diff(frames)!=1).astype(int),np.ones(history,dtype=int),'valid') == 0
    usable = (valid.sum(axis=1)>=MIN_JOINTS).astype(int)
    enough = np.convolve(usable,np.ones(history+1,dtype=int),'valid') == history+1
    return np.arange(history,len(frames))[no_gap & enough]

def validation_sources(paths):
    """Hold out whole source files within each camera; never split tracks/windows."""
    cameras = {}
    for p in paths:
        cameras.setdefault(p.name.split('_')[0],[]).append(p)
    held = set()
    for camera, files in sorted(cameras.items()):
        ranked = sorted(files,key=lambda p:hashlib.sha256(('split-v1:'+p.name).encode()).hexdigest())
        n = min(len(files)-1,max(1,round(len(files)*0.2)))
        held.update(p.name for p in ranked[:n])
    return held
