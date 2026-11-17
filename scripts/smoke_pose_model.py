"""Two-step CPU plumbing check, not model evaluation or deployable training."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pose_data import load_tracks,window_ends,HISTORY

class PosePredictor(nn.Module):
    def __init__(self):
        super().__init__();self.a=nn.Conv1d(68,16,3);self.b=nn.Conv1d(16,16,3,dilation=2);self.head=nn.Linear(16,34)
    def encode(self,x):
        x=F.relu(self.a(F.pad(x,(2,0))));return F.relu(self.b(F.pad(x,(4,0))))
    def forward(self,x): return self.head(self.encode(x.transpose(1,2))[:,:,-1])

def main():
    root=Path(__file__).resolve().parents[1];audit=json.loads((root/'artifacts/data_audit/audit.json').read_text())
    samples=[]
    for f in audit['files']:
        if f['group']!='fit':continue
        for _,frames,_,features,target,valid in load_tracks(root/'data/poselift/original/pose/train'/f['source']):
            for end in window_ends(frames,valid)[:2]:
                samples.append((features[end-HISTORY:end],target[end],np.repeat(valid[end],2)))
                if len(samples)>=16:break
            if len(samples)>=16:break
        if len(samples)>=16:break
    assert len(samples)==16
    torch.manual_seed(17);torch.set_num_threads(2)
    x=torch.from_numpy(np.stack([s[0] for s in samples]));y=torch.from_numpy(np.stack([s[1] for s in samples]));mask=torch.from_numpy(np.stack([s[2] for s in samples]))
    model=PosePredictor();optimizer=torch.optim.Adam(model.parameters(),lr=0.001);losses=[]
    initial=model.head.weight.detach().clone();start=time.perf_counter()
    for _ in range(2):
        optimizer.zero_grad();pred=model(x);loss=(F.smooth_l1_loss(pred,y,reduction='none')*mask).sum()/mask.sum()
        assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step();losses.append(float(loss.detach()))
    assert not torch.equal(initial,model.head.weight.detach())
    # Earlier encoder outputs cannot depend on later input frames.
    original=x.transpose(1,2);changed=original.clone();changed[:,:,5:]+=100
    assert torch.allclose(model.encode(original)[:,:,:5],model.encode(changed)[:,:,:5],atol=1e-6)
    report={'kind':'two-step smoke test only','device':'cpu','batch':16,'parameters':sum(p.numel() for p in model.parameters()),'losses':losses,'seconds':time.perf_counter()-start,'finite_gradients':True,'parameters_updated':True,'causal_prefix_check':True,'official_test_used_for_fitting':False,'torch':torch.__version__,'numpy':np.__version__,'python':sys.version,'isolated_prefix':sys.prefix,'checkpoint_saved':False}
    out=root/'artifacts/smoke';out.mkdir(parents=True,exist_ok=True);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
