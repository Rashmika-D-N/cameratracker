"""Render sparse contact sheets for review; not temporal ground-truth annotation."""
from pathlib import Path
import json
import cv2
import numpy as np

def main():
    root=Path(__file__).resolve().parents[1];base=root/'data/video_pilot/mnnit_v1'
    manifest=json.loads((base/'manifest.json').read_text());out=root/'artifacts/video_review';out.mkdir(parents=True,exist_ok=True)
    metadata=[]
    for label in ('normal','shoplifting'):
        rows=[]
        for item in manifest['files']:
            if item['provider_label']!=label:continue
            p=root/item['file'];cap=cv2.VideoCapture(str(p));fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            tiles=[]
            for frac in (.05,.23,.41,.59,.77,.95):
                cap.set(cv2.CAP_PROP_POS_FRAMES,int((n-1)*frac));ok,image=cap.read()
                if not ok:raise ValueError('Could not decode preview: '+p.name)
                tile=cv2.resize(image,(200,150));cv2.putText(tile,f'{p.stem} {n/fps*frac:.1f}s',(3,13),cv2.FONT_HERSHEY_SIMPLEX,.32,(0,255,255),1);tiles.append(tile)
            cap.release();rows.append(np.hstack(tiles));metadata.append({'clip':p.stem,'source_fps':fps,'frames':n,'duration_seconds':n/fps,'provider_label':label})
        cv2.imwrite(str(out/(label+'_contact.jpg')),np.vstack(rows))
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8');print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
