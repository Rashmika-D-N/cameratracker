"""Describe longer-context availability; do not invent four-class accuracy."""
import json
from pathlib import Path
import numpy as np
from video_features import interaction_proxies,contiguous_contexts

def main():
    root=Path(__file__).resolve().parents[1];folder=root/'artifacts/video_pose/mnnit_yolo11_v1'
    results=[]
    for path in sorted(folder.glob('*.jsonl')):
        tracks={};rows=[json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            for person in row['people']:
                track=person['track_id']
                if track is None:continue
                history=tracks.setdefault(track,[])
                prev=history[-1] if history else None
                features=interaction_proxies(person,prev['person'] if prev else None,row['timestamp_seconds']-prev['time'] if prev else None)
                history.append({'time':row['timestamp_seconds'],'person':person,'features':features})
        contexts={str(seconds):sum(contiguous_contexts(track,seconds) for track in tracks.values()) for seconds in (.8,2.4,4.8)}
        available=[v['features'] for tr in tracks.values() for v in tr if v['features'] is not None]
        keys=sorted({key for item in available for key in item})
        results.append({'clip':path.stem,'samples':len(rows),'tracks':len(tracks),'observations_with_complete_proxy_joints':len(available),'eligible_context_windows_seconds':contexts,'proxy_medians':{key:float(np.median([v[key] for v in available if key in v])) for key in keys}})
    assert results,'No extracted videos'
    out={'clips':results,'status':'feature/context availability pilot only','labels':'Provider normal/shoplifting labels; not reviewed four-action labels','accuracy_comparison_performed':False,'limits':['Wrist-to-body distances are proxies; no bag/product detector or real hand-object interaction ground truth was used.','Longer windows trade coverage for context; this report does not show better concealment detection.','No splits or accuracy are invented from eight provider-labeled clips.']}
    (folder/'context_report.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
