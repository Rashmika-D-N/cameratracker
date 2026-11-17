"""Extract timestamped YOLO11 pose tracks from the provider-labeled video pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

def main(args):
    root=Path(__file__).resolve().parents[1]
    os.environ['YOLO_CONFIG_DIR']=str(root/'runtime/ultralytics')
    os.environ['YOLO_AUTOINSTALL']='false'
    os.environ['MPLCONFIGDIR']=str(root/'runtime/matplotlib')
    os.environ['TORCH_HOME']=str(root/'artifacts/torch_cache')
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    weights=root/'artifacts/models/yolo11n-pose.pt'
    if not weights.exists():raise ValueError('Download the official model first; no implicit model downloads')
    manifest=json.loads((root/'data/video_pilot/mnnit_v1/manifest.json').read_text())
    out=root/'artifacts/video_pose'/args.run
    if out.exists():raise ValueError('Use a new output run name')
    out.mkdir(parents=True)
    results=[]
    for clip in manifest['files']:
        path=root/clip['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==clip['sha256']
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not 0<fps<500:raise ValueError('Unreadable video or invalid frame rate')
        model=YOLO(str(weights));frame=0;next_time=0.;processed=0;observed=0;tracked=0;times=[]
        destination=out/(path.stem+'.jsonl');last_ts=-1.
        with destination.open('w',encoding='utf-8') as stream:
            while True:
                ok,image=cap.read()
                if not ok:break
                ts=frame/fps
                if ts+1e-6<next_time:frame+=1;continue
                next_time+=1/args.sample_fps
                started=time.perf_counter()
                # New model per source resets tracker state between clips.
                r=model.track(image,persist=True,tracker='bytetrack.yaml',device='cpu',imgsz=args.imgsz,conf=.25,verbose=False)[0]
                times.append((time.perf_counter()-started)*1000)
                people=[]
                if r.keypoints is not None and r.boxes is not None:
                    points=r.keypoints.data.cpu().numpy();boxes=r.boxes.xyxy.cpu().numpy();conf=r.boxes.conf.cpu().numpy()
                    ids=r.boxes.id.cpu().numpy() if r.boxes.id is not None else None
                    for i,k in enumerate(points):
                        people.append({'track_id':int(ids[i]) if ids is not None else None,'bbox_xyxy':boxes[i].tolist(),'box_confidence':float(conf[i]),'keypoints':k.tolist()})
                record={'sample_index':processed,'source_frame':frame,'timestamp_seconds':ts,'timestamp_basis':'source_frame/source_fps','width':image.shape[1],'height':image.shape[0],'people':people}
                assert ts>last_ts;last_ts=ts
                stream.write(json.dumps(record,allow_nan=False)+'\n')
                processed+=1;observed+=int(bool(people));tracked+=int(any(p['track_id'] is not None for p in people));frame+=1
        cap.release()
        if processed==0:raise ValueError('No frames extracted')
        result={'clip':path.stem,'provider_label':clip['provider_label'],'source_sha256':clip['sha256'],'source_fps':fps,'decoded_frames':frame,'sampled_frames':processed,'frames_with_pose':observed,'frames_with_track':tracked,'sample_fps':args.sample_fps,'imgsz':args.imgsz,'model_and_tracker_mean_ms':sum(times)/len(times)}
        results.append(result);print(json.dumps(result),flush=True)
    report={'model':'yolo11n-pose.pt','weight_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'ultralytics':ultralytics.__version__,'device':'cpu','threads':2,'clips':results,'limits':['Provider clip labels only; action boundaries and phone/clothing labels unverified.','Extraction and tracking timings are laptop measurements, not low-spec certification.','Timestamp mapping assumes source constant frame rate; not proof of encoded PTS accuracy.','No trained classifier or person-level ground truth is created by pose extraction.']}
    (out/'extraction_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='mnnit_yolo11_v1');p.add_argument('--sample-fps',type=float,default=10.);p.add_argument('--imgsz',type=int,default=416);main(p.parse_args())
