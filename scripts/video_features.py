"""Pose-derived interaction proxies, not verified hand-object interactions."""
import numpy as np

def interaction_proxies(person,previous=None,dt=None):
    k=np.asarray(person['keypoints'],dtype=float)
    if k.shape!=(17,3):raise ValueError('Expected COCO17 x/y/confidence triplets')
    usable=np.isfinite(k).all(1)&(k[:,2]>=.2)
    required=[5,6,9,10,11,12] # shoulders, wrists, hips: COCO17 convention
    if not usable[required].all():return None
    shoulder=(k[5,:2]+k[6,:2])/2;hip=(k[11,:2]+k[12,:2])/2
    scale=np.linalg.norm(shoulder-hip)
    if scale<1e-6:return None
    wrist=k[[9,10],:2]
    result={'left_wrist_to_hip':float(np.linalg.norm(wrist[0]-hip)/scale),'right_wrist_to_hip':float(np.linalg.norm(wrist[1]-hip)/scale),'wrists_apart':float(np.linalg.norm(wrist[0]-wrist[1])/scale),'left_wrist_to_shoulder':float(np.linalg.norm(wrist[0]-shoulder)/scale),'right_wrist_to_shoulder':float(np.linalg.norm(wrist[1]-shoulder)/scale)}
    if previous is not None and dt is not None and 0<dt<=.25:
        pk=np.asarray(previous['keypoints'],dtype=float)
        if pk.shape==(17,3) and np.isfinite(pk[[9,10]]).all() and (pk[[9,10],2]>=.2).all():
            result['wrist_speed_per_torso_per_second']=float(np.linalg.norm(wrist-pk[[9,10],:2],axis=1).mean()/scale/dt)
    return result

def contiguous_contexts(observations,seconds):
    """Count causal track windows with no sample gap >250ms and adequate quality."""
    starts=0;count=0
    for end,obs in enumerate(observations):
        if obs['features'] is None:
            starts=end+1;continue
        if end and obs['time']-observations[end-1]['time']>.25:starts=end
        while starts<end and obs['time']-observations[starts]['time']>seconds+.15:starts+=1
        if starts<=end and obs['time']-observations[starts]['time']>=seconds:count+=1
    return count
