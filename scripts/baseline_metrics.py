"""Frame-level metrics with explicit abstention and grouped score ties."""
import numpy as np

def ranking_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) == 0 or len(np.unique(labels)) != 2:
        return {'roc_auc': None, 'average_precision': None}
    order = np.argsort(-scores, kind='stable')
    y = labels[order]; s = scores[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    tp = np.cumsum(y)[ends]; fp = (ends+1)-tp
    recall = tp/y.sum(); fpr = fp/(len(y)-y.sum())
    auc = np.trapezoid(np.r_[0, recall], np.r_[0, fpr])
    ap = np.sum(np.diff(np.r_[0, recall]) * (tp/(tp+fp)))
    return {'roc_auc': float(auc), 'average_precision': float(ap)}

def aggregate_frames(entries, lengths):
    """Each entry is (source, frame, score); uncovered frames remain NaN."""
    output = {name: np.full(length, np.nan) for name, length in lengths.items()}
    for name, frame, score in entries:
        if frame < 0 or frame >= len(output[name]):
            raise ValueError('Frame outside scene label bounds')
        old = output[name][frame]
        output[name][frame] = score if not np.isfinite(old) else max(old, score)
    return output

def frame_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    covered = np.isfinite(scores)
    y = labels[covered]; alerts = scores[covered] > threshold
    tp = int(((y==1)&alerts).sum()); fp = int(((y==0)&alerts).sum())
    fn = int(((y==1)&~alerts).sum()); tn = int(((y==0)&~alerts).sum())
    positive = int((labels==1).sum()); negative = int((labels==0).sum())
    return {
        'total_frames': len(labels), 'scored_frames': int(covered.sum()),
        'coverage': float(covered.mean()), 'positive_frames': positive,
        'covered_positive_frames': int((y==1).sum()),
        'covered_negative_frames': int((y==0).sum()),
        'uncovered_positive_frames': int(((labels==1)&~covered).sum()),
        'covered_frame_confusion': {'tp':tp,'fp':fp,'fn':fn,'tn':tn},
        'precision_on_scored_frames': tp/(tp+fp) if tp+fp else None,
        'recall_on_scored_positives': tp/(tp+fn) if tp+fn else None,
        'operational_recall_if_abstentions_emit_no_alert': tp/positive if positive else None,
        'covered_normal_frame_exceedance_rate': fp/(fp+tn) if fp+tn else None,
        'overall_positive_prevalence': positive/(positive+negative),
        **ranking_metrics(y, scores[covered]),
    }
