#!/usr/bin/env python3
import json
import numpy as np
from collections import defaultdict

def load_scores(path):
    data = json.load(open(path,'r'))
    scores = []
    per_token = defaultdict(list)
    for rec in data:
        for c in rec.get('candidates', []):
            s = c.get('clip_probe_score')
            if s is None:
                continue
            scores.append(s)
            per_token[rec.get('tokenId')].append(s)
    return np.array(scores), per_token

if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv)>1 else '../data/precomputed_run2_with_scores_full_scored.json'
    scores, per_token = load_scores(p)
    if len(scores)==0:
        print('No scores found')
        sys.exit(1)
    qs = [10,25,50,75,90,95,99]
    vals = np.percentile(scores, qs)
    for q,v in zip(qs,vals):
        print(f'{q}th percentile: {v:.4f}')
    print('min', scores.min(), 'max', scores.max(), 'mean', scores.mean())
    # tokens with median low scores
    low_meds = sorted(((tid, np.median(arr)) for tid,arr in per_token.items()), key=lambda x: x[1])[:5]
    print('\nSample tokens with lowest median scores:')
    for tid,m in low_meds:
        print(tid, m)
