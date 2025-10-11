#!/usr/bin/env python3
"""
Compute CLIP image embeddings for candidates and run DBSCAN clustering per-token.
Writes a JSON with cluster assignments and medoid indices per cluster.

Usage:
  python clip_cluster.py --in ../precomputed_merged_run2.json --out ../precomputed_merged_run2_clip_clusters.json --eps 0.35
"""

import argparse
import json
import requests
import io
from PIL import Image
import clip
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm


def download_image(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert('RGB')
    except Exception as e:
        print(f"download failed {url}: {e}")
        return None


def compute_embeddings(model, preprocess, imgs, device):
    batch = torch.stack([preprocess(im) for im in imgs]).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch)
    feats = feats.cpu().numpy()
    # normalize
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / (norms + 1e-10)
    return feats


def medoid_index(embs, indices):
    # choose medoid (min sum distance) among indices
    sub = embs[indices]
    dists = 1 - (sub @ sub.T)
    sums = dists.sum(axis=1)
    return indices[int(np.argmin(sums))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out', dest='outfile', required=True)
    parser.add_argument('--eps', type=float, default=0.35)
    parser.add_argument('--min-samples', type=int, default=2)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-B/32', device=device)

    data = json.load(open(args.infile, 'r'))
    out = []
    total = 0
    for rec in tqdm(data, desc='tokens'):
        token_out = {k: rec.get(k) for k in ('tokenId','tokenQuery','cocktail')}
        cands = rec.get('candidates', [])
        imgs = []
        urls = []
        idx_map = []
        for i, c in enumerate(cands):
            u = (c.get('urls') or {}).get('regular') or (c.get('urls') or {}).get('small')
            if not u:
                u = (rec.get('unsplash') or {}).get('urls', {}).get('regular') or (rec.get('unsplash') or {}).get('urls', {}).get('small')
            if not u:
                continue
            img = download_image(u)
            if img is None:
                continue
            imgs.append(img)
            urls.append(u)
            idx_map.append(i)
            total += 1
            if args.limit and total >= args.limit:
                break
        if len(imgs) == 0:
            token_out['clusters'] = []
            out.append(token_out)
            continue
        embs = compute_embeddings(model, preprocess, imgs, device)
        # DBSCAN with cosine distance -> use metric='cosine' in sklearn
        clustering = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric='cosine').fit(embs)
        labels = clustering.labels_
        clusters = {}
        for i, lab in enumerate(labels):
            clusters.setdefault(int(lab), []).append(i)
        token_clusters = []
        for lab, inds in clusters.items():
            if lab == -1:
                # noise: treat each as its own cluster
                for ii in inds:
                    token_clusters.append({'label': -1, 'members': [idx_map[ii]], 'medoid': idx_map[ii], 'url': urls[ii]})
            else:
                med = medoid_index(embs, inds)
                token_clusters.append({'label': int(lab), 'members': [idx_map[ii] for ii in inds], 'medoid': idx_map[med], 'url': urls[inds[0]]})
        token_out['clusters'] = token_clusters
        out.append(token_out)
    with open(args.outfile, 'w') as f:
        json.dump(out, f, indent=2)
    print('Wrote', len(out), 'tokens with clusters to', args.outfile)


if __name__ == '__main__':
    main()
