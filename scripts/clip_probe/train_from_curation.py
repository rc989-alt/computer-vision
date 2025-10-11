#!/usr/bin/env python3
"""
Train a CLIP linear probe using a curated selections file.

Inputs:
  - --precomputed: path to a precomputed JSON containing tokens with "candidates"
  - --curation:    path to exported curation.json (array of {tokenId, chosen: [{id, reason, ...}], notes})

Logic:
  - For each token with curated selections, mark selected candidate IDs as positives
  - Sample negatives from the remaining candidates of the same token (hard negatives)
  - Encode images with CLIP (cached) and train a LogisticRegression probe
  - Optionally run StratifiedKFold CV and write metrics

Usage:
  python train_from_curation.py \
    --precomputed ../data/precomputed_with_yolo_medoid.json \
    --curation ~/Downloads/curation.json \
    --out-model clip_probe_balanced.joblib \
    --cache embedding_cache.joblib \
    --cv 5 --neg-per-pos 2 --timeout 15 --max-per-token 10
"""
import argparse
import json
import io
import time
import random
from typing import Dict, List, Tuple, Set

import requests
from PIL import Image
import numpy as np
import joblib
from tqdm import tqdm
import clip
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report

try:
    from embedding_cache import load_cache, save_cache, ensure_embedding
except Exception:
    from .embedding_cache import load_cache, save_cache, ensure_embedding


def _cand_url(c: dict) -> str:
    u = (c.get('urls') or {}).get('regular') or (c.get('urls') or {}).get('small') or (c.get('urls') or {}).get('raw')
    return u or ''


def build_sets(precomputed: List[dict], curation: List[dict], neg_per_pos: int = 2, max_per_token: int = None) -> Tuple[List[Tuple[str, int]], Dict[str, List[str]]]:
    """
    Returns:
      samples: list of (url, label) pairs
      token_pos_ids: map tokenId -> list of positive candidate ids
    """
    curated_map: Dict[str, Set[str]] = {}
    for row in curation:
        tid = row.get('tokenId')
        raw = row.get('chosen')
        # Normalize chosen into a list of objects with an 'id' field
        if isinstance(raw, list):
            chosen = raw
        elif isinstance(raw, dict):
            chosen = [raw]
        elif isinstance(raw, (str, int, float)):
            chosen = [{ 'id': str(raw) }]
        else:
            chosen = []
        ids = {str(x.get('id')) for x in chosen if isinstance(x, dict) and x.get('id') is not None}
        if tid and ids:
            curated_map[tid] = ids

    samples: List[Tuple[str, int]] = []
    token_pos_ids: Dict[str, List[str]] = {}

    rng = random.Random(42)

    for rec in precomputed:
        tid = rec.get('tokenId')
        if not tid or tid not in curated_map:
            continue
        pos_ids = curated_map[tid]
        token_pos_ids[tid] = list(pos_ids)
        cands = list(rec.get('candidates') or [])
        pos = [c for c in cands if str(c.get('id')) in pos_ids]
        neg_pool = [c for c in cands if str(c.get('id')) not in pos_ids]

        # limit per token if requested (applies to positives to keep balance controlled)
        if max_per_token is not None and max_per_token > 0:
            rng.shuffle(pos)
            pos = pos[:max_per_token]

        # sample negatives per positive
        needed = neg_per_pos * len(pos)
        rng.shuffle(neg_pool)
        neg = neg_pool[:needed] if needed > 0 else []

        # add to global samples
        for c in pos:
            u = _cand_url(c)
            if u:
                samples.append((u, 1))
        for c in neg:
            u = _cand_url(c)
            if u:
                samples.append((u, 0))

    return samples, token_pos_ids


def encode_samples(samples: List[Tuple[str, int]], model, preprocess, device: str, cache, timeout: int = 10):
    X = []
    y = []
    kept = 0
    for (u, label) in tqdm(samples, desc='encoding'):
        feats = ensure_embedding(u, model, preprocess, device, timeout=timeout, cache=cache)
        if feats is None:
            continue
        X.append(feats)
        y.append(label)
        kept += 1
    if kept == 0:
        raise RuntimeError('No images could be encoded; check connectivity/URLs')
    return np.stack(X), np.array(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--precomputed', required=True)
    p.add_argument('--curation', required=True)
    p.add_argument('--out-model', default='clip_probe_balanced.joblib')
    p.add_argument('--cache', default='embedding_cache.joblib')
    p.add_argument('--neg-per-pos', type=int, default=2)
    p.add_argument('--max-per-token', type=int, default=None, help='cap positive samples per token before sampling negatives')
    p.add_argument('--cv', type=int, default=5)
    p.add_argument('--auto-cv', action='store_true', help='auto-reduce CV folds to feasible value based on smallest class count; disable if <2')
    p.add_argument('--save-report', action='store_true', help='write a small human-readable training_report.txt next to the model')
    p.add_argument('--timeout', type=int, default=10)
    args = p.parse_args()

    precomputed = json.load(open(args.precomputed, 'r'))
    curation = json.load(open(args.curation, 'r'))

    samples, token_pos_ids = build_sets(precomputed, curation, neg_per_pos=args.neg_per_pos, max_per_token=args.max_per_token)
    print('Training pairs:', len(samples), 'tokens with curation:', len(token_pos_ids))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device', device)

    model_clip, preprocess = clip.load('ViT-B/32', device=device)
    cache = load_cache(args.cache)

    X, y = encode_samples(samples, model_clip, preprocess, device, cache, timeout=args.timeout)
    binc = np.bincount(y)
    print('Dataset:', X.shape, 'labels distribution', binc)
    if len(np.unique(y)) < 2:
        raise RuntimeError('Need both positive and negative samples to train')

    # Metrics container
    metrics = {'n': int(len(y)), 'pos': int(int(np.sum(y))), 'neg': int(int(len(y) - np.sum(y)))}

    # Determine effective CV folds
    eff_cv = args.cv
    if args.auto_cv:
        min_count = int(binc.min()) if len(binc) > 1 else 0
        eff_cv = max(0, min(args.cv or 5, min_count))

    # Optional CV
    if eff_cv and eff_cv > 1:
        skf = StratifiedKFold(n_splits=eff_cv, shuffle=True, random_state=42)
        aucs = []
        for fold, (tr, va) in enumerate(skf.split(X, y), 1):
            clf_cv = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
            clf_cv.fit(X[tr], y[tr])
            proba = clf_cv.predict_proba(X[va])[:, 1]
            auc = roc_auc_score(y[va], proba)
            aucs.append(float(auc))
            print(f'Fold {fold} ROC-AUC: {auc:.4f}')
        metrics['cv_folds'] = int(eff_cv)
        metrics['cv_auc_mean'] = float(np.mean(aucs))
        metrics['cv_auc_std'] = float(np.std(aucs))

    # Final fit on all data
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    clf.fit(X, y)
    print(classification_report(y, clf.predict(X)))

    payload = {
        'model': clf,
        'device': device,
        'trained_at': time.time(),
        'args': vars(args),
        'metrics': metrics,
    }
    joblib.dump(payload, args.out_model)
    print('Saved model to', args.out_model)

    save_cache(cache, args.cache)
    # also save metrics alongside
    metrics_path = args.out_model + '.metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(payload.get('metrics', {}), f, indent=2)
    print('Saved metrics to', metrics_path)

    # Optional human-readable report
    if args.save_report:
        report_path = args.out_model + '.training_report.txt'
        lines = []
        lines.append('CLIP Probe Training Report')
        lines.append('===========================')
        lines.append(f"Samples: {metrics['n']} (pos={metrics['pos']}, neg={metrics['neg']})")
        if 'cv_folds' in metrics:
            lines.append(f"CV folds: {metrics['cv_folds']}")
            lines.append(f"CV AUC mean: {metrics.get('cv_auc_mean', 'n/a')}")
            lines.append(f"CV AUC std: {metrics.get('cv_auc_std', 'n/a')}")
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print('Saved training report to', report_path)


if __name__ == '__main__':
    main()
