#!/usr/bin/env python3
"""
Train a CLIP linear probe with balanced positives and negatives.

Positives: candidate has detection OR alt_description contains positive cues.
Negatives: candidate has no detection AND (alt_description contains negative cues OR score is very low). If not enough negatives, sample ambiguous candidates as negatives to balance.

Now includes:
- Embedding cache to avoid re-encoding/downloading across runs.
- Optional Stratified K-Fold CV reporting ROC-AUC.
- Optional metrics.json output.

Usage:
    python train_clip_probe_balanced.py --in precomputed_with_yolo.json --out-model clip_probe_balanced.joblib --per-class 100 --cache embedding_cache.joblib --cv 5 --save-metrics
"""

import argparse
import json
import io
import random
import requests
from PIL import Image
import numpy as np
import clip
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from tqdm import tqdm

try:
    from embedding_cache import load_cache, save_cache, ensure_embedding
except Exception:
    # fallback if run as module
    from .embedding_cache import load_cache, save_cache, ensure_embedding


def download_image(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert('RGB')
    except Exception as e:
        return None


def has_positive_text(txt):
    txt = (txt or '').lower()
    positives = ['cocktail','drink','glass','beverage','martini','coupe','highball','garnish','served','bar']
    return any(p in txt for p in positives)


def has_negative_text(txt):
    txt = (txt or '').lower()
    negatives = ['flower','petal','bloom','blossom','plant','floral']
    return any(n in txt for n in negatives)


def collect_candidates(data):
    positives = []
    negatives = []
    ambiguous = []
    for rec in data:
        for cand in rec.get('candidates', []):
            u = (cand.get('urls') or {}).get('regular') or (cand.get('urls') or {}).get('small')
            if not u:
                continue
            detected = bool(cand.get('detected')) if isinstance(cand, dict) else False
            alt = cand.get('alt_description') if isinstance(cand, dict) else ''
            score = cand.get('score') or 0
            if detected or has_positive_text(alt):
                positives.append((u, cand))
            else:
                if has_negative_text(alt) or (score and score < 20):
                    negatives.append((u, cand))
                else:
                    ambiguous.append((u, cand))
    return positives, negatives, ambiguous


def build_dataset_from_lists(positives, negatives, ambiguous, device, model, preprocess, cache, timeout=10, per_class_limit=None):
    # ensure balanced: take up to per_class_limit from each; if negatives too few, sample ambiguous to fill
    if per_class_limit is None:
        per_class_limit = max(len(positives), len(negatives))
    # shuffle
    random.seed(42)
    random.shuffle(positives)
    random.shuffle(negatives)
    random.shuffle(ambiguous)

    pos_sel = positives[:per_class_limit]
    neg_sel = negatives[:per_class_limit]
    if len(neg_sel) < per_class_limit:
        need = per_class_limit - len(neg_sel)
        neg_sel += ambiguous[:need]

    # cut to same size
    n = min(len(pos_sel), len(neg_sel))
    pos_sel = pos_sel[:n]
    neg_sel = neg_sel[:n]

    X = []
    y = []
    urls = []

    total = len(pos_sel) + len(neg_sel)
    for (u, cand) in tqdm(pos_sel + neg_sel, desc='encoding'):
        feats = ensure_embedding(u, model, preprocess, device, timeout=timeout, cache=cache)
        if feats is None:
            continue
        X.append(feats)
        label = 1 if (u, cand) in pos_sel else 0
        y.append(label)
        urls.append(u)

    if len(X) == 0:
        raise RuntimeError('No images encoded')
    X = np.stack(X)
    y = np.array(y)
    return X, y, urls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out-model', dest='out_model', default='clip_probe_balanced.joblib')
    parser.add_argument('--limit', type=int, default=None, help='max total samples to consider when collecting candidates')
    parser.add_argument('--per-class', type=int, default=100, help='max samples per class')
    parser.add_argument('--cache', default='embedding_cache.joblib', help='path to joblib cache for CLIP embeddings')
    parser.add_argument('--cv', type=int, default=5, help='Stratified K-folds for validation (0 to disable)')
    parser.add_argument('--save-metrics', action='store_true')
    parser.add_argument('--timeout', type=int, default=10)
    args = parser.parse_args()

    data = json.load(open(args.infile, 'r'))
    positives, negatives, ambiguous = collect_candidates(data)
    print('Collected', len(positives), 'positives,', len(negatives), 'negatives,', len(ambiguous), 'ambiguous')

    # optionally limit candidates
    if args.limit:
        total_candidates = positives[:args.limit] + negatives[:args.limit] + ambiguous[:args.limit]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device', device)

    # model and cache
    model_clip, preprocess = clip.load('ViT-B/32', device=device)
    cache = load_cache(args.cache)

    X, y, urls = build_dataset_from_lists(positives, negatives, ambiguous, device, model_clip, preprocess, cache, timeout=args.timeout, per_class_limit=args.per_class)
    print('Dataset shapes', X.shape, y.shape, 'labels distribution', np.bincount(y))

    if len(np.unique(y)) < 2:
        raise RuntimeError('Need two classes')

    # optional CV
    metrics = {}
    if args.cv and args.cv > 1:
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
        aucs = []
        for fold, (tr, va) in enumerate(skf.split(X, y), 1):
            clf_cv = LogisticRegression(max_iter=2000)
            clf_cv.fit(X[tr], y[tr])
            proba = clf_cv.predict_proba(X[va])[:, 1]
            auc = roc_auc_score(y[va], proba)
            aucs.append(float(auc))
            print(f'Fold {fold} ROC-AUC: {auc:.4f}')
        metrics['cv_auc_mean'] = float(np.mean(aucs))
        metrics['cv_auc_std'] = float(np.std(aucs))
        print('CV ROC-AUC mean=', metrics['cv_auc_mean'], 'std=', metrics['cv_auc_std'])

    # final fit
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)
    print(classification_report(y, clf.predict(X)))

    joblib.dump({'model': clf, 'device': device}, args.out_model)
    print('Saved model to', args.out_model)

    # persist cache and metrics
    save_cache(cache, args.cache)
    if args.save_metrics:
        metrics_path = args.out_model + '.metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print('Saved metrics to', metrics_path)
