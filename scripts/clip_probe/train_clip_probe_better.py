#!/usr/bin/env python3
"""
Train a CLIP linear probe using medoid/cluster outputs and improved weak labeling heuristics.

Labels logic (weak):
 - Positive if candidate has detection (non-empty `detected`) OR alt_description contains positive cues (cocktail, drink, glass, martini, coupe, highball, garnish, beverage) OR candidate.medoid is true and any cluster rep had detection.
 - Negative if alt_description suggests floral-only (flower, petal, bloom, leaf, plant) and no detections and no positive cues.

Usage:
    python train_clip_probe_better.py --in ../data/precomputed_merged_run2_with_yolo_medoid.json --out-model clip_probe_better.joblib --limit 2000
"""

import argparse
import json
import io
import requests
from PIL import Image
import numpy as np
import clip
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from tqdm import tqdm


def download_image(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert('RGB')
    except Exception as e:
        print(f"download failed {url}: {e}")
        return None


def has_positive_text(txt):
    txt = (txt or '').lower()
    positives = ['cocktail','drink','glass','beverage','martini','coupe','highball','garnish','served','bar']
    return any(p in txt for p in positives)


def has_negative_text(txt):
    txt = (txt or '').lower()
    negatives = ['flower','petal','bloom','blossom','plant','floral']
    return any(n in txt for n in negatives)


def build_dataset(in_path, device, limit=None):
    data = json.load(open(in_path, 'r'))
    model, preprocess = clip.load('ViT-B/32', device=device)
    X = []
    y = []
    urls = []
    count = 0
    for rec in data:
        for cand in rec.get('candidates', []):
            # prefer candidate.urls.regular then small
            u = None
            if isinstance(cand, dict):
                u = (cand.get('urls') or {}).get('regular') or (cand.get('urls') or {}).get('small')
            if not u:
                u = (rec.get('unsplash') or {}).get('urls', {}).get('regular') or (rec.get('unsplash') or {}).get('urls', {}).get('small')
            if not u:
                continue
            img = download_image(u)
            if img is None:
                continue
            try:
                img_inp = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feats = model.encode_image(img_inp)
                feats = feats.cpu().numpy().reshape(-1)
            except Exception as e:
                print(f"CLIP encoding failed for {u}: {e}")
                continue

            # decide label using heuristics (avoid treating medoid-only as positive)
            detected = (isinstance(cand, dict) and cand.get('detected') and len(cand.get('detected'))>0)
            alt = cand.get('alt_description') if isinstance(cand, dict) else None
            pos_text = has_positive_text(alt)
            neg_text = has_negative_text(alt)

            label = None
            if detected or pos_text:
                label = 1
            elif neg_text and not detected:
                label = 0
            else:
                # ambiguous — skip to avoid noisy labels
                continue

            X.append(feats)
            y.append(label)
            urls.append(u)
            count += 1
            if limit and count >= limit:
                break
        if limit and count >= limit:
            break

    if len(X) == 0:
        raise RuntimeError('No training data found with these heuristics')
    X = np.stack(X)
    y = np.array(y)
    return X, y, urls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out-model', dest='out_model', default='clip_probe_better.joblib')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device', device)
    X, y, urls = build_dataset(args.infile, device, limit=args.limit)
    print('Dataset:', X.shape, 'labels distribution:', np.bincount(y))

    # simple train/test
    if len(np.unique(y)) < 2:
        raise RuntimeError('Need at least two label classes for training; adjust heuristics or provide more data')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    joblib.dump({'model': clf, 'urls_test': urls, 'device': device}, args.out_model)
    print('Saved model to', args.out_model)
