#!/usr/bin/env python3
"""
Score candidates in a precomputed JSON using a saved CLIP probe model.

Writes two outputs:
 - <out_prefix>_scored.json  : original structure with candidate['clip_probe_score'] added
 - <out_prefix>_filtered.json: same structure but candidates filtered by threshold or top_k per token

Example:
 python run_clip_probe_inference.py --in ../data/precomputed_merged_run2_with_yolo.json --model ../clip_probe/clip_probe_balanced.joblib --out-prefix ../data/precomputed_run2_with_scores --threshold 0.5 --top-k 3
"""
import argparse
import json
import io
import requests
from PIL import Image
import numpy as np
import clip
import torch
import joblib
from tqdm import tqdm

try:
    from embedding_cache import load_cache, save_cache, ensure_embedding
except Exception:
    from .embedding_cache import load_cache, save_cache, ensure_embedding


def download_image(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert('RGB')
    except Exception:
        return None


def encode_image_from_url(url, model, preprocess, device, cache, timeout=10):
    feats = ensure_embedding(url, model, preprocess, device, timeout=timeout, cache=cache)
    return feats


def _normalize(v):
    n = np.linalg.norm(v) + 1e-12
    return v / n


def build_text_prompt(rec, template=None):
    """Build a text prompt from record.cocktail using a format template."""
    cocktail = rec.get('cocktail') or {}
    name = cocktail.get('name') or ''
    visual = cocktail.get('visual') or {}
    color = visual.get('color') or ''
    glass = visual.get('glass') or visual.get('cup') or ''
    garnish = visual.get('garnish') or ''
    if not template:
        template = 'a photo of a {color} cocktail drink in a {glass} with {garnish}, bar setting, high quality, no flowers'
    return template.format(name=name, color=color, glass=glass, garnish=garnish).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--model', dest='model', required=True)
    parser.add_argument('--out-prefix', dest='out_prefix', required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--top-k', type=int, default=3, help='keep top-k candidates per token after thresholding')
    parser.add_argument('--cache', default='embedding_cache.joblib')
    parser.add_argument('--timeout', type=int, default=10)
    # Optional CLIP text guidance
    parser.add_argument('--use-text', action='store_true', help='blend CLIP text-image similarity with probe score')
    parser.add_argument('--text-weight', type=float, default=0.4, help='weight for text-image similarity when blending')
    parser.add_argument('--probe-weight', type=float, default=0.6, help='weight for probe score when blending')
    parser.add_argument('--text-template', type=str, default=None, help='Python format string using {name},{color},{glass},{garnish}')
    parser.add_argument('--neg-prompts', type=str, default='flower,petal,blossom,bloom', help='comma-separated negative prompts to penalize')
    parser.add_argument('--neg-weight', type=float, default=0.2, help='weight for negative prompt similarity to subtract')
    # Detection-aware gating/weighting
    parser.add_argument('--require-detection', action='store_true', help='drop candidates without target detections')
    parser.add_argument('--det-classes', type=str, default='cup,wine glass,glass,bottle', help='comma-separated target classes')
    parser.add_argument('--det-min', type=float, default=0.2, help='minimum detection score to count')
    parser.add_argument('--det-weight', type=float, default=0.2, help='weight to add max detection score to final')
    # Alt-text negative penalty
    parser.add_argument('--alt-neg-terms', type=str, default='flower,petal,blossom,bloom,botanical', help='comma-separated negative words to penalize in alt text')
    parser.add_argument('--alt-neg-weight', type=float, default=0.15, help='penalty weight if alt text contains any negative term')
    args = parser.parse_args()

    data = json.load(open(args.infile, 'r'))
    probe_pack = joblib.load(args.model)
    clf = probe_pack.get('model') if isinstance(probe_pack, dict) else probe_pack

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device', device)
    model_clip, preprocess = clip.load('ViT-B/32', device=device)
    cache = load_cache(args.cache)
    text_ctx = None

    # iterate tokens and candidates
    for rec in tqdm(data, desc='tokens'):
        cands = rec.get('candidates', [])
        scores = []
        text_sim = None
        neg_sims = None
        if args.use_text:
            prompt = build_text_prompt(rec, template=args.text_template)
            texts = [prompt]
            negs = [t.strip() for t in (args.neg_prompts or '').split(',') if t.strip()]
            if negs:
                texts.extend(negs)
            with torch.no_grad():
                text_tokens = clip.tokenize(texts).to(device)
                text_feats = model_clip.encode_text(text_tokens).cpu().numpy()
                text_feats = np.stack([_normalize(x) for x in text_feats])
            text_sim = text_feats[0]
            neg_sims = text_feats[1:] if len(texts) > 1 else None
        # Prepare detection class set and neg terms
        det_targets = {c.strip().lower() for c in (args.det_classes or '').split(',') if c.strip()}
        neg_terms = [t.strip().lower() for t in (args.alt_neg_terms or '').split(',') if t.strip()]

        for cand in cands:
            u = (cand.get('urls') or {}).get('regular') or (cand.get('urls') or {}).get('small')
            if not u:
                scores.append(None)
                continue
            feats = encode_image_from_url(u, model_clip, preprocess, device, cache, timeout=args.timeout)
            if feats is None:
                scores.append(None)
                continue
            # predict_proba expects 2D
            try:
                proba = clf.predict_proba(feats.reshape(1, -1))[0][1]
            except Exception:
                # some saved objects might be sklearn wrappers
                proba = float(clf.predict(feats.reshape(1, -1))[0])
            base_score = float(proba)
            if args.use_text and text_sim is not None:
                v = _normalize(feats.reshape(-1))
                sim_pos = float(np.dot(v, text_sim))  # cosine since vectors normalized
                sim_neg = 0.0
                if neg_sims is not None and len(neg_sims) > 0:
                    sim_neg = float(np.max(neg_sims @ v))  # max similarity to negative prompts
                blended = args.probe_weight * base_score + args.text_weight * sim_pos - args.neg_weight * sim_neg
                final_score = blended
                cand['clip_text_sim'] = sim_pos
                cand['clip_text_neg_sim'] = sim_neg
                cand['clip_probe_score'] = base_score
            else:
                final_score = base_score

            # Detection-aware gating/weighting
            detected = cand.get('detected') or []
            det_max = 0.0
            for d in detected:
                name = str(d.get('name') or '').lower()
                sc = float(d.get('score', 0.0))
                if name in det_targets and sc >= args.det_min:
                    det_max = max(det_max, sc)
            if args.require_detection and det_max <= 0.0:
                scores.append(None)
                continue
            if det_max > 0.0 and args.det_weight > 0:
                final_score += args.det_weight * det_max

            # Alt-text penalty
            alt = (cand.get('alt_description') or '')
            alt_l = alt.lower()
            if any(term in alt_l for term in neg_terms):
                final_score -= args.alt_neg_weight

            scores.append(final_score)

        # attach scores
        for cand, s in zip(cands, scores):
            cand['clip_probe_score'] = s

        # filter: only keep candidates above threshold, then top-k by score (None scores go last)
        # Threshold applies to final score; ensure value is present
        for cand, s in zip(cands, scores):
            cand['clip_final_score'] = s
        filtered = [c for c in cands if (c.get('clip_final_score') is not None and c.get('clip_final_score') >= args.threshold)]
        filtered.sort(key=lambda x: x.get('clip_final_score', 0), reverse=True)
        rec['candidates_filtered'] = filtered[:args.top_k]

    out_scored = args.out_prefix + '_scored.json'
    out_filtered = args.out_prefix + '_filtered.json'
    json.dump(data, open(out_scored, 'w'), indent=2)
    json.dump(data, open(out_filtered, 'w'), indent=2)
    save_cache(cache, args.cache)
    print('Wrote', out_scored, 'and', out_filtered)


if __name__ == '__main__':
    main()
