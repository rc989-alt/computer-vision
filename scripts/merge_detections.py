#!/usr/bin/env python3
"""
Merge YOLO detection outputs into precompute JSON (prefer YOLO when present).

Usage:
  python merge_detections.py --pre precomputed_..._with_detections.json --yolo detection_out_yolo_local.json --out precomputed_..._with_yolo.json
"""
import argparse
import json
import sys


def normalize_label(n):
    if not n:
        return n
    n = n.lower()
    if 'bottle' in n:
        return 'bottle'
    if 'wine' in n:
        return 'wine glass'
    if 'cup' in n or 'glass' in n or 'mug' in n:
        return 'cup'
    return n


def load_indexed_by_token(path):
    data = json.load(open(path, 'r'))
    idx = {}
    for rec in data:
        tid = rec.get('tokenId')
        idx[tid] = rec
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre', required=True, help='precompute JSON with detections (Faster R-CNN)')
    parser.add_argument('--yolo', required=True, help='YOLO detection JSON (token-aligned)')
    parser.add_argument('--out', required=True, help='output merged JSON')
    args = parser.parse_args()

    pre_idx = load_indexed_by_token(args.pre)
    yolo_idx = load_indexed_by_token(args.yolo)

    merged = []
    stats = {'tokens': 0, 'candidates_total': 0, 'candidates_yolo': 0, 'candidates_frcnn_only': 0}

    for tid, pre_rec in pre_idx.items():
        stats['tokens'] += 1
        y_rec = yolo_idx.get(tid)
        out_rec = {k: pre_rec.get(k) for k in ('tokenId','tokenQuery','cocktail')}
        out_cands = []
        pre_cands = {c.get('id'): c for c in pre_rec.get('candidates', [])}
        y_cands = {c.get('id'): c for c in (y_rec.get('candidates', []) if y_rec else [])}

        for cid, pre_c in pre_cands.items():
            stats['candidates_total'] += 1
            y_c = y_cands.get(cid)
            if y_c and y_c.get('detected'):
                # prefer YOLO detections
                dets = []
                for d in y_c.get('detected', []):
                    nm = normalize_label(d.get('name'))
                    dets.append({'name': nm, 'score': float(d.get('score', 0)), 'box': d.get('box')})
                out_cands.append({
                    'id': cid,
                    'detected': dets,
                    'score': pre_c.get('score'),
                    'urls': pre_c.get('urls'),
                    'alt_description': pre_c.get('alt_description')
                })
                stats['candidates_yolo'] += 1
            else:
                # fall back to precomputed (Faster R-CNN) detections, normalize labels
                dets = []
                for d in pre_c.get('detected', []) or []:
                    nm = normalize_label(d.get('name'))
                    dets.append({'name': nm, 'score': float(d.get('score', 0)), 'box': d.get('box')})
                out_cands.append({
                    'id': cid,
                    'detected': dets,
                    'score': pre_c.get('score'),
                    'urls': pre_c.get('urls'),
                    'alt_description': pre_c.get('alt_description')
                })
                if dets:
                    stats['candidates_frcnn_only'] += 1

        out_rec['candidates'] = out_cands
        merged.append(out_rec)

    json.dump(merged, open(args.out, 'w'), indent=2)
    print('Wrote merged JSON to', args.out)
    print('Stats:', stats)


if __name__ == '__main__':
    main()
