#!/usr/bin/env python3
"""
Lightweight YOLO wrapper: load a yolov8 model and run detection per-candidate URL.
Writes a detection-annotated JSON similar to the Faster R-CNN outputs used earlier.

Usage:
  python yolo_detector.py --in ../precomputed_merged_run2.json --out ../detection_out_yolo.json --model yolov8n.pt --conf 0.35
"""

import argparse
import json
from ultralytics import YOLO
import torch


def map_cls_to_name(name):
    n = name.lower()
    if 'bottle' in n:
        return 'bottle'
    if 'wine' in n:
        return 'wine glass'
    if 'cup' in n or 'mug' in n or 'glass' in n:
        return 'cup'
    return name


def run(in_path, out_path, model_path='yolov8n.pt', conf=0.35, imgsz=640, limit=None, device=None):
    model = YOLO(model_path)
    data = json.load(open(in_path, 'r'))
    out = []
    count = 0
    for rec in data:
        token_out = {k: rec.get(k) for k in ('tokenId','tokenQuery','cocktail')}
        token_out['candidates'] = []
        for cand in rec.get('candidates', []):
            u = (cand.get('urls') or {}).get('regular') or (cand.get('urls') or {}).get('small')
            if not u:
                u = (rec.get('unsplash') or {}).get('urls', {}).get('regular') or (rec.get('unsplash') or {}).get('urls', {}).get('small')
            cand_out = {'id': cand.get('id'), 'score': cand.get('score'), 'urls': cand.get('urls'), 'alt_description': cand.get('alt_description')}
            if not u:
                cand_out['detected'] = []
                token_out['candidates'].append(cand_out)
                continue
            try:
                results = model.predict(source=u, imgsz=imgsz, conf=conf)
                if len(results) == 0:
                    cand_out['detected'] = []
                else:
                    r = results[0]
                    dets = []
                    boxes = getattr(r, 'boxes', None)
                    if boxes is None or len(boxes) == 0:
                        cand_out['detected'] = []
                    else:
                        xyxy = boxes.xyxy.cpu().numpy().tolist()
                        confs = boxes.conf.cpu().numpy().tolist()
                        cls = boxes.cls.cpu().numpy().tolist()
                        names = [model.model.names[int(c)] if hasattr(model, 'model') and hasattr(model.model, 'names') else str(int(c)) for c in cls]
                        for b, s, n in zip(xyxy, confs, names):
                            mapped = map_cls_to_name(n)
                            if mapped in ('bottle','cup','wine glass'):
                                dets.append({'name': mapped, 'score': float(s), 'box': [float(b[0]), float(b[1]), float(b[2]), float(b[3])]})
                        cand_out['detected'] = dets
            except Exception as e:
                cand_out['detected'] = []
                cand_out['error'] = str(e)
            token_out['candidates'].append(cand_out)
            count += 1
            if limit and count >= limit:
                break
        out.append(token_out)
        if limit and count >= limit:
            break
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('Wrote', len(out), 'tokens to', out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out', dest='outfile', required=True)
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--conf', type=float, default=0.35)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    run(args.infile, args.outfile, model_path=args.model, conf=args.conf, imgsz=args.imgsz, limit=args.limit, device=args.device)


if __name__ == '__main__':
    main()
