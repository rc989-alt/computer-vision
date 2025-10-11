#!/usr/bin/env python3
"""
Download candidate images from a precomputed JSON and run YOLO on the downloaded folder.

Usage:
  python download_and_run_yolo.py --in ../precomputed_merged_run2.json --out ../detection_out_yolo_local.json --limit 200

This avoids URL-as-stream warnings and reduces network stalls by downloading images first.
"""

import argparse
import json
import os
import re
import requests
import io
from PIL import Image
from ultralytics import YOLO
import traceback


def safe_name(s):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', s)[:180]


def download(url, path, timeout=15):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
        return True
    except Exception as e:
        print('download failed', url, e)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out', dest='outfile', required=True)
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--dir', default='/tmp/cocktail_images')
    parser.add_argument('--model', default='yolov8n.pt')
    parser.add_argument('--conf', type=float, default=0.35)
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    data = json.load(open(args.infile, 'r'))
    downloaded = []
    count = 0
    for rec in data:
        token = rec.get('tokenId') or 'tok'
        for cand in rec.get('candidates', []):
            if count >= args.limit:
                break
            u = (cand.get('urls') or {}).get('regular') or (cand.get('urls') or {}).get('small')
            if not u:
                u = (rec.get('unsplash') or {}).get('urls', {}).get('regular') or (rec.get('unsplash') or {}).get('urls', {}).get('small')
            if not u:
                continue
            name = safe_name(f"{token}__{cand.get('id')}")
            ext = os.path.splitext(u.split('?')[0])[1]
            if ext.lower() not in ('.jpg','.jpeg','.png','.webp'):
                ext = '.jpg'
            path = os.path.join(args.dir, name + ext)
            if os.path.exists(path):
                downloaded.append((path, token, cand))
                count += 1
                continue
            ok = download(u, path)
            if ok:
                downloaded.append((path, token, cand))
                count += 1
    print('Downloaded', len(downloaded), 'images to', args.dir)

    if len(downloaded) == 0:
        print('No images downloaded; exiting')
        return

    # run YOLO on directory
    model = YOLO(args.model)
    print('Running YOLO on folder', args.dir)
    try:
        results = model.predict(source=args.dir, imgsz=640, conf=args.conf)
    except Exception:
        traceback.print_exc()
        return

    # results correspond to processed files in order; map filenames to detections
    out_map = {}
    for r in results:
        try:
            path = r.orig_img_path if hasattr(r, 'orig_img_path') else None
            if not path:
                # try r.path
                path = getattr(r, 'path', None)
            if not path:
                continue
            boxes = getattr(r, 'boxes', None)
            dets = []
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy().tolist()
                confs = boxes.conf.cpu().numpy().tolist()
                cls = boxes.cls.cpu().numpy().tolist()
                names = [model.model.names[int(c)] if hasattr(model, 'model') and hasattr(model.model, 'names') else str(int(c)) for c in cls]
                for b, s, n in zip(xyxy, confs, names):
                    dets.append({'name': n, 'score': float(s), 'box': [float(b[0]), float(b[1]), float(b[2]), float(b[3])], 'file': path})
            out_map[path] = dets
        except Exception:
            traceback.print_exc()
            continue

    # now produce token-aligned output similar to previous detector outputs
    out = []
    for rec in data:
        token_out = {k: rec.get(k) for k in ('tokenId','tokenQuery','cocktail')}
        token_out['candidates'] = []
        for cand in rec.get('candidates', []):
            u = (cand.get('urls') or {}).get('regular') or (cand.get('urls') or {}).get('small')
            if not u:
                u = (rec.get('unsplash') or {}).get('urls', {}).get('regular') or (rec.get('unsplash') or {}).get('urls', {}).get('small')
            if not u:
                token_out['candidates'].append({'id': cand.get('id'), 'detected': [], 'score': cand.get('score'), 'urls': cand.get('urls'), 'alt_description': cand.get('alt_description')})
                continue
            name = safe_name(f"{rec.get('tokenId','tok')}__{cand.get('id')}")
            # find file path with any supported ext
            found = None
            for ext in ('.jpg','.jpeg','.png','.webp'):
                p = os.path.join(args.dir, name + ext)
                if os.path.exists(p):
                    found = p; break
            if not found:
                token_out['candidates'].append({'id': cand.get('id'), 'detected': [], 'score': cand.get('score'), 'urls': cand.get('urls'), 'alt_description': cand.get('alt_description')})
                continue
            dets = out_map.get(found, [])
            # map names to simplified labels
            mapped = []
            for d in dets:
                n = d['name'].lower()
                if 'bottle' in n:
                    nm = 'bottle'
                elif 'wine' in n:
                    nm = 'wine glass'
                elif 'cup' in n or 'glass' in n or 'mug' in n:
                    nm = 'cup'
                else:
                    nm = d['name']
                mapped.append({'name': nm, 'score': d['score'], 'box': d['box']})
            token_out['candidates'].append({'id': cand.get('id'), 'detected': mapped, 'score': cand.get('score'), 'urls': cand.get('urls'), 'alt_description': cand.get('alt_description')})
        out.append(token_out)
    with open(args.outfile, 'w') as f:
        json.dump(out, f, indent=2)
    print('Wrote detection results to', args.outfile)


if __name__ == '__main__':
    main()
