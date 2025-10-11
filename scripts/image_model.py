"""
Simple image scoring pipeline:
- Accepts a JSON list of tokens/candidate image URLs (from precompute outputs)
- For each candidate it downloads the 'regular' url, runs a pretrained torchvision detector
  (Faster R-CNN) and checks for presence of relevant classes (cup, wine glass, bottle)
- Outputs a scored JSON with detections and confidence so the precompute script can prefer
  images that actually contain glass/drink objects.

Usage:
  python image_model.py --in ./precomputed_cocktails_run7.json --out ./precomputed_with_detections.json

Note: this script is for local use and requires a machine with PyTorch installed. On macOS CPU-only
it will be slower but should run for small batches.
"""

import argparse
import json
import os
import io
import requests
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm

COCO_GLASS_CLASSES = set([44, 47, 39])
# COCO ids: 44 = bottle, 47 = cup, 39 = wine glass (verify mapping depending on model)


def load_model(device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(device)
    return model


def download_image(url, timeout=10):
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert('RGB')
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None


def detect_image(model, device, pil_image, threshold=0.5):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_t = transform(pil_image).to(device)
    with torch.no_grad():
        outputs = model([img_t])
    out = outputs[0]
    detections = []
    scores = out['scores'].cpu().numpy()
    labels = out['labels'].cpu().numpy()
    boxes = out['boxes'].cpu().numpy()
    for i, score in enumerate(scores):
        if score >= threshold:
            label = int(labels[i])
            box = boxes[i].tolist()  # [x1, y1, x2, y2]
            detections.append({'label': label, 'score': float(score), 'box': [float(box[0]), float(box[1]), float(box[2]), float(box[3])]})
    return detections


def map_label_to_name(label):
    # Minimal COCO mapping for common drink-related objects
    mapping = {39: 'wine glass', 44: 'bottle', 47: 'cup'}
    return mapping.get(label, str(label))


def process_file(in_path, out_path, device, limit=None):
    data = json.load(open(in_path, 'r'))
    out = []
    count = 0
    model = load_model(device)
    for record in tqdm(data, desc='tokens'):
        token_out = {k: record.get(k) for k in ('tokenId','tokenQuery','cocktail')}
        token_out['candidates'] = []
        for cand in record.get('candidates', []):
            if 'urls' not in cand or 'regular' not in cand['urls']:
                continue
            img = download_image(cand['urls']['regular'])
            if img is None:
                cand_out = {'id': cand.get('id'), 'detected': [], 'score': cand.get('score'), 'urls': cand.get('urls'), 'alt_description': cand.get('alt_description')}
            else:
                detections = detect_image(model, device, img, threshold=0.4)
                # keep only drink-related labels
                drink_dets = [d for d in detections if d['label'] in COCO_GLASS_CLASSES]
                cand_out = {
                    'id': cand.get('id'),
                    'detected': [{'name': map_label_to_name(d['label']), 'score': d['score'], 'box': d.get('box')} for d in drink_dets],
                    'score': cand.get('score'),
                    'urls': cand.get('urls'),
                    'alt_description': cand.get('alt_description')
                }
            token_out['candidates'].append(cand_out)
        out.append(token_out)
        count += 1
        if limit and count >= limit:
            break
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(out)} token entries to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out', dest='outfile', default='./precomputed_with_detections.json')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    process_file(args.infile, args.outfile, device, limit=args.limit)


if __name__ == '__main__':
    main()
