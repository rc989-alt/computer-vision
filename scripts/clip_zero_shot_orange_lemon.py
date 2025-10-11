#!/usr/bin/env python3
"""
Zero-shot CLIP classification: orange vs lemon for inspirations-like JSON.
Writes a compliance-like field: compliance.zshots_orange_lemon = { label, probs }

Usage:
  python clip_zero_shot_orange_lemon.py --in api/image_service/data/mined_candidates_with_yolo_YYYYMMDD.json --out api/image_service/data/mined_candidates_zs_orange_lemon_YYYYMMDD.json
"""
import argparse
import io
import json
import requests
import torch
import clip  # type: ignore
from PIL import Image
from tqdm import tqdm


PROMPTS = [
  'a cocktail with an orange slice garnish',
  'a cocktail with a lemon slice garnish',
]
CLASSES = ['orange', 'lemon']


def download_image(url: str, timeout: int = 10):
  try:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert('RGB')
  except Exception as e:  # noqa: BLE001
    print('download failed', url, e)
    return None


def url_of(c):
  u = (c.get('urls') or {})
  return u.get('regular') or u.get('small') or u.get('full') or u.get('raw') or ''


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--in', dest='infile', required=True)
  ap.add_argument('--out', dest='outfile', required=True)
  ap.add_argument('--clip-model', default='ViT-B/32')
  args = ap.parse_args()

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model, preprocess = clip.load(args.clip_model, device=device)
  with open(args.infile, 'r') as f:
    data = json.load(f)

  with torch.no_grad():
    text_tokens = clip.tokenize(PROMPTS).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

  out = []
  for rec in tqdm(data, desc='zero-shot orange/lemon'):
    nrec = {**rec}
    ninsp = []
    for insp in (rec.get('inspirations') or []):
      ni = {**insp}
      ncands = []
      for c in (insp.get('candidates') or []):
        url = url_of(c)
        comp = dict(c.get('compliance') or {})
        label = None
        probs_map = {}
        if url:
          img = download_image(url)
          if img is not None:
            try:
              inp = preprocess(img).unsqueeze(0).to(device)
              with torch.no_grad():
                image_features = model.encode_image(inp)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = image_features @ text_features.T
                probs = logits.softmax(dim=-1)[0].cpu().numpy().tolist()
                best = int(max(range(len(probs)), key=lambda i: probs[i]))
                label = CLASSES[best]
                probs_map = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
            except Exception as e:  # noqa: BLE001
              print('clip failed', e)
        comp['zshots_orange_lemon'] = {'label': label, 'probs': probs_map}
        ncands.append({**c, 'compliance': comp})
      ni['candidates'] = ncands
      ninsp.append(ni)
    nrec['inspirations'] = ninsp
    out.append(nrec)

  with open(args.outfile, 'w') as f:
    json.dump(out, f, indent=2)
  print('Wrote', len(out), 'records to', args.outfile)


if __name__ == '__main__':
  main()
