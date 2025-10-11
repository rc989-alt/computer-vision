#!/usr/bin/env python3
"""
Generic zero-shot CLIP classifier for inspirations-like JSON.
Writes a compliance-like field under compliance[family] = { label, probs }.

Usage:
  python clip_zero_shot_generic.py \
    --in api/.../inspirations.json \
    --out api/.../inspirations_zs_attr.json \
    --family garnish_citrus_form \
    --classes orange: "a cocktail with an orange slice garnish" \
              lemon: "a cocktail with a lemon slice garnish" \
    [--clip-model ViT-B/32]
"""
import argparse
import io
import json
import requests
import torch
import clip  # type: ignore
from PIL import Image
from tqdm import tqdm


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
    ap.add_argument('--family', required=True)
    ap.add_argument('--clip-model', default='ViT-B/32')
    # Provide classes as repeated --class name:prompt
    ap.add_argument('--class', dest='classes', action='append', required=True,
                    help='Repeated option: --class name:prompt')
    args = ap.parse_args()

    # Parse classes
    names = []
    prompts = []
    for item in args.classes:
        if ':' not in item:
            raise SystemExit('Each --class must be name:prompt')
        name, prompt = item.split(':', 1)
        name = name.strip()
        prompt = prompt.strip()
        if not name or not prompt:
            raise SystemExit('Empty class name or prompt')
        names.append(name)
        prompts.append(prompt)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load(args.clip_model, device=device)
    with open(args.infile, 'r') as f:
        data = json.load(f)

    with torch.no_grad():
        text_tokens = clip.tokenize(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    out = []
    for rec in tqdm(data, desc=f'zero-shot {args.family}'):
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
                                label = names[best]
                                probs_map = {names[i]: float(probs[i]) for i in range(len(names))}
                        except Exception as e:  # noqa: BLE001
                            print('clip failed', e)
                comp[args.family] = {'label': label, 'probs': probs_map}
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
