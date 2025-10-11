#!/usr/bin/env python3
"""
Utility for caching CLIP image embeddings by URL (or local path).

Cache format: joblib dict { key -> np.ndarray(float32, 512) }
  - key is the original URL or a local file path string.
  - Embedding vectors are stored as float32 to reduce size.

Public API:
  - load_cache(path) -> dict
  - save_cache(cache, path)
  - ensure_embedding(key, model, preprocess, device, *, timeout=10, cache=None) -> np.ndarray | None

Notes:
  - For URLs, uses requests to download; for local files, opens via PIL.Image.
  - Returns None on failure; callers may choose to skip such samples.
"""
import io
import os
from typing import Dict, Optional
import requests
import joblib
import numpy as np
from PIL import Image
import torch


def load_cache(path: str) -> Dict[str, np.ndarray]:
    try:
        return joblib.load(path)
    except Exception:
        return {}


def save_cache(cache: Dict[str, np.ndarray], path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    # ensure float32 for size
    to_save = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) else v) for k, v in cache.items()}
    joblib.dump(to_save, path)


def _open_image(key: str, timeout: int = 10) -> Optional[Image.Image]:
    # If key points to a local file, open from disk
    if os.path.exists(key):
        try:
            return Image.open(key).convert('RGB')
        except Exception:
            return None
    # Otherwise treat as URL
    try:
        r = requests.get(key, timeout=timeout)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert('RGB')
    except Exception:
        return None


def ensure_embedding(key: str, model, preprocess, device: str, *, timeout: int = 10, cache: Optional[Dict[str, np.ndarray]] = None) -> Optional[np.ndarray]:
    if cache is not None and key in cache:
        return cache[key]
    img = _open_image(key, timeout=timeout)
    if img is None:
        return None
    try:
        inp = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = model.encode_image(inp).cpu().numpy().reshape(-1).astype(np.float32)
        if cache is not None:
            cache[key] = feats
        return feats
    except Exception:
        return None
