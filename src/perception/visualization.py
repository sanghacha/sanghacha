"""Persistent ID-based colors for segmentation review images."""
import hashlib
import numpy as np


def instance_color(ident):
    seed=int.from_bytes(hashlib.sha256(ident.encode()).digest()[:8],'little')
    return np.random.default_rng(seed).integers(40,255,3)
