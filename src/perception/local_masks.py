"""Explicit CPU color-block segmentation baseline, using RGB pixels only.

Not a replacement for SAM/DINO evaluation on general scenes. No simulator
segmentation render, object colors, names, poses or IDs are read here.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label


def predict_color_masks(capture_directory, prediction_directory):
    capture_directory, prediction_directory = Path(capture_directory), Path(prediction_directory)
    meta = json.loads((capture_directory/'capture.json').read_text(encoding='utf-8'))
    for name in meta['cameras']:
        path = capture_directory/f'{name}_rgb.png'
        image = Image.open(path).convert('RGB')
        hsv = np.asarray(image.convert('HSV'))
        hue = hsv[:, :, 0].astype(int)
        valid = (hsv[:, :, 1] > 95) & (hsv[:, :, 2] > 30)
        histogram = np.bincount(hue[valid], minlength=256)
        peaks = []
        for index in np.argsort(histogram)[::-1]:
            if histogram[index] < 20:
                break
            if all(min(abs(index-p), 256-abs(index-p)) >= 6 for p in peaks):
                peaks.append(int(index))
        masks = []
        if peaks:
            distances = np.abs(hue[:, :, None]-np.array(peaks))
            distances = np.minimum(distances, 256-distances)
            owner = distances.argmin(axis=2)
            for i in range(len(peaks)):
                parts, count = label(valid & (owner == i) & (distances[:, :, i] <= 5))
                for j in range(1, count+1):
                    mask = parts == j
                    if mask.sum() >= 25:
                        masks.append(mask)
        out = prediction_directory/name
        out.mkdir(parents=True, exist_ok=True)
        array = np.stack(masks) if masks else np.zeros((0, image.height, image.width), dtype=bool)
        np.savez_compressed(out/'masks.npz', masks=array, scores=np.ones(len(array), np.float32))
        detections = []
        for mask in masks:
            y, x = np.where(mask)
            detections.append(dict(box=[int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)],
                                   label='box', score=1.))
        (out/'detections.json').write_text(json.dumps(dict(
            rgb_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), detections=detections,
            method='rgb_color_components_baseline', scores_note='constant, not model confidence')),
            encoding='utf-8')
