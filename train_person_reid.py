#!/usr/bin/env python3
"""
train_person_reid.py -- Train a body-based person classifier.

Reads body crops from PERSON_TRAINING_DIR/<name>/*.jpg, embeds them via
OSNet (ONNX, CPU), applies augmentation, and saves a centroid+gallery
classifier to PERSON_CLASSIFIER_PATH.

Usage:
    python3 train_person_reid.py
"""

import os
import sys
import cv2
import pickle
import numpy as np

sys.path.append('/home/medpal/pyorbbecsdk_v1/build')
import config

try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime not installed (pip install onnxruntime)")
    sys.exit(1)


class BodyEmbedder:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Body ReID model not found: {model_path}")
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.input_h = 256
        self.input_w = 128
        if len(shape) == 4:
            if isinstance(shape[2], int) and shape[2] > 0:
                self.input_h = shape[2]
            if isinstance(shape[3], int) and shape[3] > 0:
                self.input_w = shape[3]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        print(f"BodyEmbedder: input {self.input_w}x{self.input_h}")

    def embed(self, crop):
        if crop is None or crop.size == 0:
            return None
        img = cv2.resize(crop, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0).astype(np.float32)
        out = self.session.run(None, {self.input_name: img})[0]
        vec = out.flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


def _augment(img):
    """Body crop augmentations (10 variants per crop)."""
    variants = [img]
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # Horizontal flip
    variants.append(cv2.flip(img, 1))

    # Brightness +/- 20
    for delta in (-20, 20):
        variants.append(np.clip(img.astype(np.int16) + delta,
                                0, 255).astype(np.uint8))

    # Small rotations +/- 8 degrees
    for angle in (-8, 8):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        variants.append(cv2.warpAffine(img, M, (w, h),
                                       borderMode=cv2.BORDER_REFLECT_101))

    # Zoom 0.92 and 1.08
    for scale in (0.92, 1.08):
        M = cv2.getRotationMatrix2D(center, 0, scale)
        variants.append(cv2.warpAffine(img, M, (w, h),
                                       borderMode=cv2.BORDER_REFLECT_101))

    # Distance simulation: down to 48x48 then back
    small = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
    variants.append(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC))

    return variants


def main():
    train_dir = config.PERSON_TRAINING_DIR
    if not os.path.isdir(train_dir):
        print(f"No training directory at {train_dir}")
        print(f"Collect body crops first, e.g.:")
        print(f"  python3 person_collect.py --name Carl "
              f"--output {train_dir}/Carl")
        return

    people = sorted([d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d))])
    if not people:
        print(f"No person folders in {train_dir}")
        return

    print(f"Found {len(people)} people: {people}")
    embedder = BodyEmbedder(config.REID_PATH)

    names = []
    centroids = []
    galleries = []
    stats_p10 = []
    stats_p50 = []
    stats_p90 = []

    for person in people:
        pdir = os.path.join(train_dir, person)
        files = sorted([f for f in os.listdir(pdir)
                        if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        if not files:
            print(f"  {person}: no crops, skipping")
            continue

        print(f"  {person}: processing {len(files)} crops...")
        embs = []
        for fname in files:
            img = cv2.imread(os.path.join(pdir, fname))
            if img is None:
                continue
            for var in _augment(img):
                e = embedder.embed(var)
                if e is not None:
                    embs.append(e)

        if len(embs) < 2:
            print(f"    -> only {len(embs)} embeddings, skipping")
            continue

        X = np.stack(embs)
        mu = X.mean(axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-9)

        sims = X @ mu
        p10, p50, p90 = np.percentile(sims, [10, 50, 90])

        names.append(person)
        centroids.append(mu)
        galleries.append(X)
        stats_p10.append(float(p10))
        stats_p50.append(float(p50))
        stats_p90.append(float(p90))

        variants_per = len(embs) // len(files)
        print(f"    -> {len(embs)} embeddings "
              f"({len(files)} crops x {variants_per} variants)")
        print(f"       centroid sim: p10={p10:.3f} p50={p50:.3f} p90={p90:.3f}")

    if not names:
        print("No valid training data.")
        return

    os.makedirs(os.path.dirname(config.PERSON_CLASSIFIER_PATH), exist_ok=True)
    with open(config.PERSON_CLASSIFIER_PATH, "wb") as f:
        pickle.dump({
            "model_type": "body_centroid_gallery",
            "names": names,
            "centroids": np.stack(centroids),
            "galleries": galleries,
            "stats_p10": np.array(stats_p10, dtype=np.float32),
            "stats_p50": np.array(stats_p50, dtype=np.float32),
            "stats_p90": np.array(stats_p90, dtype=np.float32),
        }, f)

    print(f"\nSaved person classifier to {config.PERSON_CLASSIFIER_PATH}")
    print(f"People: {names}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nFATAL: {e}\n")
        sys.exit(1)