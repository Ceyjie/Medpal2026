#!/usr/bin/env python3
"""
train_face_svm.py -- Build a centroid+gallery classifier from aligned
112x112 crops using ArcFace embeddings.

Reads FACE_TRAINING_DIR/<name>/*.jpg (already aligned), embeds each
with ArcFace, computes centroid + within-class percentiles, saves a
pickle at config.SVM_MODEL_PATH.

Run standalone or spawned by the web button.
"""

import os
import sys

# Limit BLAS/OpenMP threads so training doesn't starve the tracker
# when spawned from the web button.
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
cv2.setNumThreads(3)

import pickle
import numpy as np

import config
from arcface_embedder import ArcFaceEmbedder


def _augment(img):
    """Return 10 light variants of a 112x112 aligned face."""
    variants = [img]
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    variants.append(cv2.flip(img, 1))

    for delta in (-20, 20):
        variants.append(
            np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        )

    for angle in (-10, 10):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        variants.append(cv2.warpAffine(img, M, (w, h),
                                       borderMode=cv2.BORDER_REFLECT_101))

    for scale in (0.92, 1.08):
        M = cv2.getRotationMatrix2D(center, 0, scale)
        variants.append(cv2.warpAffine(img, M, (w, h),
                                       borderMode=cv2.BORDER_REFLECT_101))

    small = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
    variants.append(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC))

    variants.append(cv2.GaussianBlur(img, (5, 5), 1.2))

    return variants


def main():
    train_dir = config.FACE_TRAINING_DIR
    if not os.path.isdir(train_dir):
        print(f"No training dir at {train_dir}")
        return

    people = sorted(d for d in os.listdir(train_dir)
                    if os.path.isdir(os.path.join(train_dir, d)))
    if not people:
        print("No enrolled people.")
        return

    print(f"Found {len(people)}: {people}")

    # 3 threads for training so the tracker stays responsive
    embedder = ArcFaceEmbedder(config.ARCFACE_MODEL_PATH, num_threads=3)

    names = []
    centroids = []
    galleries = []
    p10s = []
    p50s = []
    p90s = []

    for person in people:
        pdir = os.path.join(train_dir, person)
        files = sorted(f for f in os.listdir(pdir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if not files:
            print(f"  {person}: no crops, skipping")
            continue

        print(f"  {person}: {len(files)} crops (x10 augment) ...")
        embs = []
        for fname in files:
            img = cv2.imread(os.path.join(pdir, fname))
            if img is None:
                continue
            # Crops are saved aligned. Just embed directly.
            for var in _augment(img):
                v = embedder.embed(var)
                if v is not None:
                    embs.append(v)

        if len(embs) < 10:
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
        p10s.append(float(p10))
        p50s.append(float(p50))
        p90s.append(float(p90))

        variants_per_crop = len(embs) // len(files) if files else 0
        print(f"    -> {len(embs)} embeddings "
              f"({len(files)} crops x {variants_per_crop} variants)")
        print(f"       centroid sim: p10={p10:.3f} p50={p50:.3f} p90={p90:.3f}")

    if not names:
        print("No valid training data.")
        return

    out = {
        "model_type": "centroid_gallery",
        "names": names,
        "centroids": np.stack(centroids).astype(np.float32),
        "galleries": [g.astype(np.float32) for g in galleries],
        "stats_p10": np.array(p10s, dtype=np.float32),
        "stats_p50": np.array(p50s, dtype=np.float32),
        "stats_p90": np.array(p90s, dtype=np.float32),
    }
    os.makedirs(os.path.dirname(config.SVM_MODEL_PATH), exist_ok=True)
    with open(config.SVM_MODEL_PATH, "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved: {config.SVM_MODEL_PATH}")
    print(f"Enrolled: {names}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nFATAL: {e}\n")
        sys.exit(1)
