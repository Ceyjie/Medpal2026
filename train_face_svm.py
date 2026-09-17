#!/usr/bin/env python3
"""
train_face_svm.py -- Train a centroid + gallery face classifier.

Per enrolled person:
  - Load each crop
  - Generate 11 variants per crop via _augment() (flip, brightness, rotation,
    zoom, distance simulation, blur)
  - Embed all variants to build a dense gallery
  - Compute centroid and within-class similarity statistics

Saved classifier contains:
  - centroid: mean of all embeddings, re-normalized
  - gallery:  every embedding (original + augmented)
  - stats:    p10/p50/p90 of within-class similarity to centroid
"""

import os
import sys
import cv2
import pickle
import numpy as np

sys.path.append('/home/medpal/pyorbbecsdk_v1/build')

import config

CORAL_EDGETPU = False
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    try:
        delegate = load_delegate('libedgetpu.so.1')
        CORAL_EDGETPU = True
        print("Coral Edge TPU delegate loaded.")
    except Exception:
        print("Coral Edge TPU unavailable, using CPU.")
except ImportError:
    print("tflite_runtime not installed.")
    sys.exit(1)


# ============================================================
# FaceEmbedder -- MUST match the tracker's preprocessing exactly.
# ============================================================
class FaceEmbedder:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Embedding model not found: {model_path}")

        if CORAL_EDGETPU:
            delegate = load_delegate('libedgetpu.so.1')
            self.interpreter = Interpreter(model_path,
                                           experimental_delegates=[delegate])
        else:
            cpu_path = model_path.replace('_edgetpu', '')
            self.interpreter = Interpreter(cpu_path)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        shape = self.input_details[0]['shape']
        self.input_h = 224
        self.input_w = 224
        if len(shape) == 4:
            if int(shape[1]) >= 32 and int(shape[2]) >= 32:
                self.input_h, self.input_w = int(shape[1]), int(shape[2])
            elif int(shape[2]) >= 32 and int(shape[3]) >= 32:
                self.input_h, self.input_w = int(shape[2]), int(shape[3])

        self.input_dtype = self.input_details[0]['dtype']
        self.embedding_dim = int(self.output_details[0]['shape'][-1])
        print(f"FaceEmbedder: {self.input_w}x{self.input_h} "
              f"dtype={self.input_dtype.__name__}, "
              f"output {self.embedding_dim}-D")

    def embed(self, face_crop):
        h, w = face_crop.shape[:2]
        if h == 0 or w == 0:
            return None

        # Pad small crops with a reflected border before upscaling so the
        # upscale doesn't stretch edge pixels. This mimics how the same
        # face looks if the detector had returned a slightly larger box.
        if w < 96 or h < 96:
            pad_x = max(0, (96 - w) // 2)
            pad_y = max(0, (96 - h) // 2)
            face_crop = cv2.copyMakeBorder(
                face_crop, pad_y, pad_y, pad_x, pad_x,
                cv2.BORDER_REFLECT_101,
            )

        # Interpolation depends on direction
        h2, w2 = face_crop.shape[:2]
        if self.input_w >= w2 and self.input_h >= h2:
            interp = cv2.INTER_CUBIC
        else:
            interp = cv2.INTER_AREA

        img = cv2.resize(face_crop, (self.input_w, self.input_h),
                         interpolation=interp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.input_dtype == np.uint8:
            img = img.astype(np.uint8)
        else:
            img = img.astype(np.float32)
            img = (img - 127.5) / 128.0

        img = np.expand_dims(img, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        vec = self.interpreter.tensor(self.output_details[0]['index'])().flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


def _augment(img):
    """Return 11 light variants of a face crop, including distance simulation."""
    variants = [img]

    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # Horizontal flip
    variants.append(cv2.flip(img, 1))

    # Brightness +/- 20
    for delta in (-20, 20):
        variants.append(np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8))

    # Rotations +/- 10 degrees
    for angle in (-10, 10):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        variants.append(cv2.warpAffine(img, M, (w, h),
                                       borderMode=cv2.BORDER_REFLECT_101))

    # Zoom out / zoom in
    for scale in (0.92, 1.08):
        M = cv2.getRotationMatrix2D(center, 0, scale)
        variants.append(cv2.warpAffine(img, M, (w, h),
                                       borderMode=cv2.BORDER_REFLECT_101))

    # --- Distance simulation ---
    # Downscale to 48x48 then back up. This produces the same soft, low-res
    # look the embedder sees when a face is detected far from the camera.
    small = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
    upscaled = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    variants.append(upscaled)

    # Gaussian blur to simulate softness
    variants.append(cv2.GaussianBlur(img, (5, 5), 1.2))

    return variants


def main():
    train_dir = config.FACE_TRAINING_DIR
    if not os.path.isdir(train_dir):
        print(f"No training directory at {train_dir}")
        return

    people = sorted([d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d))])
    if len(people) < 1:
        print("No enrolled people found. Run: python3 tracker_geminiV3.py --enroll NAME")
        return

    print(f"Found {len(people)} people: {people}")
    embedder = FaceEmbedder(config.MOBILEFACENET_MODEL)

    names = []
    centroids = []
    galleries = []
    stats_p10 = []
    stats_p50 = []
    stats_p90 = []

    for person in people:
        person_dir = os.path.join(train_dir, person)
        files = sorted([f for f in os.listdir(person_dir)
                        if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        if not files:
            print(f"  {person}: no crops, skipping")
            continue

        print(f"  {person}: processing {len(files)} crops (with augmentation)...")
        embs = []
        for fname in files:
            img = cv2.imread(os.path.join(person_dir, fname))
            if img is None:
                continue
            for variant in _augment(img):
                v = embedder.embed(variant)
                if v is not None:
                    embs.append(v)

        if len(embs) < 2:
            print(f"    -> only {len(embs)} valid embeddings, skipping")
            continue

        X = np.stack(embs)

        mu = X.mean(axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-9)

        sims_cent = X @ mu
        p10, p50, p90 = np.percentile(sims_cent, [10, 50, 90])

        sims_gallery = X @ X.T
        np.fill_diagonal(sims_gallery, -1.0)
        best_match = sims_gallery.max(axis=1)

        names.append(person)
        centroids.append(mu)
        galleries.append(X)
        stats_p10.append(float(p10))
        stats_p50.append(float(p50))
        stats_p90.append(float(p90))

        variants_per_crop = len(embs) // len(files) if files else 0
        print(f"    -> {len(embs)} embeddings "
              f"({len(files)} crops x {variants_per_crop} variants)")
        print(f"       centroid sim:  p10={p10:.3f}  p50={p50:.3f}  p90={p90:.3f}")
        print(f"       best-match:    min={best_match.min():.3f}  "
              f"mean={best_match.mean():.3f}  max={best_match.max():.3f}")

    if not names:
        print("No valid training data.")
        return

    centroids = np.stack(centroids)
    stats_p10 = np.array(stats_p10, dtype=np.float32)
    stats_p50 = np.array(stats_p50, dtype=np.float32)
    stats_p90 = np.array(stats_p90, dtype=np.float32)

    os.makedirs(os.path.dirname(config.SVM_MODEL_PATH), exist_ok=True)
    with open(config.SVM_MODEL_PATH, "wb") as f:
        pickle.dump({
            "model_type": "centroid_gallery",
            "names": names,
            "centroids": centroids,
            "galleries": galleries,
            "stats_p10": stats_p10,
            "stats_p50": stats_p50,
            "stats_p90": stats_p90,
        }, f)

    print(f"\nSaved centroid+gallery classifier to {config.SVM_MODEL_PATH}")
    print(f"Enrolled people: {names}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nFATAL: {e}\n")
        sys.exit(1)
